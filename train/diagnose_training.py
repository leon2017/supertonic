"""诊断训练瓶颈：测量数据加载、前向、反向、优化器各阶段耗时。"""
import argparse
import time
import yaml
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from train_autoencoder import AudioDataset
from models.autoencoder import SpeechAutoencoder
from models.discriminator import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from losses.reconstruction import MultiResolutionMelLoss
from losses.adversarial import generator_loss, discriminator_loss, feature_matching_loss


def diagnose(config_path: str, num_iters: int = 20):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Model
    model = SpeechAutoencoder(**config['model']).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    disc_count = (sum(p.numel() for p in mpd.parameters()) + sum(p.numel() for p in mrd.parameters())) / 1e6
    print(f"Generator: {param_count:.2f}M params")
    print(f"Discriminator: {disc_count:.2f}M params")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    disc_optimizer = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()), lr=2e-4, weight_decay=0.01
    )

    mel_loss_fn = MultiResolutionMelLoss().to(device)
    scaler = GradScaler('cuda') if config['training']['fp16'] else None

    # Dataset
    dataset = AudioDataset(
        config['data']['train_dir'],
        config['data']['segment_length'],
        config['data']['sample_rate']
    )
    dataloader = DataLoader(
        dataset, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    print(f"Dataset: {len(dataset)} samples")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Batches per epoch: {len(dataset) // config['training']['batch_size']}")
    print()

    model.train()
    mpd.train()
    mrd.train()

    data_iter = iter(dataloader)
    weights = config['loss_weights']
    accum_steps = config['training']['gradient_accumulation']

    timings = {'data': [], 'forward': [], 'gen_backward': [], 'disc_forward': [], 'disc_backward': [], 'optim': []}

    print(f"Running {num_iters} iterations for diagnosis...")
    print("-" * 60)

    for i in range(num_iters):
        torch.cuda.synchronize()

        # Data loading
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        audio = batch['audio'].to(device)
        mel = batch['mel'].to(device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['data'].append(t1 - t0)

        # Forward pass (generator)
        with autocast('cuda', enabled=config['training']['fp16']):
            wav_pred, latent = model(mel)
            min_len = min(wav_pred.shape[1], audio.shape[1])
            wav_pred = wav_pred[:, :min_len]
            audio_target = audio[:, :min_len]
            recon_loss = mel_loss_fn(wav_pred, audio_target)

            mpd_fake_out, mpd_fake_feat = mpd(wav_pred)
            mrd_fake_out, mrd_fake_feat = mrd(wav_pred)
            with torch.no_grad():
                mpd_real_out, mpd_real_feat = mpd(audio_target)
                mrd_real_out, mrd_real_feat = mrd(audio_target)

            gen_loss = generator_loss(mpd_fake_out) + generator_loss(mrd_fake_out)
            fm_loss = feature_matching_loss(mpd_real_feat, mpd_fake_feat) + feature_matching_loss(mrd_real_feat, mrd_fake_feat)
            total_gen_loss = (
                weights['reconstruction'] * recon_loss +
                weights['adversarial'] * gen_loss +
                weights['feature_matching'] * fm_loss
            ) / accum_steps

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        timings['forward'].append(t2 - t1)

        # Generator backward
        if scaler:
            scaler.scale(total_gen_loss).backward()
        else:
            total_gen_loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        timings['gen_backward'].append(t3 - t2)

        # Discriminator forward
        with autocast('cuda', enabled=config['training']['fp16']):
            mpd_real_out, _ = mpd(audio_target.detach())
            mrd_real_out, _ = mrd(audio_target.detach())
            mpd_fake_out, _ = mpd(wav_pred.detach())
            mrd_fake_out, _ = mrd(wav_pred.detach())
            disc_loss = (
                discriminator_loss(mpd_real_out, mpd_fake_out) +
                discriminator_loss(mrd_real_out, mrd_fake_out)
            ) / accum_steps
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        timings['disc_forward'].append(t4 - t3)

        # Discriminator backward
        if scaler:
            scaler.scale(disc_loss).backward()
        else:
            disc_loss.backward()
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        timings['disc_backward'].append(t5 - t4)

        # Optimizer step (every accum_steps)
        if (i + 1) % accum_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(mrd.parameters()), 1.0)
                scaler.step(disc_optimizer)
                scaler.update()
            else:
                optimizer.step()
                disc_optimizer.step()
            optimizer.zero_grad()
            disc_optimizer.zero_grad()
            torch.cuda.synchronize()
            t6 = time.perf_counter()
            timings['optim'].append(t6 - t5)
        else:
            timings['optim'].append(0)

        total = t5 - t0 + timings['optim'][-1]
        mem = torch.cuda.max_memory_allocated() / 1024**3

        print(f"[{i+1:3d}/{num_iters}] "
              f"data={timings['data'][-1]:.3f}s "
              f"fwd={timings['forward'][-1]:.3f}s "
              f"gen_bwd={timings['gen_backward'][-1]:.3f}s "
              f"disc_fwd={timings['disc_forward'][-1]:.3f}s "
              f"disc_bwd={timings['disc_backward'][-1]:.3f}s "
              f"optim={timings['optim'][-1]:.3f}s "
              f"total={total:.3f}s "
              f"peak_mem={mem:.2f}GB "
              f"recon={recon_loss.item():.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY (excluding first 3 warmup iterations)")
    print("=" * 60)
    for key in timings:
        vals = timings[key][3:]
        if vals:
            avg = np.mean(vals)
            print(f"  {key:15s}: avg={avg:.3f}s  min={min(vals):.3f}s  max={max(vals):.3f}s")

    total_avg = sum(np.mean(timings[k][3:]) for k in timings if timings[k][3:])
    print(f"\n  {'TOTAL':15s}: avg={total_avg:.3f}s per iteration")
    print(f"  Estimated time to iteration 1000: {total_avg * 1000 / 3600:.1f} hours")
    print(f"  Estimated time to iteration 10000: {total_avg * 10000 / 3600:.1f} hours")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/autoencoder.yaml')
    parser.add_argument('--num_iters', type=int, default=20)
    args = parser.parse_args()
    diagnose(args.config, args.num_iters)
