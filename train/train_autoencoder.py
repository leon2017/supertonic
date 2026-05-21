import argparse
import sys
import time
import yaml
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np

from models.autoencoder import SpeechAutoencoder
from models.discriminator import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from losses.reconstruction import MultiResolutionMelLoss
from losses.adversarial import generator_loss, discriminator_loss, feature_matching_loss
from utils.checkpoint import CheckpointManager


class AudioDataset(Dataset):
    def __init__(self, data_dir: str, segment_length: int = 44100, sample_rate: int = 44100):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.sample_rate = sample_rate

        import pandas as pd
        metadata_path = self.data_dir / 'metadata.csv'
        self.metadata = pd.read_csv(metadata_path)
        self._preload_paths()

    def _preload_paths(self):
        """Precompute and validate all file paths to catch errors early."""
        self._audio_paths = []
        self._mel_paths = []
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            audio_path = self.data_dir.parent / row['audio_path']
            mel_path = self.data_dir.parent / row['mel_path']
            self._audio_paths.append(str(audio_path))
            self._mel_paths.append(str(mel_path))

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        mel = np.load(self._mel_paths[idx])

        import soundfile as sf
        audio, sr = sf.read(self._audio_paths[idx])

        # Random crop to segment_length
        if len(audio) > self.segment_length:
            start = np.random.randint(0, len(audio) - self.segment_length)
            audio = audio[start:start + self.segment_length]
            mel_start = start // 512
            mel_len = self.segment_length // 512
            mel = mel[:, mel_start:mel_start + mel_len]
        else:
            # Pad if shorter
            audio = np.pad(audio, (0, self.segment_length - len(audio)))
            mel_len = self.segment_length // 512
            mel = np.pad(mel, ((0, 0), (0, max(0, mel_len - mel.shape[1]))))
            mel = mel[:, :mel_len]

        return {
            'audio': torch.FloatTensor(audio),
            'mel': torch.FloatTensor(mel),
        }


def train(config_path: str, resume: str | None = None):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Model
    model = SpeechAutoencoder(**config['model']).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    gen_params = sum(p.numel() for p in model.parameters()) / 1e6
    disc_params = (sum(p.numel() for p in mpd.parameters()) + sum(p.numel() for p in mrd.parameters())) / 1e6
    print(f"模型参数量: {gen_params:.2f}M")
    print(f"判别器参数量: {disc_params:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    disc_optimizer = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=config['training']['disc_learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300000, gamma=0.5)

    # Loss
    mel_loss_fn = MultiResolutionMelLoss().to(device)

    # Scaler for fp16
    use_fp16 = config['training']['fp16'] and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_fp16 else None

    # Checkpoint
    ckpt_manager = CheckpointManager(**config['checkpoint'])

    # Resume
    start_iteration = 0
    best_loss = float('inf')

    if resume:
        checkpoint = ckpt_manager.load(resume)
        if checkpoint:
            start_iteration = ckpt_manager.resume(
                checkpoint, model, optimizer, scheduler, disc_optimizer
            )
            best_loss = checkpoint.get('best_loss', float('inf'))

    # Dataset
    num_workers = config['training'].get('num_workers', 0)
    if sys.platform == 'win32' and num_workers > 0:
        print(f"警告: Windows 下 num_workers={num_workers} 可能导致 DataLoader 死锁，已自动设为 0")
        num_workers = 0

    dataset = AudioDataset(
        config['data']['train_dir'],
        config['data']['segment_length'],
        config['data']['sample_rate']
    )
    dataloader = DataLoader(
        dataset, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda'),
        drop_last=True, persistent_workers=(num_workers > 0)
    )

    print(f"数据集: {len(dataset)} 样本, {len(dataset) // config['training']['batch_size']} batches/epoch")

    # Sanity check: load one batch and run one forward pass
    print("\n运行 sanity check...")
    try:
        test_batch = next(iter(dataloader))
        test_mel = test_batch['mel'].to(device)
        test_audio = test_batch['audio'].to(device)
        with torch.no_grad(), autocast('cuda', enabled=use_fp16):
            test_wav, test_latent = model(test_mel)
        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  前向 pass 成功, 峰值显存: {peak_mem:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        del test_batch, test_mel, test_audio, test_wav, test_latent
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        print("  Sanity check 通过")
    except Exception as e:
        print(f"  Sanity check 失败: {e}")
        raise

    # Training loop
    accum_steps = config['training']['gradient_accumulation']
    log_interval = config['logging']['log_interval']
    total_iterations = config['training']['iterations']

    model.train()
    mpd.train()
    mrd.train()

    data_iter = iter(dataloader)

    print(f"\n开始训练 (iteration {start_iteration} → {total_iterations})")
    print(f"Batch size: {config['training']['batch_size']}, Accumulation: {accum_steps}")
    print(f"等效 batch size: {config['training']['batch_size'] * accum_steps}")
    sys.stdout.flush()

    iter_start_time = time.perf_counter()

    for iteration in range(start_iteration, total_iterations):
        # Check interrupt
        if ckpt_manager.interrupted:
            state = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
                     'disc_optimizer': disc_optimizer, 'best_loss': best_loss}
            ckpt_manager.save(state, iteration)
            print(f"训练已暂停于 iteration {iteration}")
            return

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        audio = batch['audio'].to(device)
        mel = batch['mel'].to(device)

        # --- Generator step ---
        with autocast('cuda', enabled=use_fp16):
            wav_pred, latent = model(mel)

            # Match lengths
            min_len = min(wav_pred.shape[1], audio.shape[1])
            wav_pred = wav_pred[:, :min_len]
            audio_target = audio[:, :min_len]

            # Reconstruction loss
            recon_loss = mel_loss_fn(wav_pred, audio_target)

            # Adversarial loss
            mpd_fake_out, mpd_fake_feat = mpd(wav_pred)
            mrd_fake_out, mrd_fake_feat = mrd(wav_pred)

            with torch.no_grad():
                mpd_real_out, mpd_real_feat = mpd(audio_target)
                mrd_real_out, mrd_real_feat = mrd(audio_target)

            gen_loss = generator_loss(mpd_fake_out) + generator_loss(mrd_fake_out)
            fm_loss = feature_matching_loss(mpd_real_feat, mpd_fake_feat) + feature_matching_loss(mrd_real_feat, mrd_fake_feat)

            # Total generator loss
            weights = config['loss_weights']
            total_gen_loss = (
                weights['reconstruction'] * recon_loss +
                weights['adversarial'] * gen_loss +
                weights['feature_matching'] * fm_loss
            ) / accum_steps

        if scaler:
            scaler.scale(total_gen_loss).backward()
        else:
            total_gen_loss.backward()

        # --- Discriminator step ---
        with autocast('cuda', enabled=use_fp16):
            mpd_real_out, _ = mpd(audio_target.detach())
            mrd_real_out, _ = mrd(audio_target.detach())
            mpd_fake_out, _ = mpd(wav_pred.detach())
            mrd_fake_out, _ = mrd(wav_pred.detach())

            disc_loss = (
                discriminator_loss(mpd_real_out, mpd_fake_out) +
                discriminator_loss(mrd_real_out, mrd_fake_out)
            ) / accum_steps

        if scaler:
            scaler.scale(disc_loss).backward()
        else:
            disc_loss.backward()

        # Gradient accumulation step
        if (iteration + 1) % accum_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(mrd.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.step(disc_optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(mrd.parameters()), 1.0)
                disc_optimizer.step()

            optimizer.zero_grad()
            disc_optimizer.zero_grad()
            scheduler.step()

        # Logging — first 10 iterations always log, then follow log_interval
        if iteration < start_iteration + 10 or (iteration + 1) % log_interval == 0:
            elapsed = time.perf_counter() - iter_start_time
            iter_speed = (iteration - start_iteration + 1) / elapsed
            mem_str = ""
            if device.type == 'cuda':
                mem_str = f", VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
            print(f"[{iteration+1}/{total_iterations}] "
                  f"recon: {recon_loss.item():.4f}, "
                  f"gen: {gen_loss.item():.4f}, "
                  f"fm: {fm_loss.item():.4f}, "
                  f"disc: {disc_loss.item() * accum_steps:.4f}, "
                  f"speed: {iter_speed:.2f} it/s{mem_str}")
            sys.stdout.flush()

        # Save latest
        if ckpt_manager.should_save_latest(iteration + 1):
            state = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
                     'disc_optimizer': disc_optimizer, 'best_loss': best_loss}
            ckpt_manager.save_latest(state, iteration + 1)

        # Save periodic
        if ckpt_manager.should_save(iteration + 1):
            current_loss = recon_loss.item()
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss

            state = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
                     'disc_optimizer': disc_optimizer, 'best_loss': best_loss}
            ckpt_manager.save(state, iteration + 1, is_best)

    print("\n训练完成！")


def main():
    parser = argparse.ArgumentParser(description='训练 Speech Autoencoder')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练 (latest/best/path)')
    args = parser.parse_args()

    train(args.config, args.resume)


if __name__ == '__main__':
    main()
