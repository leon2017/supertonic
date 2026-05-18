import argparse
import json
import yaml
import torch
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd

from models.text_encoder import TextEncoder
from models.reference_encoder import ReferenceEncoder
from models.vf_estimator import VFEstimator
from models.autoencoder import LatentEncoder
from losses.flow_matching import flow_matching_loss, compute_target_velocity, sample_zt
from utils.checkpoint import CheckpointManager


class TextToLatentDataset(Dataset):
    def __init__(self, data_dir: str, unicode_indexer_path: str, n_mels: int = 228):
        self.data_dir = Path(data_dir)
        self.n_mels = n_mels

        with open(unicode_indexer_path, 'r', encoding='utf-8') as f:
            self.unicode_indexer = json.load(f)

        metadata_path = self.data_dir / 'metadata.csv'
        self.metadata = pd.read_csv(metadata_path)

    def text_to_ids(self, text: str) -> list[int]:
        ids = []
        for char in text:
            cp = str(ord(char))
            if cp in self.unicode_indexer:
                ids.append(self.unicode_indexer[cp])
            else:
                ids.append(1)  # <unk>
        return ids

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        # Load text
        text_path = self.data_dir.parent / row['text_path']
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        text_ids = self.text_to_ids(text)

        # Load mel
        mel_path = self.data_dir.parent / row['mel_path']
        mel = np.load(mel_path)

        return {
            'text_ids': torch.LongTensor(text_ids),
            'mel': torch.FloatTensor(mel),
            'duration': row['duration'],
        }


def collate_fn(batch):
    max_text_len = max(item['text_ids'].shape[0] for item in batch)
    max_mel_len = max(item['mel'].shape[1] for item in batch)

    text_ids = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    text_mask = torch.zeros(len(batch), 1, max_text_len)
    mels = torch.zeros(len(batch), batch[0]['mel'].shape[0], max_mel_len)
    mel_mask = torch.zeros(len(batch), 1, max_mel_len)
    durations = []

    for i, item in enumerate(batch):
        tl = item['text_ids'].shape[0]
        ml = item['mel'].shape[1]
        text_ids[i, :tl] = item['text_ids']
        text_mask[i, 0, :tl] = 1.0
        mels[i, :, :ml] = item['mel']
        mel_mask[i, 0, :ml] = 1.0
        durations.append(item['duration'])

    return {
        'text_ids': text_ids,
        'text_mask': text_mask,
        'mel': mels,
        'mel_mask': mel_mask,
        'duration': torch.FloatTensor(durations),
    }


def train(config_path: str, autoencoder_checkpoint: str, resume: str | None = None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # Load frozen autoencoder encoder
    ae_encoder = LatentEncoder(
        mel_channels=config['data']['n_mels'],
        latent_dim=config['model']['latent_dim']
    ).to(device)

    ae_ckpt = torch.load(autoencoder_checkpoint, map_location=device, weights_only=False)
    ae_state = {k.replace('encoder.', ''): v for k, v in ae_ckpt['model_state_dict'].items() if k.startswith('encoder.')}
    ae_encoder.load_state_dict(ae_state)
    ae_encoder.eval()
    for p in ae_encoder.parameters():
        p.requires_grad = False
    print("Autoencoder encoder 已加载并冻结")

    # Models
    text_encoder = TextEncoder(
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['char_embedding_dim'],
        channels=config['model']['text_channels'],
        num_heads=config['model']['text_num_heads'],
        use_checkpoint=config['model']['use_checkpoint']
    ).to(device)

    ref_encoder = ReferenceEncoder(
        mel_channels=config['data']['n_mels'],
        channels=config['model']['text_channels']
    ).to(device)

    Kc = config['model']['temporal_compression']
    compressed_dim = config['model']['latent_dim'] * Kc  # 24 * 6 = 144

    vf_estimator = VFEstimator(
        latent_dim=compressed_dim,
        channels=config['model']['vf_channels'],
        intermediate=config['model']['vf_intermediate'],
        num_blocks=config['model']['vf_num_blocks'],
        text_channels=config['model']['text_channels'],
        use_checkpoint=config['model']['use_checkpoint']
    ).to(device)

    total_params = (
        sum(p.numel() for p in text_encoder.parameters())
        + sum(p.numel() for p in ref_encoder.parameters())
        + sum(p.numel() for p in vf_estimator.parameters())
    )
    print(f"Text-to-Latent 参数量: {total_params / 1e6:.2f}M")

    # Optimizer
    all_params = (
        list(text_encoder.parameters())
        + list(ref_encoder.parameters())
        + list(vf_estimator.parameters())
    )
    optimizer = torch.optim.AdamW(
        all_params,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300000, gamma=0.5)

    scaler = GradScaler('cuda') if config['training']['fp16'] else None

    ckpt_manager = CheckpointManager(**config['checkpoint'])

    # Resume
    start_iteration = 0
    best_loss = float('inf')

    if resume:
        checkpoint_data = ckpt_manager.load(resume)
        if checkpoint_data:
            text_encoder.load_state_dict(checkpoint_data['text_encoder_state_dict'])
            ref_encoder.load_state_dict(checkpoint_data['ref_encoder_state_dict'])
            vf_estimator.load_state_dict(checkpoint_data['vf_estimator_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            if checkpoint_data.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            start_iteration = checkpoint_data['iteration']
            best_loss = checkpoint_data.get('best_loss', float('inf'))
            print(f"从 iteration {start_iteration} 恢复训练")

    # Dataset
    dataset = TextToLatentDataset(
        config['data']['train_dir'],
        config['data']['unicode_indexer'],
        config['data']['n_mels']
    )
    dataloader = DataLoader(
        dataset, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True,
        drop_last=True, collate_fn=collate_fn
    )

    # Training loop
    accum_steps = config['training']['gradient_accumulation']
    cfg_dropout = config['training']['cfg_dropout']
    total_iterations = config['training']['iterations']

    text_encoder.train()
    ref_encoder.train()
    vf_estimator.train()

    data_iter = iter(dataloader)

    print(f"\n开始训练 (iteration {start_iteration} → {total_iterations})")

    for iteration in range(start_iteration, total_iterations):
        if ckpt_manager.interrupted:
            _save_ttl_checkpoint(
                ckpt_manager, text_encoder, ref_encoder, vf_estimator,
                optimizer, scheduler, best_loss, iteration
            )
            print(f"训练已暂停于 iteration {iteration}")
            return

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        text_ids = batch['text_ids'].to(device)
        text_mask = batch['text_mask'].to(device)
        mel = batch['mel'].to(device)
        mel_mask = batch['mel_mask'].to(device)

        with autocast('cuda', enabled=config['training']['fp16']):
            # Encode mel to latent (frozen)
            with torch.no_grad():
                z1 = ae_encoder(mel)  # (B, 24, T)

            # Temporal compression: (B, 24, T) -> (B, 144, T/6)
            B, C, T = z1.shape
            T_compressed = T // Kc
            z1 = z1[:, :, :T_compressed * Kc]
            z1 = z1.reshape(B, C * Kc, T_compressed)

            # Latent mask
            latent_mask = mel_mask[:, :, :T_compressed]

            # Reference encoding
            ref_emb = ref_encoder(mel)  # (B, C, num_tokens)

            # CFG dropout: randomly zero out reference to enable classifier-free guidance
            if np.random.random() < cfg_dropout:
                ref_emb = torch.zeros_like(ref_emb)

            # Text encoding
            text_emb = text_encoder(text_ids, ref_emb, text_mask)  # (B, C, L)

            # Flow matching
            t = torch.rand(B, device=device)
            z0 = torch.randn_like(z1)
            zt = sample_zt(z0, z1, t)

            target_velocity = compute_target_velocity(z1, z0)
            predicted_velocity = vf_estimator(zt, text_emb, ref_emb, t, latent_mask)

            loss = flow_matching_loss(predicted_velocity, target_velocity, latent_mask) / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (iteration + 1) % accum_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

        # Logging
        if (iteration + 1) % config['logging']['log_interval'] == 0:
            print(f"[{iteration+1}/{total_iterations}] flow_loss: {loss.item() * accum_steps:.6f}")

        # Save latest checkpoint
        if ckpt_manager.should_save_latest(iteration + 1):
            _save_ttl_checkpoint(
                ckpt_manager, text_encoder, ref_encoder, vf_estimator,
                optimizer, scheduler, best_loss, iteration + 1, latest_only=True
            )

        # Save periodic checkpoint
        if ckpt_manager.should_save(iteration + 1):
            current_loss = loss.item() * accum_steps
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss
            _save_ttl_checkpoint(
                ckpt_manager, text_encoder, ref_encoder, vf_estimator,
                optimizer, scheduler, best_loss, iteration + 1, is_best=is_best
            )

    print("\n训练完成！")


def _save_ttl_checkpoint(
    ckpt_manager: CheckpointManager,
    text_encoder: TextEncoder,
    ref_encoder: ReferenceEncoder,
    vf_estimator: VFEstimator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_loss: float,
    iteration: int,
    latest_only: bool = False,
    is_best: bool = False
) -> None:
    checkpoint_data = {
        'iteration': iteration,
        'text_encoder_state_dict': text_encoder.state_dict(),
        'ref_encoder_state_dict': ref_encoder.state_dict(),
        'vf_estimator_state_dict': vf_estimator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    latest_path = ckpt_manager.save_dir / "checkpoint_latest.pth"
    torch.save(checkpoint_data, latest_path)

    if not latest_only:
        periodic_path = ckpt_manager.save_dir / f"checkpoint_iter_{iteration:07d}.pth"
        torch.save(checkpoint_data, periodic_path)
        print(f"Checkpoint 已保存: {periodic_path}")

        if is_best:
            best_path = ckpt_manager.save_dir / "checkpoint_best.pth"
            torch.save(checkpoint_data, best_path)
            print("Best checkpoint 已更新")


def main():
    parser = argparse.ArgumentParser(description='训练 Text-to-Latent')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True, help='Autoencoder checkpoint 路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练 (latest/best/path)')
    args = parser.parse_args()

    train(args.config, args.autoencoder_checkpoint, args.resume)


if __name__ == '__main__':
    main()
