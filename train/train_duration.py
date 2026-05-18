import argparse
import json
import yaml
import torch
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd

from models.duration_predictor import DurationPredictor
from utils.checkpoint import CheckpointManager


class DurationDataset(Dataset):
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
                ids.append(1)
        return ids

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]

        text_path = self.data_dir.parent / row['text_path']
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        text_ids = self.text_to_ids(text)
        mel_path = self.data_dir.parent / row['mel_path']
        mel = np.load(mel_path)

        return {
            'text_ids': torch.LongTensor(text_ids),
            'mel': torch.FloatTensor(mel),
            'duration': torch.FloatTensor([row['duration']]),
        }


def collate_fn(batch):
    max_text_len = max(item['text_ids'].shape[0] for item in batch)
    max_mel_len = max(item['mel'].shape[1] for item in batch)

    text_ids = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    text_mask = torch.zeros(len(batch), 1, max_text_len)
    mels = torch.zeros(len(batch), batch[0]['mel'].shape[0], max_mel_len)
    durations = torch.zeros(len(batch))

    for i, item in enumerate(batch):
        tl = item['text_ids'].shape[0]
        ml = item['mel'].shape[1]
        text_ids[i, :tl] = item['text_ids']
        text_mask[i, 0, :tl] = 1.0
        mels[i, :, :ml] = item['mel']
        durations[i] = item['duration'][0]

    return {'text_ids': text_ids, 'text_mask': text_mask, 'mel': mels, 'duration': durations}


def train(config_path: str, resume: str | None = None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = DurationPredictor(**config['model']).to(device)
    print(f"Duration Predictor 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5)

    scaler = GradScaler('cuda') if config['training']['fp16'] else None
    ckpt_manager = CheckpointManager(**config['checkpoint'])

    start_iteration = 0
    best_loss = float('inf')

    if resume:
        checkpoint = ckpt_manager.load(resume)
        if checkpoint:
            start_iteration = ckpt_manager.resume(checkpoint, model, optimizer, scheduler)
            best_loss = checkpoint.get('best_loss', float('inf'))

    dataset = DurationDataset(config['data']['train_dir'], config['data']['unicode_indexer'], config['data']['n_mels'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    accum_steps = config['training']['gradient_accumulation']
    total_iterations = config['training']['iterations']

    model.train()
    data_iter = iter(dataloader)

    print(f"\n开始训练 (iteration {start_iteration} → {total_iterations})")

    for iteration in range(start_iteration, total_iterations):
        if ckpt_manager.interrupted:
            state = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler, 'best_loss': best_loss}
            ckpt_manager.save(state, iteration)
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
        target_duration = batch['duration'].to(device)

        with autocast('cuda', enabled=config['training']['fp16']):
            pred_duration = model(text_ids, mel, text_mask)
            loss = torch.abs(pred_duration - target_duration).mean() / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (iteration + 1) % accum_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if (iteration + 1) % config['logging']['log_interval'] == 0:
            print(f"[{iteration+1}/{total_iterations}] duration_loss: {loss.item() * accum_steps:.6f}")

        if ckpt_manager.should_save_latest(iteration + 1):
            state = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler, 'best_loss': best_loss}
            ckpt_manager.save_latest(state, iteration + 1)

        if ckpt_manager.should_save(iteration + 1):
            current_loss = loss.item() * accum_steps
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss
            state = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler, 'best_loss': best_loss}
            ckpt_manager.save(state, iteration + 1, is_best)

    print("\n训练完成！")


def main():
    parser = argparse.ArgumentParser(description='训练 Duration Predictor')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练 (latest/best/path)')
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == '__main__':
    main()
