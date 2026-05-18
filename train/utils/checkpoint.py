import signal
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, save_dir: str, save_interval: int = 10000, latest_interval: int = 1000, keep_last_n: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.latest_interval = latest_interval
        self.keep_last_n = keep_last_n
        self._interrupted = False

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        print("\n收到中断信号，正在保存 checkpoint...")
        self._interrupted = True

    @property
    def interrupted(self) -> bool:
        return self._interrupted

    def save(self, state: dict[str, Any], iteration: int, is_best: bool = False) -> None:
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': state['model'].state_dict(),
            'optimizer_state_dict': state['optimizer'].state_dict(),
            'scheduler_state_dict': state['scheduler'].state_dict() if state.get('scheduler') else None,
            'best_loss': state.get('best_loss', float('inf')),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        if 'disc_optimizer' in state:
            checkpoint['disc_optimizer_state_dict'] = state['disc_optimizer'].state_dict()

        # Save periodic checkpoint
        filepath = self.save_dir / f"checkpoint_iter_{iteration:07d}.pth"
        torch.save(checkpoint, filepath)
        print(f"Checkpoint 已保存: {filepath}")

        # Save latest
        latest_path = self.save_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.save_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint 已更新: {best_path}")

        # Cleanup old checkpoints
        self._cleanup()

    def save_latest(self, state: dict[str, Any], iteration: int) -> None:
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': state['model'].state_dict(),
            'optimizer_state_dict': state['optimizer'].state_dict(),
            'scheduler_state_dict': state['scheduler'].state_dict() if state.get('scheduler') else None,
            'best_loss': state.get('best_loss', float('inf')),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        if 'disc_optimizer' in state:
            checkpoint['disc_optimizer_state_dict'] = state['disc_optimizer'].state_dict()

        latest_path = self.save_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)

    def load(self, filepath: str | None = None) -> dict[str, Any] | None:
        if filepath == 'latest':
            filepath = self.save_dir / "checkpoint_latest.pth"
        elif filepath == 'best':
            filepath = self.save_dir / "checkpoint_best.pth"
        elif filepath is None:
            filepath = self.save_dir / "checkpoint_latest.pth"
        else:
            filepath = Path(filepath)

        if not filepath.exists():
            return None

        print(f"加载 checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        return checkpoint

    def resume(self, checkpoint: dict, model, optimizer, scheduler=None, disc_optimizer=None) -> int:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if disc_optimizer and checkpoint.get('disc_optimizer_state_dict'):
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])

        if checkpoint.get('rng_state') is not None:
            torch.set_rng_state(checkpoint['rng_state'])

        if checkpoint.get('cuda_rng_state') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

        iteration = checkpoint['iteration']
        print(f"从 iteration {iteration} 恢复训练")
        return iteration

    def _cleanup(self):
        checkpoints = sorted(self.save_dir.glob("checkpoint_iter_*.pth"))
        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                ckpt.unlink()

    def should_save(self, iteration: int) -> bool:
        return iteration % self.save_interval == 0

    def should_save_latest(self, iteration: int) -> bool:
        return iteration % self.latest_interval == 0
