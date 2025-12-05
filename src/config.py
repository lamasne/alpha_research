import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import torch

ROOT = Path(__file__).resolve().parent.parent

tracking_dir = ROOT/"resources/logs/train_tracking"


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    opt_type: str = "puts" # calls or puts
    epochs: int = 50
    lr: float = 5e-4 # learning rate
    train_frac: float = 0.8 # train/val split fraction
    patience: int = 10 # early stopping patience
    is_standardize: bool = False
    is_lr_scheduler: bool = True
    is_target_log: bool = False
    DET_range: tuple[int, int] | None = None
    K_dist_pct_max: float | None = None
    date_range: tuple[str, str] | None = None
    volume_pctl_thresh: float = 0.1   # percentile
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# Setup logging
class MyLogger:
    def __init__(self):
        log_filename = ROOT / 'resources/logs' / f'{datetime.now().strftime("%Y-%m-%d")}_training_report.txt'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(message)s',
            datefmt='%H:%M:%S',  # Only hours:minutes:seconds in log
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler() # Also log to console
            ]
        )
        self._logger = logging.getLogger()
    
    def __getattr__(self, name):
        # Forward all other calls to the real logger
        return getattr(self._logger, name)
    
    def blank_line(self):
        self._logger.info("")

full_logger = MyLogger()
