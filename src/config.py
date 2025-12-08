import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import torch
import json
from dataclasses import asdict


ROOT = Path(__file__).resolve().parent.parent

logs_dir = ROOT/"resources/logs"
inputs_dir = ROOT / "resources/data/inputs"
outputs_dir = ROOT / "resources/data/outputs"
dataset_dir = inputs_dir / "dataset1"

for dir in [logs_dir, inputs_dir, dataset_dir]:
    Path(dir).mkdir(exist_ok=True)

dataset_filename = "SPY Options 2010-2023 EOD.csv"
# dataset_filename = "spy-daily-eod-options-quotes-2020-2022.csv"

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    opt_type: str = "puts" # calls or puts
    epochs: int = 100 # 100
    lr: float = 5e-6# 5e-6 # learning rate
    # train_frac: float = 0.8 # train/val split fraction
    patience: int = 10 # early stopping patience
    is_standardize: bool = True
    is_lr_scheduler: bool = True
    is_target_log: bool = False
    DTE_range: tuple[int, int] | None = (2, 30)  # days to expiration
    K_dist_pct_max: float | None = 5 # max strike distance percentage
    date_range: tuple[str, str] | None = None # ("2010-01-01", "2018-12-31")
    volume_pctl_thresh: float = 0.1   # percentile
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def save(self, path: Path):
        """Save THIS instance's config to file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)



# Setup logging
class MyLogger:
    def __init__(self):
        log_filename = logs_dir / f'{datetime.now().strftime("%Y-%m-%d")}_training_report.txt'
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
