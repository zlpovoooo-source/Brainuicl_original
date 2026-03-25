import logging
import os
from datetime import datetime
import sys


def setup_logging(params=None, log_dir: str = "logs", level: int = logging.INFO) -> str:
    """Configure root logger with CMuST-style filename and format."""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if params is not None:
        dataset = getattr(params, "dataset", "run")
        algorithm = getattr(params, "algorithm", "default")
        seed = getattr(params, "seed", "na")
        log_name = f"{dataset}_{algorithm}_seed{seed}_{ts}.log"
    else:
        log_name = f"run_seedna_{ts}.log"
    log_path = os.path.join(log_dir, log_name)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    if params is not None:
        setattr(params, "log_path", log_path)
    return log_path
