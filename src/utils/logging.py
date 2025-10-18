import logging
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """Structured logging for ML experiments"""

    def __init__(self, experiment_name: str, log_dir: str = 'logs'):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)

        log_file = self.log_dir / f"{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, message: str):
        self.logger.info(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)
