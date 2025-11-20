from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Logger:
    """
    Handles logging to TensorBoard and console.
    """
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.console_logs = {}

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)

    def log_metrics(self, metrics: dict, step: int, prefix: str = ''):
        """Log a dictionary of metrics."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}{k}", v, step)
            self.console_logs[f"{prefix}{k}"] = v

    def print_console_logs(self, step: int):
        """Prints the stored console logs."""
        log_str = f"Step: {step} | "
        for k, v in self.console_logs.items():
            if k.endswith('loss'):
                log_str += f"{k}: {v:.4f} | "
            else:
                log_str += f"{k}: {v:.2f} | "
        print(log_str)
        self.console_logs = {} # Clear after printing

    def close(self):
        self.writer.close()