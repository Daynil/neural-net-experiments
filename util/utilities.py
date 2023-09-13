from datetime import datetime
from statistics import mean
from time import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def format_timer_human(seconds: float):
    """
    Format timer results from raw seconds to human readable min/sec
    """
    if int(seconds // 60) == 0:
        return f"{int(seconds % 60)} sec"
    else:
        return f"{int(seconds // 60)} min {int(seconds % 60)} sec"


class Timer:
    """
    Usage:
    timer = Timer()
    timer.show_elapsed_total()
    """

    elapsed_loop_times: List[float] = []

    def __init__(self) -> None:
        self.reset_timer()

    def reset_timer(self):
        self.start_time_s = time()
        self.start_date_time = datetime.today()
        self.last_time_s = time()

    def show_elapsed_total(self, raw: bool = False):
        """
        Get elapsed time since last check
        Args:
            raw: min/sec string default, float seconds if raw
        """
        elapsed = time() - self.start_time_s
        self.last_time_s = time()
        if raw:
            return elapsed
        else:
            return format_timer_human(elapsed)

    def show_elapsed_last(self, raw: bool = False):
        """
        Get elapsed time since last check
        Args:
            raw: min/sec string default, float seconds if raw
        """
        elapsed = time() - self.last_time_s
        self.last_time_s = time()
        if raw:
            return elapsed
        else:
            return format_timer_human(elapsed)

    def track_elapsed_loop(self):
        """
        Tracks the time elapsed since the last call.
        Use for getting averge times for loops.

        Returns:
            Last loop time in seconds
        """

        elapsed = time() - self.last_time_s
        self.last_time_s = time()
        self.elapsed_loop_times.append(elapsed)
        return elapsed

    def elapsed_loop_average(self):
        return mean(self.elapsed_loop_times)

    def fmt_elapsed_loop_average(self):
        return format_timer_human(self.elapsed_loop_average())


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(y[0]) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # multi-class
        c = y.argmax(1).to("cpu")
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary
        c = y

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)  # type: ignore
    plt.scatter(X[:, 0], X[:, 1], c=c, s=40, cmap=plt.cm.RdYlBu)  # type: ignore
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
