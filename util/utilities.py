from datetime import datetime
from statistics import mean
from time import time
from typing import List


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
