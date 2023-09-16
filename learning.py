from dataclasses import dataclass
from datetime import datetime
from time import sleep
from typing import Any, Callable, Generic, Literal, Optional, TypeVar

import matplotlib.pyplot as plt
import torch
from colorama import deinit
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column, Table
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from message_queue import ClientRequestQueue, LearnerStats, MessageQueue
from util.random_names_generator import generate_random_name

# from message_queue import message_bus, msg_queue
from util.utilities import Timer, format_timer_human

T = TypeVar("T")

# torch for some reason initializes colorama and breaks rich colors
# https://github.com/Textualize/rich/issues/1201#issuecomment-829959040
deinit()


@dataclass
class DataLoaders(Generic[T]):
    train: DataLoader[T]
    valid: DataLoader[T]
    test: DataLoader[T]


@dataclass
class EpochData:
    train_loss: float
    valid_loss: float
    metric: float
    time_sec: float


def accuracy(predictions: Tensor, labels: Tensor) -> Tensor:
    return (predictions.argmax(1) == labels.argmax(1).type(torch.float)).sum()


class Learner:
    run_id: str
    current_total_epochs: int
    current_epoch_num: int
    task_id: Optional[TaskID] = None
    epoch_data: list[EpochData]
    max_epoch_table_rows = 10

    def __init__(
        self,
        data_loaders: DataLoaders[tuple[Tensor, ...]],
        model: nn.Module,
        loss_function: Callable[[Tensor, Tensor], Tensor],
        optimizer: Any,
        metric_function: Callable[[Tensor, Tensor], Tensor],
        device: Literal["cuda", "cpu"],
        scheduler: Optional[Any] = None,
    ) -> None:
        self.data_loaders = data_loaders
        self.model = model
        self.model.to(device)
        self.loss_function = loss_function
        self.metric_function = metric_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.timer = Timer()
        self.run_id = generate_random_name()
        self.epoch_data = []

    def get_epoch_data_table(self):
        epoch_table = Table(title="Epochs")
        epoch_table.add_column("Epoch")
        epoch_table.add_column("Train Loss")
        epoch_table.add_column("Valid Loss")
        epoch_table.add_column("Metric")
        epoch_table.add_column("Time")

        start_row = (
            0
            if len(self.epoch_data) <= self.max_epoch_table_rows
            else len(self.epoch_data) - self.max_epoch_table_rows
        )
        for idx, epoch in enumerate(self.epoch_data[start_row:]):
            epoch_table.add_row(
                str(idx + 1 + start_row),
                f"{epoch.train_loss:>8f}",
                f"{epoch.valid_loss:>8f}",
                f"{100*epoch.metric:>0.1f}%",
                format_timer_human(epoch.time_sec, clock_style=True),
            )
        return epoch_table

    def plot_epoch_data(self):
        plt.plot(
            range(len(self.epoch_data)),
            [epoch.train_loss for epoch in self.epoch_data],
            label="Train",
        )
        plt.plot(
            range(len(self.epoch_data)),
            [epoch.valid_loss for epoch in self.epoch_data],
            label="Valid",
        )
        plt.legend()
        plt.title("Loss")

    def train_loop(self, progress: Progress):
        # Setup progress bars and timers
        num_batches = len(self.data_loaders.train)
        batch_size = 0
        if self.task_id is None:
            self.task_id = progress.add_task(
                "[green]Training", total=num_batches, speed=0.0, loss=0.0
            )
        else:
            progress.reset(self.task_id)
            progress.update(
                self.task_id, description="[green]Training", total=num_batches
            )

        train_loss, avg_loss = 0, 0

        # Main training loop
        self.model.train()
        self.timer.reset_timer()
        for batch, (xb, yb) in enumerate(self.data_loaders.train):
            if batch == 0:
                batch_size = len(xb)

            # Forward pass to get batch predictions
            predictions = self.model(xb.to(self.device))
            # Calculate loss on batch predictions based on batch labels
            loss = self.loss_function(predictions, yb.to(self.device))
            train_loss += loss.item()

            # Gradients accumulate by default - reset for next batch.
            self.optimizer.zero_grad()
            # Backpropagation on the loss
            # (compute the gradient of the loss function with respect to every model parameter)
            loss.backward()
            # Update the parameters/weights based on the loss gradients and the learning rate
            # E.g. with stochastic gradient descent or adam
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # Finish progress bars and timers
            speed = ((batch + 1) * batch_size) / float(
                self.timer.show_elapsed_total(raw=True)
            )

            avg_loss = train_loss / ((batch + 1) * batch_size)

            progress.update(
                self.task_id,
                advance=1,
                speed=round(speed, 2),
                loss=round(avg_loss, 4),
            )

        return avg_loss

    def valid_loop(self, progress: Progress):
        # Setup progress bars and timers
        valid_loader = (
            self.data_loaders.valid
            if self.data_loaders.valid
            else self.data_loaders.test
        )
        size = len(valid_loader.dataset)  # type: ignore
        num_batches = len(valid_loader)
        batch_size = 0
        valid_loss, avg_loss, metric = 0, 0, 0
        if self.task_id is None:
            self.task_id = progress.add_task(
                "[cyan]Validating", total=num_batches, speed="--", loss="--"
            )
        else:
            progress.reset(self.task_id)
            progress.update(
                self.task_id, description="[cyan]Validating", total=num_batches
            )

        # Validation loop
        # torch.no_grad is the old context manager
        # with torch.no_grad():
        # Also set model to eval mode
        self.model.eval()
        with torch.inference_mode():
            self.timer.reset_timer()
            for batch, (xb, yb) in enumerate(valid_loader):
                if batch == 0:
                    batch_size = len(xb)

                predictions = self.model(xb.to(self.device))
                yb = yb.to(self.device)
                valid_loss += self.loss_function(predictions, yb).item()
                metric += self.metric_function(predictions, yb).item()

                speed = ((batch + 1) * batch_size) / float(
                    self.timer.show_elapsed_total(raw=True)
                )

                avg_loss = valid_loss / ((batch + 1) * batch_size)

                progress.update(
                    self.task_id,
                    advance=1,
                    speed=round(speed, 2),
                    loss=round(avg_loss, 4),
                )

        # Average metric
        metric /= size

        return avg_loss, metric

    def train_model(self, epochs: int):
        self.timer.reset_timer()
        total_timer = Timer()
        total_timer.reset_timer()
        self.current_total_epochs = epochs

        console = Console(record=True)

        progress_epoch = Progress(
            SpinnerColumn(),
            TextColumn(
                "[progress.description]{task.description}",
                table_column=Column(min_width=10),
            ),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.completed} / {task.total}"),
            TextColumn("{task.fields[loss]} loss"),
            console=console,
            transient=False,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[progress.description]{task.description}",
                table_column=Column(min_width=10),
            ),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[speed]} items/sec"),
            TextColumn("{task.fields[loss]} loss"),
            console=console,
            transient=False,
        )

        with Live(
            Group(progress_epoch, progress, self.get_epoch_data_table()),
            refresh_per_second=20,
        ) as live:
            training_epochs = progress_epoch.add_task(
                "[dark_violet]Epochs", total=epochs, loss=0.0
            )
            for epoch in range(1, epochs + 1):
                self.current_epoch_num = epoch
                train_loss = self.train_loop(progress)
                valid_loss, metric = self.valid_loop(progress)
                progress_epoch.update(
                    training_epochs, advance=1, loss=round(valid_loss, 4)
                )

                self.epoch_data.append(
                    EpochData(
                        train_loss, valid_loss, metric, total_timer.track_elapsed_loop()
                    )
                )

                live.update(
                    Group(progress_epoch, progress, self.get_epoch_data_table())
                )

        # Final loss doesn't print in table for some reason
        console.print(self.get_epoch_data_table())


if __name__ == "__main__":
    epochs = 100

    with Progress() as progress:
        training_epochs = progress.add_task("[red]Epochs", total=epochs)
        for i in range(100):
            progress.update(training_epochs, advance=1)
            sleep(0.1)

        print("Training complete")
