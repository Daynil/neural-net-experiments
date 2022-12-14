from dataclasses import dataclass
from time import sleep
from typing import Any, Callable, Generic, Literal, Optional, TypeVar

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
from util.utilities import Timer

T = TypeVar("T")

# torch for some reason initializes colorama and breaks rich colors
# https://github.com/Textualize/rich/issues/1201#issuecomment-829959040
deinit()


@dataclass
class DataLoaders(Generic[T]):
    train: DataLoader[T]
    valid: DataLoader[T]
    test: DataLoader[T]


class Learner:
    run_id: str
    current_total_epochs: int
    current_epoch_num: int
    task_id: Optional[TaskID] = None

    def __init__(
        self,
        data_loaders: DataLoaders[tuple[Tensor, Tensor]],
        model: nn.Module,
        loss_function: Callable[[Tensor, Tensor], Tensor],
        optimizer: Any,
        device: Literal["cuda", "cpu"],
        scheduler: Optional[Any] = None,
    ) -> None:
        self.data_loaders = data_loaders
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.timer = Timer()
        self.run_id = generate_random_name()

    def train_loop(self, progress: Progress):
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
        for batch, (xb, yb) in enumerate(self.data_loaders.train):
            if batch == 0:
                batch_size = len(xb)
            # Compute prediction and loss
            predictions = self.model(xb.to(self.device))
            loss = self.loss_function(predictions, yb.to(self.device))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            self.timer.track_elapsed_loop()

            speed = 1 / (self.timer.elapsed_loop_average() / batch_size)

            loss_value, current = loss.item(), batch * batch_size

            progress.update(
                self.task_id,
                advance=1,
                speed=round(speed, 2),
                loss=round(loss_value, 2),
            )

    def valid_loop(self, progress: Progress):
        valid_loader = (
            self.data_loaders.valid
            if self.data_loaders.valid
            else self.data_loaders.test
        )
        size = len(valid_loader.dataset)  # type: ignore
        num_batches = len(valid_loader)
        batch_size = 0
        valid_loss, correct = 0, 0
        if self.task_id is None:
            self.task_id = progress.add_task(
                "[cyan]Validating", total=num_batches, speed="--", loss="--"
            )
        else:
            progress.reset(self.task_id)
            progress.update(
                self.task_id, description="[cyan]Validating", total=num_batches
            )

        with torch.no_grad():
            for batch, (xb, yb) in enumerate(valid_loader):
                if batch == 0:
                    batch_size = len(xb)

                predictions = self.model(xb.to(self.device))
                yb = yb.to(self.device)
                valid_loss += self.loss_function(predictions, yb).item()
                correct += (
                    (predictions.argmax(1) == yb.argmax(1))
                    .type(torch.float)
                    .sum()
                    .item()
                )

                self.timer.track_elapsed_loop()

                speed = 1 / (self.timer.elapsed_loop_average() / batch_size)

                progress.update(self.task_id, advance=1, speed=round(speed, 2))

        # Average loss and accuracy so far
        valid_loss /= num_batches
        correct /= size

        return correct, valid_loss

    def train_model(self, epochs: int):
        self.timer.reset_timer()
        self.current_total_epochs = epochs

        console = Console(record=True)

        epoch_table = Table(title="Epochs")
        epoch_table.add_column("Epoch")
        epoch_table.add_column("Loss")
        epoch_table.add_column("Accuracy")

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

        group = Group(progress_epoch, progress, epoch_table)

        with Live(group, refresh_per_second=10):
            training_epochs = progress_epoch.add_task("[red]Epochs", total=epochs)
            for epoch in range(epochs):
                self.current_epoch_num = epoch + 1
                self.train_loop(progress)
                progress_epoch.update(training_epochs, advance=1)
                correct, valid_loss = self.valid_loop(progress)
                epoch_table.add_row(
                    str(epoch), f"{valid_loss:>8f}", f"{100*correct:>0.1f}%"
                )


if __name__ == "__main__":
    epochs = 100

    with Progress() as progress:
        training_epochs = progress.add_task("[red]Epochs", total=epochs)
        for i in range(100):
            progress.update(training_epochs, advance=1)
            sleep(0.1)

        print("Training complete")
