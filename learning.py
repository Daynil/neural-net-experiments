from dataclasses import dataclass
from time import strftime
from typing import Any, Callable, Literal, Optional, TypeVar

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from message_queue import ClientRequestQueue, LearnerStats, MessageQueue
from util.random_names_generator import generate_random_name

# from message_queue import message_bus, msg_queue
from util.utilities import Timer

T = TypeVar("T")


@dataclass
class DataLoaders:
    train: DataLoader[T]
    valid: Optional[DataLoader[T]] = None
    test: Optional[DataLoader[T]] = None


class Learner:
    run_id: str
    current_total_epochs: int
    current_epoch_num: int

    def __init__(
        self,
        data_loaders: DataLoaders,
        model: nn.Module,
        loss_function: Callable[[Tensor, Tensor], Tensor],
        optimizer: Any,
        device: Literal["cuda", "cpu"],
        message_queue: MessageQueue,
        client_request_queue: ClientRequestQueue,
    ) -> None:
        self.data_loaders = data_loaders
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.timer = Timer()
        self.message_queue = message_queue
        self.run_id = generate_random_name()
        self.client_request_queue = client_request_queue

    def train_loop(self):
        size = len(self.data_loaders.train.dataset)
        num_batches = len(self.data_loaders.train)
        batch_size = 0
        for batch, (xb, yb) in enumerate(self.data_loaders.train):
            if batch == 0:
                batch_size = len(xb)

            if self.client_request_queue.qsize() > 0:
                request = self.client_request_queue.get()
                if request == "cancel":
                    return

            # Compute prediction and loss
            predictions = self.model(xb.to(self.device))
            loss = self.loss_function(predictions, yb.to(self.device))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.timer.track_elapsed_loop()

            loss_value, current = loss.item(), batch * batch_size

            self.message_queue.put(
                LearnerStats(
                    run_id=self.run_id,
                    total_epochs=self.current_total_epochs,
                    current_epoch=self.current_epoch_num,
                    current_item=current,
                    total_items=size,
                    num_batches=num_batches,
                    loss_value=loss_value,
                    seconds_per_item=float(
                        self.timer.show_elapsed_loop_average(raw=True) / batch_size
                    ),
                    loop_type="train",
                )
            )

            if batch % 100 == 0 or batch == num_batches - 1:
                print(
                    f"loss: {loss_value:>7f} [{current:>5d}|{size:>5d}] | Speed: {float(self.timer.show_elapsed_loop_average(raw=True) / batch_size) * 1000:.3f} ms/image"
                )

    def valid_loop(self):
        valid_loader = (
            self.data_loaders.valid
            if self.data_loaders.valid
            else self.data_loaders.test
        )
        size = len(valid_loader.dataset)
        num_batches = len(valid_loader)
        valid_loss, correct = 0, 0

        with torch.no_grad():
            for xb, yb in valid_loader:
                if self.client_request_queue.qsize() > 0:
                    request = self.client_request_queue.get()
                    if request == "cancel":
                        return

                predictions = self.model(xb.to(self.device))
                yb = yb.to(self.device)
                valid_loss += self.loss_function(predictions, yb).item()
                correct += (
                    (predictions.argmax(1) == yb.argmax(1))
                    .type(torch.float)
                    .sum()
                    .item()
                )

        # Average loss and accuracy so far
        valid_loss /= num_batches
        correct /= size

        self.message_queue.put(
            LearnerStats(
                run_id=self.run_id,
                total_epochs=self.current_total_epochs,
                current_epoch=self.current_epoch_num,
                current_item=1,
                total_items=size,
                num_batches=num_batches,
                loss_value=valid_loss,
                seconds_per_item=float(self.timer.show_elapsed_loop_average(raw=True)),
                loop_type="valid",
                accuracy=correct,
            )
        )

        print(
            f"Valid Error: \n Accuracy: {100*correct:>0.1f}%, Avg loss: {valid_loss:>8f} \n"
        )

    def train_model(self, epochs: int):
        self.timer.reset_timer()
        self.current_total_epochs = epochs
        for epoch in range(epochs):
            if self.client_request_queue.qsize() > 0:
                request = self.client_request_queue.get()
                if request == "cancel":
                    return

            self.current_epoch_num = epoch + 1
            print(f"Epoch {epoch + 1}\n------------------")
            self.train_loop()
            self.valid_loop()

        print("Training complete")


if __name__ == "__main__":
    # learner = Learner()
    pass
