from pathlib import Path

import torch
import torch.utils.data
from torch import Tensor, nn
from torchvision.transforms import Lambda

import data_loaders
import learning
from message_queue import ClientRequestQueue, MessageQueue

num_labels = 10  # 0 through 9 are the labels


def get_data():
    dataset = data_loaders.MNISTImageDataset(
        annotations_file=Path(r"C:\Users\Danny\ml_datasets\mnist\train\train.txt"),
        img_dir=Path(r"C:\Users\Danny\ml_datasets\mnist\train"),
        # one-hot encode the label (so a 6 turns to tensor 0,0,0,0,0,0,1,0,0,0)
        target_transform=Lambda(
            lambda y: torch.zeros(num_labels, dtype=torch.float).scatter_(
                dim=0, index=torch.tensor(y), value=1
            )
        ),
    )

    train_dataset, valid_dataset = data_loaders.split_datasets(dataset, 0.1)

    test_dataset = data_loaders.MNISTImageDataset(
        annotations_file=Path(r"C:\Users\Danny\ml_datasets\mnist\test\test.txt"),
        img_dir=Path(r"C:\Users\Danny\ml_datasets\mnist\test"),
        # one-hot encode the label (so a 6 turns to tensor 0,0,0,0,0,0,1,0,0,0)
        target_transform=Lambda(
            lambda y: torch.zeros(num_labels, dtype=torch.float).scatter_(
                dim=0, index=torch.tensor(y), value=1
            )
        ),
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    return train_loader, valid_loader, test_loader


class SimpleNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(SimpleNeuralNetwork, self).__init__()
        # This does exactly what .view(-1, total_pixels_image) does -
        # This makes it from a list of 28x28 matrices
        #   to a list of vectors of 784 pixels
        # This is required because the linear function
        # operates on a single array (x@w)
        self.flatten = nn.Flatten()
        total_pixels_per_image = 28 * 28
        hidden_neurons = 30
        total_labels = 10  # 0 through 9 are the labels
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(total_pixels_per_image, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, total_labels),
        )

    def forward(self, x: Tensor):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def manual_mnist_loss(predictions: Tensor, targets: Tensor):
    # Simple manual msnist loss function attempt
    # Take every prediction's softmax and subtract from target
    # This gets you how far from the target you were
    # Then square it to prevent negatives cancelling out on sum
    predictions = predictions.softmax(dim=1)
    return ((targets - predictions) ** 2).sum()


def run_model(message_queue: MessageQueue, client_request_queue: ClientRequestQueue):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device")

    train_loader, valid_loader, test_loader = get_data()

    # data_loaders.preview_data_sample(train_loader)

    simple_model = SimpleNeuralNetwork().to(device)
    optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.001)

    learner = learning.Learner(
        # data_loaders=learning.DataLoaders(train_loader, valid_loader, test_loader),
        data_loaders=learning.DataLoaders(test_loader, valid_loader, test_loader),
        model=simple_model,
        loss_function=manual_mnist_loss,
        optimizer=optimizer,
        device=device,
        message_queue=message_queue,
        client_request_queue=client_request_queue,
    )

    # asyncio.create_task(learner.train_model(1))
    learner.train_model(3)


if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device")

    train_loader, valid_loader, test_loader = get_data()

    # data_loaders.preview_data_sample(train_loader)

    simple_model = SimpleNeuralNetwork().to(device)
    optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.001)

    # learner = learning.Learner(
    #     data_loaders=learning.DataLoaders(train_loader, valid_loader, test_loader),
    #     model=simple_model,
    #     loss_function=manual_mnist_loss,
    #     optimizer=optimizer,
    #     device=device,
    #     message_queue==msg_queue,
    # )

    # learner.train_model(10)
