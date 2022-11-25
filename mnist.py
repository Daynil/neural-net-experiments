from pathlib import Path

import torch
import torch.utils.data
from torch import Tensor, nn
from torchvision.models import resnet18
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
        # limit_data=5000,
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
        # limit_data=10000,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=512)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

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
        hidden_neurons = 512
        total_labels = 10  # 0 through 9 are the labels
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(total_pixels_per_image, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, total_labels),
        )

    def forward(self, x: Tensor):  # type: ignore
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


def run_model(epochs: int, save_model_name: str = ""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")

    torch.cuda.empty_cache()

    train_loader, valid_loader, test_loader = get_data()

    # data_loaders.preview_data_sample(train_loader.dataset)

    # model = SimpleNeuralNetwork().to(device)
    # model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
    # model = resnet18(pretrained=False, num_classes=10).to(device)
    model = resnet18(weights=None, num_classes=10).to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # Unclear why OneCycleLR isn't detected by types
    # It is present in the module checking manually
    scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
        optimizer, max_lr=1, steps_per_epoch=len(train_loader), epochs=epochs
    )

    learner = learning.Learner(
        # data_loaders=learning.DataLoaders(train_loader, valid_loader, test_loader),
        data_loaders=learning.DataLoaders(train_loader, valid_loader, test_loader),
        model=model,
        # loss_function=manual_mnist_loss,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # asyncio.create_task(learner.train_model(1))
    learner.train_model(epochs)
    if save_model_name:
        torch.save(model.state_dict(), f"{save_model_name}.pth")


def load_and_test_model(model_name: str):
    train_loader, valid_loader, test_loader = get_data()

    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(torch.load(f"{model_name}.pth"))
    # print(model.eval())
    model.eval()

    data_loaders.preview_tested_data_sample(test_loader.dataset, model)
    return

    test_image, label = test_loader.dataset[0]
    logits = model(test_image.unsqueeze(0))
    model_probabilities: Tensor = nn.Softmax(dim=1)(logits)
    model_prediction = model_probabilities.argmax(1)
    # print(model_probabilities)
    # print(model_prediction)
    model_confidence = model_probabilities[0][model_prediction]
    print(
        f"Model prediction: {model_prediction.item()}, Model confidence: {model_confidence.item() * 100}%"
    )
    print("Label: ", label.argmax(0).item())


if __name__ == "__main__":
    # run_model(5)
    # run_model(epochs=5, save_model_name="mnist_resnet18_onecyclelr")
    load_and_test_model(model_name="mnist_resnet18_onecyclelr")

    # train_loader, valid_loader, test_loader = get_data()

    # data_loaders.preview_data_sample(train_loader.dataset)
    # image, label = train_loader.dataset[0]
