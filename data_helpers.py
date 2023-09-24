import math
from pathlib import Path
from random import randint, random
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    label_ranking_average_precision_score,
)
from torch import Generator, Tensor, nn, randperm
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset
from torchvision.io import ImageReadMode, read_image

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


# TODO: make this a way to simplify stuff like get_label_name from a TensorDataset
# so I don't have to overwrite what a TensorDataset is everywhere
class DatasetHelpers:
    def __init__(self) -> None:
        pass


class MNISTImageDataset(TensorDataset):
    """
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

    def __init__(
        self,
        annotations_file: Path,
        img_dir: Path,
        transform: Union[Callable[[Tensor], Tensor], None] = None,
        target_transform: Union[Callable[[Union[Tensor, Any]], Tensor], None] = None,
        limit_data: Optional[int] = None,
    ):
        img_labels = pd.read_csv(annotations_file, header=None, delimiter=" ")
        if limit_data:
            img_labels = img_labels[:limit_data]
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        file_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        img_path = self.img_dir.joinpath(str(label)).joinpath(str(file_name))
        # image = read_image(str(img_path)).float() / 255
        image = read_image(str(img_path), ImageReadMode.RGB).float() / 255
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = torch.tensor(label)
        return image, label


def split_datasets(
    dataset: TensorDataset,
    validation_percent: float,
    validation_transforms: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
):
    training_num, validation_num = math.floor(
        (1 - validation_percent) * len(dataset)
    ), math.floor(0.1 * len(dataset))
    training_num = training_num + (len(dataset) - (training_num + validation_num))

    train, valid = torch.utils.data.random_split(
        dataset,
        [training_num, validation_num],
        generator=torch.Generator().manual_seed(42),
    )

    if validation_transforms:
        # My datasets will always have transform
        valid.transform = validation_transforms  # type: ignore

    return train, valid


def label_name_one_hot(label: str, label_dict: dict[str, int]) -> Tensor:
    """
    One-hot encode the target label.
    Creates a torch of zeroes to length of labels, map label name to label index, make that index a 1.

    Args:
        label: String label
        label_dict: Dict mapping strings to integers

    Returns:
        One hot encoded tensor
    """
    return torch.zeros(len(label_dict.items()), dtype=torch.float).scatter_(
        0, torch.tensor(label_dict[label]), value=1
    )


def label_idx_one_hot(label_idx: int, label_dict: dict[str, int]) -> Tensor:
    """
    One-hot encode the target label's index.

    Args:
        label_idx:
        label_dict: Dict mapping strings to integers

    Returns:
        One hot encoded tensor
    """
    return torch.zeros(len(label_dict.items()), dtype=torch.float).scatter_(
        0, torch.tensor(label_idx), value=1
    )


def get_label_name(label: Tensor, label_dict: dict[str, int]):
    label_idx = int(label.argmax().item())
    return list(label_dict.keys())[list(label_dict.values()).index(label_idx)]


def preview_data_sample(dataset: TensorDataset, label_dict: dict[str, int]):
    figure = plt.figure(figsize=(12, 12))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = randint(0, len(dataset))
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        # plt.title(dataset.get_label_name(label.argmax().item()))  # type: ignore
        plt.title(get_label_name(label, label_dict), fontsize="medium")
        plt.axis("off")
        """
        plt.imshow expects the image tensor to have the shape (height, width, channels), 
        while PyTorch tensors are usually channel first, meaning they have the shape (channels, height, width). 
        Therefore, you need to use img.permute(1, 2, 0) 
        to reshape the image tensor to match the expected shape for plt.imshow.
        This call to permute means we are moving:
            - channels (in pytorch the first dim, 0) to the last
            - height (in pytorch the 2nd dim, 1) to first
            - width (in pytorch, the 3rd dim, 2) to second
        
        Matplotlib also cannot load from gpu, so we transfer our image tensor to cpu.
        """
        plt.imshow(img.permute(1, 2, 0).cpu())
    plt.show()


def preview_tested_data_sample(
    dataset: TensorDataset,
    model: nn.Module,
    label_dict: dict[str, int],
    device: Literal["cuda", "cpu"],
):
    figure = plt.figure(figsize=(14, 14))
    cols, rows = 3, 3

    for i in range(1, cols * rows + 1):
        sample_idx = randint(0, len(dataset))
        test_image, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)

        logits = model(test_image.unsqueeze(0).to(device))
        model_probabilities: Tensor = nn.Softmax(dim=1)(logits)
        model_prediction = int(model_probabilities.argmax(1).item())
        model_prediction_oh = label_idx_one_hot(model_prediction, label_dict)
        model_confidence = model_probabilities[0][model_prediction]

        actual_label_name = get_label_name(label, label_dict)
        pred_label_name = get_label_name(model_prediction_oh, label_dict)

        plt.title(
            f"Label: {actual_label_name}\nPred (prob): {pred_label_name} ({round(model_confidence.item() * 100,2)}%)",
            color="green" if actual_label_name == pred_label_name else "red",
            fontsize="medium",
        )
        plt.axis("off")
        plt.imshow(test_image.permute(1, 2, 0).cpu())
    plt.show()


def plot_confusion_matrix(
    preds: Tensor, labels: Tensor, label_names: list[str], figsize=(16, 16)
):
    cm = confusion_matrix(labels.argmax(dim=1).tolist(), preds.argmax(dim=1).tolist())
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)

    _fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, cmap="Blues")

    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    plt.title("Confusion Matrix")
    plt.show()


def most_confused(preds: Tensor, labels: Tensor, label_names: list[str], min_val=1):
    cm = confusion_matrix(labels.argmax(dim=1).tolist(), preds.argmax(dim=1).tolist())

    max_by_row = []
    for row_num, row in enumerate(cm):
        row_no_identity = row.copy()
        row_no_identity[row_num] = -1
        max = row_no_identity.max()
        max_idx = row_no_identity.tolist().index(max)
        max_by_row.append((row_num, max_idx, max))

    max_by_row = [
        (label_names[x[0]], label_names[x[1]], x[2])
        for x in sorted(max_by_row, key=lambda x: x[2], reverse=True)
        if x[2] >= min_val
    ]

    return max_by_row
