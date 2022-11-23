import math
from pathlib import Path
from typing import Any, Callable, List, TypeVar, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split
from torchvision.io import read_image

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class MNISTImageDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        annotations_file: Path,
        img_dir: Path,
        transform: Union[Callable[[Tensor], Tensor], None] = None,
        target_transform: Union[Callable[[Union[Tensor, Any]], Tensor], None] = None,
    ):
        self.img_labels = pd.read_csv(annotations_file, header=None, delimiter=" ")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        file_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        img_path = self.img_dir.joinpath(str(label)).joinpath(str(file_name))
        image = read_image(str(img_path)).float() / 255
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, torch.tensor(label)


def split_datasets(dataset: Dataset[T], validation_percent: float) -> List[Subset[T]]:

    training_num, validation_num = math.floor(
        (1 - validation_percent) * len(dataset)
    ), math.floor(0.1 * len(dataset))
    training_num = training_num + (len(dataset) - (training_num + validation_num))

    return random_split(
        dataset,
        [training_num, validation_num],
        generator=torch.Generator().manual_seed(42),
    )


def preview_data_sample(data_loader: torch.utils.data.DataLoader[T]):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data_loader.dataset), size=(1,)).item()
        img, label = data_loader.dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label.argmax())
        plt.axis("off")
        plt.imshow(img.squeeze().cpu(), cmap="gray")
    plt.show()
