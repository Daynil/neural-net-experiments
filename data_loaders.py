import math
from pathlib import Path
from random import random, randint
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset, random_split
from torchvision.io import ImageReadMode, read_image

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


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


def split_datasets(dataset: TensorDataset, validation_percent: float):
    training_num, validation_num = math.floor(
        (1 - validation_percent) * len(dataset)
    ), math.floor(0.1 * len(dataset))
    training_num = training_num + (len(dataset) - (training_num + validation_num))

    return random_split(
        dataset,
        [training_num, validation_num],
        generator=torch.Generator().manual_seed(42),
    )


def preview_data_sample(dataset: TensorDataset):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = randint(0, len(dataset))
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(str(label.argmax().item()))
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


def preview_tested_data_sample(dataset: TensorDataset, model: nn.Module):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = randint(0, len(dataset))
        test_image, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)

        logits = model(test_image.unsqueeze(0))
        model_probabilities: Tensor = nn.Softmax(dim=1)(logits)
        model_prediction = model_probabilities.argmax(1)
        # print(model_probabilities)
        # print(model_prediction)
        model_confidence = model_probabilities[0][model_prediction]

        plt.title(
            f"Label: {label.argmax(0).item()} - Pred (prob): {model_prediction.item()} ({round(model_confidence.item() * 100, 3)}%)"
        )
        plt.axis("off")
        plt.imshow(test_image.permute(1, 2, 0).cpu(), cmap="gray")
    plt.show()
