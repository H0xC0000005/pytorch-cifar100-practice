import torchvision
from torch.distributions.beta import Beta
import random
import torch
from PIL import Image


class CIFAR100WithMixup(torchvision.datasets.CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool,
        transform=None,
        target_transform=None,
        download: bool = False,
        alpha: float = 0.1,
        beta: float = 0.1,
    ):
        assert (
            0 <= alpha < 1
        ), f"must have alpha value: 0 <= a < 1 where a=0 is no mixup"
        super().__init__(root, train, transform, target_transform, download)
        self.alpha = alpha
        self.beta = beta
        self.distribution = Beta(alpha, beta)

    def __len__(self):
        return len(self.data)

    def mixup(self, data, label):
        # Select another random sample
        idx = random.randint(0, len(self.data) - 1)
        data2 = self.data[idx]
        # need to be PIL.Image before transform. suppose data2 is python list or numpy array
        if not isinstance(data2, Image.Image):
            data2 = Image.fromarray(data2)
        label2 = self.targets[idx]
        if self.transform:
            data2 = self.transform(data2)

        if self.target_transform is not None:
            label2 = self.target_transform(label2)
        one_hot1 = torch.zeros(100)  # one hot
        one_hot2 = torch.zeros(100)
        one_hot1[label] = 1
        one_hot2[label2] = 1
        lam = self.distribution.sample()  # random weight
        lam = max(lam, 1 - lam)
        mixed_data = lam * data + (1 - lam) * data2
        mixed_label = lam * one_hot1 + (1 - lam) * one_hot2
        # mixed_label = mixed_label.to("cpu").detach().item()

        # since this mixup use bceloss but not cross entropy, label size should be (batch, 100) for cifar100
        assert isinstance(mixed_label, torch.Tensor), f"get label type {mixed_label}"
        assert (
            mixed_label.shape[-1] == 100
        ), f"get label last dim shape {mixed_label.shape[-1]}"
        # assert len(mixed_label.shape) == 2, f"get label shape dim not 2 {mixed_label.shape}"
        pass
        return mixed_data, mixed_label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(data, Image.Image):
            data = Image.fromarray(data)
        pass
        if self.transform:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # Apply Mixup (after transform)
        pass
        if self.alpha >= 0 and self.beta >= 0:
            data, label = self.mixup(data, label)
        # if soft:
        #     assert 0 < epsilon < 0.5, f"epsilon must be in range (0.0.5)"
        #     label = epsilon + (1 - epsilon) * label
        return data, label
