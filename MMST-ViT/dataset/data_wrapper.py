import torch
from sklearn import preprocessing
import torchvision.transforms as transforms


def simclr_pipeline_transform(img_size=224, s=1, kernel_size=9):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return transforms.Compose([transforms.RandomResizedCrop(size=img_size, antialias=True),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomApply([color_jitter], p=0.8),
                               transforms.RandomGrayscale(p=0.2),
                               transforms.GaussianBlur(kernel_size=kernel_size),
                               ])


class SimCLRViews(object):
    """float [0,1] 图像 (N,3,H,W) -> 两组独立增强视图；逐张采样增强参数（与原 PIL 逐图管线一致）。"""

    def __init__(self, img_size=224, s=1, kernel_size=9):
        self.transform = simclr_pipeline_transform(img_size, s, kernel_size)

    def __call__(self, x):
        x = x.to(torch.float32)
        xi = torch.stack([self.transform(img) for img in x])
        xj = torch.stack([self.transform(img) for img in x])
        return xi, xj


class DataWrapper(object):
    def __init__(self, img_size=224, s=1, kernel_size=9, train=True):
        self.img_size = img_size
        self.s = s
        self.kernel_size = kernel_size

        if train:
            self.transform = self.get_simclr_pipeline_transform()
        else:
            self.transform = self.get_transform_val()

    def __call__(self, x):
        x = x.to(torch.float32)
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj

    def get_simclr_pipeline_transform(self):
        return simclr_pipeline_transform(self.img_size, self.s, self.kernel_size)

    def get_transform_val(self):
        data_transforms = transforms.Compose([transforms.CenterCrop(size=self.img_size)])
        return data_transforms


class ScalarNorm(object):
    def __init__(self):
        self.norm = preprocessing.StandardScaler()

    def __call__(self, x, reverse=False):
        if not reverse:
            x = self.norm.fit_transform(x)
        else:
            x = self.norm.inverse_transform(x)

        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        return x
