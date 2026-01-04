import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import os.path
from pathlib import Path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from PIL import Image

from dataset.vision import VisionDataset

import json





def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx



IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_all_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


class ImageNet(VisionDataset):
    def __init__(
        self, 
        images_dir, 
        config_dir=None,
        has_featuers=False,
        loader: Callable[[str], Any] = default_loader,
        transform: Optional[Callable] = None,
        dir_to_class_config = None,
        return_idx=False,
        return_class_text=False,
        ):
        super().__init__(images_dir, transform=transform)

        self.images_dir = images_dir
        classes, class_to_idx = self.find_classes(self.images_dir)

        if config_dir:
            samples=[]
            if config_dir.endswith('.txt'):
                with open(config_dir, 'r', encoding='utf-8') as file:
                    for line in file:
                        if line.strip().endswith('.JPEG'):
                            samples.append(line.strip())  # 使用 strip() 去除每行末尾的换行符
            else:
                with open(config_dir, 'r', encoding='utf-8') as file:
                    for line in file:
                        # 解析每一行的 JSON 数据
                        record = json.loads(line)
                        
                        if record['image_path'].strip().endswith('.JPEG'):
                            samples.append(os.path.join(images_dir,record['image_path']))  # 使用 strip() 去除每行末尾的换行符

        else:
            samples = get_all_files(images_dir)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.transform = transform
        self.loader = loader
        self.return_idx=return_idx


        # 定义一个空字典
        self.dir_to_class = {}

        # 读取文件
        with open(dir_to_class_config, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行末的换行符并按照逗号分割
                key_value = line.strip().split(',')
                if len(key_value) == 2:  # 确保我们有两个部分
                    key = key_value[0]
                    value = key_value[1]
                    # 将键值对放入字典
                    self.dir_to_class[key] = value

        self.return_class_text = return_class_text
        self.class_to_index={}
        for idx,i in enumerate(self.dir_to_class.keys()):
            self.class_to_index[i] = idx




    def find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __len__(self):
        # assert len(self.feature_files) == len(self.label_files), \
            # "Number of feature files and label files should be same"
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        dir_name = os.path.dirname(path)
        dir_name = dir_name.split('/')[-1]
        class_txt = self.dir_to_class[dir_name]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.return_idx:
            return sample, class_txt,idx
        elif self.return_class_text:
            return sample, class_txt
        else:
            return sample, self.class_to_index[dir_name]



        # if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
        #     feature_dir = self.aug_feature_dir
        #     label_dir = self.aug_label_dir
        # else:
        #     feature_dir = self.feature_dir
        #     label_dir = self.label_dir
                   
        # feature_file = self.feature_files[idx]
        # label_file = self.label_files[idx]

        # features = np.load(os.path.join(feature_dir, feature_file))
        # if self.flip:
        #     aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
        #     features = features[:, aug_idx]
        # labels = np.load(os.path.join(label_dir, label_file))
        # return torch.from_numpy(features), torch.from_numpy(labels)


def build_imagenet_raw(args, transform):
    return ImageNet(args.data_path, transform=transform, config_dir=args.data_anno, dir_to_class_config=args.imagenet_class,return_idx=args.return_idx,return_class_text=args.return_class_text)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)