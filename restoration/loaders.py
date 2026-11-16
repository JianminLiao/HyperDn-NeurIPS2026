
import os
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image as _PILImage  # Preload Pillow before torchvision on local Windows conda envs.

class DatasetLoader(ABC):
    @abstractmethod
    def get_image(self, index: int) -> np.ndarray:
        """Return image at index as float32 numpy array [C, H, W]."""
        pass
        
    @abstractmethod
    def __len__(self) -> int:
        pass
        
    def load(self):
        """Optional setup/loading step."""
        return self


class WEDLoader(DatasetLoader):
    """Loader for WED PNG roots with train/val subdirs and on-demand image loading."""
    def __init__(self, root: str = './data/WED_crop', split: str = 'test', cache_size: int = 8):
        from glob import glob

        if split == 'train':
            subdir = 'train'
        elif split == 'test' or split == 'val':
            subdir = 'val'
        else:
            subdir = split

        self.root = os.path.join(root, subdir)
        self.dataset_name = os.path.basename(os.path.normpath(root)) or "WED_crop"
        self.split = split
        self.cache_size = max(0, int(cache_size))
        self.image_paths = sorted(glob(os.path.join(self.root, '*.png')))
        if not self.image_paths:
            raise FileNotFoundError(f"No images in {self.root}")

        self._cache = {}
        self._cache_order = []
        print(f"Indexed {len(self.image_paths)} WED images from {self.root}")

    def get_name(self) -> str:
        return self.dataset_name

    def get_size(self) -> int:
        return len(self.image_paths)

    def get_split(self) -> str:
        return self.split

    def load(self):
        return self

    def _remember(self, index: int, image_chw: np.ndarray) -> None:
        if self.cache_size <= 0:
            return
        if index in self._cache:
            self._cache_order.remove(index)
        self._cache[index] = image_chw
        self._cache_order.append(index)
        while len(self._cache_order) > self.cache_size:
            evict_index = self._cache_order.pop(0)
            self._cache.pop(evict_index, None)

    def _load_image(self, index: int) -> np.ndarray:
        from PIL import Image

        path = self.image_paths[index]
        with Image.open(path) as img:
            image_hwc = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
        return image_hwc.transpose(2, 0, 1)

    def get_image(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self.image_paths):
            raise IndexError(f"WED image index out of range: {index}")
        cached = self._cache.get(index)
        if cached is not None:
            if self.cache_size > 0:
                self._cache_order.remove(index)
                self._cache_order.append(index)
            return cached
        image_chw = self._load_image(index)
        self._remember(index, image_chw)
        return image_chw

    def __len__(self) -> int:
        return len(self.image_paths)
