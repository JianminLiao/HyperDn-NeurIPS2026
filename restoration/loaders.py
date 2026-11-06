
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

class STL10Loader(DatasetLoader):
    def __init__(self, root='./data', split='train', download=True):
        import os
        import torchvision
        import torchvision.transforms as transforms
        # Use absolute path to avoid OSError [Errno 22] with synced-folder file locks on Windows
        abs_root = os.path.abspath(root)
        transform = transforms.Compose([transforms.ToTensor()])
        # note: STL10 uses 'split' instead of 'train' bool
        # splits: 'train', 'test', 'unlabeled', 'train+unlabeled'
        self.dataset = torchvision.datasets.STL10(root=abs_root, split=split, download=download, transform=transform)
        
    def load(self):
        return self
        
    def get_image(self, index: int) -> np.ndarray:
        return self.dataset[index][0].numpy()
        
    def __len__(self) -> int:
        return len(self.dataset)


class CIFAR10Loader(DatasetLoader):
    def __init__(self, root='./data', train=True, download=True):
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        
    def load(self):
        # Already loaded in __init__
        return self
        
    def get_image(self, index: int) -> np.ndarray:
        # self.dataset[index] returns (image, label)
        # image is Tensor [C, H, W]
        return self.dataset[index][0].numpy()
        
    def __len__(self) -> int:
        return len(self.dataset)


class Set12Loader(DatasetLoader):
    """Loader for Set12 benchmark (12 grayscale images)"""
    def __init__(self, root: str = './data/Set12', split: str = 'test'):
        from PIL import Image
        from glob import glob
        
        self.root = root
        self.split = split
        image_paths = sorted(glob(os.path.join(root, '*.png')))
        if not image_paths:
            raise FileNotFoundError(f"No images in {root}")
        
        self.images = []
        for path in image_paths:
            img = Image.open(path).convert('L')  # Grayscale
            img_rgb = Image.merge('RGB', (img, img, img))  # Gray->RGB
            self.images.append(np.array(img_rgb) / 255.0)  # [H, W, 3], float
        
        print(f"Loaded {len(self.images)} Set12 images")
    
    def get_name(self) -> str:
        return "Set12"
    
    def get_size(self) -> int:
        return len(self.images)
    
    def get_split(self) -> str:
        return self.split
    
    def load(self):
        return self
    
    def get_image(self, index: int) -> np.ndarray:
        # Return [H, W, C], transpose to [C, H, W] for solver
        return self.images[index].transpose(2, 0, 1)
    
    def __len__(self) -> int:
        return len(self.images)


class Set14Loader(DatasetLoader):
    """Loader for Set14 benchmark (14 images, usually evaluated as Y channel in YCbCr or Gray)"""
    def __init__(self, root: str = './data/Set14', split: str = 'test'):
        from PIL import Image
        from glob import glob
        
        self.root = root
        self.split = split
        image_paths = sorted(glob(os.path.join(root, '*.png')) + glob(os.path.join(root, '*.bmp')))
        if not image_paths:
            raise FileNotFoundError(f"No images in {root}")
        
        self.images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')  # Color
            self.images.append(np.asarray(img, dtype=np.float32) / 255.0)  # [H, W, 3], float
        
        print(f"Loaded {len(self.images)} Set14 images")
    
    def get_name(self) -> str:
        return "Set14"
    
    def get_size(self) -> int:
        return len(self.images)
    
    def get_split(self) -> str:
        return self.split
    
    def load(self):
        return self
    
    def get_image(self, index: int) -> np.ndarray:
        # Return [H, W, C], transpose to [C, H, W] for solver
        return self.images[index].transpose(2, 0, 1)
    
    def __len__(self) -> int:
        return len(self.images)

class Set5Loader(DatasetLoader):
    """Loader for Set5 benchmark (5 color test images)"""
    def __init__(self, root: str = './data/Set5', split: str = 'test'):
        from PIL import Image
        from glob import glob
        
        self.root = root
        self.split = split
        image_paths = sorted(glob(os.path.join(root, '*.png')))
        if not image_paths:
            raise FileNotFoundError(f"No images in {root}")
        
        self.images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')  # Color
            self.images.append(np.asarray(img, dtype=np.float32) / 255.0)  # [H, W, 3], float
        
        print(f"Loaded {len(self.images)} Set5 images")
    
    def get_name(self) -> str:
        return "Set5"
    
    def get_size(self) -> int:
        return len(self.images)
    
    def get_split(self) -> str:
        return self.split
    
    def load(self):
        return self
    
    def get_image(self, index: int) -> np.ndarray:
        # Return [H, W, C], transpose to [C, H, W] for solver
        return self.images[index].transpose(2, 0, 1)
    
    def __len__(self) -> int:
        return len(self.images)


class CBSD68Loader(DatasetLoader):
    """Loader for CBSD68 benchmark (68 color images from BSD)"""
    def __init__(self, root: str = './data/CBSD68', split: str = 'test'):
        from PIL import Image
        from glob import glob
        
        self.root = root
        self.split = split
        image_paths = sorted(glob(os.path.join(root, '*.png')))
        if not image_paths:
            raise FileNotFoundError(f"No images in {root}")
        
        self.images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')  # Color images
            self.images.append(np.asarray(img, dtype=np.float32) / 255.0)  # [H, W, 3], float
        
        print(f"Loaded {len(self.images)} CBSD68 images")
    
    def get_name(self) -> str:
        return "CBSD68"
    
    def get_size(self) -> int:
        return len(self.images)
    
    def get_split(self) -> str:
        return self.split
    
    def load(self):
        return self
    
    def get_image(self, index: int) -> np.ndarray:
        # Return [H, W, C], transpose to [C, H, W] for solver
        return self.images[index].transpose(2, 0, 1)
    
    def __len__(self) -> int:
        return len(self.images)


class Kodak24Loader(DatasetLoader):
    """Loader for Kodak24 benchmark (24 color images)"""
    def __init__(self, root: str = './data/Kodak24', split: str = 'test'):
        from PIL import Image
        from glob import glob
        
        self.root = root
        self.split = split
        # Support both .png and .bmp formats
        image_paths = sorted(glob(os.path.join(root, '*.png')) + glob(os.path.join(root, '*.bmp')))
        if not image_paths:
            raise FileNotFoundError(f"No images in {root}")
        
        self.images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')  # Color
            self.images.append(np.asarray(img, dtype=np.float32) / 255.0)  # [H, W, 3], float
        
        print(f"Loaded {len(self.images)} Kodak24 images")
    
    def get_name(self) -> str:
        return "Kodak24"
    
    def get_size(self) -> int:
        return len(self.images)
    
    def get_split(self) -> str:
        return self.split
    
    def load(self):
        return self
    
    def get_image(self, index: int) -> np.ndarray:
        # Return [H, W, C], transpose to [C, H, W] for solver
        return self.images[index].transpose(2, 0, 1)
    
    def __len__(self) -> int:
        return len(self.images)


class Urban100Loader(DatasetLoader):
    """Loader for Urban100 benchmark (100 color images of urban scenes)"""
    def __init__(self, root: str = './data/Urban100', split: str = 'test'):
        from PIL import Image
        from glob import glob
        
        self.root = root
        self.split = split
        image_paths = sorted(glob(os.path.join(root, '*.png')))
        if not image_paths:
            raise FileNotFoundError(f"No images in {root}")
        
        self.images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')  # Color
            self.images.append(np.asarray(img, dtype=np.float32) / 255.0)  # [H, W, 3], float
        
        print(f"Loaded {len(self.images)} Urban100 images")
    
    def get_name(self) -> str:
        return "Urban100"
    
    def get_size(self) -> int:
        return len(self.images)
    
    def get_split(self) -> str:
        return self.split
    
    def load(self):
        return self
    
    def get_image(self, index: int) -> np.ndarray:
        # Return [H, W, C], transpose to [C, H, W] for solver
        return self.images[index].transpose(2, 0, 1)
    
    def __len__(self) -> int:
        return len(self.images)


class DIV2KLoader(DatasetLoader):
    """Loader for DIV2K validation/test subset"""
    def __init__(self, root: str = './data/Div2k_val_4', split: str = 'test'):
        from PIL import Image
        from glob import glob
        
        self.root = root
        self.split = split
        image_paths = sorted(glob(os.path.join(root, '*.png')))
        if not image_paths:
            raise FileNotFoundError(f"No images in {root}")
        
        self.images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')  # Color
            self.images.append(np.asarray(img, dtype=np.float32) / 255.0)  # [H, W, 3], float
        
        print(f"Loaded {len(self.images)} DIV2K images")
    
    def get_name(self) -> str:
        return "Div2k_val_4"
    
    def get_size(self) -> int:
        return len(self.images)
    
    def get_split(self) -> str:
        return self.split
    
    def load(self):
        return self
    
    def get_image(self, index: int) -> np.ndarray:
        # Return [H, W, C], transpose to [C, H, W] for solver
        return self.images[index].transpose(2, 0, 1)
    
    def __len__(self) -> int:
        return len(self.images)


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


class FolderImageLoader(DatasetLoader):
    """Loader for a local folder of 2D microscopy images."""

    def __init__(self, root: str, split: str = '', cache_size: int = 8):
        from glob import glob

        split = str(split or '').strip()
        candidate_root = os.path.join(root, split) if split else root
        if split and os.path.isdir(candidate_root):
            self.root = candidate_root
        else:
            self.root = root
        self.split = split or 'all'
        self.cache_size = max(0, int(cache_size))
        patterns = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.npy', '*.npz')
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob(os.path.join(self.root, pattern)))
        self.image_paths = sorted(image_paths)
        if not self.image_paths:
            raise FileNotFoundError(f"No supported images found in {self.root}")

        self._cache = {}
        self._cache_order = []
        print(f"Indexed {len(self.image_paths)} folder images from {self.root}")

    def get_name(self) -> str:
        return 'Folder'

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

    def _normalize_to_float32(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            arr = arr.astype(np.float32) / float(info.max)
        else:
            arr = arr.astype(np.float32)
            vmax = float(np.max(arr)) if arr.size else 1.0
            if vmax > 1.0:
                arr = arr / vmax
        return arr.astype(np.float32, copy=False)

    def _load_image(self, index: int) -> np.ndarray:
        from PIL import Image
        import tifffile

        path = self.image_paths[index]
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            arr = np.load(path)
        elif ext == '.npz':
            data = np.load(path)
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"No arrays found in {path}")
            arr = data[keys[0]]
        elif ext in ('.tif', '.tiff'):
            arr = tifffile.imread(path)
        else:
            with Image.open(path) as img:
                arr = np.asarray(img)

        arr = self._normalize_to_float32(arr)
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            arr = arr[..., :3]
            return arr.transpose(2, 0, 1)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            return arr
        raise ValueError(
            f"FolderImageLoader currently supports 2D grayscale/RGB images only, got shape={arr.shape} for {path}"
        )

    def get_image(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self.image_paths):
            raise IndexError(f"Folder image index out of range: {index}")
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
