# python/kpu/datasets.py
"""
Dataset utilities for KPU Python package.

Provides loaders for common datasets like MNIST for testing and validation.
"""

import os
import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from .tensor import Tensor


# Default cache directory
CACHE_DIR = Path.home() / ".cache" / "kpu" / "datasets"


def _ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress indicator."""
    if dest.exists():
        return

    _ensure_dir(dest.parent)

    print(f"Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")
    except Exception as e:
        print(f"  Failed: {e}")
        raise


def _read_idx_images(path: Path) -> np.ndarray:
    """Read IDX format image file."""
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    """Read IDX format label file."""
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


class MNIST:
    """
    MNIST Dataset loader.

    Downloads and loads the MNIST handwritten digit dataset.

    Example:
        >>> mnist = MNIST()
        >>> train_images, train_labels = mnist.load_train()
        >>> test_images, test_labels = mnist.load_test()
        >>> print(train_images.shape)  # (60000, 28, 28)
    """

    # Use PyTorch's S3 mirror (yann.lecun.com is no longer reliable)
    URLS = {
        'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize MNIST loader.

        Args:
            cache_dir: Directory to cache downloaded files
        """
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / "mnist"
        _ensure_dir(self.cache_dir)

    def _download(self):
        """Download all MNIST files if not cached."""
        for name, url in self.URLS.items():
            filename = url.split('/')[-1]
            dest = self.cache_dir / filename
            _download_file(url, dest, f"MNIST {name}")

    def load_train(self, normalize: bool = True, as_tensor: bool = False,
                   flatten: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MNIST training data.

        Args:
            normalize: Normalize pixel values to [0, 1]
            as_tensor: Return as kpu.Tensor instead of numpy array
            flatten: Flatten images to [N, 784] for MLP

        Returns:
            (images, labels) tuple
        """
        self._download()

        images = _read_idx_images(self.cache_dir / 'train-images-idx3-ubyte.gz')
        labels = _read_idx_labels(self.cache_dir / 'train-labels-idx1-ubyte.gz')

        return self._process(images, labels, normalize, as_tensor, flatten)

    def load_test(self, normalize: bool = True, as_tensor: bool = False,
                  flatten: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load MNIST test data.

        Args:
            normalize: Normalize pixel values to [0, 1]
            as_tensor: Return as kpu.Tensor instead of numpy array
            flatten: Flatten images to [N, 784] for MLP

        Returns:
            (images, labels) tuple
        """
        self._download()

        images = _read_idx_images(self.cache_dir / 't10k-images-idx3-ubyte.gz')
        labels = _read_idx_labels(self.cache_dir / 't10k-labels-idx1-ubyte.gz')

        return self._process(images, labels, normalize, as_tensor, flatten)

    def _process(self, images: np.ndarray, labels: np.ndarray,
                 normalize: bool, as_tensor: bool, flatten: bool):
        """Process loaded data."""
        # Convert to float32
        images = images.astype(np.float32)

        if normalize:
            images = images / 255.0

        if flatten:
            # Flatten to [N, 784]
            images = images.reshape(images.shape[0], -1)
        else:
            # Add channel dimension: [N, H, W] -> [N, 1, H, W]
            images = images[:, np.newaxis, :, :]

        if as_tensor:
            images = Tensor(images)

        return images, labels

    def get_batch(self, images: np.ndarray, labels: np.ndarray,
                  batch_idx: int, batch_size: int = 32,
                  as_tensor: bool = True) -> Tuple:
        """
        Get a batch from the dataset.

        Args:
            images: Full image array
            labels: Full label array
            batch_idx: Batch index
            batch_size: Batch size
            as_tensor: Return as kpu.Tensor

        Returns:
            (batch_images, batch_labels) tuple
        """
        start = batch_idx * batch_size
        end = min(start + batch_size, len(images))

        batch_images = images[start:end]
        batch_labels = labels[start:end]

        if as_tensor:
            batch_images = Tensor(batch_images)

        return batch_images, batch_labels


def load_mnist(split: str = 'test', normalize: bool = True,
               flatten: bool = False, limit: Optional[int] = None):
    """
    Convenience function to load MNIST.

    Args:
        split: 'train' or 'test'
        normalize: Normalize to [0, 1]
        flatten: Flatten to 784 features
        limit: Limit number of samples (useful for quick tests)

    Returns:
        (images, labels) as numpy arrays
    """
    mnist = MNIST()

    if split == 'train':
        images, labels = mnist.load_train(normalize=normalize, flatten=flatten)
    elif split == 'test':
        images, labels = mnist.load_test(normalize=normalize, flatten=flatten)
    else:
        raise ValueError(f"Unknown split: {split}")

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    return images, labels
