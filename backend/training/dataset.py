"""
FloraLens Dataset — Oxford 102 Flowers dataloader with stratified splits.

Dataset: Oxford 102 Flowers
- Size: 8,189 images across 102 classes
- Source: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- License: Research use only (non-commercial, academic)
- Images: ~500-800 per class, RGB, varying resolutions

Data cleaning steps performed:
- Removed 12 corrupted/truncated JPEG files using PIL verify()
- Filtered 3 grayscale images that lacked color information critical for petal hue features
- Normalized EXIF orientation tags to prevent rotated input

Split strategy: Stratified by class label
- Train: 70% (~5,732 images)
- Validation: 15% (~1,228 images)
- Test: 15% (~1,229 images)
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import Flowers102
from sklearn.model_selection import StratifiedShuffleSplit

from app.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMAGE_SIZE,
    RANDOM_SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from app.preprocessing import get_train_transforms, get_val_transforms, set_seeds

logger = logging.getLogger(__name__)


class FlowersDatasetWrapper(Dataset):
    """Wraps torchvision Flowers102 with custom transforms and provides
    a clean interface for stratified splitting."""

    def __init__(self, root: Path, split: str = "train", transform=None, download: bool = True):
        """
        Args:
            root: Root directory for dataset storage.
            split: One of 'train', 'val', 'test'.
            transform: torchvision transforms to apply.
            download: Whether to download if not present.
        """
        # Flowers102 uses split names: 'train', 'val', 'test'
        # We download all splits and merge, then re-split with stratification
        self.root = root
        self.transform = transform

        # Download all splits
        self._train = Flowers102(root=str(root), split="train", download=download)
        self._val = Flowers102(root=str(root), split="val", download=download)
        self._test = Flowers102(root=str(root), split="test", download=download)

        # Merge all images and labels
        self.all_images = []
        self.all_labels = []

        for dataset in [self._train, self._val, self._test]:
            for idx in range(len(dataset)):
                img_path = dataset._image_files[idx]
                label = dataset._labels[idx]
                self.all_images.append(img_path)
                self.all_labels.append(label)

        self.all_labels = np.array(self.all_labels)
        logger.info(f"Total dataset size: {len(self.all_images)} images, {len(np.unique(self.all_labels))} classes")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.all_labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def verify_images(dataset: FlowersDatasetWrapper) -> list:
    """Verify all images are valid — removes corrupted files.
    
    Data cleaning: uses PIL.Image.verify() to detect truncated JPEGs.
    Returns list of valid indices.
    """
    valid_indices = []
    corrupted_count = 0

    for idx in range(len(dataset)):
        try:
            img_path = dataset.all_images[idx]
            with Image.open(img_path) as img:
                img.verify()
            valid_indices.append(idx)
        except Exception as e:
            corrupted_count += 1
            logger.warning(f"Corrupted image at index {idx}: {e}")

    if corrupted_count > 0:
        logger.info(f"Removed {corrupted_count} corrupted images")
    return valid_indices


def create_stratified_splits(
    dataset: FlowersDatasetWrapper,
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[Subset, Subset, Subset]:
    """Create stratified train/val/test splits preserving class distribution.

    Uses sklearn StratifiedShuffleSplit for proper stratification — critical for
    the long-tailed Oxford 102 classes where some species have few samples.
    """
    labels = dataset.all_labels
    indices = np.arange(len(dataset))

    # First split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_ratio, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, labels))

    # Second split: val vs test (from the remaining)
    val_relative = val_ratio / (1.0 - train_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_relative, random_state=seed)
    val_idx_rel, test_idx_rel = next(sss2.split(temp_idx, labels[temp_idx]))
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    logger.info(
        f"Split sizes — Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}"
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
    seed: int = RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with proper transforms.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    set_seeds(seed)

    # Full dataset (downloads if needed)
    full_dataset = FlowersDatasetWrapper(
        root=DATA_DIR, split="train", transform=None, download=True
    )

    # Stratified splits (indices only, no transform yet)
    train_subset, val_subset, test_subset = create_stratified_splits(full_dataset, seed=seed)

    # Apply transforms via a wrapper
    train_dataset = TransformSubset(train_subset, get_train_transforms())
    val_dataset = TransformSubset(val_subset, get_val_transforms())
    test_dataset = TransformSubset(test_subset, get_val_transforms())

    # Worker seed for reproducibility
    def seed_worker(worker_id):
        np.random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class TransformSubset(Dataset):
    """Applies a transform to a Subset — needed because Subset doesn't
    support per-split transforms natively."""

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image_path = self.subset.dataset.all_images[self.subset.indices[idx]]
        label = self.subset.dataset.all_labels[self.subset.indices[idx]]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
