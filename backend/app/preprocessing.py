"""
FloraLens Preprocessing — Image transforms for training, validation, and inference.
Handles resizing, normalization, and augmentation with deterministic seeding.
"""
import io
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.config import IMAGE_SIZE, RANDOM_SEED

# ImageNet normalization statistics (used by all timm pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """Training augmentation pipeline with aggressive augmentations to
    combat overfitting on the small Oxford-102 dataset (~8k images).

    Key choices:
    - RandomResizedCrop forces scale invariance (flowers at different distances)
    - ColorJitter simulates lighting variation in outdoor photography
    - RandomErasing acts as a regularizer similar to Cutout
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_val_transforms() -> transforms.Compose:
    """Deterministic validation/test transforms — resize + center crop only."""
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.1)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform() -> transforms.Compose:
    """Inference transform matching validation — used by the API."""
    return get_val_transforms()


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert raw uploaded bytes → preprocessed numpy array for ONNX inference.

    Args:
        image_bytes: Raw image bytes from the upload.

    Returns:
        np.ndarray of shape (1, 3, IMAGE_SIZE, IMAGE_SIZE), float32.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)
    return tensor.numpy().astype(np.float32)


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image → preprocessed numpy array for ONNX inference."""
    image = image.convert("RGB")
    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0)
    return tensor.numpy().astype(np.float32)


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set seeds for reproducibility across all libraries.

    Note: torch.backends.cudnn.benchmark is left True for performance.
    This is one source of remaining nondeterminism we accept — cuDNN's
    autotuner may select different algorithms across runs, causing small
    (<0.1%) variance in validation metrics.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Sacrifice speed for determinism
    import random
    random.seed(seed)
