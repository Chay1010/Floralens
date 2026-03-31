"""
FloraLens Preprocessing — Image transforms for inference (lightweight, no PyTorch).

Uses only PIL and NumPy to replicate the torchvision validation transforms:
  1. Resize to IMAGE_SIZE * 1.1
  2. Center crop to IMAGE_SIZE x IMAGE_SIZE
  3. Convert to float32 [0, 1]
  4. Normalize with ImageNet mean/std
  5. Transpose to (1, 3, H, W) for ONNX
"""
import io
from typing import Optional

import numpy as np
from PIL import Image

from app.config import IMAGE_SIZE

# ImageNet normalization statistics (used by all timm pretrained models)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_and_center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    """Resize so the shorter edge = crop_size * 1.1, then center-crop to crop_size."""
    target_size = int(crop_size * 1.1)

    # Resize preserving aspect ratio (shorter edge = target_size)
    w, h = image.size
    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image


def _normalize(img_array: np.ndarray) -> np.ndarray:
    """Normalize a (H, W, 3) float32 array with ImageNet stats, return (1, 3, H, W)."""
    # Scale [0, 255] -> [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    # Normalize per channel
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW, add batch dim -> (1, 3, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert raw uploaded bytes -> preprocessed numpy array for ONNX inference.

    Args:
        image_bytes: Raw image bytes from the upload.

    Returns:
        np.ndarray of shape (1, 3, IMAGE_SIZE, IMAGE_SIZE), float32.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = _resize_and_center_crop(image, IMAGE_SIZE)
    img_array = np.array(image)
    return _normalize(img_array)


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image -> preprocessed numpy array for ONNX inference."""
    image = image.convert("RGB")
    image = _resize_and_center_crop(image, IMAGE_SIZE)
    img_array = np.array(image)
    return _normalize(img_array)
