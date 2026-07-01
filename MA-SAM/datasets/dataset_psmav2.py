import sys
import pickle
import random
from pathlib import Path

import numpy as np
import numpy.core.numeric

import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


sys.modules["numpy._core.numeric"] = numpy.core.numeric


def read_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    vmin = image.min()
    vmax = image.max()
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image


def random_rot_flip(image: np.ndarray, label: np.ndarray):
    """
    image, label: (D, H, W)

    Apply in-plane augmentation on each slice by rotating/flipping
    over the H-W plane, i.e. axes (1, 2).
    """
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2)).copy()
    label = np.rot90(label, k, axes=(1, 2)).copy()

    axis = np.random.choice([1, 2])
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: np.ndarray, angle_range=(-15, 15)):
    """
    image, label: (D, H, W)

    Rotate over the H-W plane, i.e. axes (1, 2).
    """
    angle = np.random.randint(angle_range[0], angle_range[1] + 1)

    image = ndimage.rotate(
        image,
        angle,
        axes=(1, 2),
        reshape=False,
        order=3,
        mode="nearest",
    )

    label = ndimage.rotate(
        label,
        angle,
        axes=(1, 2),
        reshape=False,
        order=0,
        mode="nearest",
    )
    return image, label


def resize_longest_and_pad(image: np.ndarray, label: np.ndarray, output_size):
    """
    image, label: (D, H, W)
    output_size: (target_h, target_w)

    Resizes while preserving aspect ratio so that the longest spatial
    dimension fits within output_size, then pads to exact output_size.
    """
    d, h, w = image.shape
    target_h, target_w = output_size

    scale = min(target_h / h, target_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    if (new_h, new_w) != (h, w):
        image = zoom(image, (1.0, new_h / h, new_w / w), order=3)
        label = zoom(label, (1.0, new_h / h, new_w / w), order=0)

    pad_h = target_h - new_h
    pad_w = target_w - new_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    image = np.pad(
        image,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    label = np.pad(
        label,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    return image, label


def merge_axes(arr, ax1, ax2):
    shape = arr.shape
    axes = list(range(arr.ndim))
    
    # put ax1 and ax2 together in front
    remaining = [a for a in axes if a not in (ax1, ax2)]
    new_order = [ax1, ax2] + remaining
    
    transposed = arr.transpose(new_order)
    new_shape = (shape[ax1] * shape[ax2],) + tuple(shape[a] for a in remaining)
    
    return transposed.reshape(new_shape)


def undo_merge(y, original_shape, ax1, ax2):
    """
    Undo a merge performed as:
        x.permute(ax1, ax2, *remaining).reshape(...)
    
    Args:
        y: torch.Tensor after merge
        original_shape: tuple/list of original shape
        ax1, ax2: axes that were merged, in the same order used in merge
    
    Returns:
        Recovered tensor with shape == original_shape
    """
    ndim = len(original_shape)
    axes = list(range(ndim))
    remaining = [a for a in axes if a not in (ax1, ax2)]

    # shape after permute, before merge
    premerge_shape = (
        original_shape[ax1],
        original_shape[ax2],
        *[original_shape[a] for a in remaining]
    )

    z = y.reshape(premerge_shape)

    # forward permute order used during merge
    forward_order = [ax1, ax2] + remaining

    # inverse permutation
    inverse_order = [0] * ndim
    for i, a in enumerate(forward_order):
        inverse_order[a] = i

    return z.permute(*inverse_order)


class TrainTransform:
    def __init__(self, output_size, low_res):
        """
        output_size: (H, W)
        low_res: (H_low, W_low)
        Input image/label shape: (D, H, W)
        """
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)

        if random.random() > 0.5:
            image, label = random_rotate(image, label)

        image, label = resize_longest_and_pad(image, label, self.output_size)

        # _, label_h, label_w = label.shape
        # low_h, low_w = self.low_res
        # low_res_label = zoom(
        #     label,
        #     (1.0, low_h / label_h, low_w / label_w),
        #     order=0,
        # )

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.int64))
        # low_res_label = torch.from_numpy(low_res_label.astype(np.int64))

        return {
            "image": image,                 # (D, H, W)
            "label": label,                 # (D, H, W)
            # "low_res_label": low_res_label, # (D, H_low, W_low)
        }


class ValTransform:
    def __init__(self, output_size, low_res):
        """
        output_size: (H, W)
        low_res: (H_low, W_low)
        Input image/label shape: (D, H, W)
        """
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        image, label = resize_longest_and_pad(image, label, self.output_size)

        # _, label_h, label_w = label.shape
        # low_h, low_w = self.low_res
        # low_res_label = zoom(
        #     label,
        #     (1.0, low_h / label_h, low_w / label_w),
        #     order=0,
        # )

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.int64))
        # low_res_label = torch.from_numpy(low_res_label.astype(np.int64))

        return {
            "image": image,                 # (D, H, W)
            "label": label,                 # (D, H, W)
            # "low_res_label": low_res_label, # (D, H_low, W_low)
        }


class PSMADataset(Dataset):
    """
    Expected directory structure:

    base_dir/
      train/
        images/
          case001/
            chunk_0000.pkl
            chunk_0001.pkl
        masks/
          case001/
            chunk_0000.pkl
            chunk_0001.pkl
      val/
        images/
        masks/

    Each pkl file should contain:
      {"data": np.ndarray of shape (D, H, W)}
    """

    def __init__(self, base_dir, split, transform=None):
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform

        self.images_dir = self.base_dir / split / "images"
        self.masks_dir = self.base_dir / split / "masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        image_files = sorted(self.images_dir.rglob("*.pkl"))
        if len(image_files) == 0:
            raise RuntimeError(f"No .pkl files found under {self.images_dir}")

        for img_path in image_files:
            rel_path = img_path.relative_to(self.images_dir)
            mask_path = self.masks_dir / rel_path

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for image:\n"
                    f"  image: {img_path}\n"
                    f"  expected mask: {mask_path}"
                )

            samples.append(
                {
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "case_name": str(rel_path),
                }
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image_obj = read_pkl(item["image_path"])
        mask_obj = read_pkl(item["mask_path"])

        # input shape is D, H, W
        image = np.asarray(image_obj, dtype=np.float32)
        mask = np.asarray(mask_obj)

        if image.ndim == 4:
            image = merge_axes(image, 0, 3)

        if mask.ndim == 4:
            mask = merge_axes(mask, 0, 3)

        if image.ndim != 3:
            raise ValueError(
                f"Expected image to have shape (D, H, W), got {image.shape} "
                f"for file {item['image_path']}"
            )

        if mask.ndim != 3:
            raise ValueError(
                f"Expected mask to have shape (D, H, W), got {mask.shape} "
                f"for file {item['mask_path']}"
            )

        if image.shape[1:] != mask.shape[1:]:
            raise ValueError(
                f"Image-mask shape mismatch:\n"
                f"  image: {item['image_path']} shape={image.shape}\n"
                f"  mask : {item['mask_path']} shape={mask.shape}"
            )

        image = normalize_image(image)
        mask = mask.astype(np.int8)

        sample = {
            "image": image,
            "label": mask,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        C = 3
        DC, H, W = sample["image"].shape
        sample["image"] = undo_merge(sample["image"], (DC // C, C, H, W), 0, 1)

        sample["case_name"] = item["case_name"]
        return sample