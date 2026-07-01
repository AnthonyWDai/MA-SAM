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

        d, h, w = image.shape
        target_h, target_w = self.output_size

        if (h, w) != (target_h, target_w):
            image = zoom(image, (1.0, target_h / h, target_w / w), order=3)
            label = zoom(label, (1.0, target_h / h, target_w / w), order=0)

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

        d, h, w = image.shape
        target_h, target_w = self.output_size

        if (h, w) != (target_h, target_w):
            image = zoom(image, (1.0, target_h / h, target_w / w), order=3)
            label = zoom(label, (1.0, target_h / h, target_w / w), order=0)

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
        image = np.asarray(image_obj["data"], dtype=np.float32)
        mask = np.asarray(mask_obj["data"])

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

        if image.shape != mask.shape:
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

        sample["case_name"] = item["case_name"]
        return sample