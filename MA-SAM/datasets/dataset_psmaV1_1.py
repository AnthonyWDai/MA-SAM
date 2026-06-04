import os
import random
from pathlib import Path

import blosc2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def load_array(path):
    """
    Load array from .npy or .b2nd.
    """
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".b2nd":
        arr = blosc2.open(path)[:]
    else:
        raise ValueError(f"Unsupported file format: {path}")

    return np.asarray(arr)


def normalize_image(image):
    image = image.astype(np.float32)
    vmin = image.min()
    vmax = image.max()
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image


def ensure_3d(arr, is_mask=False):
    """
    Convert nnU-Net preprocessed arrays to (H, W, D).

    Common possible shapes:
    - (H, W, D)
    - (1, H, W, D)
    - (C, H, W, D) -> use first channel for image if needed
    """
    arr = np.asarray(arr)

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        # nnU-Net preprocessed image often has shape (C, H, W, D)
        # mask may also appear as (1, H, W, D)
        if arr.shape[0] == 1:
            return arr[0]
        if is_mask:
            return arr[0]
        return arr[0]

    raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")


def random_flip(image, label=None):
    axis = random.choice([0, 1])  # flip H or W only
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label=None, angle_range=15):
    angle = random.uniform(-angle_range, angle_range)
    image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=1)
    if label is not None:
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False, order=0)
    return image, label


class SimpleTransform:
    def __init__(self, output_size=None, low_res=None, augment=False):
        """
        output_size: tuple (H, W) or None
        low_res: tuple (H, W) or None
        augment: apply light augmentation for training only
        """
        self.output_size = output_size
        self.low_res = low_res
        self.augment = augment

    def __call__(self, sample):
        image = sample["image"]
        label = sample.get("label", None)

        if self.augment:
            if random.random() < 0.5:
                image, label = random_flip(image, label)
            if random.random() < 0.3:
                image, label = random_rotate(image, label)

        if self.output_size is not None:
            h, w, d = image.shape
            target_h, target_w = self.output_size
            if (h, w) != (target_h, target_w):
                image = zoom(image, (target_h / h, target_w / w, 1.0), order=3)
                if label is not None:
                    label = zoom(label, (target_h / h, target_w / w, 1.0), order=0)

        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (D, H, W)

        output = {"image": image}

        if label is not None:
            label = torch.from_numpy(label.astype(np.int64)).permute(2, 0, 1)
            output["label"] = label

            if self.low_res is not None:
                lh, lw = self.low_res
                _, h, w = label.shape
                low_res_label = zoom(
                    label.numpy().transpose(1, 2, 0),
                    (lh / h, lw / w, 1.0),
                    order=0
                )
                low_res_label = torch.from_numpy(low_res_label.astype(np.int64)).permute(2, 0, 1)
                output["low_res_label"] = low_res_label

        if "case_name" in sample:
            output["case_name"] = sample["case_name"]

        return output


def find_image_mask_pairs(base_dir, exts=(".b2nd", ".npy")):
    """
    Find pairs:
      image: xxx.ext
      mask : xxx_seg.ext
    """
    base_dir = Path(base_dir)
    files = []

    for ext in exts:
        files.extend(base_dir.rglob(f"*{ext}"))

    files = [f for f in files if f.is_file()]

    image_files = []
    for f in files:
        name = f.name
        if "_seg" in f.stem:
            continue
        image_files.append(f)

    pairs = []
    for img_path in image_files:
        mask_path = img_path.with_name(f"{img_path.stem}_seg{img_path.suffix}")
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path)))
        else:
            # allow unlabeled/test-only images
            pairs.append((str(img_path), None))

    pairs.sort()
    return pairs


def create_split_if_missing(base_dir, split_ratio=0.8, seed=42):
    """
    Create masam/train.csv and masam/test.csv if missing.
    """
    base_dir = Path(base_dir)
    split_dir = base_dir / "masam"
    train_csv = split_dir / "train.csv"
    test_csv = split_dir / "test.csv"

    if train_csv.exists() and test_csv.exists():
        return

    split_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_image_mask_pairs(base_dir)

    if len(pairs) == 0:
        raise RuntimeError(f"No .b2nd or .npy files found under {base_dir}")

    random.Random(seed).shuffle(pairs)

    n_train = max(1, int(len(pairs) * split_ratio))
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:] if len(pairs) > 1 else pairs

    train_df = pd.DataFrame(train_pairs, columns=["image_pth", "mask_pth"])
    test_df = pd.DataFrame(test_pairs, columns=["image_pth", "mask_pth"])

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)


class PSMADataset(Dataset):
    def __init__(
        self,
        base_dir,
        split="train",
        transform=None,
        create_split=True,
        split_ratio=0.8,
        seed=42,
        allow_missing_mask=True,
    ):
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform = transform
        self.allow_missing_mask = allow_missing_mask

        if create_split:
            create_split_if_missing(self.base_dir, split_ratio=split_ratio, seed=seed)

        split_file_map = {
            "train": self.base_dir / "masam" / "train.csv",
            "test": self.base_dir / "masam" / "test.csv",
            "val": self.base_dir / "masam" / "test.csv",
        }

        if split not in split_file_map:
            raise ValueError(f"Unsupported split: {split}. Use 'train', 'test', or 'val'.")

        csv_path = split_file_map[split]
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if "image_pth" not in df.columns:
            raise ValueError(f"{csv_path} must contain column 'image_pth'")
        if "mask_pth" not in df.columns:
            df["mask_pth"] = None

        self.sample_list = df["image_pth"].tolist()
        self.mask_list = df["mask_pth"].tolist()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_path = self.sample_list[idx]
        mask_path = self.mask_list[idx] if idx < len(self.mask_list) else None

        image = load_array(image_path)
        image = ensure_3d(image, is_mask=False)
        image = normalize_image(image)

        sample = {
            "image": image,
            "case_name": Path(image_path).stem,
        }

        if isinstance(mask_path, str) and mask_path and os.path.exists(mask_path):
            mask = load_array(mask_path)
            mask = ensure_3d(mask, is_mask=True)
            mask = mask.astype(np.int64)

            sample["label"] = mask
        elif not self.allow_missing_mask:
            raise FileNotFoundError(f"Mask not found for image: {image_path}")

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            sample["image"] = torch.from_numpy(sample["image"].astype(np.float32)).permute(2, 0, 1)
            if "label" in sample:
                sample["label"] = torch.from_numpy(sample["label"].astype(np.int64)).permute(2, 0, 1)

        return sample