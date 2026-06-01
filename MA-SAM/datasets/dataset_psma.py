import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = Image.fromarray(image.astype(np.uint8))
        label = Image.fromarray(label.astype(np.uint8))

        resize_img = transforms.Resize(self.output_size, interpolation=Image.BILINEAR)
        resize_mask = transforms.Resize(self.output_size, interpolation=Image.NEAREST)
        resize_low_res = transforms.Resize(self.low_res, interpolation=Image.NEAREST)

        image = resize_img(image)
        label_hr = resize_mask(label)
        label_lr = resize_low_res(label)

        image = np.array(image).astype(np.float32)
        label_hr = np.array(label_hr).astype(np.uint8)
        label_lr = np.array(label_lr).astype(np.uint8)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        return {
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label_hr).long(),
            'low_res_label': torch.from_numpy(label_lr).long()
        }


class PSMADataset(Dataset):
    """
    Supports both:
      1) flat:
         split/images/*.jpg
         split/masks/*.png

      2) nested:
         split/images/<case>/*.jpg
         split/masks/<case>/*.png
    """
    def __init__(self, base_dir, split="train", transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(base_dir, split, "images")
        self.mask_dir = os.path.join(base_dir, split, "masks")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.image_paths = sorted(
            glob.glob(os.path.join(self.image_dir, "**", "*.jpg"), recursive=True)
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .jpg files found under {self.image_dir}")

        self.samples = []
        missing_masks = []

        for image_path in self.image_paths:
            rel_path = os.path.relpath(image_path, self.image_dir)
            rel_stem = os.path.splitext(rel_path)[0]

            # preferred: preserve relative subfolder structure
            mask_path = os.path.join(self.mask_dir, rel_stem + ".png")

            # fallback: mask directly under masks/ with same basename
            if not os.path.isfile(mask_path):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                fallback_mask = os.path.join(self.mask_dir, base_name + ".png")
                if os.path.isfile(fallback_mask):
                    mask_path = fallback_mask
                else:
                    missing_masks.append((image_path, mask_path))
                    continue

            self.samples.append((image_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError("No matched image-mask pairs found.")

        if len(missing_masks) > 0:
            print(f"[WARN] {len(missing_masks)} images do not have matching masks and were skipped.")
            print(f"[WARN] Example missing pair: image={missing_masks[0][0]}, expected_mask={missing_masks[0][1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = np.array(image)
        mask = np.array(mask)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        sample = {
            'image': image,
            'label': mask
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['case_name'] = os.path.basename(image_path)
        return sample