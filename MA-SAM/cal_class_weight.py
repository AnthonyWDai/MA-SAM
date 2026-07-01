import csv
import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import DiceLoss
from datasets.dataset_psmav2 import TrainTransform, ValTransform, PSMADataset


def class_weights_csv_path(root_path):
    return os.path.join(root_path, "class_weights.csv")


def save_class_weights_to_csv(weights, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    weights_cpu = weights.detach().cpu().tolist()
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "weight"])
        for class_id, weight in enumerate(weights_cpu):
            writer.writerow([class_id, float(weight)])


def load_class_weights_from_csv(csv_path, num_classes):
    expected_len = num_classes + 1
    weights = [None] * expected_len

    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header.")

        required_columns = {"class_id", "weight"}
        if not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV file {csv_path} must contain columns {required_columns}, "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            class_id = int(row["class_id"])
            weight = float(row["weight"])
            if class_id < 0 or class_id >= expected_len:
                raise ValueError(
                    f"class_id {class_id} in {csv_path} is out of valid range "
                    f"[0, {expected_len - 1}]"
                )
            weights[class_id] = weight

    missing = [i for i, w in enumerate(weights) if w is None]
    if missing:
        raise ValueError(
            f"CSV file {csv_path} is missing weights for class ids: {missing}"
        )

    return torch.tensor(weights, dtype=torch.float32)


def compute_ce_class_weights(dataset, num_classes, clamp_max=None, background_scale=1.0):
    """
    Compute inverse-frequency class weights for CE from dataset labels.
    Assumes:
      - labels contain class ids in [0, num_classes]
      - class 0 is background
      - total number of classes for CE is num_classes + 1
    """
    counts = torch.zeros(num_classes + 1, dtype=torch.float64)

    for idx in tqdm(range(len(dataset)), desc="Computing CE class weights", ncols=70, leave=False):
        sample = dataset[idx]
        label = sample["label"]

        if not torch.is_tensor(label):
            label = torch.as_tensor(label)

        label = label.long().view(-1)
        valid = (label >= 0) & (label <= num_classes)
        label = label[valid]

        if label.numel() == 0:
            continue

        bincount = torch.bincount(label, minlength=num_classes + 1).to(torch.float64)
        counts += bincount

    if counts.sum() == 0:
        raise ValueError("No valid labels found to compute CE class weights.")

    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean()

    weights[0] = weights[0] * background_scale

    if clamp_max is not None:
        weights[1:] = torch.clamp(weights[1:], max=clamp_max)

    weights = weights / weights.mean()
    return weights.float()


def get_or_create_ce_class_weights(root_path, dataset, num_classes, clamp_max=None, background_scale=1.0):
    """
    If class_weights.csv exists in root_path, load and return it.
    Otherwise compute class weights, save them to CSV, and return them.

    Original input params are preserved:
      - root_path
      - dataset
      - num_classes
      - clamp_max
      - background_scale
    """
    csv_path = class_weights_csv_path(root_path)

    if os.path.exists(csv_path):
        logging.info("Found existing class weights CSV at %s. Loading without recomputation.", csv_path)
        weights = load_class_weights_from_csv(csv_path, num_classes)
    else:
        logging.info("No class weights CSV found at %s. Computing class weights.", csv_path)
        weights = compute_ce_class_weights(
            dataset=dataset,
            num_classes=num_classes,
            clamp_max=clamp_max,
            background_scale=background_scale,
        )
        save_class_weights_to_csv(weights, csv_path)
        logging.info("Saved computed class weights to %s", csv_path)

    return weights


if __name__ == "__main__":
    # export MASAM_DATASET="xxx"
    train_transform = TrainTransform(
        output_size=(512, 512),
        low_res=(512, 512)
    )

    train_dataset = PSMADataset(
        base_dir=os.environ["MASAM_DATASET"],
        split="train",
        transform=train_transform,
    )

    ce_class_weights = get_or_create_ce_class_weights(
        root_path=os.environ["MASAM_DATASET"],
        dataset=train_dataset,
        num_classes=1,
        clamp_max=5.0,
        background_scale=0.5,
    )