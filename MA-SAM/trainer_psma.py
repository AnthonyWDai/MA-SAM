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
from datasets.dataset_psma import TrainTransform, ValTransform, PSMADataset


def recommended_num_workers(reserve=1, train=True):
    if hasattr(os, "sched_getaffinity"):
        n_cpu = len(os.sched_getaffinity(0))
    else:
        n_cpu = os.cpu_count() or 1
    usable = max(1, n_cpu - reserve)
    if train:
        return max(1, min(usable, 4))
    else:
        return max(1, min(usable // 2, 2))


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


def calc_loss(outputs, label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    logits = outputs["masks"]
    target_ce = label_batch.long()
    loss_ce = ce_loss(logits, target_ce)
    loss_dice = dice_loss(logits, label_batch.unsqueeze(1), softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def compute_seg_dice_stats(pred, target, num_classes, eps=1e-5):
    """
    pred:   [B, H, W] predicted class ids
    target: [B, H, W] ground truth class ids

    Returns:
        dice_sum: sum of Dice scores over valid (sample, class) pairs
        valid_count: number of valid (sample, class) pairs

    Valid means the class is present in pred or target.
    Absent-in-both cases are excluded from aggregation.
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"

    reduce_dims = tuple(range(1, pred.ndim))
    dice_sum = 0.0
    valid_count = 0

    for cls in range(1, num_classes + 1):
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()

        intersect = (pred_c * target_c).sum(dim=reduce_dims)
        pred_sum = pred_c.sum(dim=reduce_dims)
        target_sum = target_c.sum(dim=reduce_dims)
        denom = pred_sum + target_sum
        valid = denom > 0

        if valid.any():
            dice = (2.0 * intersect[valid] + eps) / (denom[valid] + eps)
            dice_sum += dice.sum().item()
            valid_count += valid.sum().item()

    return dice_sum, valid_count


@torch.no_grad()
def validate(args, model, valloader, ce_loss, dice_loss, multimask_output):
    model.eval()

    val_loss = 0.0
    val_ce = 0.0
    val_dice_loss = 0.0
    val_metric_dice_total = 0.0
    val_metric_dice_count = 0
    num_batches = 0

    for sampled_batch in tqdm(valloader, desc="Validation", ncols=70, leave=False):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]

        image_batch = image_batch.unsqueeze(2)
        image_batch = torch.cat((image_batch, image_batch, image_batch), dim=2)

        label_batch = label_batch.contiguous().long()

        image_batch = image_batch.cuda(non_blocking=True)
        label_batch = label_batch.cuda(non_blocking=True)
        label_batch = label_batch.reshape(-1, *label_batch.shape[2:])

        if args.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(
                    outputs, label_batch, ce_loss, dice_loss, args.dice_param
                )
        else:
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(
                outputs, label_batch, ce_loss, dice_loss, args.dice_param
            )

        pred_masks = outputs["masks"]
        pred_masks = torch.argmax(torch.softmax(pred_masks, dim=1), dim=1)

        dice_sum, dice_count = compute_seg_dice_stats(
            pred_masks,
            label_batch,
            args.num_classes,
        )

        val_loss += loss.item()
        val_ce += loss_ce.item()
        val_dice_loss += loss_dice.item()
        val_metric_dice_total += dice_sum
        val_metric_dice_count += dice_count
        num_batches += 1

    model.train()

    if num_batches == 0:
        return {
            "loss": float("inf"),
            "loss_ce": float("inf"),
            "loss_dice": float("inf"),
            "metric_dice": 0.0,
        }

    return {
        "loss": val_loss / num_batches,
        "loss_ce": val_ce / num_batches,
        "loss_dice": val_dice_loss / num_batches,
        "metric_dice": (
            val_metric_dice_total / val_metric_dice_count
            if val_metric_dice_count > 0 else 0.0
        ),
    }


def save_model(model, path):
    try:
        model.save_parameters(path)
    except AttributeError:
        model.module.save_parameters(path)


def trainer_run(args, model, snapshot_path, multimask_output, low_res):
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    validation_interval = args.validation_interval or max(1, round(args.max_epochs * 0.1))
    if validation_interval < 1:
        raise ValueError(f"validation_interval must be >= 1, got {validation_interval}")

    train_transform = TrainTransform(
        output_size=(args.img_size, args.img_size),
        low_res=(low_res, low_res)
    )
    val_transform = ValTransform(
        output_size=(args.img_size, args.img_size),
        low_res=(low_res, low_res)
    )

    train_dataset = PSMADataset(
        base_dir=args.root_path,
        split="train",
        transform=train_transform,
    )
    val_dataset = PSMADataset(
        base_dir=args.root_path,
        split="val",
        transform=val_transform,
    )

    print("The length of train set is: {}".format(len(train_dataset)))
    print("The length of val set is: {}".format(len(val_dataset)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_workers = recommended_num_workers(train=True)
    val_workers = recommended_num_workers(train=False)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    ce_class_weights = get_or_create_ce_class_weights(
        root_path=args.root_path,
        dataset=train_dataset,
        num_classes=num_classes,
        clamp_max=args.ce_weight_clamp_max,
        background_scale=args.ce_background_scale,
    ).cuda()

    logging.info("CE class weights: %s", ce_class_weights.detach().cpu().tolist())
    logging.info("Class weights CSV path: %s", class_weights_csv_path(args.root_path))

    ce_loss = CrossEntropyLoss(
        weight=ce_class_weights,
        ignore_index=-100,
    )
    dice_loss = DiceLoss(num_classes + 1)

    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    if args.AdamW:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=b_lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=b_lr,
            momentum=0.9,
            weight_decay=0.0001,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp) if args.use_amp else None
    writer = SummaryWriter(snapshot_path + "/log")

    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    best_val_dice = -1.0

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    logging.info("Validation interval: every %d epoch(s)", validation_interval)

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]

            image_batch = image_batch.unsqueeze(2)
            image_batch = torch.cat((image_batch, image_batch, image_batch), dim=2)
            label_batch = label_batch.contiguous().long()

            image_batch = image_batch.cuda(non_blocking=True)
            label_batch = label_batch.cuda(non_blocking=True)
            label_batch = label_batch.reshape(-1, *label_batch.shape[2:])

            optimizer.zero_grad()

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size)
                    loss, loss_ce, loss_dice = calc_loss(
                        outputs, label_batch, ce_loss, dice_loss, args.dice_param
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(
                    outputs, label_batch, ce_loss, dice_loss, args.dice_param
                )
                loss.backward()
                optimizer.step()

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f"Shift iter is {shift_iter}, smaller than zero"
                    decay_total = max(1, max_iterations - args.warmup_period)
                else:
                    shift_iter = iter_num
                    decay_total = max(1, max_iterations)

                lr_ = base_lr * ((1.0 - shift_iter / decay_total) ** args.lr_exp)
                lr_ = max(lr_, 0.0)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("train/total_loss", loss.item(), iter_num)
            writer.add_scalar("train/loss_ce", loss_ce.item(), iter_num)
            writer.add_scalar("train/loss_dice", loss_dice.item(), iter_num)

            logging.info(
                "iteration %d : loss : %f, loss_ce: %f, loss_dice: %f"
                % (iter_num, loss.item(), loss_ce.item(), loss_dice.item())
            )

        current_epoch = epoch_num + 1
        is_final_epoch = (epoch_num >= max_epoch - 1) or (epoch_num >= stop_epoch - 1)
        should_validate = (current_epoch % validation_interval == 0) or is_final_epoch

        if should_validate:
            val_metrics = validate(args, model, valloader, ce_loss, dice_loss, multimask_output)

            writer.add_scalar("val/total_loss", val_metrics["loss"], current_epoch)
            writer.add_scalar("val/loss_ce", val_metrics["loss_ce"], current_epoch)
            writer.add_scalar("val/loss_dice", val_metrics["loss_dice"], current_epoch)
            writer.add_scalar("val/metric_dice", val_metrics["metric_dice"], current_epoch)

            logging.info(
                "epoch %d validation : val_loss : %f, val_ce: %f, val_dice_loss: %f, val_metric_dice: %f"
                % (
                    current_epoch,
                    val_metrics["loss"],
                    val_metrics["loss_ce"],
                    val_metrics["loss_dice"],
                    val_metrics["metric_dice"],
                )
            )

            if val_metrics["metric_dice"] > best_val_dice:
                best_val_dice = val_metrics["metric_dice"]
                best_model_path = os.path.join(snapshot_path, "best_model.pth")
                save_model(model, best_model_path)
                logging.info("save best model to {}".format(best_model_path))

        save_interval = 20
        if current_epoch % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, "epoch_" + str(epoch_num) + ".pth")
            save_model(model, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if is_final_epoch:
            save_mode_path = os.path.join(snapshot_path, "epoch_" + str(epoch_num) + ".pth")
            save_model(model, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
