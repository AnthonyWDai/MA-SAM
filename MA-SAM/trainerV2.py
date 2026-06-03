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


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    low_res_logits = outputs['low_res_logits']
    # print("low_res_logits.shape:", low_res_logits.shape)
    # print("low_res_labels.shape:", low_res_label_batch.shape)

    loss_ce = ce_loss(low_res_logits, low_res_label_batch.long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


@torch.no_grad()
def validate(args, model, valloader, ce_loss, dice_loss, multimask_output):
    model.eval()

    val_loss = 0.0
    val_ce = 0.0
    val_dice = 0.0
    num_batches = 0

    for sampled_batch in tqdm(valloader, desc="Validation", ncols=70, leave=False):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch = image_batch.unsqueeze(2)
        image_batch = torch.cat((image_batch, image_batch, image_batch), dim=2)

        hw_size = image_batch.shape[-1]
        label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)
        low_res_label_batch = sampled_batch['low_res_label']

        image_batch = image_batch.cuda(non_blocking=True)
        label_batch = label_batch.cuda(non_blocking=True)
        low_res_label_batch = low_res_label_batch.cuda(non_blocking=True)

        if args.use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(
                    outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param
                )
        else:
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(
                outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param
            )

        val_loss += loss.item()
        val_ce += loss_ce.item()
        val_dice += loss_dice.item()
        num_batches += 1

    model.train()

    if num_batches == 0:
        return {
            "loss": float("inf"),
            "loss_ce": float("inf"),
            "loss_dice": float("inf"),
        }

    return {
        "loss": val_loss / num_batches,
        "loss_ce": val_ce / num_batches,
        "loss_dice": val_dice / num_batches,
    }


def trainer_run(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset_psmav2 import SimpleTransform, PSMADataset

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


    db_train = PSMADataset(
        base_dir=args.root_path,
        split="train",
        transform=SimpleTransform(
            output_size=(args.img_size, args.img_size), 
            low_res=(low_res, low_res), 
            augment=True
        ),
    )

    db_val = PSMADataset(
        base_dir=args.root_path,
        split="val",
        transform=SimpleTransform(
            output_size=(args.img_size, args.img_size), 
            low_res=(low_res, low_res), 
            augment=True
        ),
    )

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_workers = recommended_num_workers(train=True)
    val_workers = recommended_num_workers(train=False)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    valloader = DataLoader(
        db_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    ce_loss = CrossEntropyLoss(ignore_index=-100)
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
            weight_decay=0.1
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=b_lr,
            momentum=0.9,
            weight_decay=0.0001
        )

    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp) if args.use_amp else None

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    best_val_loss = float("inf")

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch = image_batch.unsqueeze(2)
            image_batch = torch.cat((image_batch, image_batch, image_batch), dim=2)

            hw_size = image_batch.shape[-1]
            label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)
            low_res_label_batch = sampled_batch['low_res_label']

            image_batch = image_batch.cuda(non_blocking=True)
            label_batch = label_batch.cuda(non_blocking=True)
            low_res_label_batch = low_res_label_batch.cuda(non_blocking=True)

            optimizer.zero_grad()

            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size)
                    loss, loss_ce, loss_dice = calc_loss(
                        outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(
                    outputs, low_res_label_batch, ce_loss, dice_loss, args.dice_param
                )
                loss.backward()
                optimizer.step()

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('train/total_loss', loss.item(), iter_num)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f'
                % (iter_num, loss.item(), loss_ce.item(), loss_dice.item())
            )

        # validation every epoch
        val_metrics = validate(args, model, valloader, ce_loss, dice_loss, multimask_output)
        writer.add_scalar('val/total_loss', val_metrics["loss"], epoch_num + 1)
        writer.add_scalar('val/loss_ce', val_metrics["loss_ce"], epoch_num + 1)
        writer.add_scalar('val/loss_dice', val_metrics["loss_dice"], epoch_num + 1)

        logging.info(
            'epoch %d validation : val_loss : %f, val_ce: %f, val_dice: %f'
            % (epoch_num + 1, val_metrics["loss"], val_metrics["loss_ce"], val_metrics["loss_dice"])
        )

        # save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_path = os.path.join(snapshot_path, 'best_model.pth')
            try:
                model.save_parameters(best_model_path)
            except:
                model.module.save_parameters(best_model_path)
            logging.info("save best model to {}".format(best_model_path))

        save_interval = 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_parameters(save_mode_path)
            except:
                model.module.save_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"