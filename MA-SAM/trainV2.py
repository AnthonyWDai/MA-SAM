import argparse
import os
import random
from importlib import import_module

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from segment_anything import sam_model_registry
from trainerV2 import trainer_run


def parse_args():
    parser = argparse.ArgumentParser(description="Train FacT-TT adapted SAM for segmentation")

    # Paths
    parser.add_argument(
        "--root_path",
        type=str,
        default="/mnt/weka/wekafs/rad-megtron/cchen/synapseCT/Training/2D_all_5slice",
        help="Root directory for training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/weka/wekafs/rad-megtron/cchen/project_results/MA_SAM/results-1",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/weka/wekafs/rad-megtron/cchen/PretrainedModel/sam_vit_h_4b8939.pth",
        help="Path to pretrained SAM checkpoint",
    )
    parser.add_argument(
        "--adapt_ckpt",
        type=str,
        default=None,
        help="Path to finetuned/adapted checkpoint",
    )

    # Model
    parser.add_argument("--vit_name", type=str, default="vit_h", help="SAM ViT backbone")
    parser.add_argument("--num_classes", type=int, default=12, help="Number of output classes")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size")
    parser.add_argument("--module", type=str, default="sam_fact_tt_image_encoder", help="Adapter module to import")
    parser.add_argument("--rank", type=int, default=32, help="Rank for FacT")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for FacT")

    # Optimization
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--n_gpu", type=int, default=8, help="Total number of GPUs")
    parser.add_argument("--base_lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=400, help="Maximum number of training epochs")
    parser.add_argument("--stop_epoch", type=int, default=300, help="Early stopping epoch")
    parser.add_argument("--dice_param", type=float, default=0.8, help="Dice loss parameter")
    parser.add_argument("--lr_exp", type=float, default=7, help="Learning rate decay exponent")
    parser.add_argument("--AdamW", action="store_true", help="Use AdamW optimizer")

    # Warmup
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm up learning rate from a lower value to base_lr",
    )
    parser.add_argument(
        "--warmup_period",
        type=int,
        default=250,
        help="Warmup iterations (used only if --warmup is enabled)",
    )

    # Reproducibility
    parser.add_argument("--deterministic", action='store_false', help="Use deterministic training")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # Acceleration
    parser.add_argument('--tf32', action='store_false', help='If activated, use tf32 to accelerate the training process')
    parser.add_argument('--compile', action='store_true', help='If activated, compile the training model for acceleration')
    parser.add_argument('--use_amp', action='store_false', help='If activated, adopt mixed precision for acceleration')
    parser.add_argument('--skip_hard', action='store_false', help='If activated, adopt mixed precision for acceleration')

    args = parser.parse_args()

    return args


def configure_runtime(args):
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_model(args):
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )

    adapter_module = import_module(args.module)
    net = adapter_module.Fact_tt_Sam(sam, args.rank, s=args.scale).cuda()

    if args.compile:
        net = torch.compile(net)

    if args.adapt_ckpt is not None:
        net.load_parameters(args.adapt_ckpt)

    return net, img_embedding_size


def save_config(args, output_dir: str):
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


def main():
    args = parse_args()
    configure_runtime(args)
    ensure_output_dir(args.output)

    net, img_embedding_size = build_model(args)

    multimask_output = args.num_classes > 1
    low_res = img_embedding_size * 4

    save_config(args, args.output)

    trainer_run(
        args=args,
        model=net,
        snapshot_path=args.output,
        multimask_output=multimask_output,
        low_res=low_res,
    )


if __name__ == "__main__":
    main()