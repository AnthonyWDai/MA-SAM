import os
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from importlib import import_module
from tqdm import tqdm
from scipy.ndimage import zoom

from segment_anything import sam_model_registry

from batchgenerators.utilities.file_and_folder_operations import (
    load_json,
    maybe_mkdir_p,
    join,
)
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.export_prediction import export_prediction_from_logits


def setup_nnunet_objects(dataset_json_file, plans_json_file, configuration_name):
    dataset_json = load_json(dataset_json_file)
    plans = load_json(plans_json_file)
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    preprocessor = configuration_manager.preprocessor_class(verbose=False)
    return dataset_json, plans_manager, configuration_manager, preprocessor


def map_channel_to_reference(src, ref, eps=1e-8):
    src = src.astype(np.float32)
    ref = ref.astype(np.float32)

    src_mean = src.mean()
    src_std = src.std()
    ref_mean = ref.mean()
    ref_std = ref.std()

    if src_std < eps:
        return np.full_like(src, ref_mean, dtype=np.float32)

    out = (src - src_mean) / (src_std + eps)
    out = out * (ref_std + eps) + ref_mean
    return out.astype(np.float32)


def masam_channel_preprocess(data):
    """
    data: [C, Z, Y, X] after nnU-Net preprocessing

    requested preprocessing:
    1. modalities become channels
    2. map other channels to first channel scale
    3. average all channels
    4. repeat to 3 channels

    returns:
        [3, Z, Y, X]
    """
    assert data.ndim == 4, f"Expected [C, Z, Y, X], got {data.shape}"

    ref = data[0].astype(np.float32)
    aligned = [ref]
    for c in range(1, data.shape[0]):
        aligned.append(map_channel_to_reference(data[c], ref))

    aligned = np.stack(aligned, axis=0)       # [C, Z, Y, X]
    avg = np.mean(aligned, axis=0, keepdims=True)  # [1, Z, Y, X]
    avg3 = np.repeat(avg, 3, axis=0)          # [3, Z, Y, X]
    return avg3.astype(np.float32)


def normalize_minmax(x, eps=1e-8):
    x = x.astype(np.float32)
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + eps)


def resize_block(block, patch_size):
    """
    block: [D, 3, H, W]
    returns:
        [D, 3, patch_h, patch_w]
    """
    d, c, h, w = block.shape
    ph, pw = patch_size
    if (h, w) == (ph, pw):
        return block
    return zoom(block, (1, 1, ph / h, pw / w), order=3)


def resize_prediction_block(pred_block, target_hw):
    """
    pred_block: [D, H, W]
    target_hw: (Y, X)
    """
    d, h, w = pred_block.shape
    Y, X = target_hw
    if (h, w) == (Y, X):
        return pred_block.astype(np.uint8)

    out = np.zeros((d, Y, X), dtype=np.uint8)
    for i in range(d):
        out[i] = zoom(pred_block[i].astype(np.float32), (Y / h, X / w), order=0).astype(np.uint8)
    return out


def get_non_overlapping_5slice_starts(z):
    """
    Example:
      z=20  -> [0, 5, 10, 15]
      z=22  -> [0, 5, 10, 15, 17]
      z=484 -> [0, 5, 10, ..., 475, 479]
    """
    if z <= 5:
        return [0]

    starts = list(range(0, z - 4, 5))
    last_start = z - 5
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def predict_case_5slice_blocks(model, volume_3ch, patch_size, multimask_output):
    """
    volume_3ch: [3, Z, Y, X]

    block inference:
    - split into non-overlapping 5-slice blocks
    - final block may overlap only to cover the tail
    - input to model: [B, D, 3, H, W], with B=1, D=5
    - output from model: [BD, C, H, W]

    returns:
        pred_seg: [Z, Y, X]
    """
    assert volume_3ch.ndim == 4 and volume_3ch.shape[0] == 3
    _, Z, Y, X = volume_3ch.shape

    starts = get_non_overlapping_5slice_starts(Z)
    pred_seg = np.zeros((Z, Y, X), dtype=np.uint8)
    written = np.zeros(Z, dtype=bool)

    model.eval()
    with torch.no_grad():
        for start in tqdm(starts, desc="Predicting 5-slice blocks"):
            end = start + 5

            block = volume_3ch[:, start:end, :, :]   # [3, 5, Y, X]
            if block.shape[1] < 5:
                # should only happen if Z < 5
                pad_n = 5 - block.shape[1]
                pad_block = np.repeat(block[:, -1:, :, :], pad_n, axis=1)
                block = np.concatenate([block, pad_block], axis=1)

            block = np.transpose(block, (1, 0, 2, 3))   # [5, 3, Y, X]
            block = normalize_minmax(block)
            block = resize_block(block, patch_size)     # [5, 3, H, W]

            # [5, 3, H, W] -> [1, 5, 3, H, W]
            inputs = torch.from_numpy(block).float().unsqueeze(0).cuda()

            outputs = model(inputs, multimask_output, patch_size[0])
            output_masks = outputs["masks"]   # expected [BD, C, H, W]

            if output_masks.ndim != 4:
                raise RuntimeError(
                    f"Expected output_masks shape [BD, C, H, W], got {tuple(output_masks.shape)}"
                )

            bd, c, h, w = output_masks.shape
            B = 1
            D = 5

            if bd != B * D:
                raise RuntimeError(
                    f"Expected first dim BD={B*D}, but got {bd}. output_masks shape={tuple(output_masks.shape)}"
                )

            output_masks = output_masks.view(B, D, c, h, w)   # [1, 5, C, H, W]
            output_masks = torch.softmax(output_masks, dim=2)
            pred = torch.argmax(output_masks, dim=2)          # [1, 5, H, W]
            pred = pred[0].cpu().numpy()                      # [5, H, W]

            pred = resize_prediction_block(pred, (Y, X))      # [5, Y, X]

            for local_idx in range(5):
                global_idx = start + local_idx
                if global_idx >= Z:
                    continue
                if not written[global_idx]:
                    pred_seg[global_idx] = pred[local_idx]
                    written[global_idx] = True

    return pred_seg


def segmentation_to_onehot_logits(seg, num_classes):
    """
    seg: [Z, Y, X]
    returns:
        [C, Z, Y, X]
    """
    z, y, x = seg.shape
    logits = np.zeros((num_classes, z, y, x), dtype=np.float32)
    for c in range(num_classes):
        logits[c] = (seg == c).astype(np.float32)
    return torch.from_numpy(logits)


def inference(args, multimask_output, model):
    dataset_json, plans_manager, configuration_manager, preprocessor = setup_nnunet_objects(
        args.dataset_json,
        args.plans_json,
        args.configuration
    )

    maybe_mkdir_p(args.output_dir)

    case_list = create_lists_from_splitted_dataset_folder(
        args.input_dir,
        dataset_json["file_ending"]
    )

    print(f"Found {len(case_list)} cases")

    for case_files in case_list:
        case_id = os.path.basename(case_files[0])[:-(len(dataset_json["file_ending"]) + 5)]
        output_file_truncated = join(args.output_dir, case_id)

        print(f"\nProcessing case: {case_id}")
        print(f"Input files: {case_files}")

        data, seg, data_properties = preprocessor.run_case(
            case_files,
            None,
            plans_manager,
            configuration_manager,
            dataset_json
        )
        # data: [C, Z, Y, X]

        volume_3ch = masam_channel_preprocess(data)   # [3, Z, Y, X]

        pred_seg = predict_case_5slice_blocks(
            model=model,
            volume_3ch=volume_3ch,
            patch_size=[args.img_size, args.img_size],
            multimask_output=multimask_output
        )   # [Z, Y, X]

        predicted_logits = segmentation_to_onehot_logits(pred_seg, args.num_classes)

        export_prediction_from_logits(
            predicted_logits,
            data_properties,
            configuration_manager,
            plans_manager,
            dataset_json,
            output_file_truncated,
            save_probabilities=False
        )

        print(f"Saved prediction to: {output_file_truncated}{dataset_json['file_ending']}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--adapt_ckpt', type=str, required=True,
                        help='MASAM adapted checkpoint')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Pretrained SAM checkpoint')

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input folder with nnU-Net style nii.gz files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output folder for predicted nii.gz')
    parser.add_argument('--dataset_json', type=str, required=True,
                        help='Path to nnU-Net dataset.json')
    parser.add_argument('--plans_json', type=str, required=True,
                        help='Path to nnU-Net plans.json')
    parser.add_argument('--configuration', type=str, required=True,
                        help='nnU-Net configuration name, e.g. 3d_fullres')

    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--vit_name', type=str, default='vit_h')
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--module', type=str, default='sam_fact_tt_image_encoder')

    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    maybe_mkdir_p(args.output_dir)

    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0., 0., 0.],
        pixel_std=[1., 1., 1.]
    )

    pkg = import_module(args.module)
    model = pkg.Fact_tt_Sam(sam, args.rank, s=args.scale).cuda()
    model.load_parameters(args.adapt_ckpt)
    model.eval()

    multimask_output = args.num_classes > 1

    inference(args, multimask_output, model)


if __name__ == '__main__':
    # python test_nnunet.py \
    # --adapt_ckpt /path/to/epoch_159.pth \
    # --ckpt /path/to/sam_vit_h_4b8939.pth \
    # --input_dir /path/to/imagesTs \
    # --output_dir /path/to/predictions \
    # --dataset_json /path/to/dataset.json \
    # --plans_json /path/to/plans.json \
    # --configuration 3d_fullres \
    # --num_classes 12 \
    # --img_size 512 \
    # --vit_name vit_h \
    # --rank 32 \
    # --scale 1.0 \
    # --module sam_fact_tt_image_encoder
    main()