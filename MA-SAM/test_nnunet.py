import os
import json
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
    """
    Match intensity scale of src to ref using mean/std normalization.
    """
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

    requested behavior:
    1. modalities -> channels (already true)
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

    aligned = np.stack(aligned, axis=0)     # [C, Z, Y, X]
    avg = np.mean(aligned, axis=0, keepdims=True)  # [1, Z, Y, X]
    avg3 = np.repeat(avg, 3, axis=0)        # [3, Z, Y, X]

    return avg3.astype(np.float32)


def normalize_minmax(x, eps=1e-8):
    x = x.astype(np.float32)
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + eps)


def build_5slice_input(volume_3ch, z_idx):
    """
    volume_3ch: [3, Z, Y, X]
    return 5-slice pseudo-3D input around z_idx:
        [3, 5, Y, X]
    edge slices are replicated
    """
    _, Z, _, _ = volume_3ch.shape

    idxs = [
        max(0, min(Z - 1, z_idx - 2)),
        max(0, min(Z - 1, z_idx - 1)),
        max(0, min(Z - 1, z_idx)),
        max(0, min(Z - 1, z_idx + 1)),
        max(0, min(Z - 1, z_idx + 2)),
    ]

    window = volume_3ch[:, idxs, :, :]  # [3, 5, Y, X]
    return window


def predict_case_5slice(model, volume_3ch, patch_size, multimask_output):
    """
    volume_3ch: [3, Z, Y, X]
    returns:
        pred_seg: [Z, Y, X]
    """
    assert volume_3ch.ndim == 4 and volume_3ch.shape[0] == 3
    _, Z, Y, X = volume_3ch.shape

    pred_seg = np.zeros((Z, Y, X), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for z in tqdm(range(Z), desc="Predicting 5-slice windows"):
            window = build_5slice_input(volume_3ch, z)  # [3, 5, Y, X]
            window = normalize_minmax(window)

            in_h, in_w = window.shape[-2], window.shape[-1]
            if [in_h, in_w] != patch_size:
                window_rs = zoom(
                    window,
                    (1, 1, patch_size[0] / in_h, patch_size[1] / in_w),
                    order=3
                )
            else:
                window_rs = window

            # [3, 5, H, W] -> [1, 5, 3, H, W]
            inputs = torch.from_numpy(window_rs).float().permute(1, 0, 2, 3).unsqueeze(0).cuda()

            outputs = model(inputs, multimask_output, patch_size[0])
            output_masks = outputs["masks"]  # [B, C, H, W]

            pred = torch.argmax(torch.softmax(output_masks, dim=1), dim=1)[0].cpu().numpy()

            out_h, out_w = pred.shape
            if (out_h, out_w) != (Y, X):
                pred = zoom(pred.astype(np.float32), (Y / out_h, X / out_w), order=0)

            pred_seg[z] = pred.astype(np.uint8)

    return pred_seg


def segmentation_to_onehot_logits(seg, num_classes):
    """
    seg: [Z, Y, X]
    returns logits-like tensor [C, Z, Y, X] for nnU-Net export
    """
    z, y, x = seg.shape
    logits = np.zeros((num_classes, z, y, x), dtype=np.float32)
    for c in range(num_classes):
        logits[c] = (seg == c).astype(np.float32)
    return torch.from_numpy(logits)


def inference(args, model, multimask_output):
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

        # nnU-Net preprocessing from raw files
        data, seg, data_properties = preprocessor.run_case(
            case_files,
            None,
            plans_manager,
            configuration_manager,
            dataset_json
        )
        # data shape expected: [C, Z, Y, X]

        # custom MASAM preprocessing after nnU-Net preprocessing
        volume_3ch = masam_channel_preprocess(data)  # [3, Z, Y, X]

        # 5-slice pseudo-3D inference
        pred_seg = predict_case_5slice(
            model=model,
            volume_3ch=volume_3ch,
            patch_size=[args.img_size, args.img_size],
            multimask_output=multimask_output
        )  # [Z, Y, X]

        # convert to logits-like tensor for nnU-Net export
        predicted_logits = segmentation_to_onehot_logits(pred_seg, args.num_classes)

        # save in original image space following nnU-Net export logic
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
                        help='Input folder with nnU-Net style nii.gz files, e.g. case_0000.nii.gz, case_0001.nii.gz')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output folder for predicted nii.gz')
    parser.add_argument('--dataset_json', type=str, required=True,
                        help='Path to nnU-Net dataset.json')
    parser.add_argument('--plans_json', type=str, required=True,
                        help='Path to nnU-Net plans.json')
    parser.add_argument('--configuration', type=str, required=True,
                        help='nnU-Net configuration name, e.g. 3d_fullres')

    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size for MASAM')
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

    # build MASAM model
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

    inference(args, model, multimask_output)


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