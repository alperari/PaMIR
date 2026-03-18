from __future__ import division, print_function

import argparse
import glob
import json
import os

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn, maskrcnn_resnet50_fpn

from main_test import main_test_texture, main_test_wo_gt_smpl_with_optm


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-prepare images (crop/mask/keypoints) and run PaMIR inference."
    )
    parser.add_argument("--input", required=True,
                        help="Path to an image file or a folder of images.")
    parser.add_argument(
        "--work-dir",
        default="./results/test_data_auto",
        help="Working folder where processed inputs and outputs are stored.",
    )
    parser.add_argument(
        "--pretrained-geo",
        default="./results/pamir_geometry/checkpoints/latest.pt",
        help="Path to PaMIR geometry checkpoint.",
    )
    parser.add_argument(
        "--pretrained-gcmr",
        default="./results/gcmr_pretrained/gcmr_2020_12_10-21_03_12.pt",
        help="Path to GraphCMR checkpoint.",
    )
    parser.add_argument(
        "--pretrained-tex",
        default="./results/pamir_texture/checkpoints/latest.pt",
        help="Path to PaMIR texture checkpoint.",
    )
    parser.add_argument(
        "--iternum",
        type=int,
        default=50,
        help="SMPL optimization iterations in geometry stage.",
    )
    parser.add_argument(
        "--person-score-thres",
        type=float,
        default=0.6,
        help="Detection score threshold for person proposals.",
    )
    parser.add_argument(
        "--mask-thres",
        type=float,
        default=0.5,
        help="Threshold for person mask logits.",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.8,
        help="Target person-height/image-height ratio after crop.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.05,
        help="Extra crop margin ratio around the person bbox.",
    )
    return parser.parse_args()


def list_input_images(input_path):
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in IMG_EXTS:
            raise ValueError(
                "Input file is not a supported image type: %s" % input_path)
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError("Input path not found: %s" % input_path)

    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(glob.glob(os.path.join(input_path, "*%s" % ext)))
        imgs.extend(glob.glob(os.path.join(input_path, "*%s" % ext.upper())))
    imgs = sorted(list(set(imgs)))
    if len(imgs) == 0:
        raise FileNotFoundError("No images found under: %s" % input_path)
    return imgs


def load_models(device):
    # Torchvision 0.8 API supports pretrained=True.
    kp_model = keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
    mask_model = maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
    return kp_model, mask_model


def image_to_tensor(image_bgr, device):
    rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    ten = transforms.ToTensor()(rgb).to(device)
    return ten


def select_best_person(pred, score_thres):
    labels = pred["labels"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    person_idx = np.where((labels == 1) & (scores >= score_thres))[0]
    if person_idx.size == 0:
        person_idx = np.where(labels == 1)[0]
    if person_idx.size == 0:
        return None

    best = person_idx[np.argmax(scores[person_idx])]
    return int(best)


def box_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def square_crop_with_padding(image, x1, y1, x2, y2, target_ratio=0.8, margin=0.05):
    h, w = image.shape[:2]

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # Set crop side so person occupies around target_ratio of final image height.
    side_by_h = bh / max(target_ratio, 1e-3)
    side_by_w = bw / max(target_ratio, 1e-3)
    side = max(side_by_h, side_by_w)
    side *= (1.0 + margin)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    crop_x1 = int(np.floor(cx - side / 2.0))
    crop_y1 = int(np.floor(cy - side / 2.0))
    crop_x2 = int(np.ceil(cx + side / 2.0))
    crop_y2 = int(np.ceil(cy + side / 2.0))

    pad_l = max(0, -crop_x1)
    pad_t = max(0, -crop_y1)
    pad_r = max(0, crop_x2 - w)
    pad_b = max(0, crop_y2 - h)

    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        image = cv.copyMakeBorder(
            image,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            borderType=cv.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    crop_x1 += pad_l
    crop_x2 += pad_l
    crop_y1 += pad_t
    crop_y2 += pad_t

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2), (pad_l, pad_t, pad_r, pad_b)


def coco17_to_body25(kps17):
    # COCO order:
    # 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
    # 5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow,
    # 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip,
    # 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
    body25 = np.zeros((25, 3), dtype=np.float32)

    def copy_kp(dst_idx, src_idx):
        body25[dst_idx, :2] = kps17[src_idx, :2]
        body25[dst_idx, 2] = kps17[src_idx, 2]

    # direct mappings
    copy_kp(0, 0)   # Nose
    copy_kp(2, 6)   # RShoulder
    copy_kp(3, 8)   # RElbow
    copy_kp(4, 10)  # RWrist
    copy_kp(5, 5)   # LShoulder
    copy_kp(6, 7)   # LElbow
    copy_kp(7, 9)   # LWrist
    copy_kp(9, 12)  # RHip
    copy_kp(10, 14)  # RKnee
    copy_kp(11, 16)  # RAnkle
    copy_kp(12, 11)  # LHip
    copy_kp(13, 13)  # LKnee
    copy_kp(14, 15)  # LAnkle
    copy_kp(15, 2)  # REye
    copy_kp(16, 1)  # LEye
    copy_kp(17, 4)  # REar
    copy_kp(18, 3)  # LEar

    # derived neck (1) and mid-hip (8)
    l_sh = kps17[5]
    r_sh = kps17[6]
    l_hip = kps17[11]
    r_hip = kps17[12]

    body25[1, :2] = 0.5 * (l_sh[:2] + r_sh[:2])
    body25[1, 2] = min(l_sh[2], r_sh[2])
    body25[8, :2] = 0.5 * (l_hip[:2] + r_hip[:2])
    body25[8, 2] = min(l_hip[2], r_hip[2])

    return body25


def save_openpose_json(body25, out_json):
    payload = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": body25.reshape(-1).tolist(),
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": [],
            }
        ],
    }
    with open(out_json, "w") as fp:
        json.dump(payload, fp)


def run_detector(model, image_bgr, device):
    ten = image_to_tensor(image_bgr, device)
    with torch.no_grad():
        pred = model([ten])[0]
    return pred


def prepare_single_image(img_path, out_dir, kp_model, mask_model, device, score_thres, mask_thres,
                         target_ratio, margin):
    img_name = os.path.basename(img_path)
    stem, _ = os.path.splitext(img_name)

    image = cv.imread(img_path)
    if image is None:
        raise RuntimeError("Failed to read image: %s" % img_path)

    # Pass 1: detect person box from keypoint detector in raw image.
    pred_kp_raw = run_detector(kp_model, image, device)
    kp_idx_raw = select_best_person(pred_kp_raw, score_thres)
    if kp_idx_raw is None:
        raise RuntimeError("No person detected in %s" % img_path)
    box_raw = pred_kp_raw["boxes"][kp_idx_raw].detach(
    ).cpu().numpy().astype(np.float32)

    crop, _, _ = square_crop_with_padding(
        image,
        box_raw[0],
        box_raw[1],
        box_raw[2],
        box_raw[3],
        target_ratio=target_ratio,
        margin=margin,
    )
    crop_512 = cv.resize(crop, (512, 512), interpolation=cv.INTER_LINEAR)

    # Pass 2: run keypoint and mask detectors on normalized 512 image.
    pred_kp = run_detector(kp_model, crop_512, device)
    kp_idx = select_best_person(pred_kp, score_thres)
    if kp_idx is None:
        raise RuntimeError("No person keypoints after crop in %s" % img_path)

    pred_mask = run_detector(mask_model, crop_512, device)
    mask_idx = select_best_person(pred_mask, score_thres)

    kp_box = pred_kp["boxes"][kp_idx].detach().cpu().numpy().astype(np.float32)

    if mask_idx is None:
        # Fallback rectangular mask from keypoint person box.
        person_mask = np.zeros((512, 512), dtype=np.uint8)
        x1, y1, x2, y2 = kp_box.astype(np.int32)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(511, x2), min(511, y2)
        person_mask[y1:y2 + 1, x1:x2 + 1] = 255
    else:
        mask_boxes = pred_mask["boxes"].detach(
        ).cpu().numpy().astype(np.float32)
        labels = pred_mask["labels"].detach().cpu().numpy()
        scores = pred_mask["scores"].detach().cpu().numpy()
        valid_mask_ids = np.where((labels == 1) & (scores >= score_thres))[0]
        if valid_mask_ids.size == 0:
            valid_mask_ids = np.where(labels == 1)[0]

        if valid_mask_ids.size > 0:
            ious = [box_iou_xyxy(kp_box, mask_boxes[i])
                    for i in valid_mask_ids]
            mask_idx = int(valid_mask_ids[int(np.argmax(np.array(ious)))])

        mask_prob = pred_mask["masks"][mask_idx, 0].detach().cpu().numpy()
        person_mask = (mask_prob >= mask_thres).astype(np.uint8) * 255

    # Keep only foreground over white background.
    mask_f = (person_mask.astype(np.float32) / 255.0)[:, :, None]
    proc = (crop_512.astype(np.float32) * mask_f +
            255.0 * (1.0 - mask_f)).astype(np.uint8)

    # COCO-17 -> BODY-25 style keypoints JSON.
    kps = pred_kp["keypoints"][kp_idx].detach(
    ).cpu().numpy().astype(np.float32)
    # Torchvision keypoints are [x, y, confidence-like value]. Clamp to [0,1].
    kps[:, 2] = np.clip(kps[:, 2], 0.0, 1.0)
    body25 = coco17_to_body25(kps)

    out_img = os.path.join(out_dir, "%s.png" % stem)
    out_mask = os.path.join(out_dir, "%s_mask.png" % stem)
    out_kpt = os.path.join(out_dir, "%s_keypoints.json" % stem)

    cv.imwrite(out_img, proc)
    cv.imwrite(out_mask, person_mask)
    save_openpose_json(body25, out_kpt)

    return out_img, out_mask, out_kpt


def main():
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)
    print("[INFO] Loading person detectors...")
    kp_model, mask_model = load_models(device)

    imgs = list_input_images(args.input)
    print("[INFO] Preparing %d image(s)..." % len(imgs))

    for i, img_path in enumerate(imgs):
        try:
            out_img, out_mask, out_kpt = prepare_single_image(
                img_path,
                args.work_dir,
                kp_model,
                mask_model,
                device,
                score_thres=args.person_score_thres,
                mask_thres=args.mask_thres,
                target_ratio=args.target_ratio,
                margin=args.crop_margin,
            )
            print("[OK] %d/%d %s" %
                  (i + 1, len(imgs), os.path.basename(out_img)))
            print("     mask: %s" % os.path.basename(out_mask))
            print("     keypoints: %s" % os.path.basename(out_kpt))
        except Exception as exc:
            print("[WARN] Skip %s: %s" % (img_path, str(exc)))

    print("[INFO] Running PaMIR geometry inference...")
    main_test_wo_gt_smpl_with_optm(
        args.work_dir,
        args.work_dir,
        pretrained_checkpoint=args.pretrained_geo,
        pretrained_gcmr_checkpoint=args.pretrained_gcmr,
        iternum=args.iternum,
    )

    print("[INFO] Running PaMIR texture inference...")
    main_test_texture(
        args.work_dir,
        args.work_dir,
        pretrained_checkpoint_pamir=args.pretrained_geo,
        pretrained_checkpoint_pamirtex=args.pretrained_tex,
    )

    print("[DONE] Outputs are in: %s" % os.path.join(args.work_dir, "results"))


if __name__ == "__main__":
    main()
