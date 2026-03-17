import argparse
import glob
import json
import os
from pathlib import Path
import torch
from torchvision import transforms
import cv2
from PIL import Image
import yaml

from daoai_classification.src.models import DaoAIClassificationModel
import re


INPUT_PATH = Path(os.getenv("INPUT_PATH", "/tmp/task/inputs"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/tmp/task/outputs"))


def rotate_image(image, angle):
    """
    Rotate image by the specified angle (in degrees).
    Positive values mean counter-clockwise rotation.
    """
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        # For arbitrary angles, use rotation matrix
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated


def extract_rotation_angle(label):
    """
    Extract rotation angle from label string.
    Assumes label contains a number representing degrees (e.g., "90", "180", "rotate_90", "deg_270").
    Returns the angle as an integer, or 0 if no valid angle is found.
    """
    # Try to find numbers in the label
    numbers = re.findall(r'\d+', label)
    if numbers:
        angle = int(numbers[0])
        # Normalize angle to 0-359 range
        angle = angle % 360
        return angle
    return 0


def inference(
    image_input, model_path, config_path, device: str = "cuda"
):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    is_multilabel = cfg["DATASET"]["MULTILABEL"]
    cls_labels = cfg["DATASET"]["CLS_LABELS"]

    # Build model without checkpoint or pretrained weights (we'll load our own),
    # then load the full state dict (which has backbone.* + head.* keys).
    cfg_copy = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    cfg_copy["MODEL"] = dict(cfg["MODEL"])
    cfg_copy["MODEL"]["PRETRAINED"] = False
    model = DaoAIClassificationModel(cfg_copy)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Create inference transform pipeline matching dataset preprocessing
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print(f"Processing images from {image_input}")
    if os.path.isdir(image_input):
        images = []
        images.extend(glob.glob(os.path.join(image_input, "**", "*.jpg"), recursive=True))
        images.extend(glob.glob(os.path.join(image_input, "**", "*.jpeg"), recursive=True))
        images.extend(glob.glob(os.path.join(image_input, "**", "*.png"), recursive=True))
    elif os.path.isfile(image_input) and (image_input.endswith('.jpg') or image_input.endswith('.jpeg') or image_input.endswith('.png')):
        images = [image_input]
    else:
        raise ValueError("Provided image input path is not a directory or a valid image file.")

    print(f"Found {len(images)} images to process")

    rotated_count = 0
    for im in images:
        img_orig = cv2.imread(im)
        if img_orig is None:
            print(f"Warning: Could not read {im}, skipping...")
            continue

        # Preprocess image to match dataset preprocessing
        img = Image.fromarray(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        img = inference_transform(img).unsqueeze(0).to(device)
        outputs = model(img)

        if is_multilabel:
            # Apply sigmoid and then threshold
            predicted = (torch.sigmoid(outputs.data) > 0.5).int()
            pred_labels = []
            for i, label in enumerate(cls_labels):
                if predicted[0][i].item():
                    pred_labels.append(label)
            pred_text = ", ".join(pred_labels) if pred_labels else "None"
        else:
            _, predicted = torch.max(outputs.data, 1)
            pred_label = cls_labels[predicted.item()]
            pred_text = pred_label

        # Rotate image based on prediction and save in place
        rotation_angle = extract_rotation_angle(pred_text)
        if rotation_angle > 0:
            rotated_img = rotate_image(img_orig, rotation_angle)
            cv2.imwrite(im, rotated_img)  # Overwrite original image
            rotated_count += 1
            print(f"Rotated {os.path.basename(im)} by {rotation_angle} degrees (predicted: {pred_text})")

    print(f"\nProcessing complete: {rotated_count} images rotated out of {len(images)} total images")


def main():
    parser = argparse.ArgumentParser(description="Rotate images in place based on model predictions.")
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config-path', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    inference(
        image_input=args.data_path,
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )


if __name__ == "__main__":
    main()
