import json
import os
import argparse
from pathlib import Path
from typing import List, Dict
import supervision as sv
from PIL import Image
from rfdetr import RFDETRNano, RFDETRMedium
from tqdm import tqdm


def collect_images(input_path):
    """Collect all image files from input path"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    images = []

    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() in image_extensions:
            images.append(str(input_path))
    elif input_path.is_dir():
        for ext in image_extensions:
            images.extend([str(p) for p in input_path.rglob(f'*{ext}')])

    return sorted(images)


def load_config_and_labels(config_path: str) -> tuple:
    """Load config.json and extract class labels, skipping 'background'.

    Returns:
        (config, class_labels) where class_labels is a dict {int: str}
        with background entries removed.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Support both flat 'class_labels' and nested 'DATASET.CLS_LABELS' formats
    labels_list = config.get('class_labels',
                             config.get('DATASET', {}).get('CLS_LABELS', []))

    if not labels_list:
        raise ValueError(f"No class labels found in {config_path}. "
                         "Expected 'class_labels' or 'DATASET.CLS_LABELS' key.")

    # Build label map, skipping background
    class_labels = {i: name for i, name in enumerate(labels_list)
                    if name != 'background'}

    return config, class_labels


def detect_model_class(config: dict) -> type:
    """Auto-detect model class from config 'architecture' field.

    Falls back to RFDETRMedium if architecture is not specified.
    Checks multiple config key conventions for 'nano' indicator.
    """
    arch = config.get('architecture', '').lower()
    model_size = config.get('model_size', '').lower()
    model_id = config.get('model_id', '').lower()
    nested_size = config.get('MODEL', {}).get('SIZE', '').lower()

    if 'nano' in (arch, model_size, model_id, nested_size):
        return RFDETRNano
    else:
        return RFDETRMedium


def detect_resolution(config: dict, model_dir_name: str = '', cli_resolution: int = None) -> int:
    """Auto-detect resolution from config or model directory name.

    Priority: CLI arg > config field > directory name heuristic > default 576.
    """
    if cli_resolution is not None:
        return cli_resolution

    # Check config for resolution field (multiple key conventions)
    if 'resolution' in config:
        return int(config['resolution'])

    train_res = config.get('train', {}).get('resolution')
    if train_res is not None:
        return int(train_res)

    preprocess_res = config.get('PREPROCESS', {}).get('IMGSZ')
    if preprocess_res is not None:
        return int(preprocess_res)

    # Heuristic: look for 224 or 448 in directory name
    for res in (224, 448):
        if str(res) in model_dir_name:
            return res

    return 576


def convert_to_labelme_format(detections, class_labels, image_path, img_size):
    """Convert RFDETR detections to LabelMe JSON format"""
    width, height = img_size
    shapes = []

    if detections.xyxy is not None and len(detections.xyxy) > 0:
        for bbox, class_id, confidence in zip(
            detections.xyxy,
            detections.class_id,
            detections.confidence
        ):
            # Skip background detections
            if class_id not in class_labels:
                continue

            x1, y1, x2, y2 = bbox
            label = class_labels[class_id]

            shape = {
                "label": label,
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "description": f"{confidence:.4f}",
                "shape_type": "rectangle",
                "flags": {}
            }
            shapes.append(shape)

    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    return labelme_data


def visualize_and_save(image, detections, class_labels, output_path):
    """Visualize detections and save annotated image"""
    labels = [
        f"{class_labels[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
        if class_id in class_labels
    ]

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    annotated_image.save(output_path)


def run_detection(model, image_paths, class_labels, output_dir, threshold, visualize):
    """Run detection on a list of images and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(image_paths)} images...")
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load image
            image = Image.open(img_path)
            img_size = image.size

            # Run detection
            detections = model.predict(image, threshold=threshold)

            if len(detections) == 0:
                print(f"No detections for {img_path}")
                continue

            # Convert to LabelMe format
            labelme_data = convert_to_labelme_format(
                detections, class_labels, img_path, img_size)

            # Save JSON annotation
            img_name = Path(img_path).stem
            json_path = output_dir / f"{img_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)

            # Optionally save visualization
            if visualize:
                viz_path = output_dir / f"{img_name}_viz.png"
                visualize_and_save(image, detections, class_labels, str(viz_path))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Done! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch object detection with RFDETR")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image file or directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for JSON annotations "
                             "(required for single-model mode)")
    parser.add_argument("--model", "-m", type=str,
                        default="./Checkpoints/rfdetr_char_agnostic.pth",
                        help="Model weights path (ignored when --models is used)")
    parser.add_argument("--config", "-c", type=str,
                        default="./Checkpoints/config.json",
                        help="Config file path (ignored when --models is used)")
    parser.add_argument("--models", nargs='+', type=str, default=None,
                        help="Checkpoint directory names under ./Checkpoints/ for "
                             "dual-model batch mode. Each directory must contain "
                             "config.json and model.pth. "
                             "Example: --models char_224 char_448")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Detection threshold")
    parser.add_argument("--resolution", "-r", type=int, default=None,
                        help="Model resolution (auto-detected if not set)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Save visualization images")

    args = parser.parse_args()

    # Collect images
    print(f"Collecting images from {args.input}...")
    image_paths = collect_images(args.input)
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return

    if args.models:
        # --- Dual/multi-model batch mode ---
        checkpoints_root = Path("./Checkpoints")
        input_path = Path(args.input).resolve()
        # Base output: strip trailing slash, append model name
        base_output = str(input_path).rstrip('/')

        for model_name in args.models:
            model_dir = checkpoints_root / model_name
            config_path = model_dir / "config.json"
            weights_path = model_dir / "model.pth"
            if not weights_path.exists():
                # Fallback: glob for any .pth file in the directory
                pth_files = sorted(model_dir.glob("*.pth"))
                if pth_files:
                    weights_path = pth_files[0]

            if not config_path.exists():
                print(f"ERROR: Config not found: {config_path}")
                continue
            if not weights_path.exists():
                print(f"ERROR: Weights not found: {weights_path}")
                continue

            print(f"\n{'='*60}")
            print(f"Running model: {model_name}")
            print(f"{'='*60}")

            # Load config and labels
            config, class_labels = load_config_and_labels(str(config_path))

            # Auto-detect model class and resolution
            ModelClass = detect_model_class(config)
            resolution = detect_resolution(config, model_name, args.resolution)

            print(f"  Architecture: {ModelClass.__name__}")
            print(f"  Resolution:   {resolution}")
            print(f"  Labels:       {len(class_labels)} classes (background excluded)")

            # Initialize model
            model = ModelClass(resolution=resolution,
                               pretrain_weights=str(weights_path))

            # Output directory: {input}_{model_name}/
            output_dir = f"{base_output}_{model_name}"

            run_detection(model, image_paths, class_labels, output_dir,
                          args.threshold, args.visualize)

    else:
        # --- Single-model mode (backward compatible) ---
        if args.output is None:
            parser.error("--output/-o is required in single-model mode")

        # Load config and class labels
        print(f"Loading config from {args.config}...")
        config, class_labels = load_config_and_labels(args.config)

        # Auto-detect model class and resolution
        ModelClass = detect_model_class(config)
        resolution = detect_resolution(config, '', args.resolution)

        # If resolution was not specified anywhere, fall back to 576
        print(f"Loading model from {args.model}...")
        print(f"  Architecture: {ModelClass.__name__}")
        print(f"  Resolution:   {resolution}")

        model = ModelClass(resolution=resolution, pretrain_weights=args.model)

        run_detection(model, image_paths, class_labels, args.output,
                      args.threshold, args.visualize)


if __name__ == "__main__":
    main()
