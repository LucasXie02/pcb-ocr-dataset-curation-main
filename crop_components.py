"""Crop individual component regions from full board images.

Reads LabelMe JSON (produced by annotation_extraction.py) with rotated_box
shapes, warps each OBB crop upright, and writes per-board crop directories
with a manifest tracking subboard/sibling relationships.

Usage:
    python crop_components.py --input Data/board_output/ --output Data/crops/
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from annotation_extraction import (
    _frame_dimensions,
    extract_label_prefix,
    rotate_point,
)


def _parse_description(description: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract region_group_id and array_index from a shape description.

    Expected formats (from annotation_extraction.py):
        "group_{region_group_id}_{array_index}"
        "group_{region_group_id}_{array_index} {component_class}"

    Returns:
        (region_group_id, array_index) or (None, None) if parsing fails.
    """
    if not description:
        return None, None
    match = re.match(r"group_(\d+)_(-?\d+)", description.strip())
    if match:
        return match.group(1), match.group(2)
    return None, None


def _obb_corners(points: List[List[float]], angle: float) -> np.ndarray:
    """Compute the four corner points of an oriented bounding box.

    Args:
        points: [[x1, y1], [x2, y2]] centre-ish corner pair.
        angle: Rotation angle in degrees.

    Returns:
        4x2 numpy array of corner coordinates (float32), ordered:
        top-left, top-right, bottom-right, bottom-left in the *rotated* frame.
    """
    (x1, y1), (x2, y2) = points
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w, h = _frame_dimensions(points, angle)
    half_w = w / 2.0
    half_h = h / 2.0

    # Axis-aligned corners relative to centre (before rotation)
    local_corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]

    # Rotate each corner around centre
    corners = []
    for lx, ly in local_corners:
        gx, gy = rotate_point(cx + lx, cy + ly, cx, cy, angle)
        corners.append([gx, gy])

    return np.array(corners, dtype=np.float32)


def crop_obb(image: np.ndarray, points: List[List[float]],
             angle: float) -> Optional[np.ndarray]:
    """Warp an OBB region from *image* into an upright axis-aligned crop.

    Steps:
        1. Compute the four OBB corners.
        2. Build an affine transform that maps those corners to an upright
           rectangle of size (width x height).
        3. Apply warpAffine and return the result.

    Returns:
        Upright crop as a numpy array, or None on failure.
    """
    w, h = _frame_dimensions(points, angle)
    if w < 1 or h < 1:
        return None

    src_corners = _obb_corners(points, angle)

    # Destination corners for an upright rectangle
    dst_corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ], dtype=np.float32)

    # Use first three points for the affine transform
    M = cv2.getAffineTransform(src_corners[:3], dst_corners[:3])
    crop = cv2.warpAffine(image, M, (int(round(w)), int(round(h))),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return crop


def process_board(json_path: Path, output_dir: Path) -> bool:
    """Process a single board JSON + image pair.

    Args:
        json_path: Path to the LabelMe JSON file.
        output_dir: Base output directory (board subdirectory will be created).

    Returns:
        True on success, False on error.
    """
    board_name = json_path.stem

    # Load LabelMe JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            labelme_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error loading JSON {json_path}: {exc}")
        return False

    # Locate the board image (try common extensions)
    image_path = None
    parent = json_path.parent
    image_name = labelme_data.get("imagePath", "")
    if image_name:
        candidate = parent / image_name
        if candidate.exists():
            image_path = candidate

    if image_path is None:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            candidate = parent / f"{board_name}{ext}"
            if candidate.exists():
                image_path = candidate
                break

    if image_path is None:
        print(f"Error: Cannot find image for {json_path}")
        return False

    # Load image with OpenCV
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Failed to read image {image_path}")
        return False

    shapes = labelme_data.get("shapes", [])
    if not shapes:
        print(f"Warning: No shapes found in {json_path}")
        return False

    # Use _mount shapes for cropping — these are the component body inspection
    # regions used for OCR. Derive the component class from the matching _full
    # shape that shares the same region_group_id.
    mount_shapes = [s for s in shapes if s.get("label", "") == "_mount"]
    if not mount_shapes:
        print(f"Warning: No _mount shapes in {json_path}, skipping.")
        return False

    # Build gid → component class lookup from _full shapes
    gid_to_class: Dict[str, str] = {}
    for s in shapes:
        lbl = s.get("label", "")
        if lbl.endswith("_full") and not lbl.startswith("_"):
            gid, _ = _parse_description(s.get("description", ""))
            if gid:
                component_class = lbl[:-5]  # strip _full
                gid_to_class[gid] = component_class

    # Create board output directory
    board_out = output_dir / board_name
    board_out.mkdir(parents=True, exist_ok=True)

    # First pass: collect group info for sibling computation
    group_members: Dict[str, List[str]] = defaultdict(list)  # gid -> [stem, ...]
    crop_entries: List[Dict] = []

    for shape in mount_shapes:
        label = shape.get("label", "")
        points = shape.get("points")
        angle = shape.get("angle", 0)
        description = shape.get("description", "")

        if not points or len(points) < 2:
            print(f"Warning: Invalid points in shape '{label}', skipping.")
            continue

        # Parse group info and look up component class from _full shape
        region_group_id, array_index = _parse_description(description)
        if region_group_id is None:
            print(f"Warning: Cannot parse description '{description}', skipping.")
            continue

        component_class = gid_to_class.get(region_group_id, "Unknown")

        # Crop filename stem (no extension)
        crop_stem = f"{component_class}_{region_group_id}_{array_index}"
        crop_filename = f"{crop_stem}.png"

        # Perform the crop
        crop_img = crop_obb(image, points, angle)
        if crop_img is None:
            print(f"Warning: Failed to crop '{crop_stem}', skipping.")
            continue

        # Write crop image
        crop_path = board_out / crop_filename
        cv2.imwrite(str(crop_path), crop_img)

        # Record for manifest
        group_members[region_group_id].append(crop_stem)
        crop_entries.append({
            "crop_file": crop_filename,
            "component_class": component_class,
            "region_group_id": region_group_id,
            "array_index": array_index,
            "_crop_stem": crop_stem,  # temporary, removed before serialisation
        })

    # Second pass: populate sibling lists
    for entry in crop_entries:
        gid = entry["region_group_id"]
        siblings = [s for s in group_members[gid] if s != entry["_crop_stem"]]
        entry["siblings"] = siblings
        del entry["_crop_stem"]

    # Write manifest
    manifest = {
        "board_name": board_name,
        "crops": crop_entries,
    }
    manifest_path = board_out / "crop_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Board '{board_name}': {len(crop_entries)} crops saved to {board_out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop individual component regions from full board images."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing board JSON + image pairs from annotation_extraction.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for crop images and manifests",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all LabelMe JSON files in the input directory
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"Error: No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s) in {input_dir}")
    print("=" * 80)

    success_count = 0
    fail_count = 0

    for json_path in json_files:
        print(f"\nProcessing: {json_path.name}")
        print("-" * 80)
        try:
            if process_board(json_path, output_dir):
                success_count += 1
            else:
                fail_count += 1
        except Exception as exc:
            print(f"Error processing {json_path.name}: {exc}")
            fail_count += 1

    print("\n" + "=" * 80)
    print(f"Done. Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
