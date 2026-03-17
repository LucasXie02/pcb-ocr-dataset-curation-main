#!/usr/bin/env python3
"""
Merge Detections
================
Merge dual-model RF-DETR outputs (224 + 448 resolution) into unified
LabelMe annotations with line assignment, OCR text generation,
acceptance gating, and optional subboard cross-validation.

Pipeline:
1. Load detection JSONs from one or two model directories
2. NMS-merge character detections across models (IoU > 0.5)
3. NMS-merge line-bbox detections across models (IoU > 0.3, union merge)
4. Assign characters to lines (center-in-box)
5. Sort chars left-to-right, generate OCR text per line
6. Apply acceptance gate (confidence, ordering, completeness)
7. Optional: subboard cross-validation via majority voting
8. Write merged LabelMe JSON + symlink crop images

Usage:
    python merge_detections.py \\
        --det-224 Data/det_224/ \\
        --det-448 Data/det_448/ \\
        --images Data/crops/ \\
        --manifest Data/crops/crop_manifest.json \\
        --output Data/merged/
"""

import argparse
import json
import os
import re
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_iou(box1: Tuple[float, float, float, float],
                box2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Return (cx, cy) of a box (x1, y1, x2, y2)."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def point_in_box(px: float, py: float,
                 box: Tuple[float, float, float, float]) -> bool:
    """Check whether point (px, py) is inside box (x1, y1, x2, y2)."""
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


def overlap_area(box1: Tuple[float, float, float, float],
                 box2: Tuple[float, float, float, float]) -> float:
    """Compute intersection area between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def union_box(box1: Tuple[float, float, float, float],
              box2: Tuple[float, float, float, float]
              ) -> Tuple[float, float, float, float]:
    """Return the union bounding box of two boxes."""
    return (min(box1[0], box2[0]), min(box1[1], box2[1]),
            max(box1[2], box2[2]), max(box1[3], box2[3]))


# ---------------------------------------------------------------------------
# Detection data structure
# ---------------------------------------------------------------------------

class Detection:
    """Lightweight detection container."""

    __slots__ = ('label', 'box', 'conf', 'source')

    def __init__(self, label: str, box: Tuple[float, float, float, float],
                 conf: float, source: str = ''):
        self.label = label
        self.box = box  # (x1, y1, x2, y2)
        self.conf = conf
        self.source = source  # e.g. "224" or "448"

    def center(self) -> Tuple[float, float]:
        return box_center(self.box)


# ---------------------------------------------------------------------------
# LabelMe JSON I/O
# ---------------------------------------------------------------------------

def load_detection_json(json_path: Path) -> Optional[Dict]:
    """Load a LabelMe detection JSON file."""
    if not json_path.exists():
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def extract_detections(data: Dict, source: str) -> Tuple[List[Detection], List[Detection]]:
    """
    Extract character and line-bbox detections from LabelMe JSON data.

    Returns:
        (char_detections, line_detections)
    """
    chars: List[Detection] = []
    lines: List[Detection] = []

    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        if len(points) != 2:
            continue

        x1, y1 = points[0]
        x2, y2 = points[1]
        box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        # Parse confidence from description (run_rfdetr stores it as plain float string)
        desc = shape.get('description', '')
        try:
            conf = float(desc)
        except (ValueError, TypeError):
            conf = 0.0

        det = Detection(label=label, box=box, conf=conf, source=source)

        if label == 'line-bbox':
            lines.append(det)
        elif re.match(r'^[0-9A-Z]$', label):
            chars.append(det)
        # Skip anything else (unexpected labels)

    return chars, lines


def extract_agnostic_detections(data: Dict, source: str = 'agnostic') -> List[Detection]:
    """Extract agnostic character detections (label='char') from LabelMe JSON."""
    chars: List[Detection] = []
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        if len(points) != 2:
            continue
        if label != 'char':
            continue
        x1, y1 = points[0]
        x2, y2 = points[1]
        box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        desc = shape.get('description', '')
        try:
            conf = float(desc)
        except (ValueError, TypeError):
            conf = 0.0
        chars.append(Detection(label='char', box=box, conf=conf, source=source))
    return chars


def cross_check_agnostic(
    merged_chars: List[Detection],
    agnostic_chars: List[Detection],
    iou_thresh: float = 0.3,
) -> List[str]:
    """
    Cross-check classifier chars against agnostic chars.

    Returns list of reason strings for any discrepancies found:
    - MISSED_BY_CLASSIFIER:N — agnostic found N chars not matched by classifier
    - FALSE_POSITIVE_SUSPECT:N — classifier has N chars not matched by agnostic
    - AGNOSTIC_COUNT_MISMATCH:XvY — X classifier chars vs Y agnostic chars
    """
    reasons: List[str] = []

    if not agnostic_chars:
        return reasons

    # Match classifier chars to agnostic chars via IoU
    matched_agnostic = set()
    matched_classifier = set()

    for ci, c_det in enumerate(merged_chars):
        best_iou = 0.0
        best_ai = -1
        for ai, a_det in enumerate(agnostic_chars):
            iou = compute_iou(c_det.box, a_det.box)
            if iou > best_iou:
                best_iou = iou
                best_ai = ai
        if best_iou >= iou_thresh and best_ai >= 0:
            matched_classifier.add(ci)
            matched_agnostic.add(best_ai)

    missed = len(agnostic_chars) - len(matched_agnostic)
    false_pos = len(merged_chars) - len(matched_classifier)

    if missed > 0:
        reasons.append(f'MISSED_BY_CLASSIFIER:{missed}')
    if false_pos > 0:
        reasons.append(f'FALSE_POSITIVE_SUSPECT:{false_pos}')
    if len(merged_chars) != len(agnostic_chars):
        reasons.append(f'AGNOSTIC_COUNT_MISMATCH:{len(merged_chars)}v{len(agnostic_chars)}')

    return reasons


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms_chars(detections: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
    """
    Non-maximum suppression for character detections.

    Sorts by confidence descending, suppresses lower-confidence duplicates
    with IoU > *iou_thresh*.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d.conf, reverse=True)
    keep: List[Detection] = []

    for det in dets:
        suppressed = False
        for kept in keep:
            if compute_iou(det.box, kept.box) > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)

    return keep


def nms_lines(detections: List[Detection], iou_thresh: float = 0.3) -> List[Detection]:
    """
    NMS for line-bbox detections with union-merge behaviour.

    If two line-bboxes overlap (IoU > *iou_thresh*), merge into the union
    bounding box and keep the higher confidence.
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d.conf, reverse=True)
    merged: List[Detection] = []

    for det in dets:
        was_merged = False
        for i, existing in enumerate(merged):
            if compute_iou(det.box, existing.box) > iou_thresh:
                # Merge: union bbox, keep higher confidence
                new_box = union_box(existing.box, det.box)
                new_conf = max(existing.conf, det.conf)
                merged[i] = Detection(
                    label='line-bbox',
                    box=new_box,
                    conf=new_conf,
                    source=existing.source if existing.conf >= det.conf else det.source,
                )
                was_merged = True
                break
        if not was_merged:
            merged.append(det)

    return merged


# ---------------------------------------------------------------------------
# Character → Line assignment
# ---------------------------------------------------------------------------

def assign_chars_to_lines(
    chars: List[Detection],
    lines: List[Detection],
    padding: float = 5.0,
) -> Tuple[Dict[int, List[Detection]], List[Detection]]:
    """
    Assign each character to a line-bbox.

    Strategy:
    - If char center falls inside exactly one line → assign there.
    - If inside multiple lines → assign to the one with most overlap area.
    - Otherwise → unassigned.

    Returns:
        (line_idx_to_chars, unassigned_chars)
    """
    assigned: Dict[int, List[Detection]] = defaultdict(list)
    unassigned: List[Detection] = []

    for char in chars:
        cx, cy = char.center()
        candidates: List[Tuple[int, float]] = []  # (line_idx, overlap)

        for li, line in enumerate(lines):
            if point_in_box(cx, cy, line.box):
                ov = overlap_area(char.box, line.box)
                candidates.append((li, ov))

        if len(candidates) == 1:
            assigned[candidates[0][0]].append(char)
        elif len(candidates) > 1:
            # Pick line with greatest overlap
            best = max(candidates, key=lambda c: c[1])
            assigned[best[0]].append(char)
        else:
            unassigned.append(char)

    return dict(assigned), unassigned


def create_auto_line(chars: List[Detection], padding: float = 5.0) -> Detection:
    """
    Create an auto line-bbox that tightly encloses *chars* with *padding*.
    """
    x1 = min(c.box[0] for c in chars) - padding
    y1 = min(c.box[1] for c in chars) - padding
    x2 = max(c.box[2] for c in chars) + padding
    y2 = max(c.box[3] for c in chars) + padding
    conf = min(c.conf for c in chars)
    return Detection(label='line-bbox', box=(x1, y1, x2, y2),
                     conf=conf, source='auto')


# ---------------------------------------------------------------------------
# OCR text generation & acceptance gate
# ---------------------------------------------------------------------------

def sort_chars_ltr(chars: List[Detection]) -> List[Detection]:
    """Sort characters left-to-right by x-center."""
    return sorted(chars, key=lambda c: c.center()[0])


def has_clear_ordering(chars: List[Detection]) -> bool:
    """
    Check that characters have clearly separated x-centers
    (no two chars have overlapping x-center positions).
    """
    if len(chars) <= 1:
        return True
    centers = [c.center()[0] for c in chars]
    for i in range(len(centers) - 1):
        if centers[i + 1] <= centers[i]:
            return False
    return True


def acceptance_gate(
    chars: List[Detection],
    conf_threshold: float = 0.8,
) -> Tuple[bool, List[str]]:
    """
    Decide whether a line should be auto-accepted or flagged for review.

    Returns:
        (needs_review, list_of_reasons)
    """
    reasons: List[str] = []

    if not chars:
        return True, ['NO_CHARS']

    # Check confidence
    if any(c.conf < conf_threshold for c in chars):
        reasons.append('LOW_CONF')

    # Check ordering
    if not has_clear_ordering(chars):
        reasons.append('ORDER_AMBIGUOUS')

    needs_review = len(reasons) > 0
    return needs_review, reasons


# ---------------------------------------------------------------------------
# Subboard cross-validation (majority voting)
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> Optional[Dict]:
    """Load crop_manifest.json."""
    if not manifest_path or not manifest_path.exists():
        return None
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading manifest {manifest_path}: {e}")
        return None


def cross_validate_siblings(
    all_results: Dict[str, List[Dict]],
    manifest: Dict,
) -> Dict[str, List[Dict]]:
    """
    Subboard cross-validation via majority voting.

    *manifest* maps image stems to metadata including ``region_group_id``.
    *all_results* maps image stem to list of line dicts
    (each with 'ocr_text', 'conf', 'needs_review', 'reasons').

    For each region_group, compare OCR texts across siblings.  If a majority
    agree, outliers are flagged with ``SIBLING_DISAGREE``.

    Returns updated *all_results* (mutated in-place and returned).
    """
    # Build region_group_id → [image_stem, ...] mapping
    groups: Dict[str, List[str]] = defaultdict(list)
    for stem, meta in manifest.items():
        gid = meta.get('region_group_id')
        if gid is not None:
            groups[str(gid)].append(stem)

    for gid, stems in groups.items():
        if len(stems) < 2:
            continue

        # Collect per-line-index OCR texts across siblings
        max_lines = max(len(all_results.get(s, [])) for s in stems)
        for line_idx in range(max_lines):
            texts: List[Tuple[str, str]] = []  # (stem, ocr_text)
            confs: List[float] = []
            for stem in stems:
                line_list = all_results.get(stem, [])
                if line_idx < len(line_list):
                    texts.append((stem, line_list[line_idx].get('ocr_text', '')))
                    confs.append(line_list[line_idx].get('conf', 0.0))

            if len(texts) < 2:
                continue

            # Majority vote
            text_counter = Counter(t for _, t in texts)
            majority_text, majority_count = text_counter.most_common(1)[0]

            # Flag outliers
            for stem, text in texts:
                line_list = all_results.get(stem, [])
                if line_idx < len(line_list):
                    line_dict = line_list[line_idx]
                    if text != majority_text:
                        if 'SIBLING_DISAGREE' not in line_dict.get('reasons', []):
                            line_dict.setdefault('reasons', []).append('SIBLING_DISAGREE')
                            line_dict['needs_review'] = True

            # Group confidence = median of per-instance line confs
            if confs:
                group_conf = statistics.median(confs)
                for stem in stems:
                    line_list = all_results.get(stem, [])
                    if line_idx < len(line_list):
                        line_list[line_idx]['group_conf'] = round(group_conf, 4)

    return all_results


# ---------------------------------------------------------------------------
# Per-image merge pipeline
# ---------------------------------------------------------------------------

def merge_single_image(
    image_stem: str,
    det_224_path: Optional[Path],
    det_448_path: Optional[Path],
    det_agnostic_path: Optional[Path] = None,
    image_width: int = 0,
    image_height: int = 0,
    image_filename: str = '',
    conf_threshold: float = 0.8,
    char_iou_thresh: float = 0.5,
    line_iou_thresh: float = 0.3,
    line_padding: float = 5.0,
) -> Tuple[Dict, List[Dict]]:
    """
    Run the full merge pipeline for a single image.

    Returns:
        (labelme_dict, line_info_list)
        where *line_info_list* contains per-line metadata for cross-validation.
    """
    # --- Step 1: load detections ---
    all_chars: List[Detection] = []
    all_lines: List[Detection] = []

    for path, source in [(det_224_path, '224'), (det_448_path, '448')]:
        if path is None:
            continue
        data = load_detection_json(path)
        if data is None:
            continue
        chars, lines = extract_detections(data, source)
        all_chars.extend(chars)
        all_lines.extend(lines)

    # --- Step 2: NMS merge characters ---
    merged_chars = nms_chars(all_chars, iou_thresh=char_iou_thresh)

    # --- Step 2b: agnostic cross-check ---
    agnostic_reasons: List[str] = []
    if det_agnostic_path is not None:
        ag_data = load_detection_json(det_agnostic_path)
        if ag_data is not None:
            ag_chars = extract_agnostic_detections(ag_data)
            agnostic_reasons = cross_check_agnostic(merged_chars, ag_chars)

    # --- Step 3: NMS merge line-bboxes ---
    merged_lines = nms_lines(all_lines, iou_thresh=line_iou_thresh)

    # --- Step 4: assign characters to lines ---
    line_char_map, unassigned = assign_chars_to_lines(
        merged_chars, merged_lines, padding=line_padding)

    # Create auto lines for unassigned chars (group nearby unassigned chars)
    if unassigned:
        # Simple approach: create one auto-line per cluster of unassigned chars
        # whose bboxes overlap vertically
        unassigned_sorted = sorted(unassigned, key=lambda c: c.center()[0])
        clusters: List[List[Detection]] = []
        for ch in unassigned_sorted:
            placed = False
            for cluster in clusters:
                # Check if this char overlaps vertically with any char in cluster
                # and is horizontally close
                last = cluster[-1]
                cy = ch.center()[1]
                ly = last.center()[1]
                h = max(ch.box[3] - ch.box[1], last.box[3] - last.box[1])
                if abs(cy - ly) < h * 1.5 and ch.box[0] - last.box[2] < h * 3:
                    cluster.append(ch)
                    placed = True
                    break
            if not placed:
                clusters.append([ch])

        for cluster in clusters:
            auto_line = create_auto_line(cluster, padding=line_padding)
            new_idx = len(merged_lines)
            merged_lines.append(auto_line)
            line_char_map[new_idx] = cluster

    # --- Step 5 & 6: sort, generate OCR text, acceptance gate ---
    shapes: List[Dict] = []
    line_info_list: List[Dict] = []

    for line_idx, line_det in enumerate(merged_lines):
        line_uid = f"{image_stem}#L{line_idx}"
        chars_in_line = line_char_map.get(line_idx, [])
        chars_sorted = sort_chars_ltr(chars_in_line)

        # Generate OCR text
        ocr_text = ''.join(c.label for c in chars_sorted)

        # Line confidence = min of char confidences (or line det conf if no chars)
        if chars_sorted:
            line_conf = min(c.conf for c in chars_sorted)
        else:
            line_conf = line_det.conf

        # Acceptance gate
        needs_review, reasons = acceptance_gate(
            chars_sorted, conf_threshold=conf_threshold)

        # Add agnostic cross-check reasons
        if agnostic_reasons:
            reasons.extend(agnostic_reasons)
            needs_review = True

        # Also flag if there were unassigned chars that formed this auto-line
        if line_det.source == 'auto' and chars_sorted:
            if 'UNASSIGNED_CHARS' not in reasons:
                reasons.append('UNASSIGNED_CHARS')
                needs_review = True

        # Build line-bbox shape
        reason_str = ','.join(reasons) if reasons else ''
        line_desc = (
            f"line_uid={line_uid};ocr={ocr_text};"
            f"conf={line_conf:.4f};needs_review={'1' if needs_review else '0'}"
        )
        if reason_str:
            line_desc += f";reason={reason_str}"

        line_shape = {
            "label": "line-bbox",
            "points": [
                [round(line_det.box[0], 2), round(line_det.box[1], 2)],
                [round(line_det.box[2], 2), round(line_det.box[3], 2)],
            ],
            "group_id": None,
            "description": line_desc,
            "shape_type": "rectangle",
            "flags": {},
        }
        shapes.append(line_shape)

        # Build char shapes
        for char_idx, char_det in enumerate(chars_sorted):
            char_desc = (
                f"line_uid={line_uid};idx={char_idx};conf={char_det.conf:.4f}"
            )
            char_shape = {
                "label": char_det.label,
                "points": [
                    [round(char_det.box[0], 2), round(char_det.box[1], 2)],
                    [round(char_det.box[2], 2), round(char_det.box[3], 2)],
                ],
                "group_id": None,
                "description": char_desc,
                "shape_type": "rectangle",
                "flags": {},
            }
            shapes.append(char_shape)

        # Collect line info for cross-validation
        line_info_list.append({
            'line_uid': line_uid,
            'ocr_text': ocr_text,
            'conf': round(line_conf, 4),
            'needs_review': needs_review,
            'reasons': list(reasons),
        })

    # Build LabelMe JSON
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }

    return labelme_data, line_info_list


# ---------------------------------------------------------------------------
# Image dimension helpers
# ---------------------------------------------------------------------------

def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Read image width and height without loading the full pixel buffer.

    Falls back to PIL if available, otherwise reads from detection JSON.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        pass
    return 0, 0


def dimensions_from_det_json(data: Optional[Dict]) -> Tuple[int, int]:
    """Extract (width, height) from a detection JSON dict."""
    if data is None:
        return 0, 0
    return data.get('imageWidth', 0), data.get('imageHeight', 0)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def collect_image_stems(
    det_224_dir: Optional[Path],
    det_448_dir: Optional[Path],
) -> List[str]:
    """
    Collect the union of image stems present in the detection directories.
    """
    stems: set = set()
    for d in [det_224_dir, det_448_dir]:
        if d and d.is_dir():
            for p in d.glob('*.json'):
                stems.add(p.stem)
    return sorted(stems)


def find_image_file(images_dir: Optional[Path], stem: str) -> Optional[Path]:
    """Find the crop image for a given stem under *images_dir*."""
    if images_dir is None or not images_dir.is_dir():
        return None
    for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def run_merge(
    det_224_dir: Optional[Path],
    det_448_dir: Optional[Path],
    images_dir: Optional[Path],
    manifest_path: Optional[Path],
    output_dir: Path,
    det_agnostic_dir: Optional[Path] = None,
    conf_threshold: float = 0.8,
    char_iou_thresh: float = 0.5,
    line_iou_thresh: float = 0.3,
    line_padding: float = 5.0,
) -> None:
    """Run the full merge pipeline across all images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = collect_image_stems(det_224_dir, det_448_dir)
    if det_agnostic_dir and det_agnostic_dir.is_dir():
        stems_set = set(stems)
        for p in det_agnostic_dir.glob('*.json'):
            stems_set.add(p.stem)
        stems = sorted(stems_set)
    if not stems:
        print("No detection JSONs found. Nothing to merge.")
        return

    print(f"Found {len(stems)} image stems to process.")

    # Load manifest for cross-validation
    manifest = None
    if manifest_path:
        raw_manifest = load_manifest(manifest_path)
        if raw_manifest:
            # Convert from {"board_name": ..., "crops": [...]} to
            # {crop_stem: {"region_group_id": ..., ...}} keyed by image stem
            crops_list = raw_manifest.get("crops", [])
            if isinstance(crops_list, list) and crops_list:
                manifest = {}
                for entry in crops_list:
                    crop_file = entry.get("crop_file", "")
                    stem_key = Path(crop_file).stem if crop_file else ""
                    if stem_key:
                        manifest[stem_key] = entry
                print(f"Loaded manifest with {len(manifest)} entries.")
            elif isinstance(raw_manifest, dict) and "crops" not in raw_manifest:
                # Already in {stem: meta} format
                manifest = raw_manifest
                print(f"Loaded manifest with {len(manifest)} entries.")

    # Per-image merge
    all_results: Dict[str, List[Dict]] = {}
    stats = {'total': 0, 'auto_accept': 0, 'needs_review': 0, 'total_lines': 0}

    for stem in stems:
        det_224_path = (det_224_dir / f"{stem}.json") if det_224_dir else None
        det_448_path = (det_448_dir / f"{stem}.json") if det_448_dir else None
        det_agnostic_path = (det_agnostic_dir / f"{stem}.json") if det_agnostic_dir else None

        if det_224_path and not det_224_path.exists():
            det_224_path = None
        if det_448_path and not det_448_path.exists():
            det_448_path = None
        if det_agnostic_path and not det_agnostic_path.exists():
            det_agnostic_path = None

        if det_224_path is None and det_448_path is None:
            continue

        # Determine image dimensions
        image_file = find_image_file(images_dir, stem)
        image_width, image_height = 0, 0
        image_filename = f"{stem}.png"

        if image_file:
            image_width, image_height = get_image_dimensions(image_file)
            image_filename = image_file.name
        else:
            # Fall back to detection JSON dimensions
            for p in [det_224_path, det_448_path]:
                if p:
                    data = load_detection_json(p)
                    w, h = dimensions_from_det_json(data)
                    if w > 0 and h > 0:
                        image_width, image_height = w, h
                        break

        # Run merge
        labelme_data, line_info_list = merge_single_image(
            image_stem=stem,
            det_224_path=det_224_path,
            det_448_path=det_448_path,
            det_agnostic_path=det_agnostic_path,
            image_width=image_width,
            image_height=image_height,
            image_filename=image_filename,
            conf_threshold=conf_threshold,
            char_iou_thresh=char_iou_thresh,
            line_iou_thresh=line_iou_thresh,
            line_padding=line_padding,
        )

        all_results[stem] = line_info_list

        # Write JSON
        out_json = output_dir / f"{stem}.json"
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)

        # Copy or symlink image
        if image_file:
            out_img = output_dir / image_filename
            if not out_img.exists():
                try:
                    os.symlink(image_file.resolve(), out_img)
                except OSError:
                    shutil.copy2(str(image_file), str(out_img))

        # Stats
        stats['total'] += 1
        for li in line_info_list:
            stats['total_lines'] += 1
            if li['needs_review']:
                stats['needs_review'] += 1
            else:
                stats['auto_accept'] += 1

    # --- Step 7: subboard cross-validation ---
    if manifest:
        print("Running subboard cross-validation...")
        all_results = cross_validate_siblings(all_results, manifest)

        # Re-write JSONs whose lines were updated by cross-validation
        for stem, line_info_list in all_results.items():
            out_json = output_dir / f"{stem}.json"
            if not out_json.exists():
                continue

            # Check if any line was modified (SIBLING_DISAGREE added)
            has_sibling_flag = any(
                'SIBLING_DISAGREE' in li.get('reasons', [])
                for li in line_info_list
            )
            if not has_sibling_flag:
                continue

            # Reload and patch
            with open(out_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            line_shape_idx = 0
            for shape in data.get('shapes', []):
                if shape.get('label') != 'line-bbox':
                    continue
                if line_shape_idx >= len(line_info_list):
                    break

                li = line_info_list[line_shape_idx]
                line_shape_idx += 1

                # Rebuild description with updated reasons
                reason_str = ','.join(li.get('reasons', []))
                needs_review = li.get('needs_review', False)
                line_uid = li.get('line_uid', '')
                ocr_text = li.get('ocr_text', '')
                conf = li.get('conf', 0.0)

                desc = (
                    f"line_uid={line_uid};ocr={ocr_text};"
                    f"conf={conf:.4f};needs_review={'1' if needs_review else '0'}"
                )
                if reason_str:
                    desc += f";reason={reason_str}"
                if 'group_conf' in li:
                    desc += f";group_conf={li['group_conf']:.4f}"

                shape['description'] = desc

            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        # Update stats after cross-validation
        review_delta = sum(
            1 for stem_lines in all_results.values()
            for li in stem_lines
            if 'SIBLING_DISAGREE' in li.get('reasons', [])
        )
        if review_delta > 0:
            print(f"  Cross-validation flagged {review_delta} additional lines.")

    # Print summary
    print(f"\nMerge complete.")
    print(f"  Images processed : {stats['total']}")
    print(f"  Total lines      : {stats['total_lines']}")
    print(f"  Auto-accepted    : {stats['auto_accept']}")
    print(f"  Needs review     : {stats['needs_review']}")
    print(f"  Output directory : {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge dual-model RF-DETR detection outputs into unified "
                    "LabelMe annotations with line assignment and acceptance gating.",
    )
    parser.add_argument(
        '--det-224', type=str, default=None,
        help='Directory containing 224-model detection JSONs',
    )
    parser.add_argument(
        '--det-448', type=str, default=None,
        help='Directory containing 448-model detection JSONs',
    )
    parser.add_argument(
        '--det-agnostic', type=str, default=None,
        help='Directory containing char_agnostic detection JSONs for cross-check',
    )
    parser.add_argument(
        '--images', type=str, default=None,
        help='Directory containing crop images (for dimensions and symlinking)',
    )
    parser.add_argument(
        '--manifest', type=str, default=None,
        help='Path to crop_manifest.json for subboard cross-validation',
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output directory for merged LabelMe JSONs',
    )
    parser.add_argument(
        '--conf-threshold', type=float, default=0.8,
        help='Confidence threshold for auto-acceptance (default: 0.8)',
    )
    parser.add_argument(
        '--char-iou', type=float, default=0.5,
        help='IoU threshold for character NMS (default: 0.5)',
    )
    parser.add_argument(
        '--line-iou', type=float, default=0.3,
        help='IoU threshold for line-bbox NMS (default: 0.3)',
    )
    parser.add_argument(
        '--line-padding', type=float, default=5.0,
        help='Padding for auto-generated line bboxes (default: 5.0)',
    )

    args = parser.parse_args()

    if args.det_224 is None and args.det_448 is None:
        parser.error("At least one of --det-224 or --det-448 must be provided.")

    det_224_dir = Path(args.det_224) if args.det_224 else None
    det_448_dir = Path(args.det_448) if args.det_448 else None
    det_agnostic_dir = Path(args.det_agnostic) if args.det_agnostic else None
    images_dir = Path(args.images) if args.images else None
    manifest_path = Path(args.manifest) if args.manifest else None
    output_dir = Path(args.output)

    # Validate directories
    for label, d in [('--det-224', det_224_dir), ('--det-448', det_448_dir),
                     ('--det-agnostic', det_agnostic_dir), ('--images', images_dir)]:
        if d is not None and not d.is_dir():
            parser.error(f"{label} directory does not exist: {d}")

    run_merge(
        det_224_dir=det_224_dir,
        det_448_dir=det_448_dir,
        images_dir=images_dir,
        manifest_path=manifest_path,
        output_dir=output_dir,
        det_agnostic_dir=det_agnostic_dir,
        conf_threshold=args.conf_threshold,
        char_iou_thresh=args.char_iou,
        line_iou_thresh=args.line_iou,
        line_padding=args.line_padding,
    )


if __name__ == '__main__':
    main()
