#!/usr/bin/env python3
"""
OCR Review Web Application
===========================
Flask-based web app for line-level OCR review.

Features:
- Three-tab UI: Review, Dashboard, Queue
- Line-level workflow with keyboard shortcuts (A/E/X)
- Real-time metrics and analytics
- Event logging for audit trail
"""

import argparse
import json
import logging
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import quote as urlquote

from flask import Flask, render_template, jsonify, request, send_file, Response, make_response
import io
import cv2
import numpy as np
import base64

from line_event_store import LineEventStore, EventType
from line_loader import (
    load_all_annotations, find_line_in_annotations,
    ImageAnnotation, LineAnnotation, BBox,
    ComponentGroup, build_component_groups, build_text_groups
)
from metrics_calculator import MetricsCalculator


# ============================================================================
# Flask App Setup
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ocr-review-secret-key'

# Global state
app_state = {
    'dataset_root': None,
    'available_classes': [],
    'current_class': None,
    'class_configs': {},  # class_name -> {fused_dir, images_dir, db_path}
    'class_data': {},  # class_name -> {event_store, all_annotations, all_metrics_calcs, available_subdirs}
    'annotations_dir': None,
    'db_path': None,
    'event_store': None,
    'annotations': [],
    'metrics_calc': None,
    'line_queue': [],  # List of line_uids in current view
    'current_idx': 0,  # Current position in queue
    'filter_status': 'all',  # Filter: all, needs_review, unreviewed, etc.
    'filter_reason': None,  # Filter by reason
    'manifest_path': None,
    'component_groups': [],  # List of ComponentGroup
    'group_mode': 'none',  # 'position' | 'text' | 'none'
}

IMAGE_QUEUE_PREFIX = "__image__:"


class ThumbnailCache:
    """LRU cache for gallery/group thumbnail JPEG bytes."""
    def __init__(self, max_entries=2000):
        self._cache = OrderedDict()
        self._max = max_entries

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def invalidate(self):
        self._cache.clear()

thumbnail_cache = ThumbnailCache(max_entries=2000)


# ============================================================================
# Helper Functions
# ============================================================================

def derive_display_status(line_status: Dict[str, Any], line_ann: Optional[LineAnnotation] = None) -> str:
    """
    Derive the display status for a line based on events and annotation data.

    **ROOT CAUSE FIX:**
    Lines that were auto-accepted during fusion have NO events in the database.
    The event store returns status="unknown" for lines with no events.
    We need to check the annotation's needs_review flag to determine the correct status.

    **LOGIC:**
    - If event store has a status (not "unknown") → use it (human has interacted)
    - If status is "unknown" AND line.needs_review is False → return "accepted" (auto-accepted)
    - If status is "unknown" AND line.needs_review is True → return "uncertain" (needs review)
    - If status is "unknown" AND no line annotation → return "unknown" (can't determine)

    Args:
        line_status: Dictionary from event_store.get_line_status()
        line_ann: Optional LineAnnotation object with needs_review flag

    Returns:
        Corrected status string
    """
    status = line_status.get('status', 'unknown')

    # If event store has a real status, use it (human has interacted with this line)
    if status != 'unknown':
        return status

    # Status is "unknown" - means no events logged yet
    # Check the annotation's needs_review flag to determine initial status
    if line_ann:
        if line_ann.needs_review:
            # Line was flagged for review during fusion
            return 'uncertain'
        else:
            # Line was auto-accepted during fusion (k == len(S))
            return 'accepted'

    # No annotation available, can't determine
    return 'unknown'


def invalidate_metrics():
    """Mark metrics as needing recalculation."""
    app_state['metrics_dirty'] = True


def ensure_metrics_fresh():
    """Rebuild metrics only for the current subdirectory if invalidated."""
    if not app_state.get('metrics_dirty', False):
        return
    event_store = app_state['event_store']
    current_subdir = app_state['current_subdir']
    current_anns = app_state['annotations']

    app_state['all_metrics_calcs'][current_subdir] = MetricsCalculator(event_store, current_anns)
    app_state['metrics_calc'] = app_state['all_metrics_calcs'][current_subdir]
    app_state['metrics_dirty'] = False


def reload_image_annotation(json_path: Path, img_ann: ImageAnnotation):
    """
    Reload an image annotation after JSON file changes and update all affected structures.

    Args:
        json_path: Path to the JSON file that was modified
        img_ann: The original ImageAnnotation object (to find it in the cache)

    Returns:
        The updated ImageAnnotation object, or None if reload failed
    """
    from line_loader import parse_image_annotation

    updated_img_ann = parse_image_annotation(json_path)
    if not updated_img_ann:
        logging.error(f"Failed to reload annotation from {json_path}")
        return None

    # Update in all subdirectories' annotations
    all_annotations = app_state['all_annotations']
    all_metrics_calcs = app_state['all_metrics_calcs']
    event_store = app_state['event_store']

    # Find which subdirectory this image belongs to and update it
    for subdir_name, subdir_anns in all_annotations.items():
        for i, ann in enumerate(subdir_anns):
            if ann.image_id == img_ann.image_id and ann.json_path == str(json_path):
                # Update this annotation
                all_annotations[subdir_name][i] = updated_img_ann
                # Recalculate metrics for this subdirectory
                all_metrics_calcs[subdir_name] = MetricsCalculator(event_store, subdir_anns)
                logging.info(f"Reloaded annotations for {subdir_name}/")
                break

    # Update current review annotations if applicable
    current_subdir = app_state['current_subdir']
    app_state['annotations'] = all_annotations[current_subdir]
    app_state['metrics_calc'] = all_metrics_calcs[current_subdir]

    return updated_img_ann


def reload_subdir_cache(subdir_name: str) -> List[ImageAnnotation]:
    """
    Reload all annotations for a subdirectory and refresh metrics.

    Args:
        subdir_name: Subdirectory name (e.g., "candidate", "final")

    Returns:
        List of ImageAnnotation for the subdirectory
    """
    fused_dir = app_state['fused_dir']
    event_store = app_state['event_store']
    subdir_path = fused_dir / subdir_name

    annotations = load_all_annotations(subdir_path)
    app_state['all_annotations'][subdir_name] = annotations
    app_state['all_metrics_calcs'][subdir_name] = MetricsCalculator(event_store, annotations)

    return annotations


def load_class_data(class_name: str):
    """
    Load annotations and metrics for a class into app_state['class_data'].
    """
    class_config = app_state['class_configs'][class_name]
    fused_dir = class_config['fused_dir']
    db_path = class_config['db_path']

    # Detect available subdirectories
    available_subdirs = []
    for subdir_name in ["candidate", "final"]:
        subdir_path = fused_dir / subdir_name
        if subdir_path.exists() and subdir_path.is_dir():
            available_subdirs.append(subdir_name)

    if not available_subdirs:
        raise RuntimeError(f"No candidate/ or final/ subdirectories found in {fused_dir}")

    event_store = LineEventStore(db_path)

    all_annotations = {}
    all_metrics_calcs = {}
    for subdir_name in available_subdirs:
        subdir_path = fused_dir / subdir_name
        subdir_annotations = load_all_annotations(subdir_path)
        all_annotations[subdir_name] = subdir_annotations
        all_metrics_calcs[subdir_name] = MetricsCalculator(event_store, subdir_annotations)

    app_state['class_data'][class_name] = {
        'event_store': event_store,
        'all_annotations': all_annotations,
        'all_metrics_calcs': all_metrics_calcs,
        'available_subdirs': available_subdirs,
    }


def set_current_class(class_name: str, subdir: Optional[str] = None):
    """
    Switch app_state to the specified class and subdir.
    """
    if class_name not in app_state['class_configs']:
        raise RuntimeError(f"Unknown class: {class_name}")

    if class_name not in app_state['class_data']:
        load_class_data(class_name)

    class_config = app_state['class_configs'][class_name]
    class_data = app_state['class_data'][class_name]

    available_subdirs = class_data['available_subdirs']
    current_subdir = subdir if subdir in available_subdirs else available_subdirs[0]

    annotations_dir = class_config['fused_dir'] / current_subdir
    annotations = class_data['all_annotations'][current_subdir]
    metrics_calc = class_data['all_metrics_calcs'][current_subdir]

    line_queue = []
    for img_ann in annotations:
        if img_ann.lines:
            for line in img_ann.lines:
                line_queue.append(line.line_uid)
        else:
            # Image with no lines — add image-level entry so user can add lines
            line_queue.append(f"{IMAGE_QUEUE_PREFIX}{img_ann.image_id}")

    app_state['current_class'] = class_name
    app_state['fused_dir'] = class_config['fused_dir']
    app_state['images_dir'] = class_config['images_dir']
    app_state['db_path'] = class_config['db_path']
    app_state['event_store'] = class_data['event_store']
    app_state['all_annotations'] = class_data['all_annotations']
    app_state['all_metrics_calcs'] = class_data['all_metrics_calcs']
    app_state['available_subdirs'] = available_subdirs
    app_state['current_subdir'] = current_subdir
    app_state['annotations_dir'] = annotations_dir
    app_state['annotations'] = annotations
    app_state['metrics_calc'] = metrics_calc
    app_state['line_queue'] = line_queue
    app_state['current_idx'] = 0

    # Load component groups for gallery view
    load_manifest_and_groups()


def load_manifest_and_groups():
    """Load crop manifest and build component groups for current class."""
    fused_dir = app_state.get('fused_dir')
    if not fused_dir:
        return
    manifest_path = fused_dir / 'crop_manifest.json'
    if not manifest_path.exists():
        # Also try parent directory
        manifest_path = fused_dir.parent / 'crop_manifest.json'
    if manifest_path.exists():
        app_state['manifest_path'] = manifest_path
        app_state['component_groups'] = build_component_groups(
            app_state['annotations'], manifest_path
        )
        app_state['group_mode'] = 'position'
    else:
        app_state['component_groups'] = []
        app_state['group_mode'] = 'none'

    # Fallback: if no position groups (no manifest) or all groups have only 1 instance,
    # try text-based grouping for batch review efficiency
    pos_groups = app_state['component_groups']
    if not pos_groups or all(len(g.instances) <= 1 for g in pos_groups):
        text_groups = build_text_groups(app_state['annotations'])
        if text_groups:
            app_state['component_groups'] = text_groups
            app_state['group_mode'] = 'text'


def update_line_description_flags(description: str, needs_review: bool,
                                  clear_reasons: bool = True) -> str:
    """
    Update needs_review (and optionally clear reasons) in a line-bbox description string.
    """
    parts = [p for p in description.split(';') if p]
    updated_parts = []
    has_needs_review = False

    for part in parts:
        if part.startswith('needs_review='):
            updated_parts.append(f'needs_review={1 if needs_review else 0}')
            has_needs_review = True
        elif clear_reasons and part.startswith('reason='):
            continue
        else:
            updated_parts.append(part)

    if not has_needs_review:
        updated_parts.append(f'needs_review={1 if needs_review else 0}')

    return ';'.join(updated_parts)


def _accept_lines_in_instance(inst, lines_to_accept):
    """Clear needs_review flag for multiple lines in one JSON read/write cycle.

    Args:
        inst: ImageAnnotation instance (has .json_path)
        lines_to_accept: list of LineAnnotation objects to accept
    Returns:
        Number of lines successfully updated
    """
    if not lines_to_accept:
        return 0
    json_path = Path(inst.json_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        line_uids = {line.line_uid for line in lines_to_accept}
        for shape in json_data.get('shapes', []):
            if shape.get('label') != 'line-bbox':
                continue
            desc = shape.get('description', '')
            for uid in list(line_uids):
                if f'line_uid={uid}' in desc:
                    shape['description'] = update_line_description_flags(desc, needs_review=False)
                    line_uids.discard(uid)
                    break
            if not line_uids:
                break

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        for line in lines_to_accept:
            line.needs_review = False
            line.reasons = []
        return len(lines_to_accept)
    except Exception as e:
        logging.warning(f"Failed to update needs_review in {json_path}: {e}")
        return 0


def find_image_in_annotations(image_id: str, annotations: List[ImageAnnotation]) -> Optional[ImageAnnotation]:
    """
    Find an image annotation by image_id.
    """
    for img_ann in annotations:
        if img_ann.image_id == image_id:
            return img_ann
    return None


def find_group_by_id(region_group_id: str) -> Optional['ComponentGroup']:
    """Find a component group by region_group_id in current app_state."""
    for group in app_state.get('component_groups', []):
        if group.region_group_id == region_group_id:
            return group
    return None


def find_image_across_classes(image_id: str):
    """Find image annotation in current class, falling back to all loaded classes."""
    annotations = app_state.get('annotations', [])
    img_ann = find_image_in_annotations(image_id, annotations)
    if not img_ann:
        for class_data in app_state.get('class_data', {}).values():
            for anns in class_data['all_annotations'].values():
                img_ann = find_image_in_annotations(image_id, anns)
                if img_ann:
                    return img_ann
    return img_ann


def is_image_queue_uid(line_uid: str) -> bool:
    return line_uid.startswith(IMAGE_QUEUE_PREFIX)


def extract_image_id(line_uid: str) -> str:
    return line_uid.replace(IMAGE_QUEUE_PREFIX, "", 1)


def generate_new_line_uid(img_ann: ImageAnnotation) -> str:
    """
    Generate a new unique line_uid for an image.
    """
    max_idx = -1
    for line in img_ann.lines:
        if '#L' in line.line_uid:
            try:
                idx = int(line.line_uid.split('#L')[-1])
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    return f"{img_ann.image_id}#L{max_idx + 1}"


def validate_bbox(bbox: dict, img_width: int, img_height: int) -> tuple[bool, str]:
    """
    Validate bbox coordinates.

    Args:
        bbox: Dictionary with keys x1, y1, x2, y2
        img_width: Image width
        img_height: Image height

    Returns:
        (is_valid, error_message) tuple
    """
    try:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']

        # Coerce None/non-numeric to float
        if any(v is None for v in (x1, y1, x2, y2)):
            return False, f"Bbox contains None: ({x1}, {y1}, {x2}, {y2})"
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        # Check for negative coordinates
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return False, f"Negative coordinates not allowed: ({x1}, {y1}, {x2}, {y2})"

        # Check bbox is within image bounds (skip if dimensions unknown)
        if img_width is not None and img_height is not None:
            if x2 > img_width or y2 > img_height:
                return False, f"Bbox exceeds image bounds: ({x2}, {y2}) > ({img_width}, {img_height})"

        # Check bbox has positive area
        if x1 >= x2 or y1 >= y2:
            return False, f"Invalid bbox dimensions: x1={x1} >= x2={x2} or y1={y1} >= y2={y2}"

        # Check bbox is not too small (at least 1 pixel)
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            return False, f"Bbox too small: width={x2-x1}, height={y2-y1}"

        return True, ""

    except (KeyError, TypeError, ValueError) as e:
        return False, f"Invalid bbox format: {str(e)}"


def rotate_bbox(bbox: BBox, width: int, height: int, direction: str) -> BBox:
    """
    Rotate a bbox around the image origin by 90-degree steps.
    direction: 'cw' or 'ccw' or '180'
    """
    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

    if direction == 'cw':
        new_x1 = height - y2
        new_y1 = x1
        new_x2 = height - y1
        new_y2 = x2
    elif direction == 'ccw':
        new_x1 = y1
        new_y1 = width - x2
        new_x2 = y2
        new_y2 = width - x1
    elif direction == '180':
        new_x1 = width - x2
        new_y1 = height - y2
        new_x2 = width - x1
        new_y2 = height - y1
    else:
        raise ValueError(f"Invalid rotation direction: {direction}")

    return BBox(
        x1=min(new_x1, new_x2),
        y1=min(new_y1, new_y2),
        x2=max(new_x1, new_x2),
        y2=max(new_y1, new_y2)
    )


def validate_char_label(label: str) -> tuple[bool, str]:
    """
    Validate character label is in allowed set [0-9A-Z].

    Args:
        label: Character label string

    Returns:
        (is_valid, error_message) tuple
    """
    if not label:
        return False, "Empty label not allowed"

    if len(label) != 1:
        return False, f"Label must be single character, got: '{label}'"

    allowed_chars = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    if label.upper() not in allowed_chars:
        return False, f"Invalid character '{label}'. Only 0-9A-Z allowed"

    return True, ""


def validate_ocr_text(text: str) -> tuple[bool, str]:
    """
    Validate OCR text contains only allowed characters [0-9A-Z].

    Args:
        text: OCR text string

    Returns:
        (is_valid, error_message) tuple
    """
    if not text:
        return True, ""  # Empty text is allowed

    allowed_chars = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    invalid_chars = set(text.upper()) - allowed_chars

    if invalid_chars:
        return False, f"Invalid characters in OCR text: {sorted(invalid_chars)}"

    return True, ""


# ============================================================================
# Image Rendering
# ============================================================================

def generate_char_color(idx: int) -> tuple:
    """
    Generate a distinct color for character index.
    Uses a color palette that's easy to distinguish.

    Args:
        idx: Character index (0-based)

    Returns:
        BGR color tuple for OpenCV
    """
    # Color palette: vibrant colors that are easy to distinguish
    colors = [
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (0, 165, 255),    # Orange
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 128, 255),    # Orange-Red
        (255, 128, 0),    # Light Blue
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Spring Green
        (255, 0, 127),    # Deep Pink
        (128, 255, 0),    # Chartreuse
    ]
    return colors[idx % len(colors)]


def find_image_path(img_ann: ImageAnnotation) -> Optional[Path]:
    """
    Resolve the image path for an annotation using JSON path or images_dir.
    """
    if img_ann.image_path and Path(img_ann.image_path).exists():
        return Path(img_ann.image_path)

    images_dir = app_state.get('images_dir')
    if images_dir:
        image_filename = Path(img_ann.json_path).stem
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            potential_path = images_dir / f"{image_filename}{ext}"
            if potential_path.exists():
                return potential_path

    return None


def get_cached_thumbnail_bytes(image_path, max_width=200):
    """Get raw JPEG thumbnail bytes with caching by (path, mtime)."""
    if not image_path or not image_path.exists():
        return None
    try:
        mtime = image_path.stat().st_mtime
        cache_key = (str(image_path), mtime, max_width)
        cached = thumbnail_cache.get(cache_key)
        if cached is not None:
            return cached

        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (max_width, int(h * scale)))
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        jpeg_bytes = buf.tobytes()
        thumbnail_cache.put(cache_key, jpeg_bytes)
        return jpeg_bytes
    except Exception:
        return None


def get_annotated_thumbnail_bytes(img_ann, max_width=200):
    """Get thumbnail with semi-transparent bbox overlays. Original image stays clearly visible."""
    image_path = find_image_path(img_ann)
    if not image_path or not image_path.exists():
        return None
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(max_width / w, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Draw on a separate overlay, then alpha-blend to keep original visible
        overlay = img.copy()
        event_store = app_state.get('event_store')

        for line in img_ann.lines:
            # Determine line color by status
            if event_store:
                ls = event_store.get_line_status(line.line_uid)
                status = ls.get('status', 'unknown')
            else:
                status = 'accepted' if not line.needs_review else 'uncertain'

            if status in ('reviewed', 'edited'):
                line_color = (0, 200, 0)   # green
            elif line.needs_review:
                line_color = (0, 100, 255)  # orange
            else:
                line_color = (200, 180, 0)  # cyan-ish

            # Semi-transparent line bbox fill + solid border
            pt1 = (int(line.bbox.x1 * scale), int(line.bbox.y1 * scale))
            pt2 = (int(line.bbox.x2 * scale), int(line.bbox.y2 * scale))
            cv2.rectangle(overlay, pt1, pt2, line_color, cv2.FILLED)

            # Char bboxes: thin solid border only (no fill)
            for char in line.chars:
                cpt1 = (int(char.bbox.x1 * scale), int(char.bbox.y1 * scale))
                cpt2 = (int(char.bbox.x2 * scale), int(char.bbox.y2 * scale))
                # Draw char border directly on img (not overlay) so it's fully opaque
                cv2.rectangle(img, cpt1, cpt2, generate_char_color(char.idx), 1)

                # Char label: small white text with dark outline for readability
                font_scale = 0.3
                tx, ty = cpt1[0], cpt1[1] - 2
                if ty < 8:
                    ty = cpt2[1] + 8  # below box if too close to top
                cv2.putText(img, char.label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, char.label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend: 20% overlay (bbox fills barely tinted), 80% original
        alpha = 0.15
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw solid thin line borders on top of blend (fully opaque)
        for line in img_ann.lines:
            if event_store:
                ls = event_store.get_line_status(line.line_uid)
                status = ls.get('status', 'unknown')
            else:
                status = 'accepted' if not line.needs_review else 'uncertain'

            if status in ('reviewed', 'edited'):
                line_color = (0, 200, 0)
            elif line.needs_review:
                line_color = (0, 100, 255)
            else:
                line_color = (200, 180, 0)

            pt1 = (int(line.bbox.x1 * scale), int(line.bbox.y1 * scale))
            pt2 = (int(line.bbox.x2 * scale), int(line.bbox.y2 * scale))
            cv2.rectangle(img, pt1, pt2, line_color, 1)

        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()
    except Exception as e:
        logging.warning(f"Failed to create annotated thumbnail: {e}")
        return None


def get_cached_thumbnail(image_path, max_width=200):
    """Get base64 data URL thumbnail (backward compat)."""
    raw = get_cached_thumbnail_bytes(image_path, max_width)
    if not raw:
        return ''
    return 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('utf-8')


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE-based contrast enhancement for display.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def render_original_image(img_ann: ImageAnnotation, enhanced: bool = False) -> str:
    """
    Render the original image without overlays.
    """
    image_path = find_image_path(img_ann)

    if image_path and image_path.exists():
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.ones((img_ann.image_height, img_ann.image_width, 3), dtype=np.uint8) * 200
            cv2.putText(img, "Failed to load image", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        img = np.ones((img_ann.image_height, img_ann.image_width, 3), dtype=np.uint8) * 200
        cv2.putText(img, "Image not found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if enhanced:
        img = enhance_contrast(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode('.png', img_rgb)
    if not success:
        return ""

    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def render_line_image(img_ann: ImageAnnotation, line_ann: LineAnnotation,
                      highlight_line: bool = True) -> str:
    """
    Render image with line and character annotations.
    Characters are shown with numeric indices only (no text labels on image).

    Args:
        img_ann: Image annotation
        line_ann: Line annotation to highlight
        highlight_line: Whether to highlight the current line

    Returns:
        Base64-encoded image data URL
    """
    # Try to find the image file
    image_path = find_image_path(img_ann)

    # Load image or create placeholder
    if image_path and image_path.exists():
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.ones((img_ann.image_height, img_ann.image_width, 3), dtype=np.uint8) * 200
            cv2.putText(img, "Failed to load image", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        # Create placeholder image
        img = np.ones((img_ann.image_height, img_ann.image_width, 3), dtype=np.uint8) * 200
        cv2.putText(img, "Image not found", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Draw all lines (faded)
    for line in img_ann.lines:
        if line.line_uid == line_ann.line_uid and not highlight_line:
            continue

        color = (100, 100, 100)  # Gray for other lines
        thickness = 1

        pt1 = (int(line.bbox.x1), int(line.bbox.y1))
        pt2 = (int(line.bbox.x2), int(line.bbox.y2))
        cv2.rectangle(img, pt1, pt2, color, thickness)

    # Draw current line (highlighted)
    if highlight_line:
        color = (0, 255, 0) if line_ann.status == "accepted" else (0, 165, 255)
        thickness = 2  # Reduced from 3 to 2

        pt1 = (int(line_ann.bbox.x1), int(line_ann.bbox.y1))
        pt2 = (int(line_ann.bbox.x2), int(line_ann.bbox.y2))
        cv2.rectangle(img, pt1, pt2, color, thickness)

    # Draw characters - colored bounding boxes ONLY (no text/indices)
    for char in line_ann.chars:
        pt1 = (int(char.bbox.x1), int(char.bbox.y1))
        pt2 = (int(char.bbox.x2), int(char.bbox.y2))

        # Get color for this character index
        color = generate_char_color(char.idx)

        # Draw colored bounding box only (thin line)
        cv2.rectangle(img, pt1, pt2, color, 1)  # Reduced from 2 to 1

    # Add white annotation area below the image
    annotation_height = 60
    img_height, img_width = img.shape[:2]

    # Create white strip
    white_strip = np.ones((annotation_height, img_width, 3), dtype=np.uint8) * 255

    # Render line-level text in the white strip
    if highlight_line:
        label = f"{line_ann.ocr_text} ({line_ann.get_char_count()} chars)"
        color = (0, 255, 0) if line_ann.status == "accepted" else (0, 165, 255)

        # Draw text centered vertically in the white strip
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (_, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

        text_x = 10
        text_y = (annotation_height + text_height) // 2

        cv2.putText(white_strip, label, (text_x, text_y), font, font_scale, color, thickness)

    # Concatenate image and white strip vertically
    img = np.vstack([img, white_strip])

    # Convert to base64
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode('.png', img_rgb)
    if not success:
        return ""

    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


# ============================================================================
# API Endpoints - Review Tab
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('ocr_review.html')


# ============================================================================
# Image Serving Endpoints — serve images as binary with HTTP caching
# ============================================================================

@app.route('/api/thumb/<path:image_id>')
def api_thumbnail(image_id):
    """Serve a thumbnail JPEG for a given image_id. ?annotated=1 draws bboxes."""
    try:
        max_width = request.args.get('w', 200, type=int)
        annotated = request.args.get('annotated', '0') == '1'
        img_ann = find_image_across_classes(image_id)
        if not img_ann:
            return Response('Not found', status=404)

        if annotated:
            jpeg_bytes = get_annotated_thumbnail_bytes(img_ann, max_width)
        else:
            image_path = find_image_path(img_ann)
            jpeg_bytes = get_cached_thumbnail_bytes(image_path, max_width)

        if not jpeg_bytes:
            return Response('Image not available', status=404)

        resp = make_response(jpeg_bytes)
        resp.headers['Content-Type'] = 'image/jpeg'
        resp.headers['Cache-Control'] = 'private, max-age=60'
        return resp
    except Exception as e:
        logging.error(f"Error serving thumbnail {image_id}: {e}")
        return Response('Error', status=500)


@app.route('/api/img/<path:image_id>')
def api_full_image(image_id):
    """Serve full-resolution image. ?enhanced=1 applies CLAHE contrast enhancement."""
    try:
        enhanced = request.args.get('enhanced', '0') == '1'
        img_ann = find_image_across_classes(image_id)
        if not img_ann:
            return Response('Not found', status=404)

        image_path = find_image_path(img_ann)
        if not image_path or not image_path.exists():
            return Response('Image not available', status=404)

        if enhanced:
            img = cv2.imread(str(image_path))
            if img is not None:
                img = enhance_contrast(img)
                _, buf = cv2.imencode('.png', img)
                resp = make_response(buf.tobytes())
                resp.headers['Content-Type'] = 'image/png'
                resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                return resp

        resp = send_file(str(image_path), mimetype='image/png')
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return resp
    except Exception as e:
        logging.error(f"Error serving image {image_id}: {e}")
        return Response('Error', status=500)


@app.route('/api/img_annotated/<path:line_uid>')
def api_annotated_image(line_uid):
    """Serve annotated line image as JPEG. Cached in memory."""
    try:
        enhanced = request.args.get('enhanced', '0') == '1'
        result = find_line_in_annotations(line_uid, app_state['annotations'])
        if not result:
            return Response('Not found', status=404)

        img_ann, line_ann = result
        # Use render_line_image but return as binary JPEG
        image_path = find_image_path(img_ann)
        if not image_path or not image_path.exists():
            return Response('Image not available', status=404)

        img = cv2.imread(str(image_path))
        if img is None:
            return Response('Failed to load', status=500)

        if enhanced:
            img = enhance_contrast(img)

        # Draw annotations on the image
        for line in img_ann.lines:
            color = (100, 100, 100)
            thickness = 1
            pt1 = (int(line.bbox.x1), int(line.bbox.y1))
            pt2 = (int(line.bbox.x2), int(line.bbox.y2))
            cv2.rectangle(img, pt1, pt2, color, thickness)

        # Current line highlighted
        color = (0, 255, 0) if line_ann.status == "accepted" else (0, 165, 255)
        pt1 = (int(line_ann.bbox.x1), int(line_ann.bbox.y1))
        pt2 = (int(line_ann.bbox.x2), int(line_ann.bbox.y2))
        cv2.rectangle(img, pt1, pt2, color, 2)

        for char in line_ann.chars:
            pt1 = (int(char.bbox.x1), int(char.bbox.y1))
            pt2 = (int(char.bbox.x2), int(char.bbox.y2))
            char_color = generate_char_color(char.idx)
            cv2.rectangle(img, pt1, pt2, char_color, 1)

        # Annotation strip
        annotation_height = 60
        img_height, img_width = img.shape[:2]
        white_strip = np.ones((annotation_height, img_width, 3), dtype=np.uint8) * 255
        label = f"{line_ann.ocr_text} ({line_ann.get_char_count()} chars)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (_, text_height), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.putText(white_strip, label, (10, (annotation_height + text_height) // 2),
                    font, 0.7, color, 2)
        img = np.vstack([img, white_strip])

        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        resp = make_response(buf.tobytes())
        resp.headers['Content-Type'] = 'image/jpeg'
        resp.headers['Cache-Control'] = 'private, max-age=60'
        return resp
    except Exception as e:
        logging.error(f"Error serving annotated image {line_uid}: {e}")
        return Response('Error', status=500)


@app.route('/api/queue')
def api_queue():
    """Get current line queue with filters applied"""
    try:
        line_queue = app_state['line_queue']
        current_idx = app_state['current_idx']

        # Get status for each line
        queue_data = []
        current_subdir = app_state.get('current_subdir', '')

        for line_uid in line_queue:
            if is_image_queue_uid(line_uid):
                image_id = extract_image_id(line_uid)
                queue_data.append({
                    'line_uid': line_uid,
                    'image_id': image_id,
                    'ocr_text': '',
                    'char_count': 0,
                    'status': 'no_lines',
                    'needs_review': False,
                    'reasons': []
                })
                continue

            result = find_line_in_annotations(line_uid, app_state['annotations'])
            if result:
                img_ann, line_ann = result
                line_status = app_state['event_store'].get_line_status(line_uid)

                # Derive correct display status (fixes "unknown" for auto-accepted lines)
                display_status = derive_display_status(line_status, line_ann)

                queue_data.append({
                    'line_uid': line_uid,
                    'image_id': img_ann.image_id,
                    'ocr_text': line_ann.ocr_text,
                    'char_count': line_ann.get_char_count(),
                    'status': display_status,
                    'needs_review': line_ann.needs_review,
                    'reasons': line_ann.reasons
                })

        return jsonify({
            'success': True,
            'queue': queue_data,
            'current_idx': current_idx,
            'total': len(line_queue),
            'current_subdir': app_state.get('current_subdir', 'candidate'),
            'available_subdirs': app_state.get('available_subdirs', []),
            'current_class': app_state.get('current_class'),
            'available_classes': app_state.get('available_classes', [])
        })

    except Exception as e:
        logging.error(f"Error getting queue: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/line/<path:line_uid>')
def api_get_line(line_uid: str):
    """Get line data including image"""
    try:
        enhanced = request.args.get('enhanced', '0') == '1'
        if is_image_queue_uid(line_uid):
            image_id = extract_image_id(line_uid)
            img_ann = find_image_in_annotations(image_id, app_state['annotations'])
            if not img_ann:
                return jsonify({'success': False, 'error': 'Image not found'})

            img_url = f'/api/img/{img_ann.image_id}'
            return jsonify({
                'success': True,
                'line': {
                    'line_uid': None,
                    'image_id': img_ann.image_id,
                    'ocr_text': '',
                    'det_text': '',
                    'char_count': 0,
                    'bbox': None,
                    'status': 'no_lines',
                    'needs_review': False,
                    'reasons': [],
                    'chars': [],
                    'image_data': img_url,
                    'image_data_raw': img_url,
                    'image_width': img_ann.image_width,
                    'image_height': img_ann.image_height
                },
                'review_history': {
                    'review_count': 0,
                    'edit_count': 0,
                    'last_event': None
                }
            })

        result = find_line_in_annotations(line_uid, app_state['annotations'])
        if not result:
            return jsonify({'success': False, 'error': 'Line not found'})

        img_ann, line_ann = result

        # Get line status from events
        line_status = app_state['event_store'].get_line_status(line_uid)

        # Derive correct display status (fixes "unknown" for auto-accepted lines)
        display_status = derive_display_status(line_status, line_ann)

        # Image URLs instead of embedded base64
        _ts = int(datetime.now().timestamp())
        enhanced_param = '&enhanced=1' if enhanced else ''
        annotated_url = f'/api/img_annotated/{urlquote(line_uid, safe="")}?t={_ts}'
        raw_url = f'/api/img/{img_ann.image_id}?t={_ts}{enhanced_param}'

        # Character data with color information
        chars = []
        for char in line_ann.chars:
            # Get BGR color from OpenCV
            bgr_color = generate_char_color(char.idx)
            # Convert BGR to RGB for HTML
            rgb_color = f'rgb({bgr_color[2]}, {bgr_color[1]}, {bgr_color[0]})'

            chars.append({
                'idx': char.idx,
                'label': char.label,
                'bbox': {'x1': char.bbox.x1, 'y1': char.bbox.y1,
                        'x2': char.bbox.x2, 'y2': char.bbox.y2},
                'conf': char.conf,
                'color': rgb_color  # RGB color for HTML display
            })

        return jsonify({
            'success': True,
            'line': {
                'line_uid': line_uid,
                'image_id': img_ann.image_id,
                'ocr_text': line_ann.ocr_text,
                'det_text': line_ann.det_text,
                'char_count': line_ann.get_char_count(),
                'bbox': {'x1': line_ann.bbox.x1, 'y1': line_ann.bbox.y1,
                        'x2': line_ann.bbox.x2, 'y2': line_ann.bbox.y2},
                'status': display_status,  # Use corrected status
                'needs_review': line_ann.needs_review,
                'reasons': line_ann.reasons,
                'chars': chars,
                'image_data': annotated_url,
                'image_data_raw': raw_url,
                'image_width': img_ann.image_width,
                'image_height': img_ann.image_height
            },
            'review_history': {
                'review_count': line_status['review_count'],
                'edit_count': line_status['edit_count'],
                'last_event': line_status['last_event_type']
            }
        })

    except Exception as e:
        logging.error(f"Error getting line {line_uid}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/line/accept', methods=['POST'])
def api_accept_line():
    """Accept current line (A key)"""
    try:
        data = request.get_json()
        line_uid = data.get('line_uid')
        if is_image_queue_uid(line_uid):
            return jsonify({'success': False, 'error': 'No line to accept'})

        result = find_line_in_annotations(line_uid, app_state['annotations'])
        if not result:
            return jsonify({'success': False, 'error': 'Line not found'})

        img_ann, line_ann = result

        _accept_lines_in_instance(img_ann, [line_ann])

        # Log REVIEWED event
        app_state['event_store'].log_event(
            img_ann.image_id,
            line_uid,
            EventType.REVIEWED,
            {
                'action': 'accept',
                'ocr_text': line_ann.ocr_text,
                'char_count': line_ann.get_char_count()
            }
        )

        # Recalculate metrics for all subdirectories
        invalidate_metrics()

        return jsonify({'success': True, 'action': 'accepted'})

    except Exception as e:
        logging.error(f"Error accepting line: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/line/skip', methods=['POST'])
def api_skip_line():
    """Skip current line (X key) - deprecated"""
    try:
        return jsonify({'success': False, 'error': 'Skip is disabled. Use delete if needed.'})

    except Exception as e:
        logging.error(f"Error skipping line: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/line/edit', methods=['POST'])
def api_edit_line():
    """Save line edits (E key → save)"""
    try:
        data = request.get_json()
        line_uid = data.get('line_uid')
        if is_image_queue_uid(line_uid):
            return jsonify({'success': False, 'error': 'No line to edit'})
        edits = data.get('edits', {})

        # Find line in annotations
        result = find_line_in_annotations(line_uid, app_state['annotations'])
        if not result:
            return jsonify({'success': False, 'error': 'Line not found'})

        img_ann, line_ann = result

        # Validate edits before applying
        # 1. Validate line bbox if provided
        if 'line_bbox' in edits:
            is_valid, error = validate_bbox(
                edits['line_bbox'],
                img_ann.image_width,
                img_ann.image_height
            )
            if not is_valid:
                return jsonify({'success': False, 'error': f'Invalid line bbox: {error}'})

        # 2. Validate OCR text if provided
        if 'ocr_text' in edits:
            is_valid, error = validate_ocr_text(edits['ocr_text'])
            if not is_valid:
                return jsonify({'success': False, 'error': f'Invalid OCR text: {error}'})

        # 3. Validate character data if provided
        if 'chars' in edits:
            for i, char_data in enumerate(edits['chars']):
                # Validate char bbox
                if 'bbox' not in char_data:
                    return jsonify({'success': False, 'error': f'Character {i} missing bbox'})

                is_valid, error = validate_bbox(
                    char_data['bbox'],
                    img_ann.image_width,
                    img_ann.image_height
                )
                if not is_valid:
                    return jsonify({'success': False, 'error': f'Character {i} bbox invalid: {error}'})

                # Validate char label
                if 'label' not in char_data:
                    return jsonify({'success': False, 'error': f'Character {i} missing label'})

                is_valid, error = validate_char_label(char_data['label'])
                if not is_valid:
                    return jsonify({'success': False, 'error': f'Character {i} label invalid: {error}'})

                # Validate idx is present and is integer
                if 'idx' not in char_data or not isinstance(char_data['idx'], int):
                    return jsonify({'success': False, 'error': f'Character {i} missing or invalid idx'})

        # All validations passed, proceed with edits

        # Load current JSON
        json_path = Path(img_ann.json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Find existing line-bbox shape if present
        line_shape = None
        for shape in json_data['shapes']:
            desc = shape.get('description', '')
            if f'line_uid={line_uid}' in desc and shape.get('label') == 'line-bbox':
                line_shape = shape
                break

        # Update or create line bbox if changed
        if 'line_bbox' in edits:
            new_bbox = edits['line_bbox']
            if line_shape:
                line_shape['points'] = [[new_bbox['x1'], new_bbox['y1']],
                                       [new_bbox['x2'], new_bbox['y2']]]
            else:
                max_group_id = max([s.get('group_id') or 0 for s in json_data['shapes']], default=0)
                line_shape = {
                    'label': 'line-bbox',
                    'points': [[new_bbox['x1'], new_bbox['y1']],
                              [new_bbox['x2'], new_bbox['y2']]],
                    'group_id': max_group_id + 1,
                    'description': f"line_uid={line_uid};src=manual;conf=1.0",
                    'shape_type': 'rectangle',
                    'flags': {}
                }
                json_data['shapes'].append(line_shape)

        # Update OCR text if changed (insert ocr= if missing)
        if 'ocr_text' in edits:
            new_ocr_text = edits['ocr_text']
            if not line_shape:
                max_group_id = max([s.get('group_id') or 0 for s in json_data['shapes']], default=0)
                bbox = edits.get('line_bbox', {
                    'x1': line_ann.bbox.x1,
                    'y1': line_ann.bbox.y1,
                    'x2': line_ann.bbox.x2,
                    'y2': line_ann.bbox.y2
                })
                line_shape = {
                    'label': 'line-bbox',
                    'points': [[bbox['x1'], bbox['y1']], [bbox['x2'], bbox['y2']]],
                    'group_id': max_group_id + 1,
                    'description': f"line_uid={line_uid};src=manual;conf=1.0",
                    'shape_type': 'rectangle',
                    'flags': {}
                }
                json_data['shapes'].append(line_shape)

            desc = line_shape.get('description', '')
            desc_parts = [p for p in desc.split(';') if p]
            has_ocr = False
            updated_parts = []
            for part in desc_parts:
                if part.startswith('ocr='):
                    updated_parts.append(f'ocr={new_ocr_text}')
                    has_ocr = True
                else:
                    updated_parts.append(part)
            if not has_ocr:
                updated_parts.append(f'ocr={new_ocr_text}')
            line_shape['description'] = ';'.join(updated_parts)

        # Clear needs_review flag on any edit
        if line_shape:
            line_shape['description'] = update_line_description_flags(
                line_shape.get('description', ''),
                needs_review=False
            )

        # Update characters (delete/modify/add)
        if 'chars' in edits:
            new_chars = edits['chars']

            # Auto-sync: if ocr_text length matches char count, update labels
            ocr_text = edits.get('ocr_text', '')
            if ocr_text and len(new_chars) == len(ocr_text):
                sorted_chars = sorted(new_chars, key=lambda c: c.get('idx', 0))
                for i, ch in enumerate(sorted_chars):
                    ch['label'] = ocr_text[i]

            # Remove old chars for this line
            json_data['shapes'] = [s for s in json_data['shapes']
                                  if not (f'line_uid={line_uid}' in s.get('description', '') and
                                         s.get('label') != 'line-bbox')]

            # Add new chars (reassign group_ids)
            max_group_id = max([s.get('group_id') or 0 for s in json_data['shapes']], default=0)
            group_id_counter = max_group_id + 1

            for char_data in new_chars:
                char_shape = {
                    'label': char_data['label'],
                    'points': [[char_data['bbox']['x1'], char_data['bbox']['y1']],
                              [char_data['bbox']['x2'], char_data['bbox']['y2']]],
                    'group_id': group_id_counter,
                    'description': f"line_uid={line_uid};idx={char_data['idx']};src=manual;conf=1.0",
                    'shape_type': 'rectangle',
                    'flags': {}
                }
                json_data['shapes'].append(char_shape)
                group_id_counter += 1

        # Save updated JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # Log EDITED event
        app_state['event_store'].log_event(
            img_ann.image_id,
            line_uid,
            EventType.EDITED,
            {
                'action': 'edit',
                'ocr_text': edits.get('ocr_text', line_ann.ocr_text),
                'char_count': len(edits.get('chars', [])) if 'chars' in edits else line_ann.get_char_count()
            }
        )

        # Reload annotations and rebuild groups to reflect changes
        reload_image_annotation(json_path, img_ann)
        load_manifest_and_groups()

        return jsonify({'success': True, 'action': 'edited'})

    except Exception as e:
        logging.error(f"Error editing line: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/line/delete', methods=['POST'])
def api_delete_line():
    """Delete entire line annotation (removes line-bbox and all chars)"""
    try:
        data = request.get_json()
        line_uid = data.get('line_uid')
        if is_image_queue_uid(line_uid):
            return jsonify({'success': False, 'error': 'No line to delete'})

        # Find line in annotations
        result = find_line_in_annotations(line_uid, app_state['annotations'])
        if not result:
            return jsonify({'success': False, 'error': 'Line not found'})

        img_ann, line_ann = result

        # Load current JSON
        json_path = Path(img_ann.json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Count shapes before deletion
        shapes_before = len(json_data['shapes'])

        # Remove all shapes (line-bbox and chars) for this line_uid
        json_data['shapes'] = [s for s in json_data['shapes']
                              if f'line_uid={line_uid}' not in s.get('description', '')]

        shapes_after = len(json_data['shapes'])
        shapes_deleted = shapes_before - shapes_after

        logging.info(f"Deleted {shapes_deleted} shapes for line {line_uid}")

        # Save updated JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # Log DELETED event
        app_state['event_store'].log_event(
            img_ann.image_id,
            line_uid,
            EventType.DELETED,
            {
                'action': 'delete',
                'shapes_deleted': shapes_deleted,
                'ocr_text': line_ann.ocr_text
            }
        )

        # Reload annotations and rebuild groups to reflect changes
        reload_image_annotation(json_path, img_ann)
        load_manifest_and_groups()

        # Remove deleted line from queue; add image-level entry if no lines left
        line_queue = app_state['line_queue']
        if line_uid in line_queue:
            line_queue.remove(line_uid)
            # If this image has no more lines, add image-level queue entry
            remaining = [uid for uid in line_queue if uid.startswith(img_ann.image_id + '#')]
            if not remaining and not any(uid == f"{IMAGE_QUEUE_PREFIX}{img_ann.image_id}" for uid in line_queue):
                idx = min(app_state['current_idx'], len(line_queue))
                line_queue.insert(idx, f"{IMAGE_QUEUE_PREFIX}{img_ann.image_id}")
            if app_state['current_idx'] >= len(line_queue):
                app_state['current_idx'] = max(len(line_queue) - 1, 0)

        return jsonify({
            'success': True,
            'action': 'deleted',
            'shapes_deleted': shapes_deleted
        })

    except Exception as e:
        logging.error(f"Error deleting line: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/image/delete', methods=['POST'])
def api_delete_image():
    """Delete an image and its JSON annotation from disk."""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        if not image_id:
            return jsonify({'success': False, 'error': 'Missing image_id'})

        img_ann = find_image_in_annotations(image_id, app_state['annotations'])
        if not img_ann:
            return jsonify({'success': False, 'error': f'Image {image_id} not found'})

        # Delete JSON file
        json_path = Path(img_ann.json_path)
        if json_path.exists():
            json_path.unlink()

        # Delete image file
        image_path = find_image_path(img_ann)
        if image_path and image_path.exists():
            image_path.unlink()

        # Log deletion event for each line
        event_store = app_state.get('event_store')
        for line in img_ann.lines:
            event_store.log_event(
                image_id, line.line_uid, EventType.DELETED,
                {'action': 'image_delete'}
            )

        # Remove from in-memory annotations
        annotations = app_state['annotations']
        app_state['annotations'] = [a for a in annotations if a.image_id != image_id]

        # Remove all queue entries for this image
        line_queue = app_state['line_queue']
        app_state['line_queue'] = [
            uid for uid in line_queue
            if not (uid.startswith(f'{image_id}#') or uid == f'{IMAGE_QUEUE_PREFIX}{image_id}')
        ]
        if app_state['current_idx'] >= len(app_state['line_queue']):
            app_state['current_idx'] = max(len(app_state['line_queue']) - 1, 0)

        # Rebuild groups and invalidate metrics
        load_manifest_and_groups()
        invalidate_metrics()
        thumbnail_cache.invalidate()

        return jsonify({
            'success': True,
            'action': 'image_deleted',
            'image_id': image_id
        })

    except Exception as e:
        logging.error(f"Error deleting image: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/line/add', methods=['POST'])
def api_add_line():
    """Add a new line annotation to an image"""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        line_bbox = data.get('line_bbox')
        ocr_text = data.get('ocr_text', '')
        chars = data.get('chars', [])

        if not image_id:
            return jsonify({'success': False, 'error': 'Missing image_id'})
        if not line_bbox:
            return jsonify({'success': False, 'error': 'Missing line_bbox'})

        img_ann = find_image_in_annotations(image_id, app_state['annotations'])
        if not img_ann:
            return jsonify({'success': False, 'error': 'Image not found'})

        # Validate line bbox
        is_valid, error = validate_bbox(line_bbox, img_ann.image_width, img_ann.image_height)
        if not is_valid:
            return jsonify({'success': False, 'error': f'Invalid line bbox: {error}'})

        # Validate OCR text
        is_valid, error = validate_ocr_text(ocr_text)
        if not is_valid:
            return jsonify({'success': False, 'error': f'Invalid OCR text: {error}'})

        # Validate characters if provided
        for i, char_data in enumerate(chars):
            if 'bbox' not in char_data:
                return jsonify({'success': False, 'error': f'Character {i} missing bbox'})
            is_valid, error = validate_bbox(char_data['bbox'], img_ann.image_width, img_ann.image_height)
            if not is_valid:
                return jsonify({'success': False, 'error': f'Character {i} bbox invalid: {error}'})
            if 'label' not in char_data:
                return jsonify({'success': False, 'error': f'Character {i} missing label'})
            is_valid, error = validate_char_label(char_data['label'])
            if not is_valid:
                return jsonify({'success': False, 'error': f'Character {i} label invalid: {error}'})
            if 'idx' not in char_data or not isinstance(char_data['idx'], int):
                return jsonify({'success': False, 'error': f'Character {i} missing or invalid idx'})

        # Load current JSON
        json_path = Path(img_ann.json_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Generate new line_uid and group_ids
        line_uid = generate_new_line_uid(img_ann)
        max_group_id = max([s.get('group_id') or 0 for s in json_data.get('shapes', [])], default=0)
        group_id_counter = max_group_id + 1

        # Add line bbox
        line_shape = {
            'label': 'line-bbox',
            'points': [[line_bbox['x1'], line_bbox['y1']], [line_bbox['x2'], line_bbox['y2']]],
            'group_id': group_id_counter,
            'description': f"line_uid={line_uid};src=manual;conf=1.0;ocr={ocr_text};needs_review=0",
            'shape_type': 'rectangle',
            'flags': {}
        }
        json_data.setdefault('shapes', []).append(line_shape)
        group_id_counter += 1

        # Add chars
        for char_data in chars:
            char_shape = {
                'label': char_data['label'],
                'points': [[char_data['bbox']['x1'], char_data['bbox']['y1']],
                          [char_data['bbox']['x2'], char_data['bbox']['y2']]],
                'group_id': group_id_counter,
                'description': f"line_uid={line_uid};idx={char_data['idx']};src=manual;conf=1.0",
                'shape_type': 'rectangle',
                'flags': {}
            }
            json_data['shapes'].append(char_shape)
            group_id_counter += 1

        # Save JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # Log EDITED event for new line
        app_state['event_store'].log_event(
            img_ann.image_id,
            line_uid,
            EventType.EDITED,
            {
                'action': 'add_line',
                'ocr_text': ocr_text,
                'char_count': len(chars)
            }
        )

        # Reload annotations and rebuild groups
        reload_image_annotation(json_path, img_ann)
        load_manifest_and_groups()

        # Rebuild queue to include the new line
        prev_idx = app_state.get('current_idx', 0)
        line_queue = []
        for ann in app_state['annotations']:
            for line in ann.lines:
                line_queue.append(line.line_uid)
        app_state['line_queue'] = line_queue
        app_state['current_idx'] = min(prev_idx, max(len(line_queue) - 1, 0))

        return jsonify({'success': True, 'line_uid': line_uid})

    except Exception as e:
        logging.error(f"Error adding line: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


def _rotate_single_image(image_id: str, direction: str) -> dict:
    """Rotate a single image and its annotations. Returns {'success': True} or {'success': False, 'error': ...}."""
    img_ann = find_image_in_annotations(image_id, app_state['annotations'])
    if not img_ann:
        return {'success': False, 'error': f'Image {image_id} not found'}

    # Block rotation if any line has REVIEWED or EDITED events
    for line in img_ann.lines:
        line_status = app_state['event_store'].get_line_status(line.line_uid)
        if line_status.get('status') in ['reviewed', 'edited']:
            return {'success': False, 'error': f'{image_id}: has reviewed/edited lines'}

    json_path = Path(img_ann.json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    width = json_data.get('imageWidth', img_ann.image_width)
    height = json_data.get('imageHeight', img_ann.image_height)

    for shape in json_data.get('shapes', []):
        points = shape.get('points', [])
        if len(points) != 2:
            continue
        bbox = BBox.from_points(points)
        rotated = rotate_bbox(bbox, width, height, direction)
        shape['points'] = [[rotated.x1, rotated.y1], [rotated.x2, rotated.y2]]

    if direction in ['cw', 'ccw']:
        json_data['imageWidth'], json_data['imageHeight'] = height, width

    # Rotate the raw image file on disk
    image_path = find_image_path(img_ann)
    if image_path and image_path.exists():
        img = cv2.imread(str(image_path))
        if img is not None:
            if direction == 'cw':
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif direction == 'ccw':
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif direction == '180':
                img = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(str(image_path), img)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    reload_image_annotation(json_path, img_ann)
    return {'success': True}


@app.route('/api/image/rotate', methods=['POST'])
def api_rotate_image():
    """Rotate an image and all its annotations by 90-degree steps."""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        direction = data.get('direction', 'cw')

        if not image_id:
            return jsonify({'success': False, 'error': 'Missing image_id'})

        result = _rotate_single_image(image_id, direction)

        if result['success']:
            thumbnail_cache.invalidate()

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error rotating image: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/group/rotate', methods=['POST'])
def api_group_rotate():
    """Rotate all images in a component group by 90-degree steps."""
    try:
        data = request.get_json()
        group_id = data.get('group_id')
        direction = data.get('direction', 'cw')

        if not group_id:
            return jsonify({'success': False, 'error': 'Missing group_id'})

        target_group = find_group_by_id(group_id)
        if not target_group:
            return jsonify({'success': False, 'error': f'Group {group_id} not found'})

        rotated_count = 0
        skipped = []
        for inst in target_group.instances:
            result = _rotate_single_image(inst.image_id, direction)
            if result['success']:
                rotated_count += 1
            else:
                skipped.append(result.get('error', inst.image_id))

        # Clear thumbnail cache after batch rotation
        thumbnail_cache.invalidate()

        return jsonify({
            'success': True,
            'rotated': rotated_count,
            'total': len(target_group.instances),
            'skipped': skipped
        })

    except Exception as e:
        logging.error(f"Error rotating image: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/navigate', methods=['POST'])
def api_navigate():
    """Navigate to next/previous line"""
    try:
        data = request.get_json()
        direction = data.get('direction', 'next')

        current_idx = app_state['current_idx']
        queue_len = len(app_state['line_queue'])

        if queue_len == 0:
            app_state['current_idx'] = 0
            return jsonify({
                'success': True,
                'current_idx': 0,
                'total': 0,
                'line_uid': None
            })

        if direction == 'next':
            new_idx = min(current_idx + 1, queue_len - 1)
        elif direction == 'prev':
            new_idx = max(current_idx - 1, 0)
        elif direction == 'first':
            new_idx = 0
        elif direction == 'last':
            new_idx = queue_len - 1
        else:
            new_idx = current_idx

        app_state['current_idx'] = new_idx

        if queue_len > 0:
            current_line_uid = app_state['line_queue'][new_idx]
        else:
            current_line_uid = None

        return jsonify({
            'success': True,
            'current_idx': new_idx,
            'total': queue_len,
            'line_uid': current_line_uid
        })

    except Exception as e:
        logging.error(f"Error navigating: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# API Endpoints - Gallery & Group Review
# ============================================================================

@app.route('/api/positions')
def api_positions():
    """List all available region_group_id positions with summary info."""
    try:
        groups = app_state.get('component_groups', [])
        event_store = app_state.get('event_store')
        positions = []
        for group in groups:
            # Determine review status
            has_unreviewed = False
            has_reviewed = False
            for inst in group.instances:
                for line in inst.lines:
                    if event_store:
                        line_status = event_store.get_line_status(line.line_uid)
                        display_status = derive_display_status(line_status, line)
                    else:
                        display_status = 'uncertain' if line.needs_review else 'accepted'
                    if display_status in ('uncertain', 'unknown'):
                        has_unreviewed = True
                    elif display_status in ('reviewed', 'edited', 'accepted'):
                        has_reviewed = True
            if has_unreviewed and has_reviewed:
                status = 'partial'
            elif has_unreviewed:
                status = 'pending'
            else:
                status = 'reviewed'
            positions.append({
                'region_group_id': group.region_group_id,
                'component_class': group.component_class,
                'sibling_count': len(group.instances),
                'ocr_text': ', '.join(group.majority_text.get(i, '') for i in sorted(group.majority_text.keys())) or '-',
                'status': status,
            })
        positions.sort(key=lambda p: p['region_group_id'] or '')
        return jsonify({
            'success': True,
            'positions': positions,
            'group_mode': app_state.get('group_mode', 'none'),
        })
    except Exception as e:
        logging.error(f"Error listing positions: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/gallery')
def api_gallery():
    """Get paginated component groups for gallery view."""
    try:
        ensure_metrics_fresh()
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        sort = request.args.get('sort', 'conf_asc')  # conf_asc|conf_desc|group_id
        filter_mode = request.args.get('filter', 'all')  # all|needs_review|reviewed
        position = request.args.get('position', '')  # filter by region_group_id

        groups = app_state.get('component_groups', [])
        event_store = app_state.get('event_store')
        _ts = int(datetime.now().timestamp())

        if not groups:
            # Fall back to listing individual images (one card per image)
            annotations = app_state.get('annotations', [])
            cards = []
            for img_ann in annotations:
                ocr_texts = [line.ocr_text or line.det_text or '' for line in img_ann.lines]
                has_unreviewed = False
                has_reviewed = False
                for line in img_ann.lines:
                    if event_store:
                        line_status = event_store.get_line_status(line.line_uid)
                        display_status = derive_display_status(line_status, line)
                    else:
                        display_status = 'uncertain' if line.needs_review else 'accepted'
                    if display_status in ('uncertain', 'unknown'):
                        has_unreviewed = True
                    elif display_status in ('reviewed', 'edited', 'accepted'):
                        has_reviewed = True

                if has_unreviewed and has_reviewed:
                    status = 'partial'
                elif has_unreviewed:
                    status = 'pending'
                else:
                    status = 'reviewed'

                if filter_mode == 'needs_review' and status == 'reviewed':
                    continue
                if filter_mode == 'reviewed' and status != 'reviewed':
                    continue

                thumb_url = f'/api/thumb/{img_ann.image_id}?annotated=1&_t={_ts}'

                cards.append({
                    'group_id': img_ann.image_id,
                    'ocr_text': ', '.join(ocr_texts) if ocr_texts else '-',
                    'thumbnail': thumb_url,
                    'confidence': 1.0,
                    'sibling_count': 1,
                    'has_outliers': False,
                    'all_agree': True,
                    'status': status,
                })

            # Sort
            status_order = {'pending': 0, 'partial': 1, 'reviewed': 2}
            if sort == 'conf_asc':
                cards.sort(key=lambda c: (status_order.get(c['status'], 1), c['confidence']))
            elif sort == 'conf_desc':
                cards.sort(key=lambda c: (status_order.get(c['status'], 1), -c['confidence']))
            elif sort == 'group_id':
                cards.sort(key=lambda c: c.get('group_id') or '')
            else:
                cards.sort(key=lambda c: (status_order.get(c['status'], 1), c['confidence']))

            total = len(cards)
            start = (page - 1) * per_page
            end = start + per_page
            page_cards = cards[start:end]

            resp = jsonify({
                'success': True,
                'groups': page_cards,
                'total_groups': total,
                'total_pages': (total + per_page - 1) // per_page if per_page > 0 else 0,
                'group_mode': app_state.get('group_mode', 'none'),
            })
            resp.headers['Cache-Control'] = 'private, max-age=300'
            return resp

        # Filter by position (region_group_id) if specified
        if position:
            groups = [g for g in groups if g.region_group_id == position]

        # Build gallery cards from component groups
        cards = []
        for group in groups:
            # Determine group review status
            has_unreviewed = False
            has_reviewed = False
            for inst in group.instances:
                for line in inst.lines:
                    if event_store:
                        line_status = event_store.get_line_status(line.line_uid)
                        display_status = derive_display_status(line_status, line)
                    else:
                        display_status = 'uncertain' if line.needs_review else 'accepted'
                    if display_status in ('uncertain', 'unknown'):
                        has_unreviewed = True
                    elif display_status in ('reviewed', 'edited', 'accepted'):
                        has_reviewed = True

            if has_unreviewed and has_reviewed:
                status = 'partial'
            elif has_unreviewed:
                status = 'pending'
            else:
                status = 'reviewed'

            if filter_mode == 'needs_review' and status == 'reviewed':
                continue
            if filter_mode == 'reviewed' and status != 'reviewed':
                continue
            if filter_mode == 'has_outliers' and group.agreement_count >= group.total_count:
                continue
            if filter_mode == 'has_agnostic_flags':
                has_agnostic = False
                for inst in group.instances:
                    for line in inst.lines:
                        for reason in (line.reasons or []):
                            if reason.startswith(('MISSED_BY_CLASSIFIER', 'FALSE_POSITIVE_SUSPECT', 'AGNOSTIC_COUNT_MISMATCH')):
                                has_agnostic = True
                                break
                        if has_agnostic:
                            break
                    if has_agnostic:
                        break
                if not has_agnostic:
                    continue

            representative = group.instances[0] if group.instances else None
            ocr_texts = []
            for li in sorted(group.majority_text.keys()):
                ocr_texts.append(group.majority_text[li])

            has_outliers = group.agreement_count < group.total_count
            all_agree = group.agreement_count >= group.total_count

            thumb_url = ''
            if representative:
                thumb_url = f'/api/thumb/{representative.image_id}?annotated=1&_t={_ts}'

            # Sibling thumbnail URLs (up to 6 for preview strip)
            sibling_thumbs = [f'/api/thumb/{inst.image_id}?w=80&annotated=1&_t={_ts}' for inst in group.instances[:6]]

            cards.append({
                'group_id': group.region_group_id,
                'component_class': group.component_class,
                'ocr_text': ', '.join(ocr_texts) if ocr_texts else '-',
                'thumbnail': thumb_url,
                'sibling_thumbs': sibling_thumbs,
                'confidence': round(group.confidence, 4),
                'sibling_count': len(group.instances),
                'agreement_ratio': f'{group.agreement_count}/{group.total_count}',
                'has_outliers': has_outliers,
                'all_agree': all_agree,
                'status': status,
            })

        # Sort
        status_order = {'pending': 0, 'partial': 1, 'reviewed': 2}
        if sort == 'conf_asc':
            cards.sort(key=lambda c: (status_order.get(c['status'], 1), c['confidence']))
        elif sort == 'conf_desc':
            cards.sort(key=lambda c: (status_order.get(c['status'], 1), -c['confidence']))
        elif sort == 'component_class':
            cards.sort(key=lambda c: (c.get('component_class', ''), status_order.get(c['status'], 1), c['confidence']))
        elif sort == 'group_id':
            cards.sort(key=lambda c: c.get('group_id') or '')
        else:
            cards.sort(key=lambda c: (status_order.get(c['status'], 1), c['confidence']))

        total = len(cards)
        start = (page - 1) * per_page
        end = start + per_page
        page_cards = cards[start:end]

        resp = jsonify({
            'success': True,
            'groups': page_cards,
            'total_groups': total,
            'total_pages': (total + per_page - 1) // per_page if per_page > 0 else 0,
            'group_mode': app_state.get('group_mode', 'none'),
        })
        resp.headers['Cache-Control'] = 'private, max-age=300'
        return resp

    except Exception as e:
        logging.error(f"Error getting gallery: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/group/<region_group_id>')
def api_get_group(region_group_id):
    """Get all sibling instances for a component group (subboard comparison)."""
    try:
        event_store = app_state.get('event_store')

        target_group = find_group_by_id(region_group_id)
        if not target_group:
            return jsonify({'success': False, 'error': f'Group {region_group_id} not found'})

        _ts = int(datetime.now().timestamp())

        # Build instance data
        instances_data = []
        for inst in target_group.instances:
            thumb_url = f'/api/thumb/{inst.image_id}?annotated=1&_t={_ts}'

            # Build line data and check outlier status
            lines_data = []
            is_outlier = False
            for li, line in enumerate(inst.lines):
                text = line.det_text or line.ocr_text or ''
                majority = target_group.majority_text.get(li, '')
                if text != majority:
                    is_outlier = True

                # Get line review status
                if event_store:
                    line_status = event_store.get_line_status(line.line_uid)
                    display_status = derive_display_status(line_status, line)
                else:
                    display_status = 'uncertain' if line.needs_review else 'accepted'

                lines_data.append({
                    'line_uid': line.line_uid,
                    'ocr_text': line.ocr_text or '',
                    'det_text': line.det_text or '',
                    'confidence': round(line.conf, 4) if line.conf else 0.0,
                    'matches_majority': text == majority,
                    'status': display_status,
                })

            instances_data.append({
                'image_id': inst.image_id,
                'thumbnail': thumb_url,
                'image_data': f'/api/img/{inst.image_id}?_t={_ts}',
                'lines': lines_data,
                'is_outlier': is_outlier,
            })

        # Build majority consensus info
        majority_info = {}
        for li in sorted(target_group.majority_text.keys()):
            majority_info[str(li)] = target_group.majority_text[li]

        resp = jsonify({
            'success': True,
            'region_group_id': target_group.region_group_id,
            'component_class': target_group.component_class,
            'confidence': round(target_group.confidence, 4),
            'agreement_ratio': f'{target_group.agreement_count}/{target_group.total_count}',
            'majority_text': majority_info,
            'instances': instances_data,
        })
        resp.headers['Cache-Control'] = 'private, max-age=300'
        return resp

    except Exception as e:
        logging.error(f"Error getting group {region_group_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/group/accept', methods=['POST'])
def api_accept_group():
    """Accept entire component group (all siblings verified)."""
    try:
        data = request.get_json()
        region_group_id = data.get('region_group_id')
        if not region_group_id:
            return jsonify({'success': False, 'error': 'region_group_id is required'})

        event_store = app_state.get('event_store')

        target_group = find_group_by_id(region_group_id)
        if not target_group:
            return jsonify({'success': False, 'error': f'Group {region_group_id} not found'})

        accepted_count = 0
        for inst in target_group.instances:
            _accept_lines_in_instance(inst, inst.lines)
            for line in inst.lines:
                event_store.log_event(
                    inst.image_id,
                    line.line_uid,
                    EventType.REVIEWED,
                    {
                        'action': 'group_accept',
                        'region_group_id': region_group_id,
                        'ocr_text': line.ocr_text,
                        'char_count': line.get_char_count(),
                    }
                )
                accepted_count += 1

        if target_group.instances:
            rep = target_group.instances[0]
            event_store.log_event(
                rep.image_id,
                f'{region_group_id}#GROUP',
                EventType.GROUP_ACCEPTED,
                {
                    'region_group_id': region_group_id,
                    'sibling_count': len(target_group.instances),
                    'accepted_lines': accepted_count,
                }
            )

        invalidate_metrics()

        return jsonify({
            'success': True,
            'action': 'group_accepted',
            'region_group_id': region_group_id,
            'accepted_count': accepted_count,
        })

    except Exception as e:
        logging.error(f"Error accepting group: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/group/accept_majority', methods=['POST'])
def api_accept_majority():
    """Accept majority, reject outliers in a component group."""
    try:
        data = request.get_json()
        region_group_id = data.get('region_group_id')
        outlier_image_ids = data.get('outlier_image_ids', [])

        if not region_group_id:
            return jsonify({'success': False, 'error': 'region_group_id is required'})

        event_store = app_state.get('event_store')

        target_group = find_group_by_id(region_group_id)
        if not target_group:
            return jsonify({'success': False, 'error': f'Group {region_group_id} not found'})

        outlier_set = set(outlier_image_ids)
        accepted_count = 0
        rejected_count = 0

        for inst in target_group.instances:
            if inst.image_id in outlier_set:
                for line in inst.lines:
                    event_store.log_event(
                        inst.image_id,
                        line.line_uid,
                        EventType.INSTANCE_REJECTED,
                        {
                            'action': 'majority_reject_outlier',
                            'region_group_id': region_group_id,
                            'ocr_text': line.ocr_text,
                        }
                    )
                    rejected_count += 1
            else:
                _accept_lines_in_instance(inst, inst.lines)
                for line in inst.lines:
                    event_store.log_event(
                        inst.image_id,
                        line.line_uid,
                        EventType.REVIEWED,
                        {
                            'action': 'majority_accept',
                            'region_group_id': region_group_id,
                            'ocr_text': line.ocr_text,
                            'char_count': line.get_char_count(),
                        }
                    )
                    accepted_count += 1

        if target_group.instances:
            rep = target_group.instances[0]
            event_store.log_event(
                rep.image_id,
                f'{region_group_id}#GROUP',
                EventType.GROUP_MAJORITY_ACCEPTED,
                {
                    'region_group_id': region_group_id,
                    'accepted_lines': accepted_count,
                    'rejected_lines': rejected_count,
                    'outlier_image_ids': outlier_image_ids,
                }
            )

        invalidate_metrics()

        return jsonify({
            'success': True,
            'action': 'majority_accepted',
            'region_group_id': region_group_id,
            'accepted_count': accepted_count,
            'rejected_count': rejected_count,
        })

    except Exception as e:
        logging.error(f"Error accepting majority: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/instance/reject', methods=['POST'])
def api_reject_instance():
    """Reject a specific instance (defective)."""
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        region_group_id = data.get('region_group_id', '')

        if not image_id:
            return jsonify({'success': False, 'error': 'image_id is required'})

        event_store = app_state.get('event_store')
        annotations = app_state.get('annotations', [])

        img_ann = find_image_in_annotations(image_id, annotations)
        if not img_ann:
            return jsonify({'success': False, 'error': f'Image {image_id} not found'})

        rejected_count = 0
        for line in img_ann.lines:
            event_store.log_event(
                image_id,
                line.line_uid,
                EventType.INSTANCE_REJECTED,
                {
                    'action': 'instance_reject',
                    'region_group_id': region_group_id,
                    'ocr_text': line.ocr_text,
                }
            )
            rejected_count += 1

        invalidate_metrics()

        return jsonify({
            'success': True,
            'action': 'instance_rejected',
            'image_id': image_id,
            'rejected_count': rejected_count,
        })

    except Exception as e:
        logging.error(f"Error rejecting instance: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/batch/accept', methods=['POST'])
def api_batch_accept():
    """Accept multiple groups at once from gallery."""
    try:
        data = request.get_json()
        region_group_ids = data.get('region_group_ids', [])

        if not region_group_ids:
            return jsonify({'success': False, 'error': 'region_group_ids is required'})

        groups = app_state.get('component_groups', [])
        event_store = app_state.get('event_store')

        # Index groups by id for fast lookup
        groups_by_id = {g.region_group_id: g for g in groups}

        total_accepted = 0
        accepted_groups = 0

        for rgid in region_group_ids:
            target_group = groups_by_id.get(rgid)
            if not target_group:
                logging.warning(f"Batch accept: group {rgid} not found, skipping")
                continue

            group_line_count = 0
            for inst in target_group.instances:
                _accept_lines_in_instance(inst, inst.lines)
                for line in inst.lines:
                    event_store.log_event(
                        inst.image_id,
                        line.line_uid,
                        EventType.REVIEWED,
                        {
                            'action': 'batch_accept',
                            'region_group_id': rgid,
                            'ocr_text': line.ocr_text,
                            'char_count': line.get_char_count(),
                        }
                    )
                    group_line_count += 1

            if target_group.instances:
                rep = target_group.instances[0]
                event_store.log_event(
                    rep.image_id,
                    f'{rgid}#GROUP',
                    EventType.GROUP_ACCEPTED,
                    {
                        'region_group_id': rgid,
                        'sibling_count': len(target_group.instances),
                        'accepted_lines': group_line_count,
                    }
                )

            total_accepted += group_line_count
            accepted_groups += 1

        invalidate_metrics()

        return jsonify({
            'success': True,
            'action': 'batch_accepted',
            'accepted_groups': accepted_groups,
            'total_accepted_lines': total_accepted,
        })

    except Exception as e:
        logging.error(f"Error batch accepting: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/batch/accept_agreeing', methods=['POST'])
def api_batch_accept_agreeing():
    """Accept all groups where all siblings agree and confidence is above threshold."""
    try:
        conf_threshold = 0.8
        component_groups = app_state.get('component_groups', [])
        event_store = app_state.get('event_store')

        accepted_groups = 0
        accepted_lines = 0
        for group in component_groups:
            if group.agreement_count < group.total_count:
                continue
            if group.confidence < conf_threshold:
                continue

            # Single pass: collect unreviewed lines per instance
            group_line_count = 0
            for inst in group.instances:
                unreviewed = []
                for line in inst.lines:
                    if event_store:
                        line_status = event_store.get_line_status(line.line_uid)
                        status = derive_display_status(line_status, line)
                    else:
                        status = 'uncertain' if line.needs_review else 'accepted'
                    if status not in ('accepted', 'reviewed', 'edited'):
                        unreviewed.append(line)
                if unreviewed:
                    _accept_lines_in_instance(inst, unreviewed)
                    for line in unreviewed:
                        event_store.log_event(
                            inst.image_id,
                            line.line_uid,
                            EventType.REVIEWED,
                            {
                                'action': 'batch_accept_agreeing',
                                'region_group_id': group.region_group_id,
                                'ocr_text': line.ocr_text,
                                'char_count': line.get_char_count(),
                            }
                        )
                        group_line_count += 1

            if group_line_count > 0 and group.instances:
                rep = group.instances[0]
                event_store.log_event(
                    rep.image_id,
                    f'{group.region_group_id}#GROUP',
                    EventType.GROUP_ACCEPTED,
                    {
                        'region_group_id': group.region_group_id,
                        'sibling_count': len(group.instances),
                        'accepted_lines': group_line_count,
                    }
                )
                accepted_groups += 1
                accepted_lines += group_line_count

        if accepted_groups > 0:
            invalidate_metrics()

        return jsonify({
            'success': True,
            'accepted_count': accepted_groups,
            'accepted_lines': accepted_lines,
        })
    except Exception as e:
        logging.error(f"Error in batch accept agreeing: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# API Endpoints - Dashboard Tab
# ============================================================================

@app.route('/api/metrics/kpis')
def api_metrics_kpis():
    """Get KPIs for specified subdirectory"""
    try:
        ensure_metrics_fresh()
        subdir = request.args.get('subdir', 'final')

        all_metrics_calcs = app_state.get('all_metrics_calcs', {})
        if subdir not in all_metrics_calcs:
            return jsonify({'success': False, 'error': f'Subdirectory "{subdir}" not found'})

        calc = all_metrics_calcs[subdir]
        kpis = calc.get_kpis()
        return jsonify({'success': True, 'kpis': kpis, 'subdir': subdir})

    except Exception as e:
        logging.error(f"Error getting KPIs: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/metrics/funnel')
def api_metrics_funnel():
    """Get acceptance funnel for specified subdirectory"""
    try:
        ensure_metrics_fresh()
        subdir = request.args.get('subdir', 'final')

        all_metrics_calcs = app_state.get('all_metrics_calcs', {})
        if subdir not in all_metrics_calcs:
            return jsonify({'success': False, 'error': f'Subdirectory "{subdir}" not found'})

        calc = all_metrics_calcs[subdir]
        funnel = calc.get_funnel()
        return jsonify({'success': True, 'funnel': funnel, 'subdir': subdir})

    except Exception as e:
        logging.error(f"Error getting funnel: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/metrics/reasons')
def api_metrics_reasons():
    """Get failure reasons breakdown for specified subdirectory"""
    try:
        ensure_metrics_fresh()
        subdir = request.args.get('subdir', 'final')

        all_metrics_calcs = app_state.get('all_metrics_calcs', {})
        if subdir not in all_metrics_calcs:
            return jsonify({'success': False, 'error': f'Subdirectory "{subdir}" not found'})

        calc = all_metrics_calcs[subdir]
        reasons = calc.get_failure_reasons()
        return jsonify({'success': True, 'reasons': reasons, 'subdir': subdir})

    except Exception as e:
        logging.error(f"Error getting reasons: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/metrics/histogram')
def api_metrics_histogram():
    """Get mismatch histogram for specified subdirectory"""
    try:
        ensure_metrics_fresh()
        subdir = request.args.get('subdir', 'final')

        all_metrics_calcs = app_state.get('all_metrics_calcs', {})
        if subdir not in all_metrics_calcs:
            return jsonify({'success': False, 'error': f'Subdirectory "{subdir}" not found'})

        calc = all_metrics_calcs[subdir]
        histogram = calc.get_mismatch_histogram()
        return jsonify({'success': True, 'histogram': histogram, 'subdir': subdir})

    except Exception as e:
        logging.error(f"Error getting histogram: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# API Endpoints - Queue Management
# ============================================================================

@app.route('/api/queue/filter', methods=['POST'])
def api_filter_queue():
    """Apply filters to queue"""
    try:
        data = request.get_json()
        filter_type = data.get('filter_type', 'all')
        filter_value = data.get('filter_value')

        calc = app_state['metrics_calc']

        # Get filtered line UIDs
        if filter_type == 'all':
            # All lines
            line_uids = []
            for img_ann in app_state['annotations']:
                for line in img_ann.lines:
                    line_uids.append(line.line_uid)
        elif filter_type == 'no_lines':
            line_uids = []
            for img_ann in app_state['annotations']:
                if not img_ann.lines:
                    line_uids.append(f"{IMAGE_QUEUE_PREFIX}{img_ann.image_id}")
        else:
            line_uids = calc.get_lines_by_filter(filter_type, filter_value)

        # Update queue
        app_state['line_queue'] = line_uids
        app_state['current_idx'] = 0
        app_state['filter_status'] = filter_type

        return jsonify({
            'success': True,
            'queue_length': len(line_uids),
            'filter_type': filter_type,
            'filter_value': filter_value
        })

    except Exception as e:
        logging.error(f"Error filtering queue: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/queue/select', methods=['POST'])
def api_queue_select():
    """Set current queue index by line_uid"""
    try:
        data = request.get_json()
        line_uid = data.get('line_uid')
        if not line_uid:
            return jsonify({'success': False, 'error': 'Missing line_uid'})

        line_queue = app_state['line_queue']
        if line_uid not in line_queue:
            return jsonify({'success': False, 'error': 'Line not in current queue'})

        app_state['current_idx'] = line_queue.index(line_uid)
        return jsonify({'success': True, 'current_idx': app_state['current_idx']})

    except Exception as e:
        logging.error(f"Error selecting queue item: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/switch_subdir', methods=['POST'])
def api_switch_subdir():
    """Switch between candidate/ and final/ subdirectories"""
    try:
        data = request.get_json()
        subdir = data.get('subdir')

        if not subdir:
            return jsonify({'success': False, 'error': 'Missing subdir parameter'})

        available_subdirs = app_state.get('available_subdirs', [])
        if subdir not in available_subdirs:
            return jsonify({
                'success': False,
                'error': f'Subdirectory "{subdir}" not available. Available: {available_subdirs}'
            })

        # Update current subdirectory
        fused_dir = app_state['fused_dir']
        annotations_dir = fused_dir / subdir

        logging.info(f"Switching to subdirectory: {subdir}/")

        # Get annotations from pre-loaded cache
        all_annotations = app_state['all_annotations']
        all_metrics_calcs = app_state['all_metrics_calcs']

        annotations = all_annotations[subdir]
        metrics_calc = all_metrics_calcs[subdir]

        total_lines = sum(img.get_line_count() for img in annotations)
        logging.info(f"Switched to {subdir}/: {len(annotations)} images, {total_lines} lines")

        # Rebuild queue (include image-level entries for images with no lines)
        line_queue = []
        for img_ann in annotations:
            if img_ann.lines:
                for line in img_ann.lines:
                    line_queue.append(line.line_uid)
            else:
                line_queue.append(f"{IMAGE_QUEUE_PREFIX}{img_ann.image_id}")

        # Update app state
        app_state['current_subdir'] = subdir
        app_state['annotations_dir'] = annotations_dir
        app_state['annotations'] = annotations
        app_state['metrics_calc'] = metrics_calc
        app_state['line_queue'] = line_queue
        app_state['current_idx'] = 0

        # Rebuild component groups for new subdir's annotations
        load_manifest_and_groups()
        invalidate_metrics()

        return jsonify({
            'success': True,
            'subdir': subdir,
            'total_images': len(annotations),
            'total_lines': total_lines,
            'queue_length': len(line_queue)
        })

    except Exception as e:
        logging.error(f"Error switching subdirectory: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/switch_class', methods=['POST'])
def api_switch_class():
    """Switch between dataset classes"""
    try:
        data = request.get_json()
        class_name = data.get('class_name')
        subdir = data.get('subdir')

        if not class_name:
            return jsonify({'success': False, 'error': 'Missing class_name parameter'})

        if class_name not in app_state.get('available_classes', []):
            return jsonify({'success': False, 'error': f'Class \"{class_name}\" not available'})

        set_current_class(class_name, subdir=subdir)

        annotations = app_state['annotations']
        total_lines = sum(img.get_line_count() for img in annotations)

        return jsonify({
            'success': True,
            'class_name': class_name,
            'current_subdir': app_state.get('current_subdir'),
            'total_images': len(annotations),
            'total_lines': total_lines,
            'queue_length': len(app_state['line_queue'])
        })

    except Exception as e:
        logging.error(f"Error switching class: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/class_stats')
def api_class_stats():
    """Return lightweight completion stats for all classes (for board picker)."""
    try:
        results = {}
        for class_name in app_state.get('available_classes', []):
            config = app_state['class_configs'][class_name]
            fused_dir = config['fused_dir']

            # Quick check: does final/ exist and have files?
            final_dir = fused_dir / 'final'
            candidate_dir = fused_dir / 'candidate'
            n_final = len(list(final_dir.glob('*.json'))) if final_dir.exists() else 0
            n_candidate = len(list(candidate_dir.glob('*.json'))) if candidate_dir.exists() else 0
            n_total = n_final + n_candidate

            # If class data is loaded, use event store for more accurate stats
            if class_name in app_state.get('class_data', {}):
                class_data = app_state['class_data'][class_name]
                event_store = class_data['event_store']
                all_anns = class_data['all_annotations']
                total_lines = 0
                reviewed_lines = 0
                for subdir_name, anns in all_anns.items():
                    for img_ann in anns:
                        for line in img_ann.lines:
                            total_lines += 1
                            line_status = event_store.get_line_status(line.line_uid)
                            status = derive_display_status(line_status, line)
                            if status in ('accepted', 'reviewed', 'edited'):
                                reviewed_lines += 1
                results[class_name] = {
                    'n_candidate': n_candidate,
                    'n_final': n_final,
                    'n_total': n_total,
                    'total_lines': total_lines,
                    'reviewed_lines': reviewed_lines,
                    'done': n_candidate == 0 and n_final > 0,
                }
            else:
                # Lightweight: just file counts
                results[class_name] = {
                    'n_candidate': n_candidate,
                    'n_final': n_final,
                    'n_total': n_total,
                    'done': n_candidate == 0 and n_final > 0,
                }

        return jsonify({'success': True, 'stats': results})
    except Exception as e:
        logging.error(f"Error getting class stats: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/review/finish', methods=['POST'])
def api_finish_review():
    """Move fully reviewed JSONs from candidate/ to final/"""
    try:
        data = request.get_json() or {}
        subdir = data.get('subdir', app_state.get('current_subdir', 'candidate'))

        if subdir != 'candidate':
            return jsonify({'success': False, 'error': 'Finish review only supports candidate/ source'})

        fused_dir = app_state['fused_dir']
        candidate_dir = fused_dir / 'candidate'
        final_dir = fused_dir / 'final'
        final_dir.mkdir(parents=True, exist_ok=True)

        annotations = app_state['all_annotations'].get('candidate', [])
        event_store = app_state['event_store']

        eligible = []
        blocked = []

        allowed_statuses = {'accepted', 'edited', 'reviewed'}

        for img_ann in annotations:
            all_ok = True
            for line in img_ann.lines:
                line_status = event_store.get_line_status(line.line_uid)
                display_status = derive_display_status(line_status, line)
                if display_status not in allowed_statuses:
                    all_ok = False
                    break
            if all_ok:
                eligible.append(img_ann)
            else:
                blocked.append(img_ann.image_id)

        moved = []
        errors = []

        for img_ann in eligible:
            src = Path(img_ann.json_path)
            if not src.exists():
                errors.append(f"Missing source JSON: {src}")
                continue
            dest = final_dir / src.name
            try:
                if dest.exists():
                    dest.unlink()
                shutil.move(str(src), str(dest))

                # Also move the image file (.png/.jpg) if it's in the same directory
                image_path = find_image_path(img_ann)
                if image_path and image_path.exists() and image_path.parent == src.parent:
                    img_dest = final_dir / image_path.name
                    if img_dest.exists():
                        img_dest.unlink()
                    shutil.move(str(image_path), str(img_dest))

                moved.append(img_ann.image_id)
            except Exception as e:
                errors.append(f"{src.name}: {e}")

        # Add final/ to available_subdirs if newly created
        if 'final' not in app_state.get('available_subdirs', []):
            app_state['available_subdirs'].append('final')
            # Also update the class_data cache
            current_class = app_state.get('current_class')
            if current_class and current_class in app_state.get('class_data', {}):
                app_state['class_data'][current_class]['available_subdirs'] = app_state['available_subdirs']

        # Reload caches and queues after moving files
        for subdir_name in app_state.get('available_subdirs', []):
            reload_subdir_cache(subdir_name)

        current_subdir = app_state.get('current_subdir', 'candidate')
        app_state['annotations'] = app_state['all_annotations'].get(current_subdir, [])
        app_state['metrics_calc'] = app_state['all_metrics_calcs'].get(current_subdir)

        line_queue = []
        for img_ann in app_state['annotations']:
            if img_ann.lines:
                for line in img_ann.lines:
                    line_queue.append(line.line_uid)
            else:
                line_queue.append(f"{IMAGE_QUEUE_PREFIX}{img_ann.image_id}")
        app_state['line_queue'] = line_queue
        app_state['current_idx'] = 0

        return jsonify({
            'success': True,
            'moved_count': len(moved),
            'eligible_count': len(eligible),
            'blocked_count': len(blocked),
            'moved_images': moved,
            'blocked_images': blocked,
            'errors': errors
        })

    except Exception as e:
        logging.error(f"Error finishing review: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/browse_directories', methods=['POST'])
def api_browse_directories():
    """Browse directories for dataset selection"""
    try:
        import os
        data = request.get_json() or {}
        current_path = data.get('path')

        # Default to home directory if no path provided
        if not current_path:
            current_path = str(Path.home())

        # Resolve path
        path = Path(current_path).expanduser().resolve()

        # Security check: ensure path exists and is a directory
        if not path.exists() or not path.is_dir():
            return jsonify({'success': False, 'error': 'Invalid directory path'})

        # Get parent directory
        parent = str(path.parent) if path.parent != path else None

        # List subdirectories
        directories = []
        try:
            for entry in sorted(path.iterdir()):
                if entry.is_dir():
                    # Check if directory is readable
                    try:
                        # Test read access
                        list(entry.iterdir())
                        has_fused = any(item.name.endswith('_fused') for item in entry.iterdir() if item.is_dir())
                        directories.append({
                            'name': entry.name,
                            'path': str(entry),
                            'has_fused': has_fused
                        })
                    except PermissionError:
                        # Skip directories without read permission
                        continue
        except PermissionError:
            return jsonify({'success': False, 'error': 'Permission denied'})

        return jsonify({
            'success': True,
            'current_path': str(path),
            'parent_path': parent,
            'directories': directories
        })

    except Exception as e:
        logging.error(f"Error browsing directories: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/set_dataset_root', methods=['POST'])
def api_set_dataset_root():
    """Set a new dataset root and reinitialize the app"""
    try:
        data = request.get_json() or {}
        new_dataset_root = data.get('dataset_root')

        if not new_dataset_root:
            return jsonify({'success': False, 'error': 'Missing dataset_root parameter'})

        dataset_root = Path(new_dataset_root)
        if not dataset_root.exists() or not dataset_root.is_dir():
            return jsonify({'success': False, 'error': 'Invalid dataset root path'})

        # Scan for classes
        class_configs = {}
        available_classes = []
        for entry in sorted(dataset_root.iterdir()):
            if not entry.is_dir():
                continue
            if not entry.name.endswith('_fused'):
                continue

            class_name = entry.name.replace('_fused', '')
            fused_dir = entry
            images_dir = dataset_root / class_name
            db_path = fused_dir / "review.db"

            if not (fused_dir / 'candidate').exists() and not (fused_dir / 'final').exists():
                continue

            class_configs[class_name] = {
                'fused_dir': fused_dir,
                'images_dir': images_dir,
                'db_path': db_path
            }
            available_classes.append(class_name)

        if not available_classes:
            return jsonify({
                'success': False,
                'error': f'No <Class>_fused directories with candidate/ or final/ found in {dataset_root}'
            })

        # Update app state
        app_state['dataset_root'] = dataset_root
        app_state['available_classes'] = available_classes
        app_state['class_configs'] = class_configs
        app_state['class_data'] = {}  # Clear cached class data

        # Load first class
        current_class = available_classes[0]
        set_current_class(current_class, subdir='candidate')

        logging.info(f"Dataset root set to: {dataset_root}")
        logging.info(f"Available classes: {available_classes}")

        return jsonify({
            'success': True,
            'dataset_root': str(dataset_root),
            'available_classes': available_classes,
            'current_class': current_class
        })

    except Exception as e:
        logging.error(f"Error setting dataset root: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OCR Review Web Application",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--dataset_root", required=False, default=None,
                       help="Dataset root containing class folders and <Class>_fused/ (optional - can be set via web UI)")
    parser.add_argument("--images_dir", default=None,
                       help="Override images directory for a single class (optional)")
    parser.add_argument("--db_path", default=None,
                       help="Override review.db path for a single class (optional)")
    parser.add_argument("--class_name", default=None,
                       help="Initial class to load (default: first discovered)")
    parser.add_argument("--subdir", default="candidate",
                       choices=["candidate", "final"],
                       help="Initial subdirectory to load (default: candidate)")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5001,
                       help="Port to bind to (default: 5001)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Initialize app state with defaults
    app_state['dataset_root'] = None
    app_state['available_classes'] = []
    app_state['class_configs'] = {}

    # Initialize paths if dataset_root is provided
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        if not dataset_root.exists():
            logging.error(f"Dataset root not found: {dataset_root}")
            return 1

        class_configs = {}
        available_classes = []
        for entry in sorted(dataset_root.iterdir()):
            if not entry.is_dir():
                continue
            if not entry.name.endswith('_fused'):
                continue

            class_name = entry.name.replace('_fused', '')
            fused_dir = entry
            images_dir = Path(args.images_dir) if args.images_dir else (dataset_root / class_name)
            db_path = Path(args.db_path) if args.db_path else (fused_dir / "review.db")

            if not (fused_dir / 'candidate').exists() and not (fused_dir / 'final').exists():
                continue

            class_configs[class_name] = {
                'fused_dir': fused_dir,
                'images_dir': images_dir,
                'db_path': db_path
            }
            available_classes.append(class_name)

        if not available_classes:
            logging.error(f"No <Class>_fused directories found in {dataset_root}")
            return 1

        current_class = args.class_name if args.class_name in available_classes else available_classes[0]

        app_state['dataset_root'] = dataset_root
        app_state['available_classes'] = available_classes
        app_state['class_configs'] = class_configs

        set_current_class(current_class, subdir=args.subdir)
        logging.info(f"Loaded dataset from: {dataset_root}")
    else:
        logging.info("No dataset loaded. Use the web UI to select a dataset root.")

    logging.info(f"Starting web server on {args.host}:{args.port}")
    logging.info("Press Ctrl+C to stop")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
