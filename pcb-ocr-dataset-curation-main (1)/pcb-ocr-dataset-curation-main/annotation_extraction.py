import argparse
import csv
import json
import os
import math
import shutil
import difflib
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from PIL import Image

# Allow large images (disable decompression bomb check)
Image.MAX_IMAGE_PIXELS = None

# Load known component classes from central JSON definition
_CLASSLIST_PATH = os.path.join(os.path.dirname(__file__), "known_classlist.json")
with open(_CLASSLIST_PATH, 'r') as _f:
    KNOWN_COMPONENT_CLASSES = json.load(_f)["KNOWN_COMPONENT_CLASSES"]


@dataclass
class LeadFootprint:
    lead_count: int
    lead_size_pixel: float
    bridge_size_percentage: float


def _get_param_value(agent_configs: Sequence[Dict[str, Any]], name: str,
                     param_key: str, default: Any) -> Any:
    """Extract simple scalar parameter from a sequence of agent configs."""
    for cfg in agent_configs:
        if not cfg:
            continue
        node = cfg.get(name)
        if not isinstance(node, dict):
            continue
        param_section = node.get(param_key)
        if not isinstance(param_section, dict):
            continue
        value = param_section.get('value')
        if value is not None:
            return value
    return default


def _parse_ignore_list(agent_configs: Sequence[Dict[str, Any]],
                       name: str) -> Set[int]:
    """Parse ignore list indices from param_vector definitions."""
    indices: Set[int] = set()
    for cfg in agent_configs:
        if not cfg:
            continue
        node = cfg.get(name)
        if not isinstance(node, dict):
            continue
        vector = node.get('param_vector') or []
        for entry in vector:
            value = None
            if isinstance(entry, int):
                value = entry
            elif isinstance(entry, dict):
                if 'param_int' in entry and isinstance(entry['param_int'], dict):
                    value = entry['param_int'].get('value')
                elif 'value' in entry:
                    value = entry.get('value')
            if value is not None:
                try:
                    indices.add(int(value))
                except (TypeError, ValueError):
                    continue
    return indices


def _estimate_lead_width_px(frame_width: float, lead_count: int) -> float:
    """Approximate pixel width of a single lead when no calibration exists."""
    if lead_count <= 1:
        return max(frame_width, 1.0)
    denominator = max(1, 2 * lead_count - 1)
    estimated = frame_width / denominator if frame_width > 0 else 1.0
    return max(estimated, 1.0)


def _build_lead_footprint(frame_width: float, params: Dict[str, Any]) -> LeadFootprint:
    lead_count = max(1, int(params.get('lead_count', 1)))
    lead_width_px = params.get('lead_width_px')
    if not isinstance(lead_width_px, (float, int)):
        lead_width_px = None
    if lead_count == 1:
        # For a single-lead side, use the full frame width to avoid left-anchored ROIs.
        lead_width_px = frame_width
    if lead_width_px is None or lead_width_px <= 1.0:
        lead_width_px = _estimate_lead_width_px(frame_width, lead_count)
    bridge_percentage = float(params.get('bridge_percentage', 50.0) or 50.0)
    return LeadFootprint(
        lead_count=lead_count,
        lead_size_pixel=float(lead_width_px),
        bridge_size_percentage=bridge_percentage,
    )


def extract_label_prefix(designator):
    """Extract component class from designator.

    Examples:
        'Capacitor_2(Auto Program)' -> 'Capacitor'
        'SOIC_SOP_TSOP_TSSOP_MSOP_3(Auto Program)' -> 'SOIC_SOP_TSOP_TSSOP_MSOP'
        'TO-263_TO-252_4(Auto Program)' -> 'TO-263_TO-252'
        'SOT_SOD_13(Auto Program)' -> 'SOT_SOD'

    Pattern: <ComponentClass>_<Index>(Auto Program)
    Where ComponentClass can contain underscores.
    """
    if not designator:
        return ""

    # Strip whitespace
    designator = designator.strip()

    # Remove '(Auto Program)' suffix if present (handle malformed variants)
    # Aggressively remove everything from '(' or 'Auto' to end of string
    designator = re.sub(r'\s*\(?Auto.*$', '', designator, flags=re.IGNORECASE)

    # Remove trailing index suffix after the final underscore.
    # Most datasets use numeric indices, but some use single-letter indices.
    # Also handles space before underscore (e.g., "Resistor _537" -> "Resistor")
    match = re.search(r'\s*_(?P<suffix>[A-Za-z0-9]+)$', designator)
    if match:
        suffix = match.group('suffix')
        if suffix.isdigit() or (len(suffix) == 1 and suffix.isalpha()):
            designator = designator[:match.start()]

    label = designator.strip()
    return normalize_component_label(label)


def _normalize_label_key(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', value.lower())


def _is_marker_label(label: str) -> bool:
    if not label:
        return False
    return label.strip().lower() == "marker"


def normalize_component_label(label: str, cutoff: float = 0.82) -> str:
    """Normalize component labels to known classes with typo-tolerant matching."""
    if not label:
        return label

    if label in KNOWN_COMPONENT_CLASSES:
        return label

    normalized_map = { _normalize_label_key(name): name for name in KNOWN_COMPONENT_CLASSES }
    label_key = _normalize_label_key(label)
    if label_key in normalized_map:
        return normalized_map[label_key]

    matches = difflib.get_close_matches(label_key, normalized_map.keys(), n=1, cutoff=cutoff)
    if matches:
        return normalized_map[matches[0]]

    return label


def parse_obb_shape(shape_str):
    """Parse the OBB shape string from CSV into labelme format"""
    try:
        shape_data = json.loads(shape_str)
        geometry = shape_data.get('geometry', {})
        angle = geometry.get('angle', 0)
        points = geometry.get('points', [])

        if len(points) >= 2:
            # Convert to labelme format: [[x1, y1], [x2, y2]]
            point1 = [points[0]['x'], points[0]['y']]
            point2 = [points[1]['x'], points[1]['y']]
            return [point1, point2], angle
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing shape: {e}")
        return None, None

    return None, None


def rotate_point(x, y, cx, cy, angle_deg):
    """Rotate point (x, y) around center (cx, cy) by angle_deg degrees"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Translate to origin
    x -= cx
    y -= cy

    # Rotate
    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a

    # Translate back
    x_new += cx
    y_new += cy

    return x_new, y_new


def _frame_dimensions(frame_bbox, frame_angle):
    """Recover axis-aligned width/height before rotation."""
    (x1, y1), (x2, y2) = frame_bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width, height


def local_roi_to_global(local_x, local_y, local_width, local_height, frame_bbox, frame_angle):
    """Convert local ROI coordinates to global image coordinates

    Args:
        local_x, local_y: Top-left corner in local frame coordinates
        local_width, local_height: Size of the ROI
        frame_bbox: [[x1, y1], [x2, y2]] - the main bounding box points
        frame_angle: Rotation angle of the main bounding box

    Returns:
        [[x1, y1], [x2, y2]], angle for labelme format
    """
    frame_x1, frame_y1 = frame_bbox[0]
    frame_x2, frame_y2 = frame_bbox[1]

    frame_cx = (frame_x1 + frame_x2) / 2
    frame_cy = (frame_y1 + frame_y2) / 2

    frame_width, frame_height = _frame_dimensions(frame_bbox, frame_angle)

    local_cx = local_x + local_width / 2
    local_cy = local_y + local_height / 2

    offset_x = local_cx - frame_width / 2
    offset_y = local_cy - frame_height / 2

    angle_rad = math.radians(frame_angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    global_offset_x = offset_x * cos_a - offset_y * sin_a
    global_offset_y = offset_x * sin_a + offset_y * cos_a

    global_cx = frame_cx + global_offset_x
    global_cy = frame_cy + global_offset_y

    # Calculate axis-aligned corner points around the global center
    # These are NOT rotated - the angle parameter handles rotation
    half_w = local_width / 2
    half_h = local_height / 2

    normalized_angle = frame_angle % 360
    angle_mod_180 = normalized_angle % 180

    if angle_mod_180 < 90:
        # Keep original orientation
        roi_angle = normalized_angle
        p1 = [global_cx - half_w, global_cy - half_h]
        p2 = [global_cx + half_w, global_cy + half_h]
    else:
        # Rotate by +90°, which means swap width and height
        roi_angle = (normalized_angle + 90) % 360
        p1 = [global_cx - half_h, global_cy - half_w]
        p2 = [global_cx + half_h, global_cy + half_w]

    return [p1, p2], roi_angle


def extract_polarity_roi(inspection_items_str: str) -> Optional[Dict[str, Any]]:
    """Extract polarity ROI from mounting inspection parameters."""
    try:
        inspection_data = json.loads(inspection_items_str)
    except json.JSONDecodeError as exc:
        print(f"Error parsing polarity ROI parameters: {exc}")
        return None

    # Look for mounting_inspection_2d agent
    mounting_inspection = inspection_data.get('mounting_inspection_2d')
    if not mounting_inspection:
        return None

    agent_config = mounting_inspection.get('agent_config')
    if not isinstance(agent_config, dict):
        return None

    # Extract polarity_roi parameter
    polarity_roi = agent_config.get('polarity_roi')
    if not isinstance(polarity_roi, dict):
        return None

    # Check if polarity detection is active
    if not polarity_roi.get('active', False):
        return None

    # Extract the param_roi structure
    param_roi = polarity_roi.get('param_roi')
    if not isinstance(param_roi, dict):
        return None

    # Get points and angle from param_roi
    points = param_roi.get('points', [])
    if len(points) < 2:
        return None

    angle = param_roi.get('angle', 0)

    # Convert points from dict format to list format
    # {"x": 22, "y": 126} -> [22, 126]
    point1 = [points[0]['x'], points[0]['y']]
    point2 = [points[1]['x'], points[1]['y']]

    return {
        'points': [point1, point2],
        'angle': angle
    }


def extract_ic_lead_params(inspection_items_str: str) -> Optional[Dict[str, Any]]:
    """Extract IC lead inspection parameters from inspection_line_items JSON."""
    try:
        inspection_data = json.loads(inspection_items_str)
    except json.JSONDecodeError as exc:
        print(f"Error parsing IC lead parameters: {exc}")
        return None

    agent_cfgs: List[Dict[str, Any]] = []
    for key in ('lead_inspection_2d_v2', 'lead_inspection_2d'):
        cfg = inspection_data.get(key)
        if not cfg:
            continue
        agent_cfg = cfg.get('agent_config')
        if isinstance(agent_cfg, dict):
            agent_cfgs.append(agent_cfg)

    if not agent_cfgs:
        return None

    params: Dict[str, Any] = {
        'ext_top': int(_get_param_value(agent_cfgs, 'ext_top', 'param_int', 0) or 0),
        'ext_bottom': int(_get_param_value(agent_cfgs, 'ext_bottom', 'param_int', 0) or 0),
        'tip_length': int(_get_param_value(agent_cfgs, 'tip_length', 'param_int', 0) or 0),
        'lead_count': int(_get_param_value(agent_cfgs, 'lead_count', 'param_int', 1) or 1),
        'lead_width_px': _get_param_value(agent_cfgs, 'lead_width_px', 'param_float', None),
        'bridge_percentage': float(
            _get_param_value(agent_cfgs, 'bridge_width_percentage', 'param_float', 50.0) or 50.0
        ),
        'lead_ignore_list': _parse_ignore_list(agent_cfgs, 'lead_ignore_list'),
        'bridge_ignore_list': _parse_ignore_list(agent_cfgs, 'bridge_ignore_list'),
    }

    if params['lead_width_px'] is None:
        lead_width_mm = _get_param_value(agent_cfgs, 'lead_width_mm', 'param_float', None)
        if lead_width_mm is not None:
            # Convert mm to pixels: 0.015mm = 1 pixel
            params['lead_width_px'] = lead_width_mm / 0.015

    return params


def _generate_roi(label: str, description: str, local_x: float, local_y: float,
                  width: float, height: float, frame_bbox: Sequence[Sequence[float]],
                  frame_angle: float) -> Dict[str, Any]:
    points, angle = local_roi_to_global(local_x, local_y, width, height, frame_bbox, frame_angle)
    return {
        'label': label,
        'points': points,
        'angle': angle,
        'shape_type': 'rotated_box',
        'group_id': None,
        'description': description,
        'flags': {},
    }


def generate_ic_lead_rois(frame_bbox: Sequence[Sequence[float]], frame_angle: float,
                          params: Dict[str, Any], debug_label: Optional[str] = None) -> List[Dict[str, Any]]:
    """Pure Python port of C++ generateRois logic."""
    annotations: List[Dict[str, Any]] = []

    frame_width, frame_height = _frame_dimensions(frame_bbox, frame_angle)

    ext_top = max(0, int(params.get('ext_top', 0)))
    ext_bottom = max(0, int(params.get('ext_bottom', 0)))
    tip_length = max(0, int(params.get('tip_length', 0)))
    footprint = _build_lead_footprint(frame_width, params)
    ignore_leads = params.get('lead_ignore_list', set())

    solder_height = frame_height - ext_top - ext_bottom
    if solder_height <= 0:
        solder_height = frame_height

    if debug_label:
        print(f"[IC_LEAD_DEBUG][{debug_label}] frame_width={frame_width}, frame_height={frame_height}, "
              f"ext_top={ext_top}, ext_bottom={ext_bottom}, tip_length={tip_length}, "
              f"lead_count={footprint.lead_count}, lead_width_px={footprint.lead_size_pixel}")

    if footprint.lead_count > 1:
        bridge_size_px = (
            (frame_width - footprint.lead_size_pixel) / (footprint.lead_count - 1)
            - footprint.lead_size_pixel
        )
    else:
        bridge_size_px = 0.0

    step_size = footprint.lead_size_pixel + bridge_size_px

    if debug_label:
        print(f"[IC_LEAD_DEBUG][{debug_label}] solder_height={solder_height}, "
              f"bridge_size_px={bridge_size_px}, step_size={step_size}")

    for idx in range(footprint.lead_count):
        if idx in ignore_leads:
            continue
        center_x = idx * step_size + footprint.lead_size_pixel / 2.0
        solder_center_y = ext_top + solder_height / 2.0
        solder_x = center_x - footprint.lead_size_pixel / 2.0
        solder_y = solder_center_y - solder_height / 2.0
        lead_full_end_y = frame_height
        lead_full_start_y = 0.0
        has_tip = False

        if debug_label:
            print(f"[IC_LEAD_DEBUG][{debug_label}] Lead {idx}: center_x={center_x}, "
                  f"solder_roi=(x={solder_x}, y={solder_y}, w={footprint.lead_size_pixel}, h={solder_height})")

        annotations.append(
            _generate_roi(
                label='_ic_lead_solder',
                description=f'Lead {idx} solder',
                local_x=solder_x,
                local_y=solder_y,
                width=footprint.lead_size_pixel,
                height=solder_height,
                frame_bbox=frame_bbox,
                frame_angle=frame_angle,
            )
        )

        if ext_top > 1:
            pad_center_y = ext_top / 2.0
            pad_x = center_x - footprint.lead_size_pixel / 2.0
            pad_y = pad_center_y - ext_top / 2.0
            if debug_label:
                print(f"[IC_LEAD_DEBUG][{debug_label}] Lead {idx}: pad_roi="
                      f"(x={pad_x}, y={pad_y}, w={footprint.lead_size_pixel}, h={ext_top})")
            annotations.append(
                _generate_roi(
                    label='_ic_lead_pad',
                    description=f'Lead {idx} pad',
                    local_x=pad_x,
                    local_y=pad_y,
                    width=footprint.lead_size_pixel,
                    height=ext_top,
                    frame_bbox=frame_bbox,
                    frame_angle=frame_angle,
                )
            )

        if tip_length > 1 and tip_length <= ext_bottom:
            tip_center_y = ext_top + solder_height + tip_length / 2.0
            tip_x = center_x - footprint.lead_size_pixel / 2.0
            tip_y = tip_center_y - tip_length / 2.0
            lead_full_start_y = tip_y
            has_tip = True
            if debug_label:
                print(f"[IC_LEAD_DEBUG][{debug_label}] Lead {idx}: tip_roi="
                      f"(x={tip_x}, y={tip_y}, w={footprint.lead_size_pixel}, h={tip_length})")
            annotations.append(
                _generate_roi(
                    label='_ic_lead_tip',
                    description=f'Lead {idx} tip',
                    local_x=tip_x,
                    local_y=tip_y,
                    width=footprint.lead_size_pixel,
                    height=tip_length,
                    frame_bbox=frame_bbox,
                    frame_angle=frame_angle,
                )
            )

        lead_full_height = max(0.0, lead_full_end_y - lead_full_start_y)
        if has_tip and lead_full_height > 0:
            annotations.append(
                _generate_roi(
                    label='_ic_lead_full',
                    description=f'Lead {idx} full',
                    local_x=solder_x,
                    local_y=lead_full_start_y,
                    width=footprint.lead_size_pixel,
                    height=lead_full_height,
                    frame_bbox=frame_bbox,
                    frame_angle=frame_angle,
                )
            )

    return annotations


def read_groups_csv(csv_path, skip_marker: bool = False):
    """Read groups.csv and extract all records"""
    annotations = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape_str = row.get('shape', '')
            designator = row.get('designator', '').strip()
            region_group_id = row.get('region_group_id', '').strip()
            array_index = row.get('array_index', '').strip()  # Instance index within the group

            points, angle = parse_obb_shape(shape_str)
            if points:
                label = extract_label_prefix(designator)
                # Components from groups.csv represent the full component bounding box
                # Add _full suffix to distinguish from individual parts (mount, solder, etc.)
                if skip_marker and _is_marker_label(label):
                    continue
                label_full = f'{label}_full'
                # Use unique combination of region_group_id + array_index
                # This uniquely identifies each physical component instance
                annotations.append({
                    'label': label_full,
                    'points': points,
                    'angle': angle,
                    'shape_type': 'rotated_box',
                    'group_id': None,
                    'description': f'group_{region_group_id}_{array_index}',  # Unique instance identifier
                    'flags': {}
                })

    return annotations


def build_group_mapping(groups_csv_path, skip_marker: bool = False):
    """Build mapping from region_group_id to designator (component name)"""
    group_mapping = {}

    if not groups_csv_path.exists():
        return group_mapping

    with open(groups_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            region_group_id = row.get('region_group_id', '')
            designator = row.get('designator', '').strip()  # Strip whitespace
            if region_group_id and designator:
                # Extract prefix from designator (e.g., "Capacitor_2(Auto Program)" -> "Capacitor")
                prefix = extract_label_prefix(designator)
                if skip_marker and _is_marker_label(prefix):
                    continue
                group_mapping[region_group_id] = {
                    'designator': designator,
                    'prefix': prefix
                }

    return group_mapping


def read_regions_csv(csv_path, group_mapping=None, skip_marker: bool = False):
    """Read regions.csv and extract all records (skip _text prefixed labels)"""
    annotations = []

    if group_mapping is None:
        group_mapping = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line_num, row in enumerate(reader, start=2):
            shape_str = row.get('shape', '')
            component_class = row.get('component_class', '')
            inspection_items = row.get('inspection_line_items', '')
            region_group_id = row.get('region_group_id', '').strip()
            array_index = row.get('array_index', '').strip()  # Instance index within the group

            # Skip if component_class starts with _text
            if component_class.startswith('_text'):
                continue
            if skip_marker and _is_marker_label(component_class):
                continue

            points, angle = parse_obb_shape(shape_str)
            if not points:
                continue

            # Build description using unique region_group_id + array_index
            description = ''
            if region_group_id and array_index:
                # Use format: "group_{region_group_id}_{array_index} {component_class}"
                # This creates a unique identifier for the parent component instance
                description = f"group_{region_group_id}_{array_index} {component_class}"

            # Special handling for _ic_lead components
            if component_class == '_ic_lead' and inspection_items:
                # IC components can have leads on multiple sides (2-sided or 4-sided)
                # Each side has a unique region_id, so we use it to distinguish sides
                region_id = row.get('region_id', '').strip()

                # Extract IC lead parameters
                params = extract_ic_lead_params(inspection_items)
                if params:
                    debug_label = None
                    if line_num == 766:
                        debug_label = f"line {line_num} region {region_id}"
                    # Generate detailed ROIs for each lead
                    lead_annotations = generate_ic_lead_rois(points, angle, params, debug_label=debug_label)
                    # Add multi-level relationship to IC lead descriptions
                    # Level 1: Parent component (region_group_id + array_index)
                    # Level 2: Side of component (region_id - distinguishes top/bottom/left/right sides)
                    # Level 3: Lead number from description
                    if region_group_id and array_index and region_id:
                        for ann in lead_annotations:
                            # Format: "group_{region_group_id}_{array_index}_side_{region_id} {original_description}"
                            # e.g., "group_1766048909_0_side_1766025435 Lead 0 pad"
                            # This preserves the component ID, side ID, and lead info
                            ann['description'] = f"group_{region_group_id}_{array_index}_side_{region_id} {ann['description']}"
                    annotations.extend(lead_annotations)
                    print(f"Generated {len(lead_annotations)} detailed annotations for IC lead with {params['lead_count']} leads")
                else:
                    # Fallback to single bbox if parameter extraction fails
                    annotations.append({
                        'label': component_class,
                        'points': points,
                        'angle': angle,
                        'shape_type': 'rotated_box',
                        'group_id': None,
                        'description': description,
                        'flags': {}
                    })
            else:
                # Normal single bbox annotation
                annotations.append({
                    'label': component_class,
                    'points': points,
                    'angle': angle,
                    'shape_type': 'rotated_box',
                    'group_id': None,
                    'description': description,
                    'flags': {}
                })

                # Special handling for _mount components: extract polarity ROI if present
                if component_class == '_mount' and inspection_items:
                    polarity_roi = extract_polarity_roi(inspection_items)
                    if polarity_roi:
                        # Polarity ROI points are in local coordinates relative to the mount component
                        # We need to transform them to global image coordinates
                        local_points = polarity_roi['points']
                        local_angle = polarity_roi['angle']

                        # Calculate local ROI dimensions and position (in mount's local coordinate system)
                        local_x = min(local_points[0][0], local_points[1][0])
                        local_y = min(local_points[0][1], local_points[1][1])
                        local_width = abs(local_points[1][0] - local_points[0][0])
                        local_height = abs(local_points[1][1] - local_points[0][1])

                        # Get mount component dimensions and center
                        mount_x1, mount_y1 = points[0]
                        mount_x2, mount_y2 = points[1]
                        mount_cx = (mount_x1 + mount_x2) / 2
                        mount_cy = (mount_y1 + mount_y2) / 2
                        mount_width = abs(mount_x2 - mount_x1)
                        mount_height = abs(mount_y2 - mount_y1)

                        # Calculate polarity center in local coordinates
                        local_polarity_cx = local_x + local_width / 2
                        local_polarity_cy = local_y + local_height / 2

                        # Calculate offset from mount's local center
                        offset_x = local_polarity_cx - mount_width / 2
                        offset_y = local_polarity_cy - mount_height / 2

                        # Rotate offset by mount's angle to get global offset
                        angle_rad = math.radians(angle)
                        cos_a = math.cos(angle_rad)
                        sin_a = math.sin(angle_rad)
                        global_offset_x = offset_x * cos_a - offset_y * sin_a
                        global_offset_y = offset_x * sin_a + offset_y * cos_a

                        # Calculate global center
                        global_cx = mount_cx + global_offset_x
                        global_cy = mount_cy + global_offset_y

                        # Create global points (axis-aligned box in global coords)
                        half_w = local_width / 2
                        half_h = local_height / 2
                        global_points = [
                            [global_cx - half_w, global_cy - half_h],
                            [global_cx + half_w, global_cy + half_h]
                        ]

                        # Use mount component's angle as the polarity angle
                        # The polarity ROI is defined in mount's local coordinate system,
                        # so it inherits the mount's rotation
                        final_angle = angle

                        # Create polarity annotation with parent component relationship
                        # Description format: "group_{region_group_id}_{array_index} _polarity"
                        # This links the polarity to its parent mount component
                        polarity_annotation = {
                            'label': '_polarity',
                            'points': global_points,
                            'angle': final_angle,
                            'shape_type': 'rotated_box',
                            'group_id': None,
                            'description': description.replace('_mount', '_polarity') if description else '_polarity',
                            'flags': {}
                        }
                        annotations.append(polarity_annotation)
                        print(f"Generated polarity annotation for mount component")

    return annotations


def get_image_dimensions(image_path):
    """Get image width and height"""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return 0, 0


def create_labelme_json(shapes, image_path, output_path):
    """Create labelme format JSON file"""
    # Get image dimensions
    width, height = get_image_dimensions(image_path)

    # Assign group_ids to shapes
    for idx, shape in enumerate(shapes):
        shape['group_id'] = idx + 1

    # Create labelme format
    labelme_data = {
        'shapes': shapes,
        'imagePath': os.path.basename(image_path),
        'imageWidth': width,
        'imageHeight': height,
        'imageData': None,
        'flags': {}
    }

    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, indent=4, ensure_ascii=False)

    print(f"Created: {output_path}")
    return labelme_data


def read_model_name_from_products(products_csv):
    """Read model_name from products.csv."""
    if not products_csv.exists():
        return None

    with open(products_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row.get('model_name', '').strip()
            if model_name:
                return model_name

    return None


def read_image_path_from_inspectables(inspectables_csv, data_dir):
    """Read color_map_uri from inspectables.csv and locate the image file."""
    if not inspectables_csv.exists():
        return None

    with open(inspectables_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            color_map_uri = row.get('color_map_uri', '')
            if not color_map_uri:
                continue

            # Extract relative path starting from 'color_map'
            # Example: C:\ProgramData\DaoAI\AOI_PCB\blob\color_map\25121810a\0\1766081551cTwKjK.jpg
            # We want: color_map/25121810a/0/1766081551cTwKjK.jpg
            if 'color_map' in color_map_uri:
                # Split by 'color_map' and take everything after it
                parts = color_map_uri.split('color_map')
                if len(parts) >= 2:
                    # Get the part after 'color_map', remove leading separators
                    relative_path = parts[-1].lstrip('\\/').replace('\\', '/')
                    # Construct full path
                    image_path = data_dir / 'color_map' / relative_path
                    if image_path.exists():
                        return image_path

    return None


def is_board_directory(dir_path):
    """Check if a directory contains the required CSV files for a board"""
    required_files = ['groups.csv', 'regions.csv', 'inspectables.csv']
    return all((dir_path / file).exists() for file in required_files)


def extract_zip_file(zip_path: Path, extract_dir: Path) -> Path:
    """
    Extract a zip file to a directory, handling Windows backslash paths.

    Args:
        zip_path: Path to zip file
        extract_dir: Directory to extract to

    Returns:
        Path to extracted board directory
    """
    print(f"  Extracting {zip_path.name}...")

    # Extract and fix Windows backslash paths
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Replace backslashes with forward slashes
            fixed_path = member.replace('\\', '/')

            # Extract to the fixed path
            target_path = extract_dir / fixed_path

            # Create parent directories
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract file if it's not a directory
            if not member.endswith('/') and not member.endswith('\\'):
                with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                    shutil.copyfileobj(source, target)

    # Check if board files are directly at extract_dir root
    if is_board_directory(extract_dir):
        return extract_dir

    # Otherwise, search recursively for board directories (handles single nested folders).
    extracted_dirs = []
    for root, _, _ in os.walk(extract_dir):
        root_path = Path(root)
        if is_board_directory(root_path):
            extracted_dirs.append(root_path)

    if len(extracted_dirs) == 1:
        return extracted_dirs[0]
    elif len(extracted_dirs) > 1:
        print(f"  Warning: Multiple board directories found, using first one")
        return extracted_dirs[0]

    # Return extract_dir and let the caller handle the error
    return extract_dir


def process_single_board(data_dir, output_dir, skip_marker: bool = False):
    """Process a single board directory"""
    groups_csv = data_dir / 'groups.csv'
    regions_csv = data_dir / 'regions.csv'
    inspectables_csv = data_dir / 'inspectables.csv'
    products_csv = data_dir / 'products.csv'

    # Get image path from inspectables.csv
    image_path = read_image_path_from_inspectables(inspectables_csv, data_dir)
    if not image_path:
        print(f"Error: Could not find image from inspectables.csv in {data_dir}")
        return False, set()

    print(f"Using image: {image_path}")

    # Get model name from products.csv
    model_name = read_model_name_from_products(products_csv)
    if model_name:
        print(f"Model name: {model_name}")
    else:
        print(f"Warning: Could not find model_name in products.csv, using original filename")
        model_name = image_path.stem

    # Read annotations from CSVs
    print("Reading groups.csv...")
    groups_annotations = read_groups_csv(groups_csv, skip_marker=skip_marker)
    print(f"Found {len(groups_annotations)} annotations from groups.csv")

    # Build group mapping for relationship tracking
    print("Building component relationship mapping...")
    group_mapping = build_group_mapping(groups_csv, skip_marker=skip_marker)
    print(f"Found {len(group_mapping)} component groups")

    print("Reading regions.csv...")
    regions_annotations = read_regions_csv(regions_csv, group_mapping, skip_marker=skip_marker)
    print(f"Found {len(regions_annotations)} annotations from regions.csv")

    # Combine all annotations
    all_shapes = groups_annotations + regions_annotations
    print(f"Total annotations: {len(all_shapes)}")

    # Collect unique labels/classes
    unique_labels = {shape['label'] for shape in all_shapes}

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get original image extension
    image_ext = image_path.suffix

    # Copy image file to output directory with model name
    output_image = output_dir / f'{model_name}{image_ext}'
    if output_image != image_path:
        shutil.copy2(image_path, output_image)
        print(f"Copied image to: {output_image}")

    # Create output JSON file with same basename as renamed image
    output_json = output_dir / f'{model_name}.json'
    create_labelme_json(all_shapes, output_image, output_json)

    print(f"\nSuccessfully created annotation file: {output_json}")
    print(f"Image: {output_image.name}")
    print(f"Total shapes: {len(all_shapes)}")
    print(f"Output directory: {output_dir}")
    return True, unique_labels


def main():
    parser = argparse.ArgumentParser(
        description="Extract annotations from groups.csv and regions.csv and create labelme JSON format."
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='Data/1216',
        help='Path to the data directory containing CSV files and images, or parent directory with board subfolders (default: Data/1216)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output directory for JSON and image files. If not provided, will use data-dir'
    )
    parser.add_argument(
        '--skip-marker',
        action='store_true',
        help='Skip annotations with component_class/designator label "marker"'
    )

    args = parser.parse_args()

    # Define paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / args.data_dir

    if not data_dir.exists():
        print(f"Error: Directory does not exist: {data_dir}")
        return

    # Check if this is a single board directory or contains multiple board subfolders
    if is_board_directory(data_dir):
        # Single board directory
        print(f"Processing single board directory: {data_dir}")
        print("=" * 80)

        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = data_dir

        success, unique_labels = process_single_board(data_dir, output_dir, skip_marker=args.skip_marker)

        # Print class list
        if success and unique_labels:
            sorted_labels = sorted(unique_labels)
            print("\n" + "=" * 80)
            print("CLASS LIST:")
            print("=" * 80)
            print(sorted_labels)
            print(f"\nTotal unique classes: {len(unique_labels)}")
            print("=" * 80)
    else:
        # Parent directory containing multiple board subfolders or zip files
        print(f"Scanning for board subfolders and zip files in: {data_dir}")

        # Find all board subdirectories
        board_dirs = [d for d in data_dir.iterdir() if d.is_dir() and is_board_directory(d)]

        # Find all zip files in this directory and in subdirectories (like 1_9, 1_11, etc.)
        zip_files = list(data_dir.glob('*.zip'))
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                zip_files.extend(list(subdir.glob('*.zip')))

        if not board_dirs and not zip_files:
            print(f"Error: No valid board directories or zip files found in {data_dir}")
            print("Each board directory must contain: groups.csv, regions.csv, inspectables.csv")
            print("Or provide zip files in subdirectories")
            return

        print(f"Found {len(board_dirs)} board directories and {len(zip_files)} zip files to process")
        print("=" * 80)

        # Determine base output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = data_dir / 'output'

        # Process each board
        success_count = 0
        failed_count = 0
        all_unique_labels = set()
        temp_dirs = []  # Track temp dirs for cleanup

        # Process existing board directories
        for board_dir in sorted(board_dirs):
            board_name = board_dir.name
            print(f"\nProcessing board directory: {board_name}")
            print("-" * 80)

            try:
                success, unique_labels = process_single_board(board_dir, output_dir, skip_marker=args.skip_marker)
                if success:
                    success_count += 1
                    all_unique_labels.update(unique_labels)
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Error processing board {board_name}: {e}")
                failed_count += 1

            print("=" * 80)

        # Process zip files
        for zip_file in sorted(zip_files):
            zip_name = zip_file.stem
            print(f"\nProcessing zip file: {zip_file.name}")
            print("-" * 80)

            try:
                # Extract to temporary directory
                temp_dir = Path(tempfile.mkdtemp(prefix="board_extract_"))
                temp_dirs.append(temp_dir)

                extracted_dir = extract_zip_file(zip_file, temp_dir)

                # Process the extracted board
                success, unique_labels = process_single_board(extracted_dir, output_dir, skip_marker=args.skip_marker)
                if success:
                    success_count += 1
                    all_unique_labels.update(unique_labels)
                else:
                    failed_count += 1

            except Exception as e:
                print(f"Error processing zip {zip_file.name}: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1

            print("=" * 80)

        # Cleanup temporary directories
        print("\nCleaning up temporary extraction directories...")
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temp dir {temp_dir}: {e}")

        print(f"\n\nProcessing complete!")
        print(f"Successfully processed: {success_count} boards")
        print(f"Failed: {failed_count} boards")
        print(f"Output directory: {output_dir}")

        # Print class list from all boards
        if all_unique_labels:
            sorted_labels = sorted(all_unique_labels)
            print("\n" + "=" * 80)
            print("CLASS LIST (across all boards):")
            print("=" * 80)
            print(sorted_labels)
            print(f"\nTotal unique classes: {len(all_unique_labels)}")
            print("=" * 80)


if __name__ == '__main__':
    main()
