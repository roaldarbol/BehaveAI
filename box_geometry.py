#!/usr/bin/env python3
"""
Shared geometry/format helpers for oriented bounding boxes (OBB), used by the
annotation tools, Regenerate_annotations.py, and the classify/track/live pipelines.

Internal oriented-box convention: a box is defined by a head point, a tail point
(head-tail axis = the object's length), and a width (perpendicular to that axis).
Corners are always returned/expected in the order
	[head_left, head_right, tail_right, tail_left]
(where "left"/"right" is relative to facing from tail toward head), which traces
the rectangle's perimeter and is the order written to/read from label files.
"""

import math
import cv2
import numpy as np


# --------------------------- label line I/O ---------------------------

def read_label_line(line, oriented):
	"""Parse one YOLO label line. Returns None if the line is empty/malformed."""
	parts = line.strip().split()
	if not parts:
		return None
	cls = int(float(parts[0]))
	if oriented:
		if len(parts) < 9:
			return None
		coords = list(map(float, parts[1:9]))
		corners_n = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
		return {'cls': cls, 'corners_n': corners_n}
	else:
		if len(parts) < 5:
			return None
		xc, yc, bw, bh = map(float, parts[1:5])
		return {'cls': cls, 'xc': xc, 'yc': yc, 'bw': bw, 'bh': bh}


def write_label_line(box, oriented):
	if oriented:
		flat = []
		for (x, y) in box['corners_n']:
			flat.append(x)
			flat.append(y)
		coords_str = " ".join(f"{v:.6f}" for v in flat)
		return f"{box['cls']} {coords_str}\n"
	else:
		return f"{box['cls']} {box['xc']:.6f} {box['yc']:.6f} {box['bw']:.6f} {box['bh']:.6f}\n"


# --------------------------- pixel <-> normalized ---------------------------

def denorm_axis(xc, yc, bw, bh, w, h):
	cx = float(xc) * w
	cy = float(yc) * h
	bw_p = float(bw) * w
	bh_p = float(bh) * h
	x1 = int(cx - bw_p / 2); y1 = int(cy - bh_p / 2)
	x2 = int(cx + bw_p / 2); y2 = int(cy + bh_p / 2)
	x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
	x2 = max(0, min(w - 1, x2)); y2 = max(0, min(h - 1, y2))
	return x1, y1, x2, y2


def norm_axis(x1, y1, x2, y2, w, h):
	xc = (x1 + x2) / 2.0 / w
	yc = (y1 + y2) / 2.0 / h
	bw = abs(x2 - x1) / w
	bh = abs(y2 - y1) / h
	return xc, yc, bw, bh


def denorm_corners(corners_n, w, h):
	return [(x * w, y * h) for (x, y) in corners_n]


def norm_corners(corners, w, h):
	return [(x / w, y / h) for (x, y) in corners]


# --------------------------- head/tail/width <-> corners ---------------------------

def _midpoint(a, b):
	return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _dist(a, b):
	return math.hypot(a[0] - b[0], a[1] - b[1])


def corners_from_head_tail_width(head, tail, width):
	hx, hy = head; tx, ty = tail
	dx, dy = hx - tx, hy - ty
	length = math.hypot(dx, dy)
	if length < 1e-6:
		ux, uy = 1.0, 0.0
	else:
		ux, uy = dx / length, dy / length
	px, py = -uy, ux  # perpendicular unit vector
	hw = width / 2.0
	head_left = (hx + px * hw, hy + py * hw)
	head_right = (hx - px * hw, hy - py * hw)
	tail_right = (tx - px * hw, ty - py * hw)
	tail_left = (tx + px * hw, ty + py * hw)
	return [head_left, head_right, tail_right, tail_left]


def head_tail_width_from_corners(corners):
	head = _midpoint(corners[0], corners[1])
	tail = _midpoint(corners[2], corners[3])
	width = _dist(corners[0], corners[1])
	return head, tail, width


def signed_perpendicular_distance(head, tail, point):
	"""Signed distance from `point` to the head-tail line, projected onto the same perpendicular
	unit vector convention used by corners_from_head_tail_width (px,py = -uy,ux). Positive/negative
	sign indicates which side of the centre line `point` falls on."""
	hx, hy = head; tx, ty = tail
	dx, dy = hx - tx, hy - ty
	length = math.hypot(dx, dy)
	if length < 1e-6:
		return 0.0
	px, py = point[0] - tx, point[1] - ty
	return (dx * py - dy * px) / length


def shift_head_tail_sideways(head, tail, amount):
	"""Translate (head, tail) sideways - perpendicular to their own axis, i.e. left/right relative
	to the centre line - by `amount` (signed, same units/convention as signed_perpendicular_distance).
	Lets the annotator nudge the box's centreline left/right during the width-setting step. Callers
	must compute width from the *original*, pre-shift (head, tail) - a sideways translation moves
	the line itself, so distance-to-the-shifted-line would no longer mean what the user drew."""
	hx, hy = head; tx, ty = tail
	dx, dy = hx - tx, hy - ty
	length = math.hypot(dx, dy)
	if length < 1e-6:
		return head, tail
	ux, uy = dx / length, dy / length
	px, py = -uy, ux  # perpendicular unit vector (matches corners_from_head_tail_width's convention)
	return (hx + amount * px, hy + amount * py), (tx + amount * px, ty + amount * py)


def head_tail_width_from_xywhr_rad(cx, cy, w, h, angle_rad):
	"""Convert an Ultralytics OBB result (results[0].obb.xywhr row, radians) into our
	head/tail/width model. The head/tail assignment is an arbitrary initial hypothesis
	(mod-180 ambiguous) - callers needing a true head must disambiguate separately."""
	length, width = (w, h) if w >= h else (h, w)
	angle = angle_rad if w >= h else angle_rad + math.pi / 2
	ux, uy = math.cos(angle), math.sin(angle)
	half = length / 2.0
	head = (cx + ux * half, cy + uy * half)
	tail = (cx - ux * half, cy - uy * half)
	return head, tail, width


def angle_length_from_head_tail(head, tail):
	dx, dy = head[0] - tail[0], head[1] - tail[1]
	return math.degrees(math.atan2(dy, dx)), math.hypot(dx, dy)


# --------------------------- geometry ops ---------------------------

def axis_aligned_bounds(corners):
	xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
	return int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))


def inset_corners(corners, px):
	"""Shrink a rotated rect by px on every side (for the hierarchical outer/inner box look)."""
	head, tail, width = head_tail_width_from_corners(corners)
	hx, hy = head; tx, ty = tail
	dx, dy = hx - tx, hy - ty
	length = math.hypot(dx, dy)
	if length < 1e-6 or width - 2 * px <= 0 or length - 2 * px <= 0:
		return corners
	ux, uy = dx / length, dy / length
	new_head = (hx - ux * px, hy - uy * px)
	new_tail = (tx + ux * px, ty + uy * px)
	new_width = width - 2 * px
	return corners_from_head_tail_width(new_head, new_tail, new_width)


def _to_cv_rotated_rect(corners):
	return cv2.minAreaRect(np.array(corners, dtype=np.float32))


def rotated_iou(corners_a, corners_b):
	rect_a = _to_cv_rotated_rect(corners_a)
	rect_b = _to_cv_rotated_rect(corners_b)
	area_a = rect_a[1][0] * rect_a[1][1]
	area_b = rect_b[1][0] * rect_b[1][1]
	if area_a <= 0 or area_b <= 0:
		return 0.0
	_, inter_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
	if inter_pts is None or len(inter_pts) < 3:
		return 0.0
	inter_area = cv2.contourArea(cv2.convexHull(inter_pts))
	union = area_a + area_b - inter_area
	if union <= 0:
		return 0.0
	return float(inter_area / union)


def point_in_corners(px, py, corners):
	pts = np.array(corners, dtype=np.float32)
	return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0


def iou(box1, box2):
	"""Axis-aligned overlap proportion between two (x1,y1,x2,y2) boxes: the larger of
	intersection/area1 and intersection/area2 - so a box entirely inside another returns 1.0,
	unlike a standard intersection-over-union."""
	xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
	xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
	inter = max(0, xb - xa) * max(0, yb - ya)
	area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) if box1[2] > box1[0] and box1[3] > box1[1] else 0
	area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) if box2[2] > box2[0] and box2[3] > box2[1] else 0
	prop1 = inter / area1 if area1 > 0 else 0
	prop2 = inter / area2 if area2 > 0 else 0
	return max(prop1, prop2)


def detection_scale(det):
	"""Estimate a detection's own body scale (used to make tracking/merge thresholds relative
	to object size instead of absolute pixels). Oriented detections use their true head-tail
	length/width (avoids the axis-aligned bbox's rotation-dependent size jitter); otherwise
	falls back to the average width/height of the axis-aligned box."""
	head, tail = det.get('head'), det.get('tail')
	if head is not None and tail is not None:
		length = _dist(head, tail)
		width = det.get('width') or 0.0
		return 0.5 * (length + width) if width else length
	x1, y1, x2, y2 = det['coords']
	return 0.5 * (abs(x2 - x1) + abs(y2 - y1))


def merge_source_detections(all_detections, centroid_merge_ratio, iou_thresh, dominant_source):
	"""Merge per-frame detections collected from the static and motion streams into one
	detection per physical object, resolving which stream's classification wins per the
	project's `dominant_source` setting. This is the single canonical implementation of that
	rule - BehaveAI_classify_track.py, BehaveAI_live.py, and BehaveAI_annotation.py's
	auto-annotate all call this instead of keeping their own copies, so the rule behaves
	identically in batch processing, live capture, and interactive auto-annotation.

	`all_detections` is a list of dicts, one per raw detection from either stream, each with:
	  'coords': (x1,y1,x2,y2) axis-aligned box (used for matching, even for oriented projects -
	            callers pass the OBB's axis-aligned envelope here and carry the true oriented
	            geometry through via the optional 'head'/'tail'/'width' keys, which are copied
	            through unchanged whenever a detection is chosen as a match's new winner)
	  'primary_class', 'primary_conf', 'source' ('static' or 'motion')
	  'primary_class_combined', 'primary_conf_combined': always '' / 0.0 on input
	  'head', 'tail', 'width': optional, oriented-mode geometry

	Two detections are matched to the same physical object when their centroids are within
	`centroid_merge_ratio` times their average detection_scale() OR their boxes overlap more
	than `iou_thresh`. For a matched pair:
	  - same source, or dominant_source=='confidence': keep whichever has higher primary_conf
	  - dominant_source=='static'/'motion': the matching-source detection always wins, regardless
	    of confidence
	Either way, the loser's class/conf is kept as *_combined (for CSV/inspection), and the merge
	is symmetric in stream order (static-then-motion or motion-then-static all_detections order
	both settle on the same winner) since each new detection is compared against the running
	merged winner using the same rule.

	Returns merged_detections: one dict per physical object, each with 'coords', 'centroid',
	'source', 'primary_class', 'primary_conf', 'primary_class_combined', 'primary_conf_combined',
	and 'head'/'tail'/'width' (None when not oriented)."""
	merged_detections = []
	for det in all_detections:
		x1, y1, x2, y2 = det['coords']
		cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
		det_scale = detection_scale(det)

		matched = False
		for md in merged_detections:
			md_cx, md_cy = md['centroid']
			# squared distance instead of np.hypot: equivalent for a non-negative threshold,
			# avoids both the sqrt and numpy's per-call overhead on a two-number distance
			dx, dy = cx - md_cx, cy - md_cy
			dist2 = dx * dx + dy * dy
			merge_thresh = centroid_merge_ratio * 0.5 * (det_scale + detection_scale(md))
			within_centroid = dist2 < merge_thresh * merge_thresh

			# IoU is only needed when the (cheap) centroid check doesn't already resolve the
			# match, since the two are combined with `or` below - skip it otherwise. This is a
			# meaningful win with many small, densely-packed detections (e.g. insects), where
			# the centroid check alone resolves most matches. NOTE: `within_centroid` must stay
			# the first operand of the `or` below - Python's short-circuit evaluation is what
			# keeps `overlap` from ever being compared while still None.
			if within_centroid:
				overlap = None
			else:
				md_x1, md_y1, md_x2, md_y2 = md['coords']
				overlap = iou((x1, y1, x2, y2), (md_x1, md_y1, md_x2, md_y2))
			ms_source = md['source']

			if within_centroid or overlap > iou_thresh:
				take_it = False
				if det['source'] == ms_source or dominant_source == 'confidence':
					# matching sources so select best, or confidence strategy used
					if 'primary_conf' not in md or det['primary_conf'] > md['primary_conf']:
						take_it = True
				elif det['source'] == 'static' and dominant_source == 'static':
					take_it = True
				elif det['source'] == 'motion' and dominant_source == 'motion':
					take_it = True

				if take_it:
					md['primary_class_combined'] = md['primary_class']  # retain the combined primary
					md['primary_conf_combined'] = md['primary_conf']
					md['primary_class'] = det['primary_class']
					md['primary_conf'] = det['primary_conf']
					md['coords'] = det['coords']  # Update to the winning detection's box
					md['centroid'] = (cx, cy)
					md['source'] = det['source']
					md['head'] = det.get('head'); md['tail'] = det.get('tail'); md['width'] = det.get('width')

				matched = True
				break

		if not matched:
			merged_detections.append({
				'coords': det['coords'],
				'centroid': (cx, cy),
				'source': det['source'],
				'primary_class': det['primary_class'],
				'primary_conf': det['primary_conf'],
				'primary_class_combined': '',
				'primary_conf_combined': 0.0,
				'head': det.get('head'), 'tail': det.get('tail'), 'width': det.get('width'),
			})

	return merged_detections


def nms_oriented(boxes, get_corners, get_conf, iou_thresh):
	if not boxes:
		return boxes
	ordered = sorted(boxes, key=get_conf, reverse=True)
	kept = []
	for b in ordered:
		bc = get_corners(b)
		if not any(rotated_iou(bc, get_corners(k)) > iou_thresh for k in kept):
			kept.append(b)
	return kept


def clip_line_to_rect(point, direction, width, height):
	"""Return the two endpoints where the infinite line through `point` in `direction` crosses
	the [0,width-1] x [0,height-1] rectangle boundary, or None if it doesn't cross it at all.
	Used to draw a full-frame guide line (e.g. a crosshair orthogonal to a box's head-tail axis)
	in the same "spans the whole canvas" style as the existing screen-aligned crosshair."""
	px, py = point
	dx, dy = direction
	t_values = []
	if abs(dx) > 1e-9:
		t_values.append((0 - px) / dx)
		t_values.append((width - 1 - px) / dx)
	if abs(dy) > 1e-9:
		t_values.append((0 - py) / dy)
		t_values.append((height - 1 - py) / dy)
	candidates = []
	for t in t_values:
		x, y = px + t * dx, py + t * dy
		if -1e-6 <= x <= width - 1 + 1e-6 and -1e-6 <= y <= height - 1 + 1e-6:
			candidates.append((t, (x, y)))
	if len(candidates) < 2:
		return None
	candidates.sort(key=lambda c: c[0])
	return candidates[0][1], candidates[-1][1]


# --------------------------- drawing ---------------------------

def label_anchor_and_angle(corners, head):
	"""Anchor the label at whichever of the box's two edges meeting at its topmost corner is
	closer to horizontal - at that edge's left-most point if it slopes down-left from the top
	corner ('left'), or at the top corner itself if it slopes down-right ('right'). Choosing by
	closeness-to-horizontal (rather than the old approach of picking whichever long edge sits
	higher on screen) keeps the label's rotation close to horizontal and its outward push
	reliably clear of the box, instead of sometimes landing a near-vertical label overlapping it.

	Returns (anchor, angle_deg, align) - align tells draw_rotated_text which of its own corners
	to pin to `anchor` so the label reads outward along this edge starting at the anchor."""
	top_idx = min(range(4), key=lambda i: corners[i][1])
	top = corners[top_idx]
	neighbors = (corners[(top_idx - 1) % 4], corners[(top_idx + 1) % 4])

	def edge_angle(p1, p2):
		dx, dy = p2[0] - p1[0], p2[1] - p1[1]
		angle_deg = math.degrees(math.atan2(dy, dx))
		if angle_deg > 90:
			angle_deg -= 180
		elif angle_deg <= -90:
			angle_deg += 180
		return angle_deg

	best_angle, best_anchor, best_align = None, None, None
	for n in neighbors:
		angle_deg = edge_angle(top, n)
		is_left = n[0] < top[0]
		anchor = min((top, n), key=lambda p: p[0]) if is_left else top
		if best_angle is None or abs(angle_deg) < abs(best_angle):
			best_angle, best_anchor, best_align = angle_deg, anchor, ('left' if is_left else 'right')

	return best_anchor, best_angle, best_align


def _alpha_paste(img, overlay_bgra, x, y):
	h_img, w_img = img.shape[:2]
	oh, ow = overlay_bgra.shape[:2]
	x0, y0 = max(0, x), max(0, y)
	x1, y1 = min(w_img, x + ow), min(h_img, y + oh)
	if x0 >= x1 or y0 >= y1:
		return
	ox0, oy0 = x0 - x, y0 - y
	ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)
	roi = img[y0:y1, x0:x1]
	ov = overlay_bgra[oy0:oy1, ox0:ox1]
	alpha = ov[:, :, 3:4].astype(np.float32) / 255.0
	roi[:] = (roi.astype(np.float32) * (1 - alpha) + ov[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)


def draw_rotated_text(img, text, anchor, angle_deg, color, align='left', font_scale=0.5, thickness=1, offset=4):
	"""Draw `text` rotated to angle_deg, pinning one corner of its (rotated) bounding box to
	`anchor` - the bottom-most corner if align=='left', the left-most corner if align=='right' -
	so the label starts exactly at the OBB corner given by label_anchor_and_angle and reads
	outward along that edge, offset a few pixels clear of the box along the outward normal.

	`text` is normally a plain string drawn in `color`, but may instead be a list of
	(substring, color) pairs - e.g. a primary/secondary class label where each part should keep
	its own class colour - concatenated left-to-right on one line before rotation."""
	# normalize to a plain 3-channel BGR tuple - callers may pass malformed/short color tuples
	# (e.g. a disabled class stream's unparsed placeholder colour), which cv2's own drawing
	# functions tolerate but the explicit color[0]/[1]/[2] indexing below does not.
	def _norm(c):
		return tuple((list(c) + [0, 0, 0])[:3])
	segments = text if isinstance(text, list) else [(text, color)]
	segments = [(t, _norm(c)) for t, c in segments]
	font = cv2.FONT_HERSHEY_SIMPLEX
	sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t, _ in segments]
	tw = sum(w for w, h in sizes)
	th = max((h for w, h in sizes), default=0)
	_, baseline = cv2.getTextSize(''.join(t for t, _ in segments), font, font_scale, thickness)
	pad = 4
	canvas_w, canvas_h = tw + 2 * pad, th + baseline + 2 * pad
	canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
	cv2.rectangle(canvas, (0, 0), (canvas_w - 1, canvas_h - 1), (0, 0, 0, 255), -1)
	cx = pad
	for (t, c), (w, h) in zip(segments, sizes):
		cv2.putText(canvas, t, (cx, pad + th), font, font_scale,
					(int(c[0]), int(c[1]), int(c[2]), 255), thickness, cv2.LINE_AA)
		cx += w

	center = (canvas_w / 2.0, canvas_h / 2.0)
	M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
	cos_a = abs(M[0, 0]); sin_a = abs(M[0, 1])
	new_w = int(canvas_h * sin_a + canvas_w * cos_a)
	new_h = int(canvas_h * cos_a + canvas_w * sin_a)
	M[0, 2] += (new_w / 2.0) - center[0]
	M[1, 2] += (new_h / 2.0) - center[1]
	rotated = cv2.warpAffine(canvas, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

	# where each local (un-rotated) canvas corner lands in the rotated buffer, so we can pin the
	# correct one to `anchor` instead of centering the whole label on it
	local_corners = ((0.0, 0.0), (canvas_w, 0.0), (0.0, canvas_h), (canvas_w, canvas_h))
	rot_corners = [(M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]) for x, y in local_corners]
	pin = max(rot_corners, key=lambda p: p[1]) if align == 'left' else min(rot_corners, key=lambda p: p[0])

	angle_rad = math.radians(angle_deg)
	nx, ny = -math.sin(angle_rad), -math.cos(angle_rad)  # outward ("up") normal of the edge
	ax, ay = anchor
	top_left_x = int(ax + nx * offset - pin[0])
	top_left_y = int(ay + ny * offset - pin[1])
	_alpha_paste(img, rotated, top_left_x, top_left_y)


def draw_label_with_background(img, segments, x, y, font_scale=0.5, thickness=1, bg_color=(0, 0, 0)):
	"""Draw a black-backed, horizontal label at axis-aligned position (x, y) (box's top-left,
	y = box top edge). `segments` is a list of (text, color) pairs drawn left-to-right on one
	line - e.g. a primary/secondary class label where each part keeps its own class colour."""
	font = cv2.FONT_HERSHEY_SIMPLEX
	sizes = [cv2.getTextSize(text, font, font_scale, thickness)[0] for text, _ in segments]
	label_w = sum(w for w, h in sizes)
	label_h = max((h for w, h in sizes), default=0)
	cv2.rectangle(img, (x - thickness, y - label_h - thickness * 4), (x + label_w + thickness * 2, y), bg_color, -1)
	cx = x
	for (text, color), (w, h) in zip(segments, sizes):
		cv2.putText(img, text, (cx, y - thickness * 2), font, font_scale, color, thickness, cv2.LINE_AA)
		cx += w
	return label_w, label_h


def draw_oriented_box(img, corners, head_pt, color, thickness=1, label=None, label_color=None):
	pts = np.array(corners, dtype=np.int32)
	cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
	cv2.circle(img, (int(head_pt[0]), int(head_pt[1])), int(max(3, thickness * 2) * 1.5), color, -1, lineType=cv2.LINE_AA)
	if label:
		anchor, angle_deg, align = label_anchor_and_angle(corners, head_pt)
		draw_rotated_text(img, label, anchor, angle_deg, label_color or color, align=align)


# --------------------------- cropping ---------------------------

def crop_black_masked(img, corners):
	"""Axis-aligned bbox crop with everything outside the rotated polygon painted black."""
	x1, y1, x2, y2 = axis_aligned_bounds(corners)
	h_img, w_img = img.shape[:2]
	x1, y1 = max(0, x1), max(0, y1)
	x2, y2 = min(w_img, x2), min(h_img, y2)
	if x2 <= x1 or y2 <= y1:
		return None
	crop = img[y1:y2, x1:x2].copy()
	local_corners = np.array([(px - x1, py - y1) for (px, py) in corners], dtype=np.int32)
	mask = np.zeros(crop.shape[:2], dtype=np.uint8)
	cv2.fillPoly(mask, [local_corners], 255)
	return cv2.bitwise_and(crop, crop, mask=mask)


def crop_rotated_upright(img, head, tail, width):
	"""Rotation-corrected crop: head at the left edge, tail at the right edge, no padding."""
	hx, hy = head; tx, ty = tail
	length = math.hypot(hx - tx, hy - ty)
	if length < 1.0 or width < 1.0:
		return None
	cx, cy = (hx + tx) / 2.0, (hy + ty) / 2.0
	theta = math.atan2(hy - ty, hx - tx)
	angle_deg = math.degrees(theta) - 180.0  # rotates tail->head to point along -x (head ends up on the left)
	M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
	h_img, w_img = img.shape[:2]
	rotated = cv2.warpAffine(img, M, (w_img, h_img), flags=cv2.INTER_LINEAR)
	x1 = int(round(cx - length / 2.0)); y1 = int(round(cy - width / 2.0))
	x2 = int(round(cx + length / 2.0)); y2 = int(round(cy + width / 2.0))
	x1, y1 = max(0, x1), max(0, y1)
	x2, y2 = min(w_img, x2), min(h_img, y2)
	if x2 <= x1 or y2 <= y1:
		return None
	return rotated[y1:y2, x1:x2]


def rotate_90_variants(img):
	"""Return (img rotated 90 CW, 180, 270 CW) - the three 'wrong' orientations of a correctly
	upright crop, used to build the null class for the whole-crop orientation classifier."""
	return (
		cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
		cv2.rotate(img, cv2.ROTATE_180),
		cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
	)


def swap_axis_hypothesis(head, tail, width):
	"""Return the perpendicular-axis interpretation of a rotated box: swaps which of the box's
	two dimensions is treated as the head-tail (length) axis vs the width. Needed because a raw
	OBB detection's longer dimension is not necessarily the true head-tail axis - a wide-bodied
	animal can be wider than it is long, e.g. wings spread perpendicular to the body."""
	cx, cy = (head[0] + tail[0]) / 2.0, (head[1] + tail[1]) / 2.0
	length = math.hypot(head[0] - tail[0], head[1] - tail[1])
	if length < 1e-6:
		return (cx, cy), (cx, cy), width
	ux, uy = (head[0] - tail[0]) / length, (head[1] - tail[1]) / length
	px, py = -uy, ux
	half_w = width / 2.0
	new_head = (cx + px * half_w, cy + py * half_w)
	new_tail = (cx - px * half_w, cy - py * half_w)
	return new_head, new_tail, length


def best_head_tail_orientation(crop_img, head, tail, width, orient_conf_fn):
	"""Resolve both the head/tail direction AND which axis is actually the head-tail axis, by
	trying all 4 hypotheses (original axis x2 directions, perpendicular axis x2 directions) and
	keeping whichever one's whole canonical upright-crop scores highest under
	orient_conf_fn(img) -> float (confidence that the crop is in the 'correct' orientation, per
	a whole-crop orientation classifier - see BehaveAI_classify_track.py's head/tail training).
	Falls back to the original (head, tail, width) if no candidate produces a usable crop."""
	head2, tail2, width2 = swap_axis_hypothesis(head, tail, width)
	candidates = [
		(head, tail, width),
		(tail, head, width),
		(head2, tail2, width2),
		(tail2, head2, width2),
	]
	best = None
	best_conf = -1.0
	for h, t, w in candidates:
		upright = crop_rotated_upright(crop_img, h, t, w)
		if upright is None or upright.size == 0:
			continue
		conf = orient_conf_fn(upright)
		if conf > best_conf:
			best_conf = conf
			best = (h, t, w)
	return best if best is not None else (head, tail, width)
