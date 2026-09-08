#!/usr/bin/env python3
"""
Regenerate motion annotation images for a BehaveAI project.

Usage:
	python regenerate_motion_dataset.py <path/to/BehaveAI_settings.ini>
or:
	python regenerate_motion_dataset.py		# will prompt for INI via file dialog

This script:
 - reads the settings INI (and resolves relative paths relative to the INI's directory)
 - rebuilds motion images (annot_motion/images/{train,val}) using the same processing
   as the annotation tool (sampling a small window of frames, computing diffs, chromatic tail, etc.)
 - applies masks and blocking boxes in the same way as your annotator
"""
import cv2
import os
import numpy as np
import configparser
import glob
import sys
import time
from collections import deque
import box_geometry as bg

# optional GUI prompt if INI not supplied
try:
	import tkinter as tk
	from tkinter import filedialog, messagebox
	_HAS_TK = True
except Exception:
	_HAS_TK = False

# -----------------------
# Helpers: path resolve / config loader
# -----------------------

def resolve_project_path(project_dir, value, fallback):
	"""Resolve a path specified in the INI: absolute or relative to project_dir."""
	if value is None or str(value).strip() == '':
		value = fallback
	value = str(value)
	if os.path.isabs(value):
		return os.path.normpath(value)
	return os.path.normpath(os.path.join(project_dir, value))


def load_config(config_path):
	"""
	Read configuration from config_path and return (params_dict, clips_dir_resolved).
	params contains numeric / strategy settings used by the image generation pipeline.
	clips_dir_resolved is an absolute (or normalized) path to the clips directory resolved
	relative to the INI's project directory.
	"""
	config = configparser.ConfigParser()
	config.optionxform = str  # preserve case
	config.read(config_path)

	project_dir = os.path.dirname(os.path.abspath(config_path))

	params = {}
	try:
		# Read parameters (same names as your previous implementation)
		params['scale_factor'] = float(config['DEFAULT'].get('scale_factor', '1.0'))
		params['expA'] = float(config['DEFAULT'].get('expA', '0.5'))
		params['expB'] = float(config['DEFAULT'].get('expB', '0.8'))
		params['strategy'] = config['DEFAULT'].get('strategy', 'exponential')
		params['chromatic_tail_only'] = config['DEFAULT'].get('chromatic_tail_only', 'false').lower()
		params['lum_weight'] = float(config['DEFAULT'].get('lum_weight', '0.7'))
		params['rgb_multipliers'] = [float(x) for x in config['DEFAULT'].get('rgb_multipliers', '2,2,2').split(',')]
		params['frame_skip'] = int(config['DEFAULT'].get('frame_skip', '0'))
		params['motion_threshold'] = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))
		params['motion_blocks_static'] = config['DEFAULT'].get('motion_blocks_static', 'false').lower()
		params['static_blocks_motion'] = config['DEFAULT'].get('static_blocks_motion', 'false').lower()
		params['save_empty_frames'] = config['DEFAULT'].get('save_empty_frames', 'false').lower()
		params['oriented'] = config['DEFAULT'].get('box_shape', 'boxes').lower() == 'oriented'

		# class name lists, needed only to map a label file's class index back to the primary
		# class name (for locating/regenerating that class's secondary-crop and orientation
		# datasets) - '0' is the sentinel for "this stream is disabled" (see CLAUDE.md conventions)
		static_classes = [c.strip() for c in config['DEFAULT'].get('primary_static_classes', '0').split(',')]
		motion_classes = [c.strip() for c in config['DEFAULT'].get('primary_motion_classes', '0').split(',')]
		params['primary_static_classes'] = [] if (not static_classes or static_classes[0] == '0') else static_classes
		params['primary_motion_classes'] = [] if (not motion_classes or motion_classes[0] == '0') else motion_classes

		# Compute base frame window size (number of sampled frames)
		base_window = 4
		if params['strategy'] == 'exponential':
			if params['expA'] > 0.2 or params['expB'] > 0.2:
				base_window = 5
			if params['expA'] > 0.5 or params['expB'] > 0.5:
				base_window = 10
			if params['expA'] > 0.7 or params['expB'] > 0.7:
				base_window = 15
			if params['expA'] > 0.8 or params['expB'] > 0.8:
				base_window = 20
			if params['expA'] > 0.9 or params['expB'] > 0.9:
				base_window = 45

		params['base_frame_window'] = base_window
		params['frame_window'] = base_window * (params['frame_skip'] + 1)

	except KeyError as e:
		raise KeyError(f"Missing configuration parameter: {e}")

	# Resolve clips_dir relative to project_dir (fallback 'clips')
	clips_dir_ini = config['DEFAULT'].get('clips_dir', 'clips')
	clips_dir = resolve_project_path(project_dir, clips_dir_ini, 'clips')

	return params, clips_dir


# -----------------------
# Image processing helpers (unchanged logic besides small improvements)
# -----------------------

def generate_base_images(video_path, frame_num, params):
	"""
	Generate static and motion images for a specific video frame.
	frame_num is interpreted as the LAST frame of the motion window to mimic the annotator.
	Returns (static_img_bgr, motion_img_bgr) or (None, None) on failure.
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Error opening video: {video_path}")
		return None, None

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if total_frames <= 0:
		print(f"Video appears empty or unreadable: {video_path}")
		cap.release()
		return None, None

	step = params['frame_skip'] + 1
	base_N = params.get('base_frame_window', 4)

	# compute start so last appended index should equal frame_num
	start_frame = int(frame_num - (base_N - 1) * step)
	start_frame = max(0, start_frame)
	if start_frame > total_frames - 1:
		print(f"Start frame {start_frame} beyond video length ({total_frames}) for {video_path}")
		cap.release()
		return None, None

	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	collected = []
	read_count = 0
	idx = start_frame
	# safety limit: don't try more than frame_window + some slack
	max_reads = params['frame_window'] + 10

	while len(collected) < base_N and idx <= total_frames - 1 and read_count <= max_reads:
		ret, frame = cap.read()
		if not ret:
			break
		if (read_count % step) == 0:
			if params['scale_factor'] != 1.0:
				frame = cv2.resize(frame, None, fx=params['scale_factor'], fy=params['scale_factor'])
			collected.append(frame.copy())
		read_count += 1
		idx += 1

	if not collected:
		cap.release()
		print(f"Could not collect frames for target {frame_num} (start {start_frame}) in {video_path}")
		return None, None

	# Process collected frames to produce diffs for the last frame
	prev_frames = [None] * 3
	static_img = None
	diffs = None
	gray = None

	for i, f in enumerate(collected):
		if f is None:
			continue
		frame_bgr = f
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

		if static_img is None:
			static_img = frame_bgr.copy()
			prev_frames = [gray.copy()] * 3
			continue

		current_diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]

		if params['strategy'] == 'exponential':
			prev_frames[0] = gray
			prev_frames[1] = cv2.addWeighted(prev_frames[1], params['expA'], gray, 1 - params['expA'], 0)
			prev_frames[2] = cv2.addWeighted(prev_frames[2], params['expB'], gray, 1 - params['expB'], 0)
		elif params['strategy'] == 'sequential':
			prev_frames[2] = prev_frames[1]
			prev_frames[1] = prev_frames[0]
			prev_frames[0] = gray

		static_img = frame_bgr.copy()
		diffs = current_diffs

	if diffs is None or gray is None:
		cap.release()
		print(f"Insufficient frames to compute diffs for {frame_num} (collected {len(collected)} frames)")
		return None, None

	# Build motion image (chromatic tail or normal)
	if params['chromatic_tail_only'] == 'true':
		tb = cv2.subtract(diffs[0], diffs[1])
		tr = cv2.subtract(diffs[2], diffs[1])
		tg = cv2.subtract(diffs[1], diffs[0])

		blue = cv2.addWeighted(gray, params['lum_weight'], tb, params['rgb_multipliers'][2], params['motion_threshold'])
		green = cv2.addWeighted(gray, params['lum_weight'], tg, params['rgb_multipliers'][1], params['motion_threshold'])
		red = cv2.addWeighted(gray, params['lum_weight'], tr, params['rgb_multipliers'][0], params['motion_threshold'])
	else:
		blue = cv2.addWeighted(gray, params['lum_weight'], diffs[0], params['rgb_multipliers'][2], params['motion_threshold'])
		green = cv2.addWeighted(gray, params['lum_weight'], diffs[1], params['rgb_multipliers'][1], params['motion_threshold'])
		red = cv2.addWeighted(gray, params['lum_weight'], diffs[2], params['rgb_multipliers'][0], params['motion_threshold'])

	motion_img = cv2.merge([blue, green, red]).astype(np.uint8)

	cap.release()
	return static_img, motion_img


def read_mask_file(mask_path):
	boxes = []
	if os.path.exists(mask_path):
		with open(mask_path, 'r') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) == 4:
					try:
						boxes.append(tuple(map(int, parts)))
					except Exception:
						pass
	return boxes


def apply_grey_boxes(image, boxes):
	result = image.copy()
	for (x1, y1, x2, y2) in boxes:
		cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), -1)
	return result


def apply_blocking_boxes(image, boxes, oriented=False):
	"""Grey-fill regions from the *other* stream's boxes so it doesn't contaminate this
	stream's training image. Oriented boxes are polygon-filled (cuts the corners, consistent
	with the black-masked secondary crops elsewhere); axis-aligned boxes fill their rectangle."""
	result = image.copy()
	if oriented:
		for corners in boxes:
			cv2.fillPoly(result, [np.array(corners, dtype=np.int32)], (128, 128, 128))
	else:
		for (x1, y1, x2, y2) in boxes:
			cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), -1)
	return result


def get_blocking_boxes(label_path, img_w, img_h, oriented=False):
	"""Return either axis-aligned (x1,y1,x2,y2) tuples or lists of 4 (x,y) pixel corners,
	depending on the project's box_shape."""
	boxes = []
	if not os.path.exists(label_path):
		return boxes
	with open(label_path, 'r') as f:
		for line in f:
			parsed = bg.read_label_line(line, oriented)
			if parsed is None:
				continue
			if oriented:
				boxes.append(bg.denorm_corners(parsed['corners_n'], img_w, img_h))
			else:
				x1, y1, x2, y2 = bg.denorm_axis(parsed['xc'], parsed['yc'], parsed['bw'], parsed['bh'], img_w, img_h)
				boxes.append((x1, y1, x2, y2))
	return boxes


def read_boxes_with_class(label_path, cls_offset, img_w, img_h, oriented):
	"""Read a primary label file back into pixel-space boxes, offsetting motion-stream class
	indices back into the combined primary_classes index space (mirrors how the annotator
	subtracts len(primary_static_classes) when writing motion labels, in reverse)."""
	boxes = []
	if not os.path.exists(label_path):
		return boxes
	with open(label_path, 'r') as f:
		for line in f:
			parsed = bg.read_label_line(line, oriented)
			if parsed is None:
				continue
			box = {'cls': parsed['cls'] + cls_offset}
			if oriented:
				box['corners'] = bg.denorm_corners(parsed['corners_n'], img_w, img_h)
			else:
				box['coords'] = bg.denorm_axis(parsed['xc'], parsed['yc'], parsed['bw'], parsed['bh'], img_w, img_h)
			boxes.append(box)
	return boxes


def find_matching_crop(class_dir, video_label, frame_num, x1, y1, tol=3):
	"""Find an existing secondary-crop file for this box under class_dir/<secondary_class>/,
	matched by its embedded x1,y1 (tolerant of the odd 1px rounding difference that can occur
	reconstructing pixel coords from a label file's normalized floats, vs. the raw ints used at
	annotation-save time). Returns (secondary_class_name, path), or (None, None) if no match -
	which just means this box was never assigned a secondary class (or hierarchical mode was off
	when it was annotated)."""
	pattern = os.path.join(class_dir, '*', f"{video_label}_{frame_num}_*_*.jpg")
	best, best_dist = None, None
	for path in glob.glob(pattern):
		stem = os.path.splitext(os.path.basename(path))[0]
		parts = stem.split('_')
		try:
			fy1, fx1 = int(parts[-1]), int(parts[-2])
		except ValueError:
			continue
		dist = abs(fx1 - x1) + abs(fy1 - y1)
		if dist <= tol and (best_dist is None or dist < best_dist):
			best_dist = dist
			best = (os.path.basename(os.path.dirname(path)), path)
	return best if best else (None, None)


def regenerate_crops_for_frame(video_label, frame_num, img_w, img_h, static_label_path, motion_label_path,
								static_final, motion_final, params):
	"""Regenerate the secondary-classifier crop datasets (annot_static_crop/annot_motion_crop) and,
	for oriented projects, the head/tail orientation datasets (annot_static_ornt/annot_motion_ornt),
	so they stay in lockstep with the regenerated primary images. A box's secondary-class
	assignment isn't stored in the primary label file, so it's recovered by matching the box's
	regenerated pixel position against whichever existing crop file it corresponds to on disk -
	only crops that already exist get overwritten; none are newly created here."""
	oriented = params['oriented']
	static_classes = params['primary_static_classes']
	motion_classes = params['primary_motion_classes']
	primary_classes = static_classes + motion_classes

	boxes = read_boxes_with_class(static_label_path, 0, img_w, img_h, oriented)
	boxes += read_boxes_with_class(motion_label_path, len(static_classes), img_w, img_h, oriented)

	for box in boxes:
		if box['cls'] >= len(primary_classes):
			continue
		primary_class_name = primary_classes[box['cls']]

		if oriented:
			corners = box['corners']
			x1, y1, _, _ = bg.axis_aligned_bounds(corners)
		else:
			x1, y1, x2, y2 = box['coords']

		# --- secondary hierarchical crops ---
		for crop_root, ann_frame in (('annot_static_crop', static_final), ('annot_motion_crop', motion_final)):
			if ann_frame is None:
				continue
			class_dir = os.path.join(crop_root, primary_class_name)
			if not os.path.isdir(class_dir):
				continue
			secondary_class_name, existing_path = find_matching_crop(class_dir, video_label, frame_num, x1, y1)
			if existing_path is None:
				continue
			if oriented:
				crop = bg.crop_black_masked(ann_frame, corners)
			else:
				crop = ann_frame[y1:y2, x1:x2]
			if crop is None or crop.size == 0:
				continue
			cv2.imwrite(existing_path, crop)
			print(f"  Regenerated secondary crop ({crop_root}/{primary_class_name}/{secondary_class_name}): {os.path.basename(existing_path)}")

		# --- orientation ('correct'/'null') crops, oriented projects only ---
		if oriented:
			head, tail, width = bg.head_tail_width_from_corners(corners)
			for ornt_root, ann_frame in (('annot_static_ornt', static_final), ('annot_motion_ornt', motion_final)):
				if ann_frame is None:
					continue
				class_dir = os.path.join(ornt_root, primary_class_name)
				correct_path = os.path.join(class_dir, 'correct', f"{video_label}_{frame_num}.jpg")
				if not os.path.exists(correct_path):
					continue
				upright = bg.crop_rotated_upright(ann_frame, head, tail, width)
				if upright is None or upright.size == 0:
					continue
				cv2.imwrite(correct_path, upright)
				null_dir = os.path.join(class_dir, 'null')
				for i, rotated in enumerate(bg.rotate_90_variants(upright)):
					null_path = os.path.join(null_dir, f"{video_label}_{frame_num}_{i}.jpg")
					if os.path.exists(null_path):
						cv2.imwrite(null_path, rotated)
				print(f"  Regenerated orientation crops ({ornt_root}/{primary_class_name}): {video_label}_{frame_num}")


# -----------------------
# Main regeneration function
# -----------------------

def regenerate_annotations(config_path):
	"""Regenerate motion and static images using parameters & clips_dir from config_path."""
	params, clips_dir = load_config(config_path)

	# Ensure we operate with project_dir as cwd to keep relative paths consistent
	project_dir = os.path.dirname(os.path.abspath(config_path))
	os.chdir(project_dir)

	print(f"Regenerating using INI: {config_path}")
	print(f"Using clips directory: {clips_dir}")

	# collect annotated frames from both motion and static label dirs
	base_dirs = [
		('annot_motion', ['train', 'val']),
		('annot_static', ['train', 'val'])
	]

	# collect unique base names from these directories
	base_names = set()
	for base_dir, splits in base_dirs:
		for split in splits:
			label_dir = os.path.join(base_dir, 'labels', split)
			if not os.path.exists(label_dir):
				continue
			for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
				if label_file.endswith('.mask.txt'):
					continue
				base_name = os.path.splitext(os.path.basename(label_file))[0]
				base_names.add((base_name, split, base_dir))

	print(f"Found {len(base_names)} annotated frames to process (motion + static).")

	# extensions to search for video files
	exts = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']

	# a given (base_name, split) can appear twice in base_names (once via annot_motion/labels,
	# once via annot_static/labels) - crop/orientation regeneration only needs to run once per frame
	processed_crop_frames = set()

	# process each unique frame
	for base_name, split, base_dir in sorted(base_names):
		parts = base_name.split('_')
		try:
			frame_num = int(parts[-1])
		except ValueError:
			print(f"Skipping {base_name}: trailing token is not an integer")
			continue
		video_name = '_'.join(parts[:-1])

		# find video in clips_dir
		video_path = None
		for ext in exts:
			test_path = os.path.join(clips_dir, video_name + ext)
			if os.path.exists(test_path):
				video_path = test_path
				break

		if not video_path:
			print(f"Video not found for {base_name}: looking in {clips_dir} for files named {video_name}.*")
			continue

		static_img, motion_img = generate_base_images(video_path, frame_num, params)
		if static_img is None and motion_img is None:
			print(f"  Could not generate images for {base_name}")
			continue

		# image dims (prefer static_img if available else motion_img)
		ref_img = static_img if static_img is not None else motion_img
		img_h, img_w = ref_img.shape[:2]

		# mask & label paths for both static and motion (may or may not exist)
		static_mask_path = os.path.join('annot_static', 'masks', split, f"{base_name}.mask.txt")
		motion_mask_path = os.path.join('annot_motion', 'masks', split, f"{base_name}.mask.txt")

		static_mask_boxes = read_mask_file(static_mask_path)
		motion_mask_boxes = read_mask_file(motion_mask_path)

		static_label_path = os.path.join('annot_static', 'labels', split, f"{base_name}.txt")
		motion_label_path = os.path.join('annot_motion', 'labels', split, f"{base_name}.txt")

		# Build the grey/blocking-applied "final" images unconditionally (cheap - just copies plus
		# box drawing) so they're available below for crop/orientation regeneration regardless of
		# which label dir (base_dir) triggered this iteration.
		static_final = None
		if static_img is not None:
			static_final = apply_grey_boxes(static_img.copy(), static_mask_boxes)
			if params.get('motion_blocks_static', 'false') == 'true':
				static_block_boxes = get_blocking_boxes(motion_label_path, img_w, img_h, oriented=params['oriented'])
				static_final = apply_blocking_boxes(static_final, static_block_boxes, oriented=params['oriented'])

		motion_final = None
		if motion_img is not None:
			motion_final = apply_grey_boxes(motion_img.copy(), motion_mask_boxes)
			if params.get('static_blocks_motion', 'false') == 'true':
				static_boxes_for_block = get_blocking_boxes(static_label_path, img_w, img_h, oriented=params['oriented'])
				motion_final = apply_blocking_boxes(motion_final, static_boxes_for_block, oriented=params['oriented'])

		# -----------------------
		# Process static images (save into annot_static/images/<split>/)
		# -----------------------
		# We regenerate static images when:
		#  - the entry came from annot_static (base_dir == 'annot_static'), OR
		#  - save_empty_frames == 'true' (keep parity with motion logic)
		if base_dir == 'annot_static' or params['save_empty_frames'] == 'true':
			if static_final is None:
				print(f"  No static image for {base_name}")
			else:
				static_img_path = os.path.join('annot_static', 'images', split, f"{base_name}.jpg")
				os.makedirs(os.path.dirname(static_img_path), exist_ok=True)
				cv2.imwrite(static_img_path, static_final)
				print(f"Regenerated static: {static_img_path}")

		# -----------------------
		# Process motion images (save into annot_motion/images/<split>/)
		# -----------------------
		# Keep original behaviour: regenerate motion when entry came from annot_motion
		# or when save_empty_frames is enabled.
		if base_dir == 'annot_motion' or params['save_empty_frames'] == 'true':
			if motion_final is None:
				print(f"  No motion image for {base_name}")
			else:
				motion_img_path = os.path.join('annot_motion', 'images', split, f"{base_name}.jpg")
				os.makedirs(os.path.dirname(motion_img_path), exist_ok=True)
				cv2.imwrite(motion_img_path, motion_final)
				print(f"Regenerated motion: {motion_img_path}")

		# -----------------------
		# Process secondary-crop and orientation-crop datasets (once per unique frame)
		# -----------------------
		crop_key = (base_name, split)
		if crop_key not in processed_crop_frames and (static_final is not None or motion_final is not None):
			processed_crop_frames.add(crop_key)
			regenerate_crops_for_frame(video_name, frame_num, img_w, img_h, static_label_path, motion_label_path,
										static_final, motion_final, params)

	print("Regeneration loop complete.")



# -----------------------
# CLI & prompt logic
# -----------------------

def choose_ini_path_via_dialog():
	if not _HAS_TK:
		return None
	root = tk.Tk()
	root.withdraw()
	path = filedialog.askopenfilename(title="Select BehaveAI settings INI", filetypes=[("INI files", "*.ini"), ("All files", "*.*")])
	root.destroy()
	return path

if __name__ == "__main__":
	# Determine config_path from command-line or prompt
	if len(sys.argv) > 1:
		arg = os.path.abspath(sys.argv[1])
		if os.path.isdir(arg):
			config_path = os.path.join(arg, "BehaveAI_settings.ini")
		else:
			config_path = arg
	else:
		config_path = choose_ini_path_via_dialog()
		if not config_path:
			# no selection: report and exit
			print("No settings INI selected — exiting.")
			sys.exit(0)

	config_path = os.path.abspath(config_path)
	if not os.path.exists(config_path):
		print(f"Config file not found: {config_path}")
		sys.exit(1)

	# Run regeneration
	start_t = time.time()
	regenerate_annotations(config_path)
	elapsed = time.time() - start_t
	print(f"Regeneration complete! Elapsed {elapsed:.1f} s")
