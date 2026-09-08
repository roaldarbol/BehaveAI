import cv2
import numpy as np
import csv
import os
import glob
from ultralytics import YOLO
import configparser
import time
import shutil
import tkinter as tk
from tkinter import messagebox, filedialog
import subprocess
# ~ import config_watcher
import sys
import box_geometry as bg
from kalman_tracker import KalmanTracker


# --- NCNN helper utilities -----------------------


def ncnn_dir_for_weights(weights_path):
	"""Return the expected NCNN export directory for a given .pt path."""
	base, ext = os.path.splitext(weights_path)
	# Ultralytics export typically creates a folder named like "<base>_ncnn_model"
	return base + "_ncnn_model"

def ncnn_files_exist(ncnn_dir):
	"""Return True if NCNN param+bin appear to exist in the export dir."""
	if not os.path.isdir(ncnn_dir):
		return False
	# Look for .param and .bin files (ncnn export creates *.param and *.bin)
	has_param = any(f.endswith(".param") for f in os.listdir(ncnn_dir))
	has_bin = any(f.endswith(".bin") for f in os.listdir(ncnn_dir))
	return has_param and has_bin

def ensure_ncnn_export(weights_path, task, timeout=300):
	"""
	Ensure an NCNN conversion exists for weights_path.
	Returns the ncnn_dir on success, None on failure (falls back to .pt).
	This will skip conversion if the ncnn folder already exists.
	"""
	ncnn_dir = ncnn_dir_for_weights(weights_path)
	if ncnn_files_exist(ncnn_dir):
		return ncnn_dir

	try:
		print(f"Exporting {weights_path} -> NCNN (this may take a while)...")
		model = YOLO(weights_path, task=task)
		# Use Ultralytics export API. This creates the folder "<base>_ncnn_model".
		# Some installs can be slow; we try and catch errors below.
		model.export(format="ncnn")
		# Wait a short time for files to appear (export is synchronous in most versions).
		start = time.time()
		while time.time() - start < timeout:
			if ncnn_files_exist(ncnn_dir):
				print(f"NCNN export complete: {ncnn_dir}")
				return ncnn_dir
			time.sleep(0.5)
		# timed out
		print(f"NCNN export timeout for {weights_path}")
		return None
	except Exception as e:
		# Don't crash — export can fail on some systems; print useful debugging info and return None
		print(f"Warning: NCNN export failed for {weights_path}: {e}")
		return None

def load_model_with_ncnn_preference(weights_path, task):
	"""
	Attempt to use NCNN if available (or convert it). If conversion or loading fails,
	fall back to the original PyTorch .pt path.
	Returns a YOLO model instance (which may wrap NCNN or .pt).
	"""
	# If a .pt was not provided (maybe already a folder), just try loading directly
	if not weights_path.endswith(".pt"):
		try:
			return YOLO(weights_path, task=task)
		except Exception as e:
			print(f"Error loading model {weights_path}: {e}")
			raise

	ncnn_dir = ncnn_dir_for_weights(weights_path)
	# prefer existing NCNN dir if present
	if ncnn_files_exist(ncnn_dir):
		try:
			print(f"Loading NCNN model from {ncnn_dir}")
			return YOLO(ncnn_dir, task=task)
		except Exception as e:
			print(f"Failed to load NCNN model at {ncnn_dir}: {e} (falling back to .pt)")

	# Otherwise attempt conversion (one-time). If it fails, fall back to .pt.
	exported = ensure_ncnn_export(weights_path, task)
	if exported:
		try:
			return YOLO(exported, task=task)
		except Exception as e:
			print(f"Failed to load NCNN-exported model {exported}: {e} (falling back to .pt)")

	# Finally, fallback to direct .pt load
	print(f"Using original weights (PyTorch) at {weights_path}")
	return YOLO(weights_path, task=task)
# --------------------------------------------------------------------



def move_to_expected(project_path, run_name="train", runs_root="runs"):
    """
    If a YOLOv2x-style run was just created under runs/.../<run_name>,
    move that run/<run_name> directory into project_path/<run_name>.
    Returns the destination path on success, or None on failure / nothing found.
    """
    # look first in runs/detect/**/train then in runs/**/train
    candidates = glob.glob(os.path.join(runs_root, "detect", "**", run_name), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(runs_root, "**", run_name), recursive=True)

    # keep only directories
    candidates = [p for p in candidates if os.path.isdir(p)]
    if not candidates:
        return None

    # pick most recently modified candidate
    candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
    src_train = candidates[0]                     # e.g. runs/detect/2026-02-24_train
    dst_train = os.path.join(project_path, run_name)  # e.g. model_primary_motion/train

    try:
        # remove existing destination so the move yields the expected layout
        if os.path.exists(dst_train):
            try:
                shutil.rmtree(dst_train)
            except Exception:
                pass

        shutil.move(src_train, dst_train)

        # best-effort: remove any now-empty ancestor dirs under runs_root
        runs_root_abs = os.path.abspath(runs_root)
        parent = os.path.abspath(os.path.dirname(src_train))
        # remove upward until we hit runs_root or a non-empty dir
        while parent.startswith(runs_root_abs):
            try:
                if os.path.isdir(parent) and not os.listdir(parent):
                    shutil.rmtree(parent)
                    parent = os.path.dirname(parent)
                else:
                    break
            except Exception:
                break

        print(f"Moved YOLO training output: '{src_train}' -> '{dst_train}'")
        return dst_train
    except Exception as e:
        print(f"Warning: failed to move YOLO run folder '{src_train}' -> '{dst_train}': {e}")
        return None



# ---------- Project-aware configuration loading --------------------------

def pick_ini_via_dialog():
	root = tk.Tk()
	root.withdraw()
	path = filedialog.askopenfilename(
		title="Select BehaveAI settings INI",
		filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
	)
	root.destroy()
	return path

# Determine config_path (accept project dir or direct INI path)
if len(sys.argv) > 1:
	arg = os.path.abspath(sys.argv[1])
	if os.path.isdir(arg):
		config_path = os.path.join(arg, "BehaveAI_settings.ini")
	else:
		config_path = arg
else:
	config_path = pick_ini_via_dialog()
	if not config_path:
		tk.messagebox.showinfo("No settings file", "No settings INI selected — exiting.")
		sys.exit(0)

config_path = os.path.abspath(config_path)
if not os.path.exists(config_path):
	tk.messagebox.showerror("Missing settings", f"Configuration file not found: {config_path}")
	sys.exit(1)

# Set project directory to the INI parent and make it the working directory
project_dir = os.path.dirname(config_path)
os.chdir(project_dir)
print(f"Working directory set to project dir: {project_dir}")
print(f"Using settings file: {config_path}")

# Load configuration
config = configparser.ConfigParser()
config.optionxform = str  # keep case
config.read(config_path)

# Helper: resolve a path from INI (absolute or relative to project_dir)
def resolve_project_path(value, fallback):
	if value is None or str(value).strip() == '':
		value = fallback
	value = str(value)
	if os.path.isabs(value):
		return os.path.normpath(value)
	return os.path.normpath(os.path.join(project_dir, value))

# Read dataset / directory keys from INI (defaults are relative names inside the project)
clips_dir_ini = config['DEFAULT'].get('clips_dir', 'clips')
input_dir_ini = config['DEFAULT'].get('input_dir', 'input')
output_dir_ini = config['DEFAULT'].get('output_dir', 'output')

clips_dir = resolve_project_path(clips_dir_ini, 'clips')
input_folder = resolve_project_path(input_dir_ini, 'input')
output_folder = resolve_project_path(output_dir_ini, 'output')


# Read parameters
try:
	
	primary_motion_classes = [name.strip() for name in config['DEFAULT']['primary_motion_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('primary_motion_colors', '').split(';') if c.strip()]
	primary_motion_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
	primary_motion_hotkeys = [key.strip() for key in config['DEFAULT']['primary_motion_hotkeys'].split(',')]
	
	secondary_motion_classes = [name.strip() for name in config['DEFAULT']['secondary_motion_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('secondary_motion_colors', '').split(';') if c.strip()]
	secondary_motion_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
	secondary_motion_hotkeys = [key.strip() for key in config['DEFAULT']['secondary_motion_hotkeys'].split(',')]
	
	primary_static_classes = [name.strip() for name in config['DEFAULT']['primary_static_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('primary_static_colors', '').split(';') if c.strip()]
	primary_static_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
	primary_static_hotkeys = [key.strip() for key in config['DEFAULT']['primary_static_hotkeys'].split(',')]
	
	secondary_static_classes = [name.strip() for name in config['DEFAULT']['secondary_static_classes'].split(',')]
	cols = [c.strip() for c in config['DEFAULT'].get('secondary_static_colors', '').split(';') if c.strip()]
	secondary_static_colors = [tuple(map(int, c.split(',')))[::-1] for c in cols]
	secondary_static_hotkeys = [key.strip() for key in config['DEFAULT']['secondary_static_hotkeys'].split(',')]

	if len(secondary_motion_classes) >= 2 or len(secondary_static_classes) >= 2:
		hierarchical_mode = True
		motion_cropped_base_dir = 'annot_motion_crop'
		static_cropped_base_dir = 'annot_static_crop'
		
		# secondary classes need more than one value, so clear if there's only one value
		if len(secondary_motion_classes) == 1:
			secondary_motion_classes = []
			secondary_motion_colors = []
			secondary_motion_hotkeys = []
					
		if len(secondary_static_classes) == 1:
			secondary_static_classes = []
			secondary_static_colors = []
			secondary_static_hotkeys = []

	else: hierarchical_mode = False


	primary_classes = primary_static_classes + primary_motion_classes
	primary_colors = primary_static_colors + primary_motion_colors
	primary_hotkeys = primary_static_hotkeys + primary_motion_hotkeys
	
	secondary_classes = secondary_static_classes + secondary_motion_classes
	secondary_colors = secondary_static_colors + secondary_motion_colors
	secondary_hotkeys = secondary_static_hotkeys + secondary_motion_hotkeys

	primary_static_project_path = 'model_primary_static'
	primary_static_model_path = os.path.join('model_primary_static', "train", "weights", "best.pt")
	primary_static_yaml_path = 'static_annotations.yaml'
	
	primary_motion_project_path = 'model_primary_motion'
	primary_motion_model_path = os.path.join('model_primary_motion', "train", "weights", "best.pt")
	primary_motion_yaml_path = 'motion_annotations.yaml'
	
	ignore_secondary = [name.strip() for name in config['DEFAULT']['ignore_secondary'].split(',')]
	dominant_source = config['DEFAULT']['dominant_source'].lower()

	primary_classifier = config['DEFAULT'].get('primary_classifier', 'yolo11s.pt') 
	primary_epochs = int(config['DEFAULT'].get('primary_epochs', '50'))
	secondary_classifier = config['DEFAULT'].get('secondary_classifier', 'yolo11s-cls.pt')  
	secondary_epochs = int(config['DEFAULT'].get('secondary_epochs', '50'))

	if hierarchical_mode:
		secondary_static_project_path = 'model_secondary_static'
		secondary_static_data_path = 'annot_static_crop'
		secondary_static_model_path = os.path.join('model_secondary_static', "train", "weights", "best.pt")
		
		secondary_motion_project_path = 'model_secondary_motion'
		secondary_motion_data_path = 'annot_motion_crop'
		secondary_motion_model_path = os.path.join('model_secondary_motion', "train", "weights", "best.pt")
		
		secondary_class_ids = list(range(len(secondary_classes)))
		paired = list(zip(secondary_classes, secondary_colors, secondary_class_ids, secondary_hotkeys))
		paired_sorted = sorted(paired, key=lambda x: x[0].lower())
		secondary_classes, secondary_colors, secondary_class_ids, secondary_hotkeys = zip(*paired_sorted)
		# Convert back to lists
		secondary_classes = list(secondary_classes)
		secondary_colors = list(secondary_colors)	
		secondary_class_ids = list(secondary_class_ids)
		secondary_hotkeys = list(secondary_hotkeys)
	
	# Common parameters
	scale_factor = float(config['DEFAULT'].get('scale_factor', '1.0'))
	expA = float(config['DEFAULT'].get('expA', '0.5'))
	expB = float(config['DEFAULT'].get('expB', '0.8'))
	lum_weight = float(config['DEFAULT'].get('lum_weight', '0.7'))
	strategy = config['DEFAULT'].get('strategy', 'exponential')
	chromatic_tail_only = config['DEFAULT']['chromatic_tail_only'].lower()
	rgb_multipliers = [float(x) for x in config['DEFAULT']['rgb_multipliers'].split(',')]
	use_ncnn = config['DEFAULT']['use_ncnn'].lower()
	primary_conf_thresh = float(config['DEFAULT'].get('primary_conf_thresh', '0.5'))
	secondary_conf_thresh = float(config['DEFAULT'].get('secondary_conf_thresh', '0.5'))
	# tracking thresholds are ratios of each track/detection's own estimated body scale
	# (box_geometry.detection_scale), not absolute pixels - see kalman_tracker.py
	match_distance_ratio = float(config['DEFAULT'].get('match_distance_ratio', '2.5'))
	delete_after_missed = float(config['DEFAULT'].get('delete_after_missed', '15'))
	centroid_merge_ratio = float(config['DEFAULT'].get('centroid_merge_ratio', '0.7'))
	iou_thresh = float(config['DEFAULT'].get('iou_thresh', '0.95'))
	merge_aware = config['DEFAULT'].get('merge_aware', 'true').lower() == 'true'
	line_thickness = int(config['DEFAULT'].get('line_thickness', '1'))
	font_size = float(config['DEFAULT'].get('font_size', '0.5'))
	frame_skip = int(config['DEFAULT'].get('frame_skip', '0'))

	process_noise_pos_ratio = float(config['kalman'].get('process_noise_pos_ratio', '0.05'))
	process_noise_vel_ratio = float(config['kalman'].get('process_noise_vel_ratio', '0.1'))
	measurement_noise_ratio = float(config['kalman'].get('measurement_noise_ratio', '0.1'))
	motion_threshold = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))

	oriented = config['DEFAULT'].get('box_shape', 'boxes').lower() == 'oriented'
	disambiguate_head_tail = oriented and config['DEFAULT'].get('disambiguate_head_tail', 'false').lower() == 'true'
	static_ornt_base_dir = 'annot_static_ornt'
	motion_ornt_base_dir = 'annot_motion_ornt'

except KeyError as e:
	raise KeyError(f"Missing configuration parameter: {e}")


# Validate configuration

if len(primary_motion_classes) != len(primary_motion_colors) or len(primary_motion_classes) != len(primary_motion_hotkeys):
	raise ValueError("Primary motion classes, colors and hotkeys must match in configuration.")
if len(secondary_motion_classes) != len(secondary_motion_colors) or len(secondary_motion_classes) != len(secondary_motion_hotkeys):
	raise ValueError("Secondary motion classes, colors and hotkeys must match in configuration.")
if len(primary_static_classes) != len(primary_static_colors) or len(primary_static_classes) != len(primary_static_hotkeys):
	raise ValueError("Primary static classes, colors and hotkeys must match in configuration.")
if len(secondary_static_classes) != len(secondary_static_colors) or len(secondary_static_classes) != len(secondary_static_hotkeys):
	raise ValueError("Secondary static classes, colors and hotkeys must match in configuration.")
if dominant_source != 'motion' and dominant_source != 'static' and dominant_source != 'confidence':
	raise ValueError("dominant_source must be motion, static, or confidence")

if len(primary_static_classes) > 0:
	if not os.path.exists(primary_static_yaml_path):
		print(f"Error: Primary static YAML file not found. Run the Annotation script once to fix this")
		sys.exit(1)

if len(primary_motion_classes) > 0:
	if not os.path.exists(primary_motion_yaml_path):
		print(f"Error: Primary motion YAML file not found. Run the Annotation script once to fix this")
		sys.exit(1)


# ~ # check whether settings have been changed, and motion annotation library needs rebuilding 
# ~ settings_changed = config_watcher.check_settings_changed(current_config_path=config_path, saved_config_path=None, model_dirs=['model_primary_motion'])
# ~ # Globals for prompting/behaviour inside maybe_retrain
# ~ regen_prompt_shown = False
# ~ force_rebuild_motion = False


global_response = 0 # if 'yes' is selected for any model re-training, retraining should be perfoemd for all models

def count_images_in_dataset(path):
	## Count images in a dataset, handling both YAML-based and directory-based datasets
	# If path is a YAML file (primary models)
	if path.endswith('.yaml'):
		try:
			import yaml
			with open(path, 'r') as f:
				data = yaml.safe_load(f)
			
			# Get the path to the training images
			train_path = data['train']
			base_dir = os.path.dirname(path)
			abs_train_path = os.path.join(base_dir, train_path)
			
			# Handle different dataset formats
			if abs_train_path.endswith('.txt'):
				# Text file with image paths
				with open(abs_train_path, 'r') as f:
					return len(f.readlines())
			else:
				# Directory with images
				image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
				return len([f for f in os.listdir(abs_train_path) 
							if os.path.splitext(f)[1].lower() in image_exts])
		except Exception as e:
			print(f"Error counting images: {e}")
			return 0
	
	# If path is a directory (secondary models)
	elif os.path.isdir(path):
		total_count = 0
		image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
		
		# Walk through all subdirectories
		for root, dirs, files in os.walk(path):
			# Only count files in leaf directories (class directories)
			if not dirs:  # This is a leaf directory (no subdirectories)
				count = sum(1 for f in files 
						   if os.path.splitext(f)[1].lower() in image_exts)
				total_count += count
				
		return total_count
	
	else:
		print(f"Unsupported dataset format: {path}")
		return 0


def maybe_retrain(model_type, yaml_path, project_path, model_path, classifier, epochs, imgsz, extra_train_kwargs=None):
	"""
	Decide whether to (re)train a model based on existence and image counts.
	- If model_path exists and the recorded train_count differs from the current dataset,
	  prompt the user to retrain (Yes/No).
	- If model_path does not exist, perform first-time training.
	extra_train_kwargs, if given, are merged into the model.train(...) call (e.g. to override
	augmentation defaults for a specific model type such as the head/tail classifier).
	Returns True if a training run was performed, False otherwise.
	"""

	# Determine whether this is a motion model by naming
	is_motion_model = ('motion' in model_type.lower()) or ('secondary_motion' in project_path.lower()) or ('primary_motion' in project_path.lower())

	# If model exists: compare recorded image count (train_count.txt) with current dataset
	if os.path.exists(model_path):
		if os.path.exists(os.path.join(project_path, 'train_count.txt')):
			try:
				with open(os.path.join(project_path, 'train_count.txt'), 'r') as f:
					last_count = int(f.read().strip())
			except Exception:
				last_count = -1
		else:
			last_count = -1

		current_count = count_images_in_dataset(yaml_path)

		if current_count != last_count:
			# Ask user whether to retrain
			root = tk.Tk(); root.withdraw()
			msg = (
				f"New annotations detected for '{model_type}' model.\n"
				f"Image count changed from {last_count} to {current_count}.\n\n"
				"Do you want to re-train this model?"
			)
			response = messagebox.askyesno("Retrain model?", msg)
			root.destroy()

			if response:
				# Backup existing model dir/project and retrain from its weights
				backup_dir = project_path + "_backup"
				i = 1
				while os.path.exists(f"{backup_dir}{i}"):
					i += 1
				final_backup = f"{backup_dir}{i}"
				try:
					shutil.copytree(project_path, final_backup)
					print(f"Existing model copied to {final_backup}")
				except Exception as e:
					print(f"Warning: failed to backup {project_path}: {e}")

				start_weights = os.path.join(final_backup, "train", "weights", "best.pt")
				print(f'Training new {model_type} model using existing weights...')
				model = YOLO(start_weights)
				model.train(
					data=yaml_path,
					epochs=epochs,
					imgsz=imgsz,
					project=project_path,
					name="train",
					exist_ok=True,
					**(extra_train_kwargs or {})
				)
				move_to_expected(project_path, run_name="train", runs_root="runs")
				print(f'Done training {model_type} model')
				# Update saved train count
				with open(os.path.join(project_path, 'train_count.txt'), 'w') as f:
					f.write(str(current_count))
				# copy existing settings ini file for reference (so you know which settings were used for each model)
				os.makedirs(project_path, exist_ok=True)
				# ~ dst = os.path.join(project_path, os.path.basename(config_path))
				dst = os.path.join(project_path, 'saved_settings.ini')
				try:
					shutil.copy2(config_path, dst)
					print(f"Saved settings snapshot to {dst}")
				except Exception as e:
					print(f"Warning: could not copy settings to model dir: {e}")
				return True

		# else counts match -> nothing to do
		return False

	else:
		# Model missing -> do first-time training
		print(f'{model_type} model not found, building it...')
		model = YOLO(classifier)
		model.train(
			data=yaml_path,
			epochs=epochs,
			imgsz=imgsz,
			project=project_path,
			name="train",
			exist_ok=True,
			**(extra_train_kwargs or {})
		)
		move_to_expected(project_path, run_name="train", runs_root="runs")
		print(f'Done training {model_type} model')

		current_count = count_images_in_dataset(yaml_path)
		os.makedirs(project_path, exist_ok=True)
		with open(os.path.join(project_path, 'train_count.txt'), 'w') as f:
			f.write(str(current_count))

		# copy existing settings ini file for reference (so you know which settings were used for each model)
		os.makedirs(project_path, exist_ok=True)
		# ~ dst = os.path.join(project_path, os.path.basename(config_path))
		dst = os.path.join(project_path, 'saved_settings.ini')
		try:
			shutil.copy2(config_path, dst)
			print(f"Saved settings snapshot to {dst}")
		except Exception as e:
			print(f"Warning: could not copy settings to model dir: {e}")

		return True


# Train secondary classifiers for each static class
secondary_static_models = None
secondary_motion_models = None

if hierarchical_mode:
	secondary_static_models = {}
	static_class_map = [[None] * len(secondary_classes) for _ in range(len(primary_classes))]
	if len(secondary_static_classes) >= 2:
		for primary_class in primary_classes:
			idx = primary_classes.index(primary_class)
			hotkey = primary_hotkeys[idx]
			if hotkey in secondary_hotkeys: 
				continue
				
			if primary_class in ignore_secondary:
				continue
			
			data_dir = os.path.join(secondary_static_data_path, primary_class)
			# Skip if directory doesn't exist
			if not os.path.isdir(data_dir):
				continue
			
			# Create model directory for this static class
			model_dir = f"model_secondary_static_{primary_class}"
			weights_path = os.path.join(model_dir, "train", "weights", "best.pt")
			
			maybe_retrain(model_dir, data_dir, model_dir, 
				weights_path, secondary_classifier, secondary_epochs, 224)

			# Load the trained model
			if use_ncnn == 'true':
				secondary_static_models[primary_class] = load_model_with_ncnn_preference(weights_path, "classify")
			else:
				secondary_static_models[primary_class] = YOLO(weights_path)


		# ~ print(f"secondary_static_models {secondary_static_models}")
		
	secondary_motion_models = {}
	motion_class_map = [[None] * len(secondary_classes) for _ in range(len(primary_classes))]
	if len(secondary_motion_classes) >= 2:
		for primary_class in primary_classes:
			idx = primary_classes.index(primary_class)
			hotkey = primary_hotkeys[idx]
			if hotkey in secondary_hotkeys: 
				continue
			
			if primary_class in ignore_secondary:
				continue			
			
			data_dir = os.path.join(secondary_motion_data_path, primary_class)
			# Skip if directory doesn't exist
			if not os.path.isdir(data_dir):
				continue
			
			# Create model directory for this static class
			model_dir = f"model_secondary_motion_{primary_class}"
			weights_path = os.path.join(model_dir, "train", "weights", "best.pt")
			
			maybe_retrain(model_dir, data_dir, model_dir, 
				weights_path, secondary_classifier, secondary_epochs, 224)

			# Load the trained model
			if use_ncnn == 'true':
				secondary_motion_models[primary_class] = load_model_with_ncnn_preference(weights_path, "classify")
			else:
				secondary_motion_models[primary_class] = YOLO(weights_path)
				
		# ~ print(f"secondary_motion_models {secondary_motion_models}")

# Train/load per-primary-class whole-crop orientation classifiers (resolves the mod-180
# orientation ambiguity inherent to OBB detections - see box_geometry.crop_rotated_upright).
# 2 classes: 'correct' (the upright head-left/tail-right crop as saved) and 'null' (that same
# crop rotated 90/180/270 degrees - see box_geometry.rotate_90_variants). Deliberately whole-crop
# rather than head-vs-tail half-crop, so the classifier can use the entire animal's shape to
# judge orientation, not just a locally-distinct head marking.
ornt_static_models = None
ornt_motion_models = None

# These crops are already rotation-normalized at save time, so the usual classify-task
# augmentation defaults are actively harmful here: Ultralytics' classify pipeline ignores
# `degrees` entirely (that only applies to detect/OBB/segment) and instead injects rotation/shear
# via `auto_augment='randaugment'` (torchvision RandAugment, no bounded magnitude control) -
# verified against the installed ultralytics source. A left-right (horizontal) flip is also
# unsafe: it swaps 'correct' with what would look like the 'right'-rotated null example. A
# top-bottom (vertical) flip is fine though - it only ever moves between the 'up' and 'down'
# rotations, both already inside the null bucket, so it never crosses the correct/null boundary.
ORNT_TRAIN_KWARGS = {'auto_augment': None, 'fliplr': 0.0, 'flipud': 0.5}

if disambiguate_head_tail:
	ornt_static_models = {}
	ornt_motion_models = {}
	for primary_class in primary_classes:
		if primary_static_classes[0] != '0':
			data_dir = os.path.join(static_ornt_base_dir, primary_class)
			if os.path.isdir(data_dir):
				model_dir = f"model_ornt_static_{primary_class}"
				weights_path = os.path.join(model_dir, "train", "weights", "best.pt")
				maybe_retrain(model_dir, data_dir, model_dir, weights_path, secondary_classifier, secondary_epochs, 224,
							  extra_train_kwargs=ORNT_TRAIN_KWARGS)
				if use_ncnn == 'true':
					ornt_static_models[primary_class] = load_model_with_ncnn_preference(weights_path, "classify")
				else:
					ornt_static_models[primary_class] = YOLO(weights_path)

		if primary_motion_classes[0] != '0':
			data_dir = os.path.join(motion_ornt_base_dir, primary_class)
			if os.path.isdir(data_dir):
				model_dir = f"model_ornt_motion_{primary_class}"
				weights_path = os.path.join(model_dir, "train", "weights", "best.pt")
				maybe_retrain(model_dir, data_dir, model_dir, weights_path, secondary_classifier, secondary_epochs, 224,
							  extra_train_kwargs=ORNT_TRAIN_KWARGS)
				if use_ncnn == 'true':
					ornt_motion_models[primary_class] = load_model_with_ncnn_preference(weights_path, "classify")
				else:
					ornt_motion_models[primary_class] = YOLO(weights_path)


def _orient_conf(model, img):
	"""Probability the given whole crop is in the 'correct' (head-left, tail-right) orientation,
	per the per-primary-class orientation classifier."""
	if img is None or img.size == 0:
		return 0.0
	res = model.predict(img, verbose=False)
	probs = res[0].probs
	if probs is None:
		return 0.0
	correct_idx = None
	for idx, name in model.names.items():
		if name == 'correct':
			correct_idx = idx
			break
	if correct_idx is None:
		return 0.0
	return float(probs.data[correct_idx])


def _disambiguate_head_tail(crop_img, head, tail, width, primary_class, is_motion):
	"""Resolve both the mod-180 direction ambiguity AND which axis is actually the head-tail
	axis (its longer dimension isn't necessarily the true head-tail axis - e.g. a wide-bodied
	animal can be wider than it is long) using the whole-crop orientation classifier. Returns
	(head, tail, width)."""
	if not disambiguate_head_tail:
		return head, tail, width
	model = (ornt_motion_models if is_motion else ornt_static_models).get(primary_class)
	if model is None:
		return head, tail, width
	return bg.best_head_tail_orientation(crop_img, head, tail, width, lambda img: _orient_conf(model, img))


#-------CHECK PRIMARY MODEL EXISTS----------
if primary_static_classes[0] != '0':
	maybe_retrain('primary static', primary_static_yaml_path, primary_static_project_path, 
		primary_static_model_path, primary_classifier, primary_epochs, 640)


if primary_motion_classes[0] != '0':
	maybe_retrain('primary motion', primary_motion_yaml_path, primary_motion_project_path, 
		primary_motion_model_path, primary_classifier, primary_epochs, 640)


# --- PARAMETERS -----------------------------------------------------------

expA2 = 1 - expA
expB2 = 1 - expB

#input_folder = "./input/"
#output_folder = "./output/"

progress_update = 10 # print progress every n frames

iou = bg.iou  # shared axis-aligned overlap-proportion helper (box_geometry.py)
	

# --- MAIN PROCESSING -----------------------------------------------------
def process_video(file):
	os.makedirs(output_folder, exist_ok=True)
	base = os.path.splitext(os.path.basename(file))[0]
	cap = cv2.VideoCapture(file)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if not cap.isOpened(): return
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale_factor)
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale_factor)
	fps = cap.get(cv2.CAP_PROP_FPS)
	writer = cv2.VideoWriter(
		os.path.join(output_folder, base + "_detected.mp4"),
		cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
	)

	detect_task = "obb" if oriented else "detect"

	if primary_static_classes[0] != '0':
		if use_ncnn == 'true':
			model_static = load_model_with_ncnn_preference(primary_static_model_path, detect_task)
		else:
			model_static = YOLO(primary_static_model_path)

	if primary_motion_classes[0] != '0':
		if use_ncnn == 'true':
			model_motion = load_model_with_ncnn_preference(primary_motion_model_path, detect_task)
		else:
			model_motion = YOLO(primary_motion_model_path)
		
		
	tracker = KalmanTracker(match_distance_ratio, delete_after_missed,
							 process_noise_pos_ratio, process_noise_vel_ratio, measurement_noise_ratio,
							 merge_aware=merge_aware, frames_per_step=frame_skip + 1)

	prev_frames, frame_idx = None, 0
	csv_file = open(os.path.join(output_folder, base + "_tracking.csv"), 'w', newline='')
	csv_writer = csv.writer(csv_file)
	# Updated CSV header with four streams
	csv_header = [
		"frame", "id", "x", "y",
		"primary_static_class", "primary_static_conf",
		"primary_motion_class", "primary_motion_conf",
		"secondary_static_class", "secondary_static_conf",
		"secondary_motion_class", "secondary_motion_conf"
	]
	if oriented:
		csv_header += ["angle_deg", "width", "length"]
	csv_writer.writerow(csv_header)

	print(f"Processing video: {file}")
	print('Initialising')
	current_frame = 0
	print_tick = 0
	start_time = time.time()

	frame_count = 0
	
	while True:
		ret, raw_frame = cap.read()
		if not ret: break
		frame_idx += 1
		if frame_count == 0:
			if scale_factor != 1.0:
				raw_frame = cv2.resize(raw_frame, None, fx=scale_factor, fy=scale_factor)
			gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
			frame = raw_frame.copy()
			if prev_frames is None:
				prev_frames = [gray.copy() for _ in range(3)]
				continue
			
			# only process motion information if necessary
			if primary_motion_classes[0] != '0':
	
				diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]
				
				if strategy == 'exponential':
					prev_frames[0] = gray
					prev_frames[1] = cv2.addWeighted(prev_frames[1], expA, gray, expA2, 0)
					prev_frames[2] = cv2.addWeighted(prev_frames[2], expB, gray, expB2, 0)
				elif strategy == 'sequential':
					prev_frames[2] = prev_frames[1]
					prev_frames[1] = prev_frames[0]
					prev_frames[0] = gray


				if chromatic_tail_only == 'true':
					tb = cv2.subtract(diffs[0], diffs[1])	
					tr = cv2.subtract(diffs[2], diffs[1])
					tg = cv2.subtract(diffs[1], diffs[0])
							
					blue = cv2.addWeighted(gray, lum_weight, tb, rgb_multipliers[2], motion_threshold)
					green = cv2.addWeighted(gray, lum_weight, tg, rgb_multipliers[1], motion_threshold)
					red = cv2.addWeighted(gray, lum_weight, tr, rgb_multipliers[0], motion_threshold)
				else:
					blue = cv2.addWeighted(gray, lum_weight, diffs[0], rgb_multipliers[2], motion_threshold)
					green = cv2.addWeighted(gray, lum_weight, diffs[1], rgb_multipliers[1], motion_threshold)
					red = cv2.addWeighted(gray, lum_weight, diffs[2], rgb_multipliers[0], motion_threshold)

				motion_image = cv2.merge((blue, green, red)).astype(np.uint8)
	
			# Collect all primary detections
			all_detections = []
			
			# Primary static detection
			if primary_static_classes[0] != '0':
				results_static = model_static.predict(frame, conf=primary_conf_thresh, verbose=False)
				if oriented:
					obb = results_static[0].obb
					if obb is not None:
						for obb_box in obb:
							class_idx = int(obb_box.cls[0])
							class_name = primary_static_classes[class_idx]
							conf = float(obb_box.conf[0])
							coords = tuple(map(int, obb_box.xyxy[0].tolist()))
							cx_, cy_, bw_, bh_, angle_ = [float(v) for v in obb_box.xywhr[0]]
							head, tail, width = bg.head_tail_width_from_xywhr_rad(cx_, cy_, bw_, bh_, angle_)
							head, tail, width = _disambiguate_head_tail(frame, head, tail, width, class_name, is_motion=False)
							all_detections.append({
								'coords': coords,
								'primary_class': class_name,
								'primary_conf': conf,
								'source': 'static',
								'primary_class_combined': '',
								'primary_conf_combined': 0.0,
								'head': head, 'tail': tail, 'width': width,
							})
				else:
					for box in results_static[0].boxes:
						coords = tuple(map(int, box.xyxy[0].tolist()))
						class_idx = int(box.cls[0])
						class_name = primary_static_classes[class_idx]
						conf = float(box.conf[0])
						all_detections.append({
							'coords': coords,
							'primary_class': class_name,
							'primary_conf': conf,
							'source': 'static',
							'primary_class_combined': '',
							'primary_conf_combined': 0.0
						})

			# Primary motion detection
			if primary_motion_classes[0] != '0':
				results_motion = model_motion.predict(motion_image, conf=primary_conf_thresh, verbose=False)
				if oriented:
					obb = results_motion[0].obb
					if obb is not None:
						for obb_box in obb:
							class_idx = int(obb_box.cls[0])
							class_name = primary_motion_classes[class_idx]
							conf = float(obb_box.conf[0])
							coords = tuple(map(int, obb_box.xyxy[0].tolist()))
							cx_, cy_, bw_, bh_, angle_ = [float(v) for v in obb_box.xywhr[0]]
							head, tail, width = bg.head_tail_width_from_xywhr_rad(cx_, cy_, bw_, bh_, angle_)
							head, tail, width = _disambiguate_head_tail(motion_image, head, tail, width, class_name, is_motion=True)
							all_detections.append({
								'coords': coords,
								'primary_class': class_name,
								'primary_conf': conf,
								'source': 'motion',
								'primary_class_combined': '',
								'primary_conf_combined': 0.0,
								'head': head, 'tail': tail, 'width': width,
							})
				else:
					for box in results_motion[0].boxes:
						coords = tuple(map(int, box.xyxy[0].tolist()))
						class_idx = int(box.cls[0])
						class_name = primary_motion_classes[class_idx]
						conf = float(box.conf[0])
						all_detections.append({
							'coords': coords,
							'primary_class': class_name,
							'primary_conf': conf,
							'source': 'motion',
							'primary_class_combined': '',
							'primary_conf_combined': 0.0
						})
	
	
			# Merge detections from the static/motion streams into one per physical object,
			# respecting dominant_source - shared with BehaveAI_live.py and the annotation
			# tool's auto-annotate so the rule behaves identically everywhere (box_geometry.py)
			merged_detections = bg.merge_source_detections(all_detections, centroid_merge_ratio, iou_thresh, dominant_source)
	
	
	
			# Run secondary classification on each primary detection
			processed_detections = []
			for det in merged_detections:
				coords = det['coords']
				primary_class = det['primary_class']
				primary_conf = det['primary_conf']
				source = det['source']
				primary_class_combined = det['primary_class_combined']
				primary_conf_combined = det['primary_conf_combined']
	
				
				if source == 'static':
					det['primary_static_class'] = primary_class
					det['primary_static_conf'] = primary_conf
					det['primary_motion_class'] = primary_class_combined
					det['primary_motion_conf'] = primary_conf_combined
				else:
					det['primary_motion_class'] = primary_class
					det['primary_motion_conf'] = primary_conf
					det['primary_static_class'] = primary_class_combined
					det['primary_static_conf'] = primary_conf_combined
	
				if hierarchical_mode:
					x1, y1, x2, y2 = coords
					
					# Determine which secondary model to use based on source and configuration
					sec_model = None
					sec_classes = []
					crop_img = None
					
					if source == 'static':
						# Use static secondary model if configured
						if len(secondary_static_classes) >= 2:
							sec_model = secondary_static_models.get(primary_class, None)
							sec_classes = secondary_static_classes
							crop_img = frame
						# Fallback to motion secondary model if static not available
						elif len(secondary_motion_classes) >= 2:
							sec_model = secondary_motion_models.get(primary_class, None)
							sec_classes = secondary_motion_classes
							crop_img = motion_image if primary_motion_classes[0] != '0' else frame
					else:  # motion source
						# Use motion secondary model if configured
						if len(secondary_motion_classes) >= 2:
							sec_model = secondary_motion_models.get(primary_class, None)
							sec_classes = secondary_motion_classes
							crop_img = motion_image
						# Fallback to static secondary model if motion not available
						elif len(secondary_static_classes) >= 2:
							sec_model = secondary_static_models.get(primary_class, None)
							sec_classes = secondary_static_classes
							crop_img = frame
					
					# Get the cropped region
					crop = None
					if crop_img is not None:
						if oriented and det.get('head') is not None:
							corners = bg.corners_from_head_tail_width(det['head'], det['tail'], det['width'])
							crop = bg.crop_black_masked(crop_img, corners)
						else:
							crop = crop_img[y1:y2, x1:x2]

					secondary_class = primary_class
					secondary_conf = 1.0
					
					# Run secondary classification if we have a model and valid crop
					if sec_model and crop is not None and crop.size > 0:
						sec_results = sec_model.predict(crop, verbose=False)
						if sec_results[0].probs is not None:
							secondary_class_idx = sec_results[0].probs.top1
							secondary_conf = sec_results[0].probs.top1conf.item()
							secondary_class = sec_model.names[secondary_class_idx]
	
					# Add secondary results to detection
					if source == 'static':
						det['secondary_static_class'] = secondary_class
						det['secondary_static_conf'] = secondary_conf
					else:  # motion source
						det['secondary_motion_class'] = secondary_class
						det['secondary_motion_conf'] = secondary_conf
					
				processed_detections.append(det)
	
	
			# Prepare for tracking - include each detection's own scale so the tracker can
			# gate/size noise relative to body size instead of absolute pixels
			dets_for_tracker = [(d['centroid'][0], d['centroid'][1], bg.detection_scale(d))
								 for d in processed_detections]
			assignment = tracker.update(dets_for_tracker)
	
			# ~ frame = motion_image ## enable this line ot save the motion video instead of static 
	
			# Process tracked objects
			for idx, det in enumerate(processed_detections):
				tid = assignment.get(idx, None)
				if tid is None or tid not in tracker.tracks:
					continue
					
				x1, y1, x2, y2 = det['coords']
				cx, cy = det['centroid']
	
				# Get all class info with default values
				ps_class = det.get('primary_static_class', '')
				ps_conf = det.get('primary_static_conf', 0)
				pm_class = det.get('primary_motion_class', '')
				pm_conf = det.get('primary_motion_conf', 0)
				ss_class = det.get('secondary_static_class', '')
				ss_conf = det.get('secondary_static_conf', 0)
				sm_class = det.get('secondary_motion_class', '')
				sm_conf = det.get('secondary_motion_conf', 0)
				p_source = det.get('source', '')
	
	
				# Create display label
				label_parts = []
				# ~ if ps_class: 
				if p_source == 'static': 
					label_parts.append(f"{ps_class.upper()}")
					primary_cls = ps_class
				else:
					label_parts.append(f"{pm_class.upper()}")
					primary_cls = pm_class
				
				primary_col = primary_colors[primary_classes.index(primary_cls)]
				secondary_col = (255, 255, 255)
				if oriented and det.get('head') is not None:
					corners = bg.corners_from_head_tail_width(det['head'], det['tail'], det['width'])
					if hierarchical_mode:
						if sm_class != '' and sm_class != primary_cls:
							secondary_cls = sm_class
							secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
						if ss_class != '' and ss_class != primary_cls:
							secondary_cls = ss_class
							secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]

						if primary_cls in ignore_secondary:
							label = f"{tid} {primary_cls.upper()}"
							bg.draw_oriented_box(frame, corners, det['head'], primary_col, thickness=line_thickness, label=label, label_color=primary_col)
						else:
							outer_th = line_thickness + 2
							outer_corners = bg.inset_corners(corners, -outer_th)
							cv2.polylines(frame, [np.array(outer_corners, dtype=np.int32)], True, primary_col, outer_th, cv2.LINE_AA)
							# Secondary box is nudged outward and drawn a shade thicker than a bare
							# line_thickness so it isn't visually swallowed by the thicker primary ring.
							inner_th = line_thickness + 1
							inner_corners = bg.inset_corners(corners, -1)
							label = [(f"{tid} {primary_cls.upper()} ", primary_col), (secondary_cls, secondary_col)]
							bg.draw_oriented_box(frame, inner_corners, det['head'], secondary_col, thickness=inner_th, label=label)
					else:
						label = f"{tid} {primary_cls}"
						bg.draw_oriented_box(frame, corners, det['head'], primary_col, thickness=line_thickness, label=label, label_color=primary_col)
				else:
	
					if hierarchical_mode:
					
						if sm_class != '' and sm_class != primary_cls:
							secondary_cls = sm_class
							secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
						if ss_class != '' and ss_class != primary_cls:
							secondary_cls = ss_class
							secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
					
	
						if primary_cls in ignore_secondary:
							label = f"{tid} {primary_cls.upper()}"
							label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
							label_w, label_h = label_size
							cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
							cv2.rectangle(frame, (x1, y1), (x2, y2), primary_col, line_thickness)
							cv2.putText(frame, label, (x1, y1 - line_thickness*2), cv2.FONT_HERSHEY_SIMPLEX, 
										font_size, primary_col, line_thickness, cv2.LINE_AA)
						else:
							# Draw outer static box (slightly larger)
							outer_thickness = line_thickness + 2
							cv2.rectangle(frame, (x1-outer_thickness, y1-outer_thickness),
										 (x2+outer_thickness, y2+outer_thickness),
										primary_col, outer_thickness)
							# Secondary box is nudged outward and drawn a shade thicker than a bare
							# line_thickness so it isn't visually swallowed by the thicker primary ring.
							inner_thickness = line_thickness + 1
							cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), secondary_col, inner_thickness)
							bg.draw_label_with_background(frame, [(f"{tid} {primary_cls.upper()} ", primary_col), (secondary_cls, secondary_col)],
														   x1, y1, font_size, line_thickness)
					else:
						label = f"{tid} {primary_cls}"
						label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
						label_w, label_h = label_size
						cv2.rectangle(frame, (x1-line_thickness, y1 - label_h - line_thickness*4), (x1 + label_w + line_thickness*2, y1), (0, 0, 0), -1)
						cv2.rectangle(frame, (x1, y1), (x2, y2), primary_col, line_thickness)
						cv2.putText(frame, label, (x1, y1 - line_thickness*3), cv2.FONT_HERSHEY_SIMPLEX, 
									font_size, primary_col, line_thickness, cv2.LINE_AA)
	
	
	
		
				# Draw motion vector (if tracking available)
				if tid in tracker.tracks:
					state_post = tracker.tracks[tid]['kf'].statePost
					x, y = state_post[0, 0], state_post[1, 0]
					vx, vy = state_post[2, 0], state_post[3, 0]
					next_x = x + vx
					next_y = y + vy
					
					light_color = tuple(int(0.8 * ch + 0.2 * 255) for ch in primary_col)
					cv2.line(frame, (int(x), int(y)), (int(next_x), int(next_y)), primary_col, line_thickness)
					cv2.circle(frame, (int(next_x), int(next_y)), 3, light_color, -line_thickness)
					cv2.circle(frame, (int(cx), int(cy)), 3, primary_col, -line_thickness)
				
				# Write to CSV
				csv_row = [
					frame_idx, tid, cx, cy,
					ps_class, f"{ps_conf:.3f}",
					pm_class, f"{pm_conf:.3f}",
					ss_class, f"{ss_conf:.3f}",
					sm_class, f"{sm_conf:.3f}"
				]
				if oriented:
					if det.get('head') is not None:
						angle_deg, length = bg.angle_length_from_head_tail(det['head'], det['tail'])
						csv_row += [f"{angle_deg:.2f}", f"{det['width']:.2f}", f"{length:.2f}"]
					else:
						csv_row += ['', '', '']
				csv_writer.writerow(csv_row)


			# ~ # print frame number
			text_color = (255, 255, 255)  # white text
			label = str(current_frame)
			label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
			label_w, label_h = label_size
			cv2.rectangle(frame, (0, 0), 
						 (label_w + line_thickness*4, label_h + line_thickness*4), (0, 0, 0), -1)
			cv2.putText(frame, label, (line_thickness*2, label_h + line_thickness*2), 
					   cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, line_thickness)
						   
			writer.write(frame)
			
			if print_tick > progress_update:
				elapsed = time.time() - start_time
				current_fps = current_frame / elapsed if elapsed > 0 else 0
				pc_done = 100 * (frame_skip+1) * current_frame / total_frames
				print(f"Progress: {pc_done:.2f}% | {current_fps:.1f} FPS", end='\r', flush=True)
				print_tick = 0
			current_frame += 1
			print_tick += 1

		frame_count += 1
		
		if frame_count > frame_skip:
			frame_count = 0

	cap.release()
	writer.release()
	csv_file.close()
	print(f"Done processing {base} | {current_fps:.1f} FPS")

if __name__ == '__main__':
	for vid in glob.glob(os.path.join(input_folder, "*.*")):
		process_video(vid)
