#!/usr/bin/env python3
"""
Shared Kalman-filter multi-object tracker, used by BehaveAI_classify_track.py and
BehaveAI_live.py so the two stay behaviourally identical (same pattern as box_geometry.py /
index_annotations.py).

All distance and noise parameters are resolved per-track against each track's own
EMA-smoothed body-scale estimate (an object's detected box size, e.g. its length), not a
single global pixel constant - so objects of different apparent size in the same scene
(different distance from camera) each get an effective threshold proportional to their own
size instead of sharing one absolute-pixel value tuned for a single scale.
"""

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class KalmanTracker:
	def __init__(self, match_distance_ratio, delete_after_missed,
				 process_noise_pos_ratio, process_noise_vel_ratio, measurement_noise_ratio,
				 merge_aware=True, merge_size_ratio=1.6, merge_gate_mult=2.2,
				 missed_growth=0.3, growth_cap=3.0, scale_ema_alpha=0.3, prune_ratio=0.5,
				 frames_per_step=1):
		"""frames_per_step: raw video frames elapsed between consecutive update() calls (i.e.
		the caller's frame_skip+1). match_distance_ratio/process_noise_*_ratio are calibrated
		as "per single raw video frame" body-length fractions - if a project only processes
		every Nth frame (frame_skip>0), an object genuinely covers ~N times as much ground
		between two consecutive update() calls, so those motion-derived quantities are scaled
		by frames_per_step here rather than needing to be manually retuned per project.
		measurement_noise_ratio is about per-detection precision, not elapsed time, so it is
		not scaled."""
		self.next_id = 1
		self.tracks = {}  # tid -> {'kf': KalmanFilter, 'missed': int, 'scale': float}
		self.match_distance_ratio = match_distance_ratio * frames_per_step
		self.delete_after_missed = delete_after_missed
		self.pnp_ratio = process_noise_pos_ratio * frames_per_step
		self.pnv_ratio = process_noise_vel_ratio * frames_per_step
		self.mn_ratio = measurement_noise_ratio
		self.merge_aware = merge_aware
		self.merge_size_ratio = merge_size_ratio
		self.merge_gate_mult = merge_gate_mult
		self.missed_growth = missed_growth
		self.growth_cap = growth_cap
		self.scale_ema_alpha = scale_ema_alpha
		self.prune_ratio = prune_ratio

	def _create_kf(self, initial_pt, scale):
		# Create a 4D state (x, y, vx, vy) Kalman Filter measuring (x, y).
		kf = cv2.KalmanFilter(4, 2)
		# State transition: x' = x + vx, y' = y + vy
		kf.transitionMatrix = np.array([[1, 0, 1, 0],
										[0, 1, 0, 1],
										[0, 0, 1, 0],
										[0, 0, 0, 1]], dtype=np.float32)
		# Measurement: we only observe x, y
		kf.measurementMatrix = np.array([[1, 0, 0, 0],
										 [0, 1, 0, 0]], dtype=np.float32)
		self._set_noise(kf, scale)
		# Initialize state
		kf.statePre  = np.array([[initial_pt[0]],
								 [initial_pt[1]],
								 [0.],
								 [0.]], dtype=np.float32)
		kf.statePost = kf.statePre.copy()
		return kf

	def _set_noise(self, kf, scale):
		# process/measurement noise scale with the track's own body size, so a filter tracking
		# a small/distant object doesn't get the same absolute pixel-variance as a large/close one
		pnp = (self.pnp_ratio * scale) ** 2
		pnv = (self.pnv_ratio * scale) ** 2
		mn = (self.mn_ratio * scale) ** 2
		kf.processNoiseCov = np.diag([pnp, pnp, pnv, pnv]).astype(np.float32)
		kf.measurementNoiseCov = (np.eye(2, dtype=np.float32) * np.float32(mn)).astype(np.float32)

	def _coast(self, tid):
		"""Commit a track's just-computed prediction as its new state without correcting it
		against any measurement - used for merge-ambiguous detections so a track keeps moving
		on its own motion model through an occlusion instead of being pulled toward another
		object's position or decaying toward deletion."""
		kf = self.tracks[tid]['kf']
		kf.statePost = kf.statePre.copy()
		kf.errorCovPost = kf.errorCovPre.copy()
		self.tracks[tid]['missed'] = 0

	def predict_all(self):
		"""
		Predict the next position for every track.
		Returns list of (tid, predicted_pt).
		"""
		preds = []
		for tid, tr in self.tracks.items():
			self._set_noise(tr['kf'], tr['scale'])
			pred = tr['kf'].predict()
			preds.append((tid, (float(pred[0, 0]), float(pred[1, 0]))))
		return preds

	def _gate(self, tid, det_size):
		"""Scale-relative association distance for (track, detection): the match radius is
		match_distance_ratio times the average of the track's own smoothed size and the
		detection's size, widened further the longer the track has gone unmatched."""
		tr = self.tracks[tid]
		eff_scale = 0.5 * (tr['scale'] + det_size)
		growth = min(self.growth_cap, 1.0 + tr['missed'] * self.missed_growth)
		return self.match_distance_ratio * eff_scale * growth

	def _prune_duplicate_tracks(self):
		"""
		Merge any two tracks whose current posteriors are very close (relative to their own
		body scale). Call this at the end of update().
		"""
		tids = list(self.tracks.keys())
		posts = {}
		for tid in tids:
			sp = self.tracks[tid]['kf'].statePost
			posts[tid] = (float(sp[0, 0]), float(sp[1, 0]))
		to_drop = set()
		for i, t1 in enumerate(tids):
			x1, y1 = posts[t1]
			for t2 in tids[i+1:]:
				x2, y2 = posts[t2]
				avg_scale = 0.5 * (self.tracks[t1]['scale'] + self.tracks[t2]['scale'])
				if np.hypot(x1-x2, y1-y2) < self.prune_ratio * avg_scale:
					# mark the higher ID for deletion
					to_drop.add(max(t1, t2))
		for tid in to_drop:
			del self.tracks[tid]

	def update(self, detections):
		"""detections: list of (x, y, size) - size is the detection's own estimated body scale
		(see box_geometry.detection_scale). Returns a dict: detection_index -> track_id."""

		# 1) Predict all tracks forward one step
		preds = self.predict_all()  # list of (tid, (px, py))

		# 1b) Merge-aware pass: identify pairs of tracks that are already close to EACH OTHER
		# (not just each independently near some detection - a single fast-moving object's own
		# motion-stream box can legitimately balloon well past its resting size as its trailing
		# tail extends, so detection size alone is not a reliable merge signal in a scene with
		# several unrelated nearby objects) and, only for such a pair, check whether there is a
		# detection near their shared midpoint that is anomalously large relative to their own
		# scale - i.e. looks like YOLO fusing the two of them into one box, rather than a real
		# single-object detection. Treat that pair as "still present" (reset missed, so they
		# aren't deleted) without correcting their state against the fused box's position, which
		# would otherwise pull one track toward the midpoint of both real objects and corrupt it
		# while the other decays untouched. The detection itself is excluded from normal
		# matching (it neither corrects a track nor spawns a new one).
		merge_coasted = set()
		if self.merge_aware and len(preds) >= 2:
			pred_pos = dict(preds)
			remaining_idx = list(range(len(detections)))
			tids = [t for t, _ in preds]
			for i in range(len(tids)):
				ta = tids[i]
				if ta in merge_coasted:
					continue
				for j in range(i + 1, len(tids)):
					tb = tids[j]
					if tb in merge_coasted:
						continue
					pa, pb = pred_pos[ta], pred_pos[tb]
					avg_scale = 0.5 * (self.tracks[ta]['scale'] + self.tracks[tb]['scale'])
					if np.hypot(pa[0] - pb[0], pa[1] - pb[1]) >= self.merge_gate_mult * avg_scale:
						continue  # these two tracks aren't close enough to plausibly be occluding each other
					mid = (0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]))
					for k in remaining_idx:
						dx, dy, dsz = detections[k]
						if (np.hypot(dx - mid[0], dy - mid[1]) < self.merge_gate_mult * avg_scale
								and dsz > self.merge_size_ratio * avg_scale):
							self._coast(ta); self._coast(tb)
							merge_coasted.add(ta); merge_coasted.add(tb)
							remaining_idx.remove(k)
							break
					if ta in merge_coasted:
						break
			detections = [detections[k] for k in remaining_idx]
			preds = [(tid, p) for tid, p in preds if tid not in merge_coasted]

		track_ids = [t[0] for t in preds]
		pred_pts   = [t[1] for t in preds]

		# 2) Build cost matrix = Euclidean distance
		if pred_pts and detections:
			cost = np.zeros((len(pred_pts), len(detections)), dtype=np.float32)
			for i, p in enumerate(pred_pts):
				for j, d in enumerate(detections):
					cost[i, j] = np.hypot(p[0] - d[0], p[1] - d[1])
			row_idx, col_idx = linear_sum_assignment(cost)
		else:
			row_idx = np.array([], dtype=int)
			col_idx = np.array([], dtype=int)

		assigned_detects = {}
		matched_tracks = set()
		matched_dets   = set()

		# 3) Associate tracks <-> detections
		for r, c in zip(row_idx, col_idx):
			tid = track_ids[r]
			dpt = detections[c]
			if cost[r, c] < self._gate(tid, dpt[2]):
				matched_tracks.add(tid)
				matched_dets.add(c)
				assigned_detects[c] = tid

				# Correct KF with the detection measurement
				meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])
				self.tracks[tid]['kf'].correct(meas)
				self.tracks[tid]['missed'] = 0
				sc = self.tracks[tid]['scale']
				self.tracks[tid]['scale'] = (1 - self.scale_ema_alpha) * sc + self.scale_ema_alpha * dpt[2]

		# 4) Process unassigned detections
		for i, dpt in enumerate(detections):
			if i in matched_dets:
				continue

			# try to find an existing track under the threshold
			best_tid, best_dist = None, float('inf')
			for tid, (px, py) in preds:
				if tid in matched_tracks:
					continue
				d = np.hypot(dpt[0]-px, dpt[1]-py)
				if d < best_dist:
					best_dist, best_tid = d, tid

			if best_tid is not None and best_dist < self._gate(best_tid, dpt[2]):
				assigned_detects[i] = best_tid
				self.tracks[best_tid]['missed'] = 0
				meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])
				self.tracks[best_tid]['kf'].correct(meas)
				sc = self.tracks[best_tid]['scale']
				self.tracks[best_tid]['scale'] = (1 - self.scale_ema_alpha) * sc + self.scale_ema_alpha * dpt[2]
				matched_tracks.add(best_tid)  # Add to matched tracks

			else:
				# New track
				tid = self.next_id
				kf = self._create_kf((dpt[0], dpt[1]), dpt[2])
				self.tracks[tid] = {'kf': kf, 'missed': 0, 'scale': dpt[2]}
				assigned_detects[i] = tid
				matched_tracks.add(tid)  # Add to matched tracks
				self.next_id += 1

		# 5) Handle unmatched tracks (merge-coasted tracks were already reset above and are
		# excluded here so they don't also get a missed-count increment)
		for tid in list(self.tracks.keys()):
			if tid not in matched_tracks and tid not in merge_coasted:
				self.tracks[tid]['missed'] += 1

				# Remove track if missed too many times
				if self.tracks[tid]['missed'] > self.delete_after_missed:
					del self.tracks[tid]

		self._prune_duplicate_tracks()

		return assigned_detects
