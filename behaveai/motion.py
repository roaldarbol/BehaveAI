"""
Motion video conversion for BehaveAI.

Converts standard video to a chromatic motion-enhanced video where RGB channels
encode temporal distance of movement:
  - Blue:  recent motion
  - Green: medium-term motion
  - Red:   older motion
"""

import os
import sys
import glob
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from torchcodec.decoders import VideoDecoder as _VideoDecoder
    try:
        from torchcodec.decoders import set_cuda_backend as _set_cuda_backend
    except (ImportError, RuntimeError):
        _set_cuda_backend = None
    _TORCHCODEC_AVAILABLE = True
except (ImportError, RuntimeError):
    _TORCHCODEC_AVAILABLE = False


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm", ".ts", ".mpg", ".mpeg"}

# OpenCV's standard pip wheel on Windows has no H.264 encoder — all H.264 variants
# remap to avc1/openh264 which requires a DLL that is rarely present. Use mp4v
# directly on Windows to avoid noisy failed attempts.
# On Linux/macOS, try H.264 first for smaller files.
if platform.system() == "Windows":
    _CODEC_PREFERENCE = ["mp4v"]
else:
    _CODEC_PREFERENCE = ["avc1", "X264", "H264", "mp4v"]


def _is_network_path(path: str) -> bool:
    """Return True for UNC paths (\\\\server\\share or //server/share).

    TorchCodec/FFmpeg must fully index an MP4 before decoding; if the moov
    atom is at the end of the file this means reading the entire file over
    the network before a single frame is produced.
    """
    return path.startswith("\\\\") or path.startswith("//")


def _copy_with_progress(src: str, dst: str, chunk_mb: int = 64) -> None:
    """Copy a file in chunks, showing a live progress bar."""
    chunk = chunk_mb * 1024 * 1024
    total = Path(src).stat().st_size
    copied = 0
    start = time.perf_counter()
    bar_w = 30
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            buf = fsrc.read(chunk)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)
            elapsed = time.perf_counter() - start
            speed   = copied / elapsed / 1_048_576
            pct     = 100 * copied / total if total else 0
            filled  = int(bar_w * copied / total) if total else 0
            bar     = _CYAN + "#" * filled + _RESET + _DIM + "-" * (bar_w - filled) + _RESET
            tot_gb  = total  / 1_073_741_824
            done_gb = copied / 1_073_741_824
            if speed > 0 and (remaining := total - copied) > 0:
                eta_s = int(remaining / 1_048_576 / speed)
                eta_str = f"{eta_s}s" if eta_s < 60 else f"{eta_s // 60}m{eta_s % 60:02d}s"
            else:
                eta_str = "--"
            sys.stderr.write(
                f"\r  [{bar}] {done_gb:.2f}/{tot_gb:.2f} GB  {pct:5.1f}%  {speed:6.0f} MB/s  ETA {eta_str}   "
            )
            sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> str:
    """Resolve 'auto' → 'cuda'/'cpu'; validate explicit choices."""
    if device == "auto":
        return "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available; cannot use --device cuda.")
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device found; use --device auto or --device cpu.")
    return device


# ---------------------------------------------------------------------------
# Progress bar (writes directly to stderr, bypassing loguru)
# ---------------------------------------------------------------------------

_CYAN  = "\033[36m"
_DIM   = "\033[2m"
_RESET = "\033[0m"


def _progress(current: int, total: int, elapsed: float, written: int) -> None:
    pct    = 100 * current / total if total > 0 else 0
    bar_w  = 30
    filled = int(bar_w * current / total) if total > 0 else 0
    bar    = _CYAN + "#" * filled + _RESET + _DIM + "-" * (bar_w - filled) + _RESET
    width  = len(str(total))
    counter = _DIM + f"{current:{width}d}/{total}" + _RESET
    fps_str = f"{written / elapsed:5.1f} fps" if elapsed > 0 else "  --- fps"
    if elapsed > 0 and current > 0 and (remaining := total - current) > 0:
        eta_s = int(remaining * elapsed / current)
        if eta_s < 60:
            eta_str = f"{eta_s}s"
        elif eta_s < 3600:
            eta_str = f"{eta_s // 60}m{eta_s % 60:02d}s"
        else:
            eta_str = f"{eta_s // 3600}h{(eta_s % 3600) // 60:02d}m"
    else:
        eta_str = "--"
    sys.stderr.write(f"\r  [{bar}] {counter} {pct:5.1f}%  {fps_str}  ETA {eta_str}   ")
    sys.stderr.flush()


def _done_line() -> None:
    sys.stderr.write("\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# FFmpeg compression
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _nvenc_available() -> bool:
    """Return True if FFmpeg was built with hevc_nvenc support."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, check=True,
        )
        return b"hevc_nvenc" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class _FFmpegPipeWriter:
    """
    Drop-in replacement for cv2.VideoWriter that pipes raw BGR24 frames to FFmpeg.

    For GPU paths this eliminates the large intermediate mp4v file and the
    separate --compress step: frames are encoded directly to H.265 (NVENC when
    available, libx265 otherwise) as they are produced.

    The writer is intentionally minimal — it mirrors the write() / release()
    interface used by cv2.VideoWriter so it can be passed into the same
    processing functions without changes.
    """

    def __init__(self, output_path: str, w: int, h: int, fps: float, crf: int | None = None):
        if _nvenc_available():
            quality = crf if crf is not None else 28
            vcodec_args = ["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", str(quality)]
            logger.debug("FFmpegPipeWriter: hevc_nvenc  cq={}", quality)
        else:
            quality = crf if crf is not None else 26
            vcodec_args = ["-c:v", "libx265", "-crf", str(quality), "-preset", "medium"]
            logger.debug("FFmpegPipeWriter: libx265  crf={}", quality)

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            *vcodec_args,
            "-movflags", "+faststart",
            output_path,
        ]
        self._stderr_tmp = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_tmp,
        )

    def isOpened(self) -> bool:
        return self._proc.poll() is None

    def write(self, frame: "np.ndarray") -> None:
        if self._proc.poll() is not None:
            raise RuntimeError("FFmpeg process exited unexpectedly during encode")
        self._proc.stdin.write(memoryview(frame))

    def release(self) -> None:
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        retcode = self._proc.wait()
        if retcode != 0:
            self._stderr_tmp.seek(0)
            err = self._stderr_tmp.read(4096).decode(errors="replace").strip()
            logger.warning("FFmpeg pipe exited with code {}: …{}", retcode, err[-200:])
        self._stderr_tmp.close()


# ---------------------------------------------------------------------------
# Async writer thread
# ---------------------------------------------------------------------------

def _start_writer_thread(writer, buf_size: int = 4):
    """
    Write frames in a background thread, decoupling GPU compute from disk I/O.

    The caller puts (H, W, 3) uint8 numpy frames on the returned queue and a
    None sentinel when finished.  Any exception raised inside the thread is
    stored in exc_holder[0] so the caller can re-raise it on the main thread.

    Memory is bounded: the queue holds at most buf_size frames (~11 MB each at
    2560×1440) and blocks the producer when full, providing natural backpressure.
    """
    q: queue.Queue = queue.Queue(maxsize=buf_size)
    exc_holder: list = [None]

    def _writer() -> None:
        try:
            while True:
                frame = q.get()
                if frame is None:
                    logger.debug("Writer thread: sentinel received, stopping")
                    return
                writer.write(frame)
        except Exception as exc:
            exc_holder[0] = exc
            logger.debug("Writer thread: exception — {}", exc)
            # Drain so the producer can unblock from q.put() and exit cleanly
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    t = threading.Thread(target=_writer, daemon=True)
    t.start()
    logger.debug("Writer thread started (buf_size={})", buf_size)
    return q, t, exc_holder


def _compress_with_ffmpeg(path: str, crf: int | None = None) -> None:
    """Re-encode a video with FFmpeg H.265 (GPU via NVENC if available, else CPU) in-place."""
    tmp = path + ".tmp_compress.mp4"
    if _nvenc_available():
        logger.info("NVENC detected — using GPU encoding (hevc_nvenc)")
        codec_args = ["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", str(crf if crf is not None else 28)]
    else:
        logger.info("NVENC not available — using CPU encoding (libx265)")
        codec_args = ["-c:v", "libx265", "-crf", str(crf if crf is not None else 26), "-preset", "medium"]
    cmd = [
        "ffmpeg", "-y", "-i", path,
        *codec_args,
        "-c:a", "copy",
        "-movflags", "+faststart",
        tmp,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp, path)
    except subprocess.CalledProcessError as e:
        logger.warning("FFmpeg compression failed: {}", e.stderr.decode(errors='replace').strip())
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# Codec selection
# ---------------------------------------------------------------------------

def _pick_codec(output_path: str, w: int, h: int, fps: float):
    """
    Try codecs in preference order, validate by writing a test frame,
    and return (codec_name, writer) for the first one that actually works.
    """
    test_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for codec in _CODEC_PREFERENCE:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            writer.release()
            continue
        writer.write(test_frame)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return codec, writer
        writer.release()
        try:
            os.remove(output_path)
        except OSError:
            pass
    # Hard fallback — mp4v always works with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return "mp4v", cv2.VideoWriter(output_path, fourcc, fps, (w, h))


# ---------------------------------------------------------------------------
# Background frame reader thread
# ---------------------------------------------------------------------------

def _start_reader_thread(cap: cv2.VideoCapture, step: int, buf_size: int = 4):
    """
    Decode frames in a background thread, feeding a bounded queue.

    Uses cap.grab() for skipped frames (avoids full pixel decode) and
    cap.read() only for frames that will be processed. Puts
    (source_frame_idx, bgr_numpy_array) tuples; terminates with None.
    """
    q: queue.Queue = queue.Queue(maxsize=buf_size)

    def _reader() -> None:
        idx = 0
        while True:
            if idx % step == 0:
                ret, frame = cap.read()
                if not ret:
                    logger.debug("Reader: cap.read() returned no frame at idx={}, stopping", idx)
                    q.put(None)
                    return
                q.put((idx, frame))
            else:
                if not cap.grab():
                    logger.debug("Reader: cap.grab() failed at idx={}, stopping", idx)
                    q.put(None)
                    return
            idx += 1

    logger.debug("Starting reader thread (step={})", step)
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return q, t


# ---------------------------------------------------------------------------
# CPU processing loop
# ---------------------------------------------------------------------------

def _process_cpu(
    cap, writer, total_frames, w, h, scale_factor,
    strategy, exp_a, exp_b, lum_weight, rgb_multipliers,
    chromatic_tail_only, frame_skip, motion_threshold,
):
    """Optimized CPU path: threaded decode with grab(), pre-allocated diff buffers."""
    step             = frame_skip + 1
    exp_a2           = 1.0 - exp_a
    exp_b2           = 1.0 - exp_b
    threshold_offset = -abs(motion_threshold)

    # Pre-allocate diff and chromatic-tail buffers to avoid per-frame heap allocation
    diffs  = [np.empty((h, w), dtype=np.uint8) for _ in range(3)]
    chroma = [np.empty((h, w), dtype=np.uint8) for _ in range(3)]

    prev_frames = None
    written     = 0
    start_time  = time.perf_counter()

    q, reader_thread = _start_reader_thread(cap, step)
    logger.debug("CPU processing loop started")
    interrupted = False
    try:
        while True:
            try:
                item = q.get(timeout=30)
            except queue.Empty:
                logger.warning("No frame received in 30s after writing {} frames — reader thread may have stalled", written)
                continue
            if item is None:
                logger.debug("CPU loop: sentinel received, {} frames written", written)
                break
            frame_idx, raw_frame = item

            if scale_factor != 1.0:
                raw_frame = cv2.resize(raw_frame, (w, h))

            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

            if prev_frames is None:
                prev_frames = [gray, gray.copy(), gray.copy()]
                continue

            # Compute diffs into pre-allocated buffers before updating accumulators
            cv2.absdiff(prev_frames[0], gray, dst=diffs[0])
            cv2.absdiff(prev_frames[1], gray, dst=diffs[1])
            cv2.absdiff(prev_frames[2], gray, dst=diffs[2])

            if strategy == "exponential":
                prev_frames[0] = gray
                cv2.addWeighted(prev_frames[1], exp_a, gray, exp_a2, 0, dst=prev_frames[1])
                cv2.addWeighted(prev_frames[2], exp_b, gray, exp_b2, 0, dst=prev_frames[2])
            else:  # sequential
                prev_frames[2] = prev_frames[1]
                prev_frames[1] = prev_frames[0]
                prev_frames[0] = gray

            if chromatic_tail_only:
                cv2.subtract(diffs[0], diffs[1], dst=chroma[0])
                cv2.subtract(diffs[1], diffs[0], dst=chroma[1])
                cv2.subtract(diffs[2], diffs[1], dst=chroma[2])
                b_src, g_src, r_src = chroma[0], chroma[1], chroma[2]
            else:
                b_src, g_src, r_src = diffs[0], diffs[1], diffs[2]

            blue  = cv2.addWeighted(gray, lum_weight, b_src, rgb_multipliers[2], threshold_offset)
            green = cv2.addWeighted(gray, lum_weight, g_src, rgb_multipliers[1], threshold_offset)
            red   = cv2.addWeighted(gray, lum_weight, r_src, rgb_multipliers[0], threshold_offset)

            writer.write(cv2.merge((blue, green, red)))
            written   += 1

            if written % 300 == 0:
                _progress(frame_idx, total_frames, time.perf_counter() - start_time, written)
    except KeyboardInterrupt:
        interrupted = True
        logger.debug("CPU loop: interrupted after {} frames written", written)
    finally:
        # Drain the queue so the reader thread can unblock from q.put() and exit
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        reader_thread.join(timeout=2)

    return written, interrupted


# ---------------------------------------------------------------------------
# GPU processing loop (PyTorch CUDA)
# ---------------------------------------------------------------------------

def _bgr_to_gray_t(t: "torch.Tensor") -> "torch.Tensor":
    """(H, W, 3) float32 BGR tensor → (H, W) float32 luminance."""
    return 0.114 * t[..., 0] + 0.587 * t[..., 1] + 0.299 * t[..., 2]


def _rgb_to_gray_t(t: "torch.Tensor") -> "torch.Tensor":
    """(H, W, 3) float32 RGB tensor → (H, W) float32 luminance (decord output)."""
    return 0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]


def _process_gpu(
    cap, writer, total_frames, w, h, scale_factor,
    strategy, exp_a, exp_b, lum_weight, rgb_multipliers,
    chromatic_tail_only, frame_skip, motion_threshold,
    torch_device: "torch.device",
):
    """
    GPU path: threaded decode with grab(), all motion arithmetic on CUDA via PyTorch.

    prev_frames are kept as float32 CUDA tensors throughout so that exponential
    accumulation retains sub-pixel precision. Frames are uploaded once per kept
    frame and the composited result is downloaded once for cv2.VideoWriter.
    """
    step             = frame_skip + 1
    exp_a2           = 1.0 - exp_a
    exp_b2           = 1.0 - exp_b
    threshold_offset = float(-abs(motion_threshold))
    rm = float(rgb_multipliers[0])
    gm = float(rgb_multipliers[1])
    bm = float(rgb_multipliers[2])

    prev_frames = None  # list of 3 float32 CUDA tensors once initialised
    written     = 0
    start_time  = time.perf_counter()

    read_q, reader_thread = _start_reader_thread(cap, step)
    write_q, writer_thread, write_exc = _start_writer_thread(writer)
    logger.debug("GPU processing loop started (device={})", torch_device)

    _t_compute = _t_transfer = _t_queue = 0.0
    interrupted = False
    try:
        while True:
            try:
                item = read_q.get(timeout=30)
            except queue.Empty:
                logger.warning("No frame received in 30s after writing {} frames — reader thread may have stalled", written)
                continue
            if item is None:
                logger.debug("GPU loop: sentinel received, {} frames written", written)
                break
            frame_idx, raw_frame = item

            if write_exc[0] is not None:
                raise write_exc[0]

            if scale_factor != 1.0:
                raw_frame = cv2.resize(raw_frame, (w, h))

            _t0 = time.perf_counter()

            # Upload BGR frame, convert to float32 grayscale on GPU
            gray = _bgr_to_gray_t(
                torch.from_numpy(raw_frame).to(torch_device, dtype=torch.float32)
            )  # (H, W) float32

            if prev_frames is None:
                prev_frames = [gray, gray.clone(), gray.clone()]
                continue

            # Compute diffs before updating accumulators
            d0 = (prev_frames[0] - gray).abs_()
            d1 = (prev_frames[1] - gray).abs_()
            d2 = (prev_frames[2] - gray).abs_()

            if strategy == "exponential":
                prev_frames[0] = gray
                # In-place exponential decay: avoids allocating new tensors each frame
                prev_frames[1].mul_(exp_a).add_(gray, alpha=exp_a2)
                prev_frames[2].mul_(exp_b).add_(gray, alpha=exp_b2)
            else:  # sequential: Python reference swaps — no copies, no in-place ops
                prev_frames[2] = prev_frames[1]
                prev_frames[1] = prev_frames[0]
                prev_frames[0] = gray

            if chromatic_tail_only:
                b_src = (d0 - d1).clamp_min_(0.0)
                g_src = (d1 - d0).clamp_min_(0.0)
                r_src = (d2 - d1).clamp_min_(0.0)
            else:
                b_src, g_src, r_src = d0, d1, d2

            blue  = (lum_weight * gray + bm * b_src + threshold_offset).clamp_(0.0, 255.0)
            green = (lum_weight * gray + gm * g_src + threshold_offset).clamp_(0.0, 255.0)
            red   = (lum_weight * gray + rm * r_src + threshold_offset).clamp_(0.0, 255.0)

            _t1 = time.perf_counter()

            # Download to CPU as uint8
            bgr_out = torch.stack([blue, green, red], dim=-1).byte().cpu().numpy()

            _t2 = time.perf_counter()

            try:
                write_q.put(bgr_out, timeout=60)
            except queue.Full:
                if write_exc[0] is not None:
                    raise write_exc[0]
                raise RuntimeError("Writer queue full after 60s — writer thread may have stalled")

            written += 1
            _t3 = time.perf_counter()

            _t_compute  += _t1 - _t0
            _t_transfer += _t2 - _t1
            _t_queue    += _t3 - _t2

            if written % 300 == 0:
                _progress(frame_idx, total_frames, time.perf_counter() - start_time, written)
                logger.debug(
                    "Per-frame avg (last 300): compute={:.1f}ms  transfer={:.1f}ms  queue_put={:.1f}ms",
                    _t_compute / 300 * 1000, _t_transfer / 300 * 1000, _t_queue / 300 * 1000,
                )
                _t_compute = _t_transfer = _t_queue = 0.0

    except KeyboardInterrupt:
        interrupted = True
        logger.debug("GPU loop: interrupted after {} frames written", written)
    finally:
        # Drain read queue so reader thread can unblock and exit
        while True:
            try:
                read_q.get_nowait()
            except queue.Empty:
                break
        reader_thread.join(timeout=2)

        # Drain write queue on interrupt so writer thread can reach the sentinel
        if interrupted:
            while True:
                try:
                    write_q.get_nowait()
                except queue.Empty:
                    break
        try:
            write_q.put(None, timeout=5)
        except queue.Full:
            pass
        writer_thread.join(timeout=30)
        if write_exc[0] is not None and not interrupted:
            raise write_exc[0]

    return written, interrupted


# ---------------------------------------------------------------------------
# GPU processing loop — decord path (hardware NVDEC decode)
# ---------------------------------------------------------------------------

def _process_gpu_torchcodec(
    input_path, writer, total_frames, w, h, scale_factor,
    strategy, exp_a, exp_b, lum_weight, rgb_multipliers,
    chromatic_tail_only, frame_skip, motion_threshold,
    torch_device: "torch.device",
):
    """
    GPU path using TorchCodec for hardware NVDEC decode.

    TorchCodec decodes directly onto the GPU and returns [C, H, W] uint8 RGB
    torch tensors, eliminating the CPU decode and host→device upload entirely.
    The beta CUDA backend delivers ~3x faster decode with ~90% NVDEC utilisation.
    """
    import contextlib
    import torch.nn.functional as F

    step             = frame_skip + 1
    exp_a2           = 1.0 - exp_a
    exp_b2           = 1.0 - exp_b
    threshold_offset = float(-abs(motion_threshold))
    rm = float(rgb_multipliers[0])
    gm = float(rgb_multipliers[1])
    bm = float(rgb_multipliers[2])

    ctx = _set_cuda_backend("beta") if _set_cuda_backend is not None else contextlib.nullcontext()
    prev_frames = None
    written     = 0
    start_time  = time.perf_counter()
    interrupted = False

    # Pre-allocate a single pinned (page-locked) CPU buffer for fast non-blocking
    # GPU→CPU transfers.  Allocated once, reused every frame — memory is stable.
    # A standalone numpy copy is made before queuing so the buffer is immediately
    # free for the next frame's transfer.
    pinned_buf = torch.empty(h, w, 3, dtype=torch.uint8).pin_memory()

    _t_compute = _t_transfer = _t_queue = 0.0

    write_q, writer_thread, write_exc = _start_writer_thread(writer)

    try:
        with ctx:
            decoder = _VideoDecoder(input_path, device=str(torch_device))
            logger.debug("TorchCodec GPU decoder opened: {} frames", total_frames)

            for frame_idx, frame_batch in enumerate(decoder):
                if frame_idx % step != 0:
                    continue

                if write_exc[0] is not None:
                    raise write_exc[0]

                _t0 = time.perf_counter()

                # frame_batch.data: [C, H, W] uint8 RGB on GPU
                frame = frame_batch.data.float()  # [C, H, W] float32

                if scale_factor != 1.0:
                    frame = F.interpolate(
                        frame.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
                    ).squeeze(0)

                # RGB CHW → gray HW: 0.299*R + 0.587*G + 0.114*B
                gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]

                if prev_frames is None:
                    prev_frames = [gray, gray.clone(), gray.clone()]
                    continue

                d0 = (prev_frames[0] - gray).abs_()
                d1 = (prev_frames[1] - gray).abs_()
                d2 = (prev_frames[2] - gray).abs_()

                if strategy == "exponential":
                    prev_frames[0] = gray
                    prev_frames[1].mul_(exp_a).add_(gray, alpha=exp_a2)
                    prev_frames[2].mul_(exp_b).add_(gray, alpha=exp_b2)
                else:
                    prev_frames[2] = prev_frames[1]
                    prev_frames[1] = prev_frames[0]
                    prev_frames[0] = gray

                if chromatic_tail_only:
                    b_src = (d0 - d1).clamp_min_(0.0)
                    g_src = (d1 - d0).clamp_min_(0.0)
                    r_src = (d2 - d1).clamp_min_(0.0)
                else:
                    b_src, g_src, r_src = d0, d1, d2

                blue  = (lum_weight * gray + bm * b_src + threshold_offset).clamp_(0.0, 255.0)
                green = (lum_weight * gray + gm * g_src + threshold_offset).clamp_(0.0, 255.0)
                red   = (lum_weight * gray + rm * r_src + threshold_offset).clamp_(0.0, 255.0)

                bgr_gpu = torch.stack([blue, green, red], dim=-1).byte()

                _t1 = time.perf_counter()

                # Non-blocking DMA into pinned buffer, then sync before reading.
                # Faster than .cpu() into pageable memory; sync is cheap once the
                # transfer is already in flight.
                pinned_buf.copy_(bgr_gpu, non_blocking=True)
                torch.cuda.current_stream(torch_device).synchronize()
                # Copy pinned view to a standalone array so the buffer is free for
                # the next frame before the writer thread has consumed this one.
                bgr_out = pinned_buf.numpy().copy()

                _t2 = time.perf_counter()

                try:
                    write_q.put(bgr_out, timeout=60)
                except queue.Full:
                    if write_exc[0] is not None:
                        raise write_exc[0]
                    raise RuntimeError("Writer queue full after 60s — writer thread may have stalled")

                written += 1
                _t3 = time.perf_counter()

                _t_compute  += _t1 - _t0
                _t_transfer += _t2 - _t1
                _t_queue    += _t3 - _t2

                if written % 300 == 0:
                    _progress(frame_idx, total_frames, time.perf_counter() - start_time, written)
                    logger.debug(
                        "Per-frame avg (last 300): compute={:.1f}ms  transfer={:.1f}ms  queue_put={:.1f}ms",
                        _t_compute / 300 * 1000, _t_transfer / 300 * 1000, _t_queue / 300 * 1000,
                    )
                    _t_compute = _t_transfer = _t_queue = 0.0

    except KeyboardInterrupt:
        interrupted = True
        logger.debug("TorchCodec GPU loop: interrupted after {} frames written", written)
    finally:
        if interrupted:
            while True:
                try:
                    write_q.get_nowait()
                except queue.Empty:
                    break
        try:
            write_q.put(None, timeout=5)
        except queue.Full:
            pass
        writer_thread.join(timeout=30)
        if write_exc[0] is not None and not interrupted:
            raise write_exc[0]

    return written, interrupted


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_motion_video(
    input_path: str,
    output_path: str,
    strategy: str = "exponential",
    exp_a: float = 0.5,
    exp_b: float = 0.8,
    lum_weight: float = 0.7,
    rgb_multipliers: tuple = (4.0, 4.0, 4.0),
    chromatic_tail_only: bool = False,
    scale_factor: float = 1.0,
    frame_skip: int = 0,
    motion_threshold: int = 0,
    compress: bool = False,
    crf: int | None = None,
    device: str = "auto",
) -> None:
    """
    Convert a single video to a motion-enhanced video.

    The output encodes motion history chromatically: blue channel shows the most
    recent movement, green shows medium-term, and red shows older movement.
    """
    resolved = resolve_device(device)
    logger.info("Opening: {}", input_path)

    # -------------------------------------------------------------------------
    # Local cache: TorchCodec/FFmpeg must index the whole file before decoding;
    # on a network share this means reading gigabytes before frame 1.  Copy
    # both the input and output through a local NVMe temp directory instead.
    # -------------------------------------------------------------------------
    cache_dir   = None
    work_input  = input_path
    work_output = output_path

    if _is_network_path(input_path) and resolved == "cuda" and _TORCHCODEC_AVAILABLE:
        cache_dir  = tempfile.mkdtemp(prefix="behaveai_")
        work_input = str(Path(cache_dir) / Path(input_path).name)
        work_output = str(Path(cache_dir) / Path(output_path).name)
        logger.info("Network path detected — caching to: {}", cache_dir)
        _copy_with_progress(input_path, work_input)

    try:
        cap = cv2.VideoCapture(work_input)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {work_input}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        w = int(src_w * scale_factor)
        h = int(src_h * scale_factor)

        os.makedirs(os.path.dirname(os.path.abspath(work_output)), exist_ok=True)

        _use_torchcodec = resolved == "cuda" and _TORCHCODEC_AVAILABLE
        # For GPU paths, pipe frames directly into FFmpeg (NVENC when available).
        # This avoids the huge intermediate mp4v file and the separate --compress
        # step — the output is already compressed H.265.
        _use_pipe = resolved == "cuda" and _ffmpeg_available()
        if _use_pipe:
            writer = _FFmpegPipeWriter(work_output, w, h, fps, crf=crf)
            codec  = "hevc_nvenc (pipe)" if _nvenc_available() else "libx265 (pipe)"
        else:
            codec, writer = _pick_codec(work_output, w, h, fps)

        logger.info(
            "{}x{}  {} frames  {:.2f} fps  strategy={}  device={}  codec={}",
            src_w, src_h, total_frames, fps, strategy,
            f"{resolved} (torchcodec)" if _use_torchcodec else resolved,
            codec,
        )
        logger.info("Output: {}", output_path)

        start_time = time.perf_counter()
        try:
            if _use_torchcodec:
                written, interrupted = _process_gpu_torchcodec(
                    work_input, writer, total_frames, w, h, scale_factor,
                    strategy, exp_a, exp_b, lum_weight, rgb_multipliers,
                    chromatic_tail_only, frame_skip, motion_threshold,
                    torch.device("cuda"),
                )
            elif resolved == "cuda":
                written, interrupted = _process_gpu(
                    cap, writer, total_frames, w, h, scale_factor,
                    strategy, exp_a, exp_b, lum_weight, rgb_multipliers,
                    chromatic_tail_only, frame_skip, motion_threshold,
                    torch.device("cuda"),
                )
            else:
                written, interrupted = _process_cpu(
                    cap, writer, total_frames, w, h, scale_factor,
                    strategy, exp_a, exp_b, lum_weight, rgb_multipliers,
                    chromatic_tail_only, frame_skip, motion_threshold,
                )
        finally:
            cap.release()
            writer.release()

        _done_line()

        if interrupted:
            logger.warning("Interrupted — discarding partial output.")
            out = Path(work_output)
            if out.exists():
                out.unlink()
                logger.debug("Removed incomplete output file: {}", work_output)
            return

        elapsed = time.perf_counter() - start_time
        out_mb  = Path(work_output).stat().st_size / 1_048_576 if Path(work_output).exists() else 0
        logger.info("Done: {} frames in {:.1f}s ({:.0f} fps)  {:.1f} MB",
                    written, elapsed, written / elapsed, out_mb)

        if compress:
            if _use_pipe:
                logger.info("--compress skipped: output was already encoded via FFmpeg pipe.")
            elif _ffmpeg_available():
                logger.info("Compressing with FFmpeg H.265...")
                _compress_with_ffmpeg(work_output, crf=crf)
                compressed_mb = Path(work_output).stat().st_size / 1_048_576 if Path(work_output).exists() else 0
                logger.info("Compressed: {:.1f} MB ({:.0f}% of original)", compressed_mb, 100 * compressed_mb / out_mb)
            else:
                logger.warning("--compress requested but ffmpeg not found in PATH; skipping.")

        if cache_dir:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            logger.info("Transferring to: {}", output_path)
            transfer_start = time.perf_counter()
            shutil.move(work_output, output_path)
            transfer_mb = Path(output_path).stat().st_size / 1_048_576
            transfer_s  = time.perf_counter() - transfer_start
            logger.info("Transfer complete: {:.1f} MB in {:.1f}s ({:.0f} MB/s)",
                        transfer_mb, transfer_s, transfer_mb / transfer_s)

    finally:
        if cache_dir:
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.debug("Cache cleaned: {}", cache_dir)


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def process_motion_batch(
    input_path: str,
    output_path: str,
    **kwargs,
) -> None:
    """
    Convert one video or a folder of videos to motion-enhanced output.

    If input_path is a directory, all video files within it are processed and
    written to output_path (created as a directory). If input_path is a single
    file, output_path is treated as the output file path.

    :param input_path: Path to a video file or directory of video files.
    :param output_path: Output file path (single video) or output directory (folder input).
    :param kwargs: Passed through to :func:`process_motion_video`.
    """
    input_path  = str(input_path)
    output_path = str(output_path)

    if os.path.isdir(input_path):
        videos = sorted(
            f for f in glob.glob(os.path.join(input_path, "*"))
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise ValueError(f"No video files found in: {input_path}")
        os.makedirs(output_path, exist_ok=True)
        logger.info("Processing {} video(s) -> {}", len(videos), output_path)
        for i, vid in enumerate(videos, 1):
            out = os.path.join(output_path, Path(vid).stem + "_motion.mp4")
            logger.info("[{}/{}] {}", i, len(videos), Path(vid).name)
            process_motion_video(vid, out, **kwargs)
    else:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        process_motion_video(input_path, output_path, **kwargs)

    logger.info("All done.")
