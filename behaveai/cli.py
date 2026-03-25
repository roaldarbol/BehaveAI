import sys
import typer
from pathlib import Path
from typing import Optional
from enum import Enum
from loguru import logger

# Remove loguru's default handler; the main() callback adds one based on --debug
logger.remove()

_LOG_FORMAT_NORMAL = (
    "<cyan>{time:HH:mm:ss}</cyan> | "
    "<level>{level: <8}</level> | "
    "{message}"
)

_LOG_FORMAT_DEBUG = (
    "<cyan>{time:HH:mm:ss.SSS}</cyan> | "
    "<level>{level: <8}</level> | "
    "<dim>{name}:{line}</dim> | "
    "{message}"
)

app = typer.Typer(help="BehaveAI - animal tracking and behaviour classification.")


class MotionStrategy(str, Enum):
    exponential = "exponential"
    sequential = "sequential"


class DeviceChoice(str, Enum):
    auto = "auto"
    cpu  = "cpu"
    cuda = "cuda"


def _default_projects_dir() -> Path:
    exe = Path(sys.executable)
    home = Path.home()

    # Pixi environment: .pixi appears in the executable path
    try:
        pixi_idx = exe.parts.index(".pixi")
        pixi_parent = Path(*exe.parts[:pixi_idx])
        # Global install: .pixi is directly under home (~/.pixi/envs/<name>/...)
        if pixi_parent == home:
            return home / "BehaveAI" / "projects"
        # Project install: .pixi is inside a project directory
        return pixi_parent / "behaveai_projects"
    except ValueError:
        pass

    # uv tool install: ~/.local/share/uv/tools (Linux/Mac)
    # or %APPDATA%\uv\tools (Windows)
    try:
        parts_lower = [p.lower() for p in exe.parts]
        next(
            i for i, p in enumerate(parts_lower) if p == "tools"
            and i > 0 and parts_lower[i - 1] == "uv"
        )
        # tools/<name>/... -> global install
        return home / "BehaveAI" / "projects"
    except StopIteration:
        pass

    # Standard venv: pyvenv.cfg two levels up from the executable
    venv_root = exe.parent.parent
    if (venv_root / "pyvenv.cfg").exists():
        return venv_root.parent / "behaveai_projects"

    # Fallback
    return home / "BehaveAI" / "projects"


DEFAULT_PROJECTS_DIR = _default_projects_dir()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    project: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=False,
        help="Path to an existing BehaveAI project directory.",
    ),
    projects_dir: Optional[Path] = typer.Option(
        None,
        help="Override the default projects directory.",
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging.", is_eager=True),
):
    """BehaveAI - animal tracking and behaviour classification."""
    if debug:
        logger.add(sys.stderr, level="DEBUG", format=_LOG_FORMAT_DEBUG, colorize=True)
    else:
        logger.add(sys.stderr, level="INFO", format=_LOG_FORMAT_NORMAL, colorize=True)

    if ctx.invoked_subcommand is None:
        from behaveai.launcher import launch
        launch(
            project_path=project,
            projects_dir=projects_dir if projects_dir else DEFAULT_PROJECTS_DIR,
        )


@app.command()
def run(
    settings: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to a settings INI file."),
    input_dir: Optional[Path] = typer.Option(
        None, "--input",
        exists=True,
        file_okay=False,
        help="Override the input directory from the settings file.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output",
        help="Override the output directory from the settings file.",
    ),
):
    """Run headless batch processing using a settings INI file."""
    from behaveai.classify_track import run_batch
    run_batch(
        config_path=str(settings),
        input_dir=str(input_dir) if input_dir else None,
        output_dir=str(output_dir) if output_dir else None,
    )


@app.command()
def motion(
    input: Path = typer.Argument(..., exists=True, help="Single video file or directory of videos."),
    output: Optional[Path] = typer.Argument(None, help="Destination path (auto-generated if not provided)."),
    output_opt: Optional[Path] = typer.Option(None, "--output", help="Destination path (alias for the positional OUTPUT argument)."),
    strategy: MotionStrategy = typer.Option(MotionStrategy.exponential, show_default=True, help="Frame accumulation strategy."),
    exp_a: float = typer.Option(0.5, "--exp-a", show_default=True, help="Exponential decay for the green (medium-term) channel."),
    exp_b: float = typer.Option(0.8, "--exp-b", show_default=True, help="Exponential decay for the red (older) channel."),
    lum_weight: float = typer.Option(0.7, "--lum-weight", show_default=True, help="Blend weight of original luminance in the output (0-1)."),
    rgb_multipliers: str = typer.Option("4.0,4.0,4.0", "--rgb-multipliers", show_default=True, help="Comma-separated scaling factors for the R,G,B motion channels."),
    chromatic_tail_only: bool = typer.Option(False, help="Show only the chromatic motion tail, suppressing base luminance from motion channels."),
    scale_factor: float = typer.Option(1.0, "--scale-factor", show_default=True, help="Resize factor applied to each frame before processing."),
    frame_skip: int = typer.Option(0, "--frame-skip", show_default=True, help="Number of frames to skip between processed frames."),
    motion_threshold: int = typer.Option(0, "--motion-threshold", show_default=True, help="Brightness offset applied to output (negative darkens low-motion areas)."),
    compress: bool = typer.Option(False, help="Re-encode the output with FFmpeg H.264 after writing (requires ffmpeg in PATH)."),
    crf: int = typer.Option(23, "--crf", show_default=True, help="H.264 quality for --compress (lower = better quality, 18-28 is typical)."),
    device: DeviceChoice = typer.Option(DeviceChoice.auto, "--device", show_default=True, help="Processing device: auto detects CUDA, falls back to CPU."),
):
    """Convert a video (or folder of videos) to a motion-enhanced output.

    INPUT can be a single video file or a directory. If a directory is given,
    all video files within it are processed and written to OUTPUT as a directory.

    The output encodes motion history chromatically: blue = recent movement,
    green = medium-term, red = older movement.

    \b
    Examples:
      behaveai motion input.mp4
      behaveai motion input.mp4 output.mp4
      behaveai motion input.mp4 output.mp4
      behaveai motion input.mp4 --output output.mp4
      behaveai motion input.mp4 output.mp4 --strategy exponential --exp-a 0.5
      behaveai motion videos/ --chromatic-tail-only
      behaveai motion videos/ motion_videos/
    """
    from behaveai.motion import process_motion_batch

    if output is not None and output_opt is not None:
        raise typer.BadParameter("Cannot specify both positional OUTPUT and --output.", param_hint="OUTPUT / --output")
    output = output or output_opt

    # If input is a file but output is an existing directory, place the auto-named file inside it
    if output is not None and input.is_file() and output.is_dir():
        output = output / (input.stem + "_motion" + input.suffix)

    # Derive default output path if not given
    if output is None:
        p = input
        if p.is_dir():
            output = p.parent / (p.name + "_motion")
        else:
            output = p.parent / (p.stem + "_motion" + p.suffix)

    logger.info("Output: {}", output)

    try:
        multipliers = tuple(float(x) for x in rgb_multipliers.split(","))
        if len(multipliers) != 3:
            raise ValueError
    except ValueError:
        raise typer.BadParameter(
            "Must be three comma-separated floats, e.g. '4.0,4.0,4.0'",
            param_hint="--rgb-multipliers",
        )

    process_motion_batch(
        input_path=str(input),
        output_path=str(output),
        strategy=strategy.value,
        exp_a=exp_a,
        exp_b=exp_b,
        lum_weight=lum_weight,
        rgb_multipliers=multipliers,
        chromatic_tail_only=chromatic_tail_only,
        scale_factor=scale_factor,
        frame_skip=frame_skip,
        motion_threshold=motion_threshold,
        compress=compress,
        crf=crf,
        device=device.value,
    )


@app.command("gpu-check")
def gpu_check():
    """Show available GPU compute backends and driver info."""
    import typer

    def ok(msg):  typer.echo(typer.style("  ✓ " + msg, fg="green"))
    def no(msg):  typer.echo(typer.style("  ✗ " + msg, fg="red", dim=True))
    def info(msg): typer.echo(f"    {msg}")

    typer.echo(typer.style("\nGPU / compute backend check", bold=True))

    # --- CUDA via PyTorch ---
    typer.echo(typer.style("\nCUDA (PyTorch)", bold=True))
    try:
        import torch
        ok(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            ok(f"CUDA available  (driver: {torch.version.cuda})")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info(f"[{i}] {p.name}  {p.total_memory // 1024**2} MB  "
                     f"compute {p.major}.{p.minor}")
        else:
            no("CUDA not available")
            info("torch.cuda.is_available() returned False")
    except ImportError:
        no("PyTorch not installed")

    # --- TorchCodec ---
    typer.echo(typer.style("\nTorchCodec (motion GPU decode)", bold=True))
    try:
        import torchcodec
        ok(f"TorchCodec {torchcodec.__version__}")
        try:
            from torchcodec.decoders import set_cuda_backend
            ok("Beta CUDA backend available (~3x faster NVDEC)")
        except (ImportError, RuntimeError):
            info("Beta CUDA backend not found (set_cuda_backend missing) — using default")
    except (ImportError, RuntimeError) as e:
        no(f"TorchCodec not available: {e.args[0].splitlines()[0] if e.args else e}")
        info("On Windows, install via: pixi run behaveai  (conda-forge build includes FFmpeg)")

    typer.echo("")
