import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Forge connection
# ---------------------------------------------------------------------------

FORGE_URL: str = os.getenv("FORGE_URL", "http://127.0.0.1:7860")

# Credentials for Forge's --api-auth flag (leave blank if auth is disabled).
FORGE_API_USER: str = os.getenv("FORGE_API_USER", "")
FORGE_API_PASSWORD: str = os.getenv("FORGE_API_PASSWORD", "")

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Fallback: an absolute path next to this file so the save location is
# predictable regardless of what working directory the server is launched from
# (e.g. when running via a .mcpb bundle with uv).
_default_output = str(Path(__file__).parent / "outputs")


def _resolve_output_dir() -> str:
    """
    Pick the output directory.

    Prefers the OUTPUT_DIR env var, then a cross-platform default under the
    user's home directory, and finally falls back to _default_output (next to
    this file) if no home directory can be determined or it cannot be written.
    """
    env_dir = os.getenv("OUTPUT_DIR")
    if env_dir:
        return env_dir

    # Cross-platform default: <home>/.cache/forge-painter/outputs
    try:
        home = Path.home()
        candidate = home / ".cache" / "forge-painter" / "outputs"
        candidate.mkdir(parents=True, exist_ok=True)
        # Confirm it is actually writable before adopting it.
        probe = candidate / ".write_probe"
        probe.touch()
        probe.unlink()
        return str(candidate)
    except (OSError, RuntimeError, ValueError):
        return _default_output


OUTPUT_DIR: Path = Path(_resolve_output_dir())
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Request logbook. Defaults to OUTPUT_DIR; override with the LOG_DIR env var.
LOG_DIR: Path = Path(os.getenv("LOG_DIR", str(OUTPUT_DIR)))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Maximum number of entries kept in the request logbook.
LOG_MAX_ENTRIES: int = int(os.getenv("LOG_MAX_ENTRIES", "100"))

# When enabled, generated images are saved to auto-generated timestamped
# paths instead of overwriting the same file. Path format:
#   <OUTPUT_DIR>/<year>/<YYYY-MM-DD>/<YYYYMMDD_HHMMSS>_<ms3>_<counter5>.png
AUTO_SAVE: bool = os.getenv("AUTO_SAVE", "1").strip().lower() in ("1", "true", "yes", "on")

# Configurable default values for generation parameters (env-overridable).
DEFAULT_WIDTH: int = int(os.getenv("DEFAULT_WIDTH", "512"))
DEFAULT_HEIGHT: int = int(os.getenv("DEFAULT_HEIGHT", "512"))
DEFAULT_SEED: int = int(os.getenv("DEFAULT_SEED", "-1"))
DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "1"))
DEFAULT_STEPS: int = int(os.getenv("DEFAULT_STEPS", "9"))
DEFAULT_CFG_SCALE: float = float(os.getenv("DEFAULT_CFG_SCALE", "1.0"))

# Defaults that are "not specified" unless set via env. Empty string means the
# parameter is left unset and Forge (or the tool) chooses its own default.
DEFAULT_SAMPLER: str = os.getenv("DEFAULT_SAMPLER", "")
DEFAULT_SAVE_PATH: str = os.getenv("DEFAULT_SAVE_PATH", "")

# Default negative prompt. Applied when a tool call passes
# use_default_negative_prompt="1"/"true"/"yes"/"on" and no explicit negative_prompt.
DEFAULT_NEGATIVE_PROMPT: str = os.getenv("DEFAULT_NEGATIVE_PROMPT", "")

# Per-tool overrides for img2img / inpaint. Each inherits the corresponding
# txt2img default (DEFAULT_STEPS/CFG_SCALE/SAMPLER) unless overridden via env.
IMG2IMG_STEPS: int = int(os.getenv("IMG2IMG_STEPS", str(DEFAULT_STEPS)))
IMG2IMG_CFG_SCALE: float = float(os.getenv("IMG2IMG_CFG_SCALE", str(DEFAULT_CFG_SCALE)))
IMG2IMG_SAMPLER: str = os.getenv("IMG2IMG_SAMPLER", DEFAULT_SAMPLER)

INPAINT_STEPS: int = int(os.getenv("INPAINT_STEPS", str(DEFAULT_STEPS)))
INPAINT_CFG_SCALE: float = float(os.getenv("INPAINT_CFG_SCALE", str(DEFAULT_CFG_SCALE)))
INPAINT_SAMPLER: str = os.getenv("INPAINT_SAMPLER", DEFAULT_SAMPLER)

# Denoising strength for img2img / inpaint (how much the source is changed).
IMG2IMG_DENOISING_STRENGTH: float = float(os.getenv("IMG2IMG_DENOISING_STRENGTH", "0.6"))
INPAINT_DENOISING_STRENGTH: float = float(os.getenv("INPAINT_DENOISING_STRENGTH", "0.75"))

# Inpainting mask controls (blur radius and fill mode).
INPAINT_MASK_BLUR: int = int(os.getenv("INPAINT_MASK_BLUR", "4"))
INPAINT_FILL: int = int(os.getenv("INPAINT_FILL", "1"))

# Upscaling default multiplier.
UPSCALE_RESIZE: float = float(os.getenv("UPSCALE_RESIZE", "2.0"))

# ---------------------------------------------------------------------------
# Timeouts (seconds)
# ---------------------------------------------------------------------------

# Long-running generation requests (txt2img, img2img, inpaint, upscale).
TIMEOUT_GENERATION: float = float(os.getenv("TIMEOUT_GENERATION", "300"))

# Model / checkpoint switching — can be slow for large models.
TIMEOUT_MODEL_SWITCH: float = float(os.getenv("TIMEOUT_MODEL_SWITCH", "120"))

# Lightweight informational requests (listing models, samplers, etc.).
TIMEOUT_INFO: float = float(os.getenv("TIMEOUT_INFO", "30"))

# Fire-and-forget control requests (interrupt, progress check).
TIMEOUT_CONTROL: float = float(os.getenv("TIMEOUT_CONTROL", "10"))
