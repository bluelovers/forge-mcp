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
