from config import (
    FORGE_API_PASSWORD,
    FORGE_API_USER,
    FORGE_URL,
    LOG_DIR,
    LOG_MAX_ENTRIES,
    OUTPUT_DIR,
    TIMEOUT_CONTROL,
    TIMEOUT_GENERATION,
    TIMEOUT_INFO,
    TIMEOUT_MODEL_SWITCH,
)
import fastmcp
import json
from logbook import LOG_FILE
from mcp_instance import mcp
from utils import forge_client, format_error


def _mask(value: str) -> str:
    """Mask a secret, keeping only a hint of its length/shape."""
    if not value:
        return "(not set)"
    return f"***** (len {len(value)})"


@mcp.tool()
async def info() -> str:
    """
    Show server runtime information: fastmcp version and configuration values.

    Credentials (account / password) are masked so their contents are never
    revealed. Also reports the request logbook location. Returns a formatted
    JSON string with line breaks and indentation.
    """
    payload = {
        "FASTMCP_VERSION": fastmcp.__version__,
        "FORGE_URL": FORGE_URL,
        "FORGE_API_USER": _mask(FORGE_API_USER),
        "FORGE_API_PASSWORD": _mask(FORGE_API_PASSWORD),
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "LOG_DIR": str(LOG_DIR),
        "LOG_FILE": str(LOG_FILE),
        "LOG_MAX_ENTRIES": LOG_MAX_ENTRIES,
        "TIMEOUTS_SECONDS": {
            "TIMEOUT_GENERATION": TIMEOUT_GENERATION,
            "TIMEOUT_MODEL_SWITCH": TIMEOUT_MODEL_SWITCH,
            "TIMEOUT_INFO": TIMEOUT_INFO,
            "TIMEOUT_CONTROL": TIMEOUT_CONTROL,
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


@mcp.tool()
async def get_progress() -> str:
    """
    Check the progress of the currently running generation in Forge.

    Returns the completion percentage and estimated time remaining.
    Returns 'idle' if nothing is generating.
    """
    async with forge_client(TIMEOUT_CONTROL) as client:
        response = await client.get("/sdapi/v1/progress")

    if response.status_code != 200:
        return format_error(response)

    data = response.json()
    progress = data.get("progress", 0)

    if progress == 0 and not data.get("state", {}).get("job_count"):
        return (
            "Forge is idle — no generation in progress.\n"
            f"Request logbook: {LOG_FILE}"
        )

    eta = data.get("eta_relative", 0)
    job = data.get("state", {}).get("job", "unknown")
    return (
        f"Generation in progress: {progress * 100:.1f}% complete\n"
        f"Current job: {job}\n"
        f"ETA: {eta:.1f}s\n"
        f"Request logbook: {LOG_FILE}"
    )


@mcp.tool()
async def interrupt_generation() -> str:
    """
    Interrupt (cancel) the currently running generation in Forge immediately.

    Use this if a generation is taking too long or if you submitted the wrong
    prompt and want to stop it before it finishes.
    """
    async with forge_client(TIMEOUT_CONTROL) as client:
        response = await client.post("/sdapi/v1/interrupt")

    if response.status_code != 200:
        return format_error(response)

    return (
        "Generation interrupted.\n"
        f"Request logbook: {LOG_FILE}"
    )
