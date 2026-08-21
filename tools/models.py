from config import TIMEOUT_INFO, TIMEOUT_MODEL_SWITCH
from mcp_instance import mcp
from utils import forge_client, format_error, is_truthy


@mcp.tool()
async def get_models() -> str:
    """
    List all Stable Diffusion checkpoints (models) available in Forge.

    Returns the model title and filename for each checkpoint so you can
    pick the right one for your art style before generating.
    """
    async with forge_client(TIMEOUT_INFO) as client:
        response = await client.get("/sdapi/v1/sd-models")

    if response.status_code != 200:
        return format_error(response)

    models = response.json()
    if not models:
        return "No models found."

    lines = [f"  [{i+1}] {m['title']}  ({m['filename']})" for i, m in enumerate(models)]
    return "Available models:\n" + "\n".join(lines)


@mcp.tool()
async def set_model(
    model_title: str,
    acknowledge_required_switch: str | bool = "",
) -> str:
    """
    Switch the active Stable Diffusion checkpoint in Forge.

    ⚠️ WARNING: Switching models arbitrarily can leave the active checkpoint
    in an unusable state and prevent image generation from succeeding. Only
    call this tool when a model switch is genuinely necessary — for example
    when the user explicitly asked for a specific checkpoint.

    Model switching can take 10-60 seconds depending on model size, so the
    caller must pass a truthy value to acknowledge the switch is required.

    Args:
        model_title: The exact title of the model as returned by get_models().
        acknowledge_required_switch: A truthy value ('1'/'true'/'yes'/'on')
                   confirming the model switch is truly required. The tool
                   refuses to switch unless this is set.
    """
    if not is_truthy(acknowledge_required_switch):
        return (
            "Refusing to switch models: arbitrarily switching the active "
            "checkpoint can prevent image generation from working. set_model "
            "should only be called when a model switch is genuinely necessary. "
            "Pass acknowledge_required_switch='1' (or 'true'/'yes'/'on') to "
            "confirm the switch is required."
        )

    payload = {"sd_model_checkpoint": model_title}

    async with forge_client(TIMEOUT_MODEL_SWITCH) as client:
        response = await client.post("/sdapi/v1/options", json=payload)

    if response.status_code != 200:
        return format_error(response)

    return f"Model switched to '{model_title}'. Give Forge a moment to load it before generating."


@mcp.tool()
async def get_current_model() -> str:
    """
    Return the name of the checkpoint that is currently loaded in Forge.
    """
    async with forge_client(TIMEOUT_INFO) as client:
        response = await client.get("/sdapi/v1/options")

    if response.status_code != 200:
        return format_error(response)

    opts = response.json()
    return f"Current model: {opts.get('sd_model_checkpoint', 'unknown')}"


@mcp.tool()
async def refresh_models() -> str:
    """
    Tell Forge to scan the models folder and reload the list of available
    checkpoints, LoRAs and embeddings.

    Run this after copying new model files into the Forge models directory
    so they appear in get_models(), get_loras(), etc. without restarting Forge.
    """
    async with forge_client(TIMEOUT_INFO) as client:
        r_ckpt = await client.post("/sdapi/v1/refresh-checkpoints")
        r_lora = await client.post("/sdapi/v1/refresh-loras")

    results = []
    results.append(
        "Checkpoints refreshed." if r_ckpt.status_code == 200 else format_error(r_ckpt)
    )
    results.append(
        "LoRAs refreshed." if r_lora.status_code == 200 else format_error(r_lora)
    )
    return "\n".join(results)
