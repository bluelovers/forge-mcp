from config import TIMEOUT_INFO, TIMEOUT_MODEL_SWITCH
from mcp_instance import mcp
from utils import forge_client, format_error, is_truthy


@mcp.tool()
async def get_models() -> str:
    """
    List all Stable Diffusion checkpoints (models) available in Forge.

    Returns the model title and filename for each checkpoint.

    Unless the user explicitly asked which models are available, there is no
    need to call this tool. This tool only LISTS checkpoints — do NOT call
    set_model() unless the user explicitly instructed a checkpoint switch.
    Never assume that a particular model suits the prompt, and never switch
    the checkpoint on your own initiative.
    """
    async with forge_client(TIMEOUT_INFO) as client:
        response = await client.get("/sdapi/v1/sd-models")

    if response.status_code != 200:
        return format_error(response)

    models = response.json()
    if not models:
        return "No models found."

    lines = [f"  [{i+1}] {m['title']}  ({m['filename']})" for i, m in enumerate(models)]
    return (
        "Available models:\n"
        + "\n".join(lines)
        + "\n\n⚠️ WARNING: this only LISTS checkpoints. Do NOT call set_model() "
        "unless the user explicitly instructed a checkpoint switch, and never "
        "assume that a particular model suits the prompt."
    )


@mcp.tool()
async def set_model(
    model_title: str,
    acknowledge_required_switch: str | bool = "",
) -> str:
    """
    Switch the active Stable Diffusion checkpoint in Forge.

    ⚠️ WARNING: Switching models arbitrarily can leave the active checkpoint
    in an unusable state and prevent image generation from succeeding. Never
    assume that a particular model fits the prompt, and never switch the
    checkpoint on your own initiative. Only call this tool when the user
    explicitly instructed the switch to a specific checkpoint.

    Model switching can take 10-60 seconds depending on model size.

    Args:
        model_title: The exact title of the model as returned by get_models().
        acknowledge_required_switch: Only set when the user explicitly
                   instructed the checkpoint switch. Do not set it on your own
                   initiative — the tool refuses to switch unless this is set.
    """
    # Guardrail intent: acknowledge_required_switch makes a checkpoint switch a
    # deliberate, acknowledged action. Without a truthy value the tool refuses
    # and never contacts Forge, so accidental or arbitrary switching — which
    # can leave the active checkpoint unusable and break image generation — is
    # blocked up front. The refusal is a calm note (not a warning) so the agent
    # is not led to believe it should pass the value; only an explicit user
    # instruction to switch should ever supply it. The success path carries a
    # severe warning instead, to discourage further arbitrary switching.
    if not is_truthy(acknowledge_required_switch):
        return (
            "set_model was not called. acknowledge_required_switch was not set, "
            "so the checkpoint was left unchanged. Only switch the checkpoint "
            "when the user explicitly instructed it — never switch it on your "
            "own initiative."
        )

    payload = {"sd_model_checkpoint": model_title}

    async with forge_client(TIMEOUT_MODEL_SWITCH) as client:
        response = await client.post("/sdapi/v1/options", json=payload)

    if response.status_code != 200:
        return format_error(response)

    return (
        f"Model switched to '{model_title}'. Give Forge a moment to load it before generating.\n\n"
        "⚠️ WARNING: arbitrarily switching checkpoints can leave the active "
        "model in an unusable state and prevent image generation from "
        "succeeding. Only switch again if the user explicitly instructs it."
    )


@mcp.tool()
async def get_current_model() -> str:
    """
    Return the name of the checkpoint that is currently loaded in Forge.

    Unless the user asked which model is currently active, there is no need
    to call this tool. This tool only REPORTS the active checkpoint — do NOT
    call set_model() unless the user explicitly instructed a checkpoint
    switch. Never switch the checkpoint on your own initiative.
    """
    async with forge_client(TIMEOUT_INFO) as client:
        response = await client.get("/sdapi/v1/options")

    if response.status_code != 200:
        return format_error(response)

    opts = response.json()
    return (
        f"Current model: {opts.get('sd_model_checkpoint', 'unknown')}\n\n"
        "⚠️ WARNING: this only REPORTS the active checkpoint. Do NOT call "
        "set_model() unless the user explicitly instructed a checkpoint switch."
    )


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
