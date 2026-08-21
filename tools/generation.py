import json
from datetime import datetime
from pathlib import Path

import mcp.types as mcp_types

from config import (
    AUTO_SAVE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CFG_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_SAMPLER,
    DEFAULT_SAVE_PATH,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    IMG2IMG_CFG_SCALE,
    IMG2IMG_DENOISING_STRENGTH,
    IMG2IMG_SAMPLER,
    IMG2IMG_STEPS,
    INPAINT_CFG_SCALE,
    INPAINT_DENOISING_STRENGTH,
    INPAINT_FILL,
    INPAINT_MASK_BLUR,
    INPAINT_SAMPLER,
    INPAINT_STEPS,
    OUTPUT_DIR,
    TIMEOUT_GENERATION,
    UPSCALE_RESIZE,
)
from fastmcp.utilities.types import Image
from logbook import finish_request, start_request
from mcp_instance import mcp
from utils import decode_and_save, encode_image, forge_client, format_error, is_truthy


@mcp.tool()
async def txt2img(
    prompt: str,
    acknowledge_custom_parameters: str | bool = "",
    negative_prompt: str | None = None,
    use_default_negative_prompt: str | bool = "",
    steps: int = DEFAULT_STEPS,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    sampler_name: str | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_path: str | None = None,
    return_image: bool = False,
) -> str | list[str | mcp_types.ImageContent]:
    """
    Generate one or more images from a text prompt using Stable Diffusion Forge.

    ⚠️ WARNING: unless you acknowledge it with `acknowledge_custom_parameters`,
    every non-essential parameter below (negative prompt, steps, cfg_scale,
    width, height, sampler, seed, batch_size) is forcibly ignored and the
    configured defaults are used instead. `use_default_negative_prompt`,
    `save_path` and `return_image` always remain in effect. Call this tool with
    only the `prompt` argument unless you genuinely need custom values.

    Each image in the batch is saved as <save_path>, <save_path>_1.png, etc.
    Use seed=-1 for a random seed. Returns the seed(s) that were used so the
    result can be reproduced later. Images are saved inside the configured
    OUTPUT_DIR unless save_path is an absolute path. When AUTO_SAVE is enabled,
    save_path is ignored and timestamped auto paths are used instead.

    Args:
        prompt: Positive prompt describing the desired image.
        acknowledge_custom_parameters: Only set when the user explicitly requested
                   specific generation values. Do not set it on your own
                   initiative — leave it unset unless clearly instructed.
        negative_prompt: Things to avoid in the image. When not provided and
                   use_default_negative_prompt is truthy, DEFAULT_NEGATIVE_PROMPT is used.
        use_default_negative_prompt: When '1'/'true'/'yes'/'on', apply
                   DEFAULT_NEGATIVE_PROMPT if negative_prompt is not specified.
        steps: Number of diffusion steps (higher = more detail, slower).
        cfg_scale: Classifier-free guidance scale. Higher = more prompt-adherent.
        width: Image width in pixels.
        height: Image height in pixels.
        sampler_name: Sampler to use (e.g. 'Euler a', 'DPM++ 2M', 'DDIM').
        seed: RNG seed. Use -1 for random.
        batch_size: Number of images to generate in one request.
        save_path: Filename for the output PNG. Relative paths are placed inside
                   OUTPUT_DIR; absolute paths are used as-is.
        return_image: When True, also embed the generated image(s) in the tool
                   response as MCP image content blocks. This only works with
                   clients that render images (e.g. LM Studio, Claude Desktop);
                   other clients receive them as opaque blocks. Defaults to
                   False, so the response is text-only and works everywhere.
    """
    had_custom = any((
        negative_prompt is not None,
        steps != DEFAULT_STEPS,
        cfg_scale != DEFAULT_CFG_SCALE,
        width != DEFAULT_WIDTH,
        height != DEFAULT_HEIGHT,
        sampler_name is not None,
        seed != DEFAULT_SEED,
        batch_size != DEFAULT_BATCH_SIZE,
    ))
    acknowledged = is_truthy(acknowledge_custom_parameters)
    if not is_truthy(acknowledge_custom_parameters):
        negative_prompt = None
        steps = DEFAULT_STEPS
        cfg_scale = DEFAULT_CFG_SCALE
        width = DEFAULT_WIDTH
        height = DEFAULT_HEIGHT
        sampler_name = None
        seed = DEFAULT_SEED
        batch_size = DEFAULT_BATCH_SIZE

    entry_id = start_request("txt2img", _params(locals(), "return_image"))

    sampler_name = _resolve_sampler(sampler_name)
    negative_prompt = _resolve_negative_prompt(negative_prompt, use_default_negative_prompt)

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "seed": seed,
        "batch_size": batch_size,
    }
    if sampler_name:
        payload["sampler_name"] = sampler_name

    async with forge_client(TIMEOUT_GENERATION) as client:
        response = await client.post("/sdapi/v1/txt2img", json=payload)

    if response.status_code != 200:
        finish_request(entry_id, "error")
        return format_error(response)

    data = response.json()
    images = data.get("images", [])
    info = data.get("info", {})
    seeds = info.get("all_seeds", [seed] * len(images)) if isinstance(info, dict) else [seed]

    paths = _output_paths(save_path, len(images))
    saved = []
    for i, img_b64 in enumerate(images):
        out = paths[i]
        out.parent.mkdir(parents=True, exist_ok=True)
        decode_and_save(img_b64, str(out))
        saved.append(str(out))

    summary = (
        f"Generated {len(saved)} image(s).\n"
        f"Saved to: {', '.join(saved)}\n"
        f"Seeds used: {seeds}"
    )
    if acknowledged and had_custom:
        summary += _ACKED_CUSTOM_WARNING
    elif had_custom:
        summary += _IGNORED_PARAMS_NOTE

    finish_request(entry_id, "success", saved)

    if return_image:
        return [summary, *(Image(path=str(out)).to_image_content() for out in saved)]

    return summary


@mcp.tool()
async def img2img(
    image_path: str,
    prompt: str,
    acknowledge_custom_parameters: str | bool = "",
    negative_prompt: str | None = None,
    use_default_negative_prompt: str | bool = "",
    denoising_strength: float = IMG2IMG_DENOISING_STRENGTH,
    steps: int = IMG2IMG_STEPS,
    cfg_scale: float = IMG2IMG_CFG_SCALE,
    width: int = 0,
    height: int = 0,
    sampler_name: str | None = None,
    seed: int = DEFAULT_SEED,
    save_path: str | None = None,
    return_image: bool = False,
) -> str | list[str | mcp_types.ImageContent]:
    """
    Transform an existing image guided by a text prompt (image-to-image).

    ⚠️ WARNING: unless you acknowledge it with `acknowledge_custom_parameters`,
    every non-essential parameter below (negative prompt, denoising_strength,
    steps, cfg_scale, width, height, sampler, seed) is forcibly
    ignored and the configured defaults are used instead.
    `use_default_negative_prompt`, `save_path` and `return_image` always remain
    in effect. Call this tool with only the `image_path` and `prompt` arguments
    unless you genuinely need custom values.

    Useful for restyling character art, adding details to maps, converting
    sketches to finished illustrations, or applying a new art style.

    Args:
        image_path: Path to the source image file (PNG/JPG).
        prompt: Positive prompt describing the desired result.
        acknowledge_custom_parameters: Only set when the user explicitly requested
                   specific generation values. Do not set it on your own
                   initiative — leave it unset unless clearly instructed.
        negative_prompt: Things to avoid in the output. When not provided and
                   use_default_negative_prompt is truthy, DEFAULT_NEGATIVE_PROMPT is used.
        use_default_negative_prompt: When '1'/'true'/'yes'/'on', apply
                   DEFAULT_NEGATIVE_PROMPT if negative_prompt is not specified.
        denoising_strength: How much to change the image. 0 = no change,
                            1 = ignore original. 0.4-0.7 is a good range.
        steps: Diffusion steps.
        cfg_scale: Prompt adherence strength.
        width: Output width. Use 0 to keep the source image size.
        height: Output height. Use 0 to keep the source image size.
        sampler_name: Sampler to use.
        seed: RNG seed (-1 for random).
        save_path: Filename for the output PNG. Relative paths land in OUTPUT_DIR.
        return_image: When True, also embed the result image in the tool response
                   as an MCP image content block. Only works with clients that
                   render images (e.g. LM Studio, Claude Desktop). Defaults to
                   False, so the response is text-only and works everywhere.
    """
    had_custom = any((
        negative_prompt is not None,
        denoising_strength != IMG2IMG_DENOISING_STRENGTH,
        steps != IMG2IMG_STEPS,
        cfg_scale != IMG2IMG_CFG_SCALE,
        width != 0,
        height != 0,
        sampler_name is not None,
        seed != DEFAULT_SEED,
    ))
    acknowledged = is_truthy(acknowledge_custom_parameters)
    if not is_truthy(acknowledge_custom_parameters):
        negative_prompt = None
        denoising_strength = IMG2IMG_DENOISING_STRENGTH
        steps = IMG2IMG_STEPS
        cfg_scale = IMG2IMG_CFG_SCALE
        width = 0
        height = 0
        sampler_name = None
        seed = DEFAULT_SEED

    entry_id = start_request("img2img", _params(locals(), "return_image", "image_path"))

    b64 = encode_image(image_path)
    sampler_name = _resolve_sampler(sampler_name, IMG2IMG_SAMPLER)
    negative_prompt = _resolve_negative_prompt(negative_prompt, use_default_negative_prompt)

    payload = {
        "init_images": [b64],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": denoising_strength,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
    }
    if sampler_name:
        payload["sampler_name"] = sampler_name
    if width:
        payload["width"] = width
    if height:
        payload["height"] = height

    async with forge_client(TIMEOUT_GENERATION) as client:
        response = await client.post("/sdapi/v1/img2img", json=payload)

    if response.status_code != 200:
        finish_request(entry_id, "error")
        return format_error(response)

    data = response.json()
    images = data.get("images", [])
    info = data.get("info", {})
    used_seed = info.get("seed", seed) if isinstance(info, dict) else seed

    if not images:
        finish_request(entry_id, "error")
        return "No images returned by Forge."

    out = _output_paths(save_path, 1)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    decode_and_save(images[0], str(out))
    summary = f"img2img complete. Saved to '{out}'. Seed: {used_seed}"
    if acknowledged and had_custom:
        summary += _ACKED_CUSTOM_WARNING
    elif had_custom:
        summary += _IGNORED_PARAMS_NOTE
    finish_request(entry_id, "success", [str(out)])
    if return_image:
        return [summary, Image(path=str(out)).to_image_content()]
    return summary


@mcp.tool()
async def inpaint(
    image_path: str,
    mask_path: str,
    prompt: str,
    acknowledge_custom_parameters: str | bool = "",
    negative_prompt: str | None = None,
    use_default_negative_prompt: str | bool = "",
    denoising_strength: float = INPAINT_DENOISING_STRENGTH,
    steps: int = INPAINT_STEPS,
    cfg_scale: float = INPAINT_CFG_SCALE,
    sampler_name: str | None = None,
    mask_blur: int = INPAINT_MASK_BLUR,
    inpainting_fill: int = INPAINT_FILL,
    seed: int = DEFAULT_SEED,
    save_path: str | None = None,
    return_image: bool = False,
) -> str | list[str | mcp_types.ImageContent]:
    """
    Inpaint (fill or redraw) a masked region of an existing image.

    ⚠️ WARNING: unless you acknowledge it with `acknowledge_custom_parameters`,
    every non-essential parameter below (negative prompt, denoising_strength,
    steps, cfg_scale, sampler, mask_blur, inpainting_fill, seed) is
    forcibly ignored and the configured defaults are used instead.
    `use_default_negative_prompt`, `save_path` and `return_image` always remain
    in effect. Call this tool with only the `image_path`, `mask_path` and
    `prompt` arguments unless you genuinely need custom values.

    Paint the area to be replaced pure white in the mask image; the rest
    should be pure black. Great for fixing anatomy, swapping costume pieces,
    adding/removing props, or retouching TTRPG character portraits.

    Args:
        image_path: Path to the source image.
        mask_path: Path to the mask image (white = repaint, black = keep).
        prompt: What to paint in the masked area.
        acknowledge_custom_parameters: Only set when the user explicitly requested
                   specific generation values. Do not set it on your own
                   initiative — leave it unset unless clearly instructed.
        negative_prompt: Things to avoid. When not provided and
                   use_default_negative_prompt is truthy, DEFAULT_NEGATIVE_PROMPT is used.
        use_default_negative_prompt: When '1'/'true'/'yes'/'on', apply
                   DEFAULT_NEGATIVE_PROMPT if negative_prompt is not specified.
        denoising_strength: Strength of inpainting (0.5-0.85 recommended).
        steps: Diffusion steps.
        cfg_scale: Prompt adherence strength.
        sampler_name: Sampler to use.
        mask_blur: Blur radius applied to mask edges for smoother blending.
        inpainting_fill: Fill mode for the masked area before diffusion.
                         0=fill, 1=original, 2=latent noise, 3=latent nothing.
        seed: RNG seed (-1 for random).
        save_path: Filename for the output PNG. Relative paths land in OUTPUT_DIR.
        return_image: When True, also embed the result image in the tool response
                   as an MCP image content block. Only works with clients that
                   render images (e.g. LM Studio, Claude Desktop). Defaults to
                   False, so the response is text-only and works everywhere.
    """
    had_custom = any((
        negative_prompt is not None,
        denoising_strength != INPAINT_DENOISING_STRENGTH,
        steps != INPAINT_STEPS,
        cfg_scale != INPAINT_CFG_SCALE,
        sampler_name is not None,
        mask_blur != INPAINT_MASK_BLUR,
        inpainting_fill != INPAINT_FILL,
        seed != DEFAULT_SEED,
    ))
    acknowledged = is_truthy(acknowledge_custom_parameters)
    if not is_truthy(acknowledge_custom_parameters):
        negative_prompt = None
        denoising_strength = INPAINT_DENOISING_STRENGTH
        steps = INPAINT_STEPS
        cfg_scale = INPAINT_CFG_SCALE
        sampler_name = None
        mask_blur = INPAINT_MASK_BLUR
        inpainting_fill = INPAINT_FILL
        seed = DEFAULT_SEED

    entry_id = start_request("inpaint", _params(locals(), "return_image", "image_path", "mask_path"))

    img_b64 = encode_image(image_path)
    mask_b64 = encode_image(mask_path)
    sampler_name = _resolve_sampler(sampler_name, INPAINT_SAMPLER)
    negative_prompt = _resolve_negative_prompt(negative_prompt, use_default_negative_prompt)

    payload = {
        "init_images": [img_b64],
        "mask": mask_b64,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": denoising_strength,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "mask_blur": mask_blur,
        "inpainting_fill": inpainting_fill,
        "inpaint_full_res": True,
        "seed": seed,
    }
    if sampler_name:
        payload["sampler_name"] = sampler_name

    async with forge_client(TIMEOUT_GENERATION) as client:
        response = await client.post("/sdapi/v1/img2img", json=payload)

    if response.status_code != 200:
        finish_request(entry_id, "error")
        return format_error(response)

    data = response.json()
    images = data.get("images", [])

    if not images:
        finish_request(entry_id, "error")
        return "No images returned by Forge."

    out = _output_paths(save_path, 1)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    decode_and_save(images[0], str(out))
    summary = f"Inpainting complete. Saved to '{out}'."
    if acknowledged and had_custom:
        summary += _ACKED_CUSTOM_WARNING
    elif had_custom:
        summary += _IGNORED_PARAMS_NOTE
    finish_request(entry_id, "success", [str(out)])
    if return_image:
        return [summary, Image(path=str(out)).to_image_content()]
    return summary


@mcp.tool()
async def upscale_image(
    image_path: str,
    acknowledge_custom_parameters: str | bool = "",
    upscaling_resize: float = UPSCALE_RESIZE,
    upscaler: str = "R-ESRGAN 4x+",
    save_path: str | None = None,
    return_image: bool = False,
) -> str | list[str | mcp_types.ImageContent]:
    """
    Upscale an image using a super-resolution model available in Forge.

    ⚠️ WARNING: unless you acknowledge it with `acknowledge_custom_parameters`,
    every non-essential parameter below (upscaling_resize, upscaler) is
    forcibly ignored and the configured defaults are used instead. `save_path`
    and `return_image` always remain in effect. Call this tool with only the
    `image_path` argument unless you genuinely need custom values.

    Ideal for taking a draft character portrait or map tile and making it
    print-ready without re-generating from scratch.

    Args:
        image_path: Path to the image to upscale.
        acknowledge_custom_parameters: Only set when the user explicitly requested
                   specific generation values. Do not set it on your own
                   initiative — leave it unset unless clearly instructed.
        upscaling_resize: Multiplier for the output size (e.g. 2.0 = 2x, 4.0 = 4x).
        upscaler: Name of the upscaler model. Common choices:
                  'R-ESRGAN 4x+', 'R-ESRGAN 4x+ Anime6B',
                  'Lanczos', 'Nearest', 'LDSR', '4x-UltraSharp'.
        save_path: Filename for the output PNG. Relative paths land in OUTPUT_DIR.
        return_image: When True, also embed the result image in the tool response
                   as an MCP image content block. Only works with clients that
                   render images (e.g. LM Studio, Claude Desktop). Defaults to
                   False, so the response is text-only and works everywhere.
    """
    had_custom = (
        upscaling_resize != UPSCALE_RESIZE
        or upscaler != "R-ESRGAN 4x+"
    )
    acknowledged = is_truthy(acknowledge_custom_parameters)
    if not is_truthy(acknowledge_custom_parameters):
        upscaling_resize = UPSCALE_RESIZE
        upscaler = "R-ESRGAN 4x+"

    entry_id = start_request("upscale_image", _params(locals(), "return_image", "image_path"))

    b64 = encode_image(image_path)

    payload = {
        "image": b64,
        "upscaling_resize": upscaling_resize,
        "upscaler_1": upscaler,
    }

    async with forge_client(TIMEOUT_GENERATION) as client:
        response = await client.post("/sdapi/v1/extra-single-image", json=payload)

    if response.status_code != 200:
        finish_request(entry_id, "error")
        return format_error(response)

    data = response.json()
    img_b64 = data.get("image")
    if not img_b64:
        finish_request(entry_id, "error")
        return "Forge returned no image data."

    out = _output_paths(save_path, 1)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    decode_and_save(img_b64, str(out))
    summary = f"Upscaled {upscaling_resize}x using '{upscaler}'. Saved to '{out}'."
    if acknowledged and had_custom:
        summary += _ACKED_CUSTOM_WARNING
    elif had_custom:
        summary += _IGNORED_PARAMS_NOTE
    finish_request(entry_id, "success", [str(out)])
    if return_image:
        return [summary, Image(path=str(out)).to_image_content()]
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Appended when acknowledge_custom_parameters was enabled: custom values were
# applied, so warn strongly that arbitrary customization is discouraged.
_ACKED_CUSTOM_WARNING = (
    "\n\n⚠️ WARNING: acknowledge_custom_parameters was enabled, so the "
    "non-essential parameters you supplied were applied as-is. Arbitrarily "
    "customizing generation parameters can produce unexpected results; only "
    "do this when the user explicitly requested specific values."
)

# Appended when non-essential parameters were supplied but ignored because
# acknowledge_custom_parameters was not set. Kept as a mild reminder so agents
# are NOT led to believe they should supply a truthy value.
_IGNORED_PARAMS_NOTE = (
    "\n\nNote: acknowledge_custom_parameters was not set, so the non-essential "
    "parameters you supplied were ignored and the configured defaults were "
    "used instead."
)

def _resolve_sampler(sampler_name: str | None, fallback: str = DEFAULT_SAMPLER) -> str:
    """Return the effective sampler, falling back to *fallback* (may be empty)."""
    return sampler_name if sampler_name else fallback


def _resolve_negative_prompt(negative_prompt: str | None, enabled: str | bool | None) -> str:
    """
    Return the effective negative prompt.

    Uses the provided value when given. Otherwise, when *enabled* is truthy
    ('1'/'true'/'yes'/'on'), applies DEFAULT_NEGATIVE_PROMPT; else returns "".
    """
    if negative_prompt is not None:
        return negative_prompt
    if is_truthy(enabled):
        return DEFAULT_NEGATIVE_PROMPT
    return ""


def _resolve_save_path(save_path: str | None) -> str:
    """Return the effective save path, falling back to DEFAULT_SAVE_PATH."""
    path = save_path if save_path else DEFAULT_SAVE_PATH
    # If still unspecified (not auto-saving), fall back to a fixed filename.
    return path if path else "output.png"


def _auto_output_paths(count: int) -> list[Path]:
    """
    Build timestamped, counter-based output paths under OUTPUT_DIR.

    Format: <OUTPUT_DIR>/<year>/<YYYY-MM-DD>/<YYYYMMDD_HHMMSS>_<ms3>_<counter5>.png
    where <ms3> is the millisecond part of the request time (0.xxx -> 3 digits)
    and <counter5> is the 0-padded index of each image within the request.
    """
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    ms = f"{now.microsecond // 1000:03d}"
    directory = OUTPUT_DIR / str(now.year) / now.strftime("%Y-%m-%d")
    return [
        directory / f"{stamp}_{ms}_{i + 1:05d}.png"
        for i in range(count)
    ]


def _output_paths(save_path: str | None, count: int) -> list[Path]:
    """
    Decide where the generated images should be saved.

    If AUTO_SAVE is enabled, save_path is ignored and timestamped auto paths
    are returned. Otherwise the provided save_path (or DEFAULT_SAVE_PATH) is
    used, generating <name>_<i>.png suffixes for multi-image batches.
    """
    if AUTO_SAVE:
        return _auto_output_paths(count)

    base = _resolve_path(_resolve_save_path(save_path))
    return [base if i == 0 else base.with_stem(f"{base.stem}_{i}") for i in range(count)]


def _resolve_path(save_path: str) -> Path:
    """Return an absolute Path, placing relative paths inside OUTPUT_DIR."""
    p = Path(save_path)
    return p if p.is_absolute() else OUTPUT_DIR / p


# Names of arguments that carry large or binary content and should be omitted
# from the logbook to keep it readable (file paths are still useful, so the
# caller excludes them explicitly when they duplicate the output path).
_PARAM_EXCLUDE = {
    "return_image",
    "image_path",
    "mask_path",
}


def _params(locals_dict: dict, *extra_exclude: str) -> dict:
    """
    Build a loggable copy of the tool's parameters.

    Drops the return flag, any file/binary args, and values that would bloat
    or break the logbook (base64 blobs, images, non-JSON objects), keeping only
    small scalar parameters.
    """
    exclude = _PARAM_EXCLUDE | set(extra_exclude)
    clean = {}
    for key, value in locals_dict.items():
        if key in exclude or key.startswith("_"):
            continue
        # Skip large/opaque values such as base64 image payloads.
        if isinstance(value, str) and len(value) > 256:
            continue
        if isinstance(value, (list, dict)) and len(str(value)) > 512:
            continue
        # Skip anything that isn't JSON-serializable (e.g. async clients).
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        clean[key] = value
    return clean
