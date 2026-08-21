import json
from pathlib import Path

import mcp.types as mcp_types

from config import OUTPUT_DIR, TIMEOUT_GENERATION
from fastmcp.utilities.types import Image
from logbook import finish_request, start_request
from mcp_instance import mcp
from utils import decode_and_save, encode_image, forge_client, format_error


@mcp.tool()
async def txt2img(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 20,
    cfg_scale: float = 7.0,
    width: int = 1024,
    height: int = 1024,
    sampler_name: str = "Euler a",
    seed: int = -1,
    batch_size: int = 1,
    save_path: str = "output.png",
    return_image: bool = False,
) -> str | list[str | mcp_types.ImageContent]:
    """
    Generate one or more images from a text prompt using Stable Diffusion Forge.

    Each image in the batch is saved as <save_path>, <save_path>_1.png, etc.
    Use seed=-1 for a random seed. Returns the seed(s) that were used so the
    result can be reproduced later. Images are saved inside the configured
    OUTPUT_DIR unless save_path is an absolute path.

    Args:
        prompt: Positive prompt describing the desired image.
        negative_prompt: Things to avoid in the image.
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
    entry_id = start_request("txt2img", _params(locals(), "return_image"))

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "sampler_name": sampler_name,
        "seed": seed,
        "batch_size": batch_size,
    }

    async with forge_client(TIMEOUT_GENERATION) as client:
        response = await client.post("/sdapi/v1/txt2img", json=payload)

    if response.status_code != 200:
        finish_request(entry_id, "error")
        return format_error(response)

    data = response.json()
    images = data.get("images", [])
    info = data.get("info", {})
    seeds = info.get("all_seeds", [seed] * len(images)) if isinstance(info, dict) else [seed]

    base = _resolve_path(save_path)
    saved = []
    for i, img_b64 in enumerate(images):
        out = base if i == 0 else base.with_stem(f"{base.stem}_{i}")
        decode_and_save(img_b64, str(out))
        saved.append(str(out))

    summary = (
        f"Generated {len(saved)} image(s).\n"
        f"Saved to: {', '.join(saved)}\n"
        f"Seeds used: {seeds}"
    )

    finish_request(entry_id, "success", saved)

    if return_image:
        return [summary, *(Image(path=str(out)).to_image_content() for out in saved)]

    return summary


@mcp.tool()
async def img2img(
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    denoising_strength: float = 0.6,
    steps: int = 20,
    cfg_scale: float = 7.0,
    width: int = 0,
    height: int = 0,
    sampler_name: str = "Euler a",
    seed: int = -1,
    save_path: str = "output_img2img.png",
    return_image: bool = False,
) -> str | list[str | mcp_types.ImageContent]:
    """
    Transform an existing image guided by a text prompt (image-to-image).

    Useful for restyling character art, adding details to maps, converting
    sketches to finished illustrations, or applying a new art style.

    Args:
        image_path: Path to the source image file (PNG/JPG).
        prompt: Positive prompt describing the desired result.
        negative_prompt: Things to avoid in the output.
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
    entry_id = start_request("img2img", _params(locals(), "return_image", "image_path"))

    b64 = encode_image(image_path)

    payload = {
        "init_images": [b64],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": denoising_strength,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "seed": seed,
    }
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

    out = _resolve_path(save_path)
    decode_and_save(images[0], str(out))
    summary = f"img2img complete. Saved to '{out}'. Seed: {used_seed}"
    finish_request(entry_id, "success", [str(out)])
    if return_image:
        return [summary, Image(path=str(out)).to_image_content()]
    return summary


@mcp.tool()
async def inpaint(
    image_path: str,
    mask_path: str,
    prompt: str,
    negative_prompt: str = "",
    denoising_strength: float = 0.75,
    steps: int = 20,
    cfg_scale: float = 7.0,
    sampler_name: str = "Euler a",
    mask_blur: int = 4,
    inpainting_fill: int = 1,
    seed: int = -1,
    save_path: str = "output_inpaint.png",
    return_image: bool = False,
) -> str | list[str | mcp_types.ImageContent]:
    """
    Inpaint (fill or redraw) a masked region of an existing image.

    Paint the area to be replaced pure white in the mask image; the rest
    should be pure black. Great for fixing anatomy, swapping costume pieces,
    adding/removing props, or retouching TTRPG character portraits.

    Args:
        image_path: Path to the source image.
        mask_path: Path to the mask image (white = repaint, black = keep).
        prompt: What to paint in the masked area.
        negative_prompt: Things to avoid.
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
    entry_id = start_request("inpaint", _params(locals(), "return_image", "image_path", "mask_path"))

    img_b64 = encode_image(image_path)
    mask_b64 = encode_image(mask_path)

    payload = {
        "init_images": [img_b64],
        "mask": mask_b64,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": denoising_strength,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "mask_blur": mask_blur,
        "inpainting_fill": inpainting_fill,
        "inpaint_full_res": True,
        "seed": seed,
    }

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

    out = _resolve_path(save_path)
    decode_and_save(images[0], str(out))
    summary = f"Inpainting complete. Saved to '{out}'."
    finish_request(entry_id, "success", [str(out)])
    if return_image:
        return [summary, Image(path=str(out)).to_image_content()]
    return summary


@mcp.tool()
async def upscale_image(
    image_path: str,
    upscaling_resize: float = 2.0,
    upscaler: str = "R-ESRGAN 4x+",
    save_path: str = "output_upscaled.png",
    return_image: bool = False,
) -> str | list[str | mcp_types.ImageContent]:
    """
    Upscale an image using a super-resolution model available in Forge.

    Ideal for taking a draft character portrait or map tile and making it
    print-ready without re-generating from scratch.

    Args:
        image_path: Path to the image to upscale.
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

    out = _resolve_path(save_path)
    decode_and_save(img_b64, str(out))
    summary = f"Upscaled {upscaling_resize}x using '{upscaler}'. Saved to '{out}'."
    finish_request(entry_id, "success", [str(out)])
    if return_image:
        return [summary, Image(path=str(out)).to_image_content()]
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
