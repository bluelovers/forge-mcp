# Forge Painter — MCP Server

An MCP server for [SD WebUI Forge (Neo Branch)](https://github.com/lllyasviel/stable-diffusion-webui-forge) that exposes image generation tools to MCP-compatible clients such as Claude Desktop, opencode, Cursor, and others.

## Tools

| Tool | Description |
|---|---|
| `txt2img` | Generate an image from a text prompt |
| `img2img` | Generate an image from a prompt + input image |
| `inpaint` | Inpaint a masked region of an image |
| `upscale_image` | Upscale an image |
| `get_models` / `set_model` / `get_current_model` / `refresh_models` | Manage checkpoints |
| `get_loras` / `get_samplers` / `get_embeddings` | List prompt-able assets (LoRAs, samplers, TI embeddings) |
| `get_upscalers` / `get_vaes` / `get_current_vae` | List upscalers and VAE models |
| `get_progress` / `interrupt_generation` | Monitor and control active jobs |
| `info` | Show runtime config (fastmcp version, URLs, defaults, timeouts) |

## Compatibility

This server targets **SD WebUI Forge Neo** and is developed and tested against it. That said, Forge Neo's API is a superset of the standard AUTOMATIC1111 API, so **sd-webui-automatic1111 (legacy A1111) may work** if you point the server URL at your A1111 instance.

> **Untested.** No compatibility testing has been done against legacy A1111. Core tools (`txt2img`, `img2img`, `inpaint`, `get_models`, `get_samplers`, etc.) map to standard `/sdapi/v1/` endpoints and are likely to work. Tools that rely on forge-specific extensions may not behave correctly or may return errors. Use at your own risk and open an issue if you encounter problems.

---

## Installation

**Prerequisites:**

- Python 3.10+
- SD WebUI Forge Neo running locally (default: `http://127.0.0.1:7860`)
- [`uv`](https://github.com/astral-sh/uv) installed and on your `PATH` (optional, see below)
- `fastmcp` 3.4.7 (installed automatically from `requirements.txt`)

### Option A — Download from Releases (recommended)

Download the latest `forge-painter.mcpb` from the [Releases page](../../releases/latest), then drag it into Claude Desktop (or double-click it). Claude Desktop will prompt for your Forge URL and optional credentials, then handle the rest automatically — no Python installation required.

| Prompt | Default | Description |
|---|---|---|
| Forge URL | `http://127.0.0.1:7860` | URL of your Forge Neo (or A1111) instance |
| API Username | _(blank)_ | Only needed if Forge was launched with `--api-auth` |
| API Password | _(blank)_ | Only needed if Forge was launched with `--api-auth` |
| Output Directory | `Documents/forge-painter` | Folder where generated images will be saved |

---

### Option B — Manual setup (development)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and edit as needed:

```powershell
copy .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `FORGE_URL` | `http://127.0.0.1:7860` | Forge Neo (or A1111) instance URL |
| `FORGE_API_USER` | _(blank)_ | Username if Forge was launched with `--api-auth` |
| `FORGE_API_PASSWORD` | _(blank)_ | Password if Forge was launched with `--api-auth` |
| `OUTPUT_DIR` | `outputs` | Directory where generated images are saved |
| `AUTO_SAVE` | `0` | When `1`/`true`/`yes`, save to timestamped auto paths |
| `DEFAULT_WIDTH` / `DEFAULT_HEIGHT` | `512` / `512` | Default txt2img size |
| `DEFAULT_SEED` | `-1` | Default seed (`-1` = random) |
| `DEFAULT_BATCH_SIZE` | `1` | Images per txt2img request |
| `DEFAULT_STEPS` | `9` | Default diffusion steps |
| `DEFAULT_CFG_SCALE` | `1.0` | Default CFG scale |
| `DEFAULT_SAMPLER` | _(blank)_ | Default sampler (e.g. `Euler a`) |
| `DEFAULT_SAVE_PATH` | `output.png` | Default output filename |
| `DEFAULT_NEGATIVE_PROMPT` | _(blank)_ | Applied when a tool passes `use_default_negative_prompt` |
| `IMG2IMG_STEPS` / `IMG2IMG_CFG_SCALE` / `IMG2IMG_SAMPLER` | inherits txt2img | Per-tool overrides for img2img |
| `IMG2IMG_DENOISING_STRENGTH` | `0.6` | img2img denoising strength |
| `INPAINT_STEPS` / `INPAINT_CFG_SCALE` / `INPAINT_SAMPLER` | inherits txt2img | Per-tool overrides for inpaint |
| `INPAINT_DENOISING_STRENGTH` | `0.75` | inpaint denoising strength |
| `INPAINT_MASK_BLUR` | `4` | inpaint mask blur radius |
| `INPAINT_FILL` | `1` | inpaint fill mode (`0`=fill, `1`=original, `2`=latent noise, `3`=latent nothing) |
| `UPSCALE_RESIZE` | `2.0` | default upscale multiplier |
| `UPSCALER` | `R-ESRGAN 4x+` | default upscaler model |
| `LOG_DIR` | `$OUTPUT_DIR` | Directory for the request logbook |
| `LOG_MAX_ENTRIES` | `100` | Max logbook entries kept |
| `TIMEOUT_GENERATION` | `300` | Seconds to wait for txt2img/img2img/inpaint/upscale |
| `TIMEOUT_MODEL_SWITCH` | `120` | Seconds to wait for a checkpoint switch |
| `TIMEOUT_INFO` | `30` | Seconds to wait for listing endpoints |
| `TIMEOUT_CONTROL` | `10` | Seconds to wait for progress/interrupt |

Then register the server with your MCP client of choice.

#### Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "forge-painter": {
      "command": "C:\\path\\to\\forge-mcp\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\forge-mcp\\server.py"],
      "cwd": "C:\\path\\to\\forge-mcp"
    }
  }
}
```

Restart Claude Desktop. **Forge Painter** will appear in the MCP tools panel.

#### opencode

Add to your `opencode.json` (project or `~/.config/opencode/opencode.json`):

```json
{
  "mcp": {
    "forge-painter": {
      "type": "local",
      "command": [
        "D:/Program Files (AI)/forge-mcp/server-start"
      ]
    }
  }
}
```

`server-start` is a wrapper script (`server-start.bat` / `.ps1` / `.sh`) that runs `server.py` from the `.venv` if present, otherwise falling back to `uv run server.py`. Point the command at your actual checkout path. Restart opencode after editing the config.

---

### Building the .mcpb yourself

One-time prerequisites:

```powershell
pip install uv
npm install -g @anthropic-ai/mcpb
```

Build and sign:

```powershell
mcpb validate manifest.json
mcpb pack . forge-painter.mcpb
$env:PATH += ";C:\Program Files\Git\usr\bin"  # provides openssl
mcpb sign forge-painter.mcpb --self-signed    # claude might not open the .mcpb when signed, do at your own peril
```

To publish a new release: bump `version` in `manifest.json` and `pyproject.toml`, rerun the build commands above, and attach the updated `.mcpb` as an asset to a new GitHub Release.

---

## Troubleshooting

- If the Forge URL is unreachable, tools return errors but the server itself still loads.
- For the `.mcpb` option, `uv` must be on your `PATH` when Claude Desktop launches the server.
- The `info` tool reports the running fastmcp version and all effective config values (credentials are masked) — use it to confirm your `.env` was picked up.
- `refresh_models` re-scans Forge's model folders after you add new checkpoints, LoRAs or embeddings.