# Habit Memory Demo — ComfyUI Video Visualization Workflow

## 1. Overview

Generate short animated videos in **French animation style** (法式动画) that visualize user daily driving behaviors extracted from mockup data. The pipeline is built as a **ComfyUI node-graph workflow**, combining LLM prompt generation nodes, Kling 3.0 Partner Nodes, and batch iteration — all orchestrated visually and exportable as JSON for headless execution.

**Core Pipeline (as ComfyUI node graph)**:

```
[Scene Data Loader] → [LLM Prompt Generator] → [Kling Image-to-Video] → [Video Save]
                                                       ↑
                                            [Character Reference Image]
```

### 1.1 Design Goals

| Goal | Description |
|------|------------|
| Visual style | French animation — soft watercolor textures, pastel-to-muted palette, hand-drawn linework, cinematic lighting, Sylvain Chomet / Gobelins influence |
| Character consistency | Reference image (Kling Text-to-Image) locked as `start_frame` across all I2V generations |
| Data fidelity | Each video reflects actual mockup data: weather, time of day, temperature, specific actions taken |
| Narrative flow | 5 days x 3 habit scenes = 15 core clips, batch-generated via ComfyUI Job Iterator |
| Reproducibility | Workflow exported as JSON (API format), version-controlled, headless-executable |

### 1.2 Technology Stack

| Component | Technology | ComfyUI Node |
|-----------|-----------|-------------|
| Prompt generation | OpenAI GPT-4o / Gemini | `OpenAI Compatible LLM` (custom node) |
| Character reference | Kling 3.0 Image | `Kling Text to Image` (Partner Node) |
| Video generation | Kling 3.0 Video | `Kling Image to Video` (Partner Node) |
| Batch iteration | Multi-scene loop | `Job Iterator` / `Prompt Iterator` (custom node) |
| Workflow orchestration | ComfyUI | Subgraphs for modular grouping |
| Output | `output/videos/` | `SaveVideo` / `VHS_VideoCombine` |

### 1.3 Node Dependencies (all verified installed)

| Package | Status | Nodes Used |
|---------|--------|------------|
| **Kling Partner Nodes** | Built-in (`comfy_api_nodes/nodes_kling.py`) | `KlingImageGenerationNode`, `KlingImage2VideoNode`, `KlingCameraControls` |
| **OpenAI Partner Nodes** | Built-in (`comfy_api_nodes/nodes_openai.py`) | `OpenAIChatNode`, `OpenAIChatConfig` |
| **Gemini Partner Nodes** | Built-in (`comfy_api_nodes/nodes_gemini.py`) | `GeminiNode` (fallback for LLM) |
| **Core Nodes** | Built-in (`nodes.py`, `comfy_extras/nodes_video.py`) | `SaveImage`, `PreviewImage`, `SaveVideo` |
| **VideoHelperSuite** | Installed (`custom_nodes/comfyui-videohelpersuite`) | `VHS_VideoCombine` (optional, for frame-based output) |

No additional custom node installation required.

---

## 2. French Animation Style Specification

### 2.1 Style Prompt Template

Appended as suffix to every Kling generation prompt:

```
French animation style, hand-painted watercolor textures,
soft pastel color palette with warm undertones,
delicate ink linework, subtle film grain,
cinematic composition, European illustration aesthetic,
inspired by Sylvain Chomet and Gobelins animation,
gentle ambient lighting, atmospheric depth
```

### 2.2 Negative Prompt Template

Shared across all nodes:

```
photorealistic, 3D render, CGI, anime style, cartoon style,
low quality, blurry, distorted face, extra limbs, deformed hands,
text, watermark, logo, signature, frame border, uncanny valley
```

### 2.3 Color Palette by Time of Day

| Time Window | Dominant Palette | Mood |
|-------------|-----------------|------|
| Morning (06:00-10:00) | Warm amber, soft peach, pale blue sky | Fresh, hopeful |
| Midday (10:00-14:00) | Bright cream, muted teal, sage green | Active, clear |
| Afternoon (14:00-17:00) | Golden ochre, dusty rose, warm gray | Settled, calm |
| Evening (17:00-21:00) | Deep indigo, warm orange glow, violet | Intimate, transitional |

### 2.4 Character Design — Mary

| Attribute | Value |
|-----------|-------|
| Gender | Female |
| Age | ~30s |
| Appearance | European, brown hair in casual updo, natural look |
| Clothing | Smart-casual — light jacket, scarf (changes subtly per day) |
| Expression | Calm, focused while driving; relaxed at home |
| Vehicle | Modern European compact EV, interior visible in most shots |

---

## 3. ComfyUI Workflow Architecture

The workflow is organized into **4 Subgraphs** (modular node groups), connected left-to-right.

### 3.1 High-Level Node Graph

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SUBGRAPH 1: Character Reference                      │
│                                                                         │
│  [Style Prompt]──┐                                                      │
│                  ├──▶[Kling Text to Image]──▶[Preview]──▶[Save Image]  │
│  [Neg Prompt]────┘         │                                            │
│                            │ IMAGE output                               │
└────────────────────────────┼────────────────────────────────────────────┘
                             │ (ref_image)
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SUBGRAPH 2: Scene Data + LLM Prompting               │
│                                                                         │
│  [Load Scene JSON]──▶[Scene Extractor]──┐                               │
│                                         ├──▶[OpenAI Compatible LLM]     │
│  [System Prompt Template]───────────────┘         │                     │
│                                                   │ STRING output       │
│                                          ┌────────┘                     │
│                                          ▼                              │
│                                  [Parse LLM JSON]                       │
│                                   │prompt  │neg_prompt  │camera         │
└───────────────────────────────────┼────────┼────────────┼───────────────┘
                                    │        │            │
                                    ▼        ▼            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SUBGRAPH 3: Video Generation                         │
│                                                                         │
│  ref_image ──────▶ start_frame                                          │
│  prompt ─────────▶ prompt        [Kling Image to Video]──▶[VIDEO]      │
│  neg_prompt ─────▶ negative_prompt     │                                │
│  camera ─────────▶ [Kling Camera Controls]                              │
│                                        │                                │
│  Fixed params:                         │                                │
│    model_name: kling-v2-master         │                                │
│    cfg_scale: 0.5                      │                                │
│    mode: pro                           │                                │
│    duration: 5s                        │                                │
│    aspect_ratio: 16:9                  │                                │
└────────────────────────────────────────┼────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SUBGRAPH 4: Output & Save                            │
│                                                                         │
│  [VIDEO]──▶[VHS_VideoCombine]──▶[SaveVideo: output/videos/mary/...]    │
│                                                                         │
│  Filename pattern: {user}/day{day}/{scene}.mp4                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Batch Loop Wrapper

For processing all 15 scenes, the Subgraphs 2-4 are wrapped in a **Job Iterator**:

```
[Job Iterator]
  ├── source: scenes_manifest.json (15 entries)
  ├── per iteration:
  │     ├── Load scene[i] data
  │     ├── Run Subgraph 2 (LLM prompt)
  │     ├── Run Subgraph 3 (Kling I2V, reuse ref_image)
  │     └── Run Subgraph 4 (save with scene-specific filename)
  └── output: 15 video files
```

---

## 4. Node Specifications

### 4.1 Kling Text to Image (Character Reference)

**ComfyUI Node**: `KlingImageGenerationNode` (Partner Node, built-in)

| Parameter | Value | Notes |
|-----------|-------|-------|
| prompt | See Section 4.1.1 | Character reference prompt |
| negative_prompt | See Section 2.2 | Shared negative template |
| image_type | `subject` | Reference image type |
| image_fidelity | `0.5` | Reference intensity |
| human_fidelity | `0.45` | Subject reference similarity |
| model_name | `kling-v2` | Enum: kling-v1, kling-v1-5, kling-v2 |
| aspect_ratio | `1:1` | Enum: 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3, 21:9 |
| n | `1` | Number of images |

**Output**: IMAGE (index 0) → saved as `output/videos/mary/reference.png`

#### 4.1.1 Character Reference Prompt

```
Character reference of Mary, a 30-year-old European woman with
light brown hair in a casual updo, warm brown eyes, natural look.
Wearing a smart-casual light blue jacket over a cream top, with
a thin wool scarf. Sitting in the driver seat of a modern compact
electric vehicle. Three-quarter portrait view showing face and
upper body clearly. Soft natural lighting from car window.
French animation style, hand-painted watercolor textures,
soft pastel color palette, delicate ink linework,
European illustration aesthetic, gentle ambient lighting.
```

### 4.2 OpenAI ChatGPT (Prompt Generator)

**ComfyUI Nodes**: `OpenAIChatConfig` + `OpenAIChatNode` (Partner Nodes, built-in)

**Node A — `OpenAIChatConfig`** (system prompt + options):

| Parameter | Value | Notes |
|-----------|-------|-------|
| truncation | `auto` | Context window strategy |
| max_output_tokens | `600` | Sufficient for scene prompt JSON |
| instructions | See Section 4.2.1 | LLM system prompt (scene writer) |

**Node B — `OpenAIChatNode`** (prompt → response):

| Parameter | Value | Notes |
|-----------|-------|-------|
| prompt | Dynamic (from Scene Extractor) | See Section 4.2.2 |
| persist_context | `false` | No multi-turn needed |
| model | `gpt-4o` | Enum: gpt-4o, gpt-4.1, o4-mini, etc. |
| advanced_options | Wired from OpenAIChatConfig | System prompt config |

**Output**: STRING (index 0) — raw LLM response (JSON text)

**Alternative**: `GeminiNode` (built-in Partner Node) with `gemini-2.5-flash` if OpenAI quota is exhausted.

#### 4.2.1 LLM System + User Prompt Structure

**System prompt** (wired as prefix to `prompt` input):

```
You are a cinematic scene writer for French-style animated short films
about daily life in Paris. Given structured driving behavior data,
write a vivid scene description suitable for AI video generation.

Rules:
- Write 2-3 sentences of visual storytelling
- Include: weather atmosphere, time-of-day lighting, character actions, mood
- Naturally reference the specific driving actions from the data
- Match the color palette to the time of day
- The character is Mary, a ~30-year-old European woman
- DO NOT describe Mary's appearance (reference image handles this)
- DO describe environment, lighting, weather, actions, dashboard, vehicle
- Always end with style keywords: "French animation style, hand-painted
  watercolor textures, soft pastel palette, delicate ink linework"

Output strictly as JSON:
{
  "prompt": "cinematic scene description for video generation",
  "negative_prompt": "elements to avoid",
  "camera_type": "static | pan_left | pan_right | zoom_in | zoom_out | tracking"
}
```

**User prompt** (dynamic, built from scene data):

```
Scene: {scene_name}
Day: {day} ({weekday})
Time: {time}
Weather: {weather}, {outside_temp_c}°C
Location: {location}
Actions performed:
{actions_list}
```

#### 4.2.2 Expected LLM Output Example

```json
{
  "prompt": "A cloudy Paris morning, soft gray light filtering through bare autumn trees lining a quiet residential street. Inside a modern EV cabin, warm amber dashboard lights glow as the climate adjusts to 23 degrees, seat warmers humming gently. A podcast interface illuminates the center screen while the car silently pulls away in eco mode, the city slowly waking beyond rain-speckled windows. French animation style, hand-painted watercolor textures, soft pastel palette, delicate ink linework.",
  "negative_prompt": "photorealistic, 3D render, CGI, anime, low quality, blurry, distorted, text, watermark",
  "camera_type": "tracking"
}
```

### 4.3 Kling Image to Video (Scene Animation)

**ComfyUI Node**: `KlingImage2VideoNode` (Partner Node, built-in)

| Parameter | Value | Source |
|-----------|-------|--------|
| start_frame | IMAGE | Wired from `KlingImageGenerationNode` output [0] |
| prompt | STRING | Wired from `OpenAIChatNode` output [0] |
| negative_prompt | STRING | Hardcoded (Section 2.2) |
| model_name | `kling-v2-1-master` | Enum: kling-v1, v1-5, v1-6, v2-master, v2-1, v2-1-master |
| cfg_scale | `0.5` | 0.0-1.0; low = more creative, style-faithful |
| mode | `pro` | Enum: std, pro |
| aspect_ratio | `16:9` | Enum: 16:9, 9:16, 1:1 |
| duration | `5` | Enum: 5, 10 (seconds) |

**Output**: VIDEO (index 0), video_id (index 1), duration (index 2)

### 4.4 Kling Camera Controls (Optional)

**ComfyUI Node**: `KlingCameraControls` → `KlingCameraControlI2VNode` (Partner Nodes)

Connected to `KlingCameraControlI2VNode` variant (instead of basic `KlingImage2VideoNode`) when camera movement is desired.

| Parameter | Source |
|-----------|--------|
| camera_control | Output from `KlingCameraControls` node |

### 4.5 Kling OmniVideo (Alternative Path)

**ComfyUI Node**: `Kling Omni Video` (Partner Node)

Used as fallback for complex multi-element scenes where I2V quality is insufficient.

| Parameter | Value |
|-----------|-------|
| input image | ref_image |
| input text | LLM-generated scene prompt |
| model_name | `kling-video-o1` |
| duration | `5s` |

**When to use**: If I2V produces inconsistent character appearance or the scene requires complex camera/environment interaction that basic I2V cannot handle.

---

## 5. Workflow JSON Structure (API Format)

The ComfyUI workflow is exported as API-format JSON for headless execution. Each node is identified by a unique numeric string ID.

### 5.1 Schema Overview

```json
{
  "<node_id>": {
    "inputs": {
      "<param_name>": "<literal_value>",
      "<linked_param>": ["<source_node_id>", <output_index>]
    },
    "class_type": "<NodeClassName>",
    "_meta": {
      "title": "<Display Name>"
    }
  }
}
```

- **Literal values**: Strings, numbers, booleans set directly
- **Linked inputs**: `["source_node_id", output_index]` — references another node's output
- **`class_type`**: Exact registered node class name
- **`_meta.title`**: Human-readable label in the graph editor

### 5.2 Core Workflow JSON (Simplified)

```json
{
  "1": {
    "inputs": {
      "prompt": "Character reference of Mary, a 30-year-old European woman... French animation style, hand-painted watercolor textures...",
      "negative_prompt": "photorealistic, 3D render, CGI, anime style...",
      "model_name": "kling-v2-master",
      "aspect_ratio": "1:1"
    },
    "class_type": "KlingTextToImage",
    "_meta": {"title": "Ref Image: Mary Character Sheet"}
  },

  "10": {
    "inputs": {
      "prompt": "Scene: morning_commute\nDay: 1 (Monday)\nTime: 07:48\nWeather: cloudy, 7.3°C\nLocation: Departing from home (Paris)\nActions:\n- Sets AC to 23°C\n- Seat heating level 2\n- Plays podcast Tech Daily\n- Navigates to work (fastest)\n- Eco driving mode",
      "endpoint": "https://api.openai.com/v1/chat/completions",
      "api_token": "OPENAI_API_KEY",
      "model": "gpt-4o",
      "max_tokens": 500,
      "temperature": 0.8
    },
    "class_type": "OpenAI Compatible LLM",
    "_meta": {"title": "LLM Prompt Generator"}
  },

  "20": {
    "inputs": {
      "start_frame": ["1", 0],
      "prompt": ["10", 0],
      "negative_prompt": "photorealistic, 3D render, CGI, anime style, low quality, blurry, distorted face, text, watermark",
      "model_name": "kling-v2-master",
      "cfg_scale": 0.5,
      "mode": "pro",
      "aspect_ratio": "16:9",
      "duration": "5s"
    },
    "class_type": "KlingImage2Video",
    "_meta": {"title": "Scene Video: Day1 Morning Commute"}
  },

  "30": {
    "inputs": {
      "video": ["20", 0],
      "filename_prefix": "mary/day1/morning_commute",
      "format": "video/h264-mp4"
    },
    "class_type": "VHS_VideoCombine",
    "_meta": {"title": "Save Video"}
  }
}
```

> **Note**: Node `"10"` output → Node `"20"` prompt connection uses `["10", 0]` notation. In practice, the LLM output (raw JSON string) needs a **JSON Parse** intermediate node to extract `.prompt` field. The simplified example above assumes direct wiring for clarity.

### 5.3 Full Workflow with JSON Parse

In production, add an intermediate `StringFunction` or custom parse node:

```json
  "15": {
    "inputs": {
      "text": ["10", 0],
      "expression": "json.loads(text)['prompt']"
    },
    "class_type": "StringFunction",
    "_meta": {"title": "Extract: scene prompt"}
  },
  "16": {
    "inputs": {
      "text": ["10", 0],
      "expression": "json.loads(text)['negative_prompt']"
    },
    "class_type": "StringFunction",
    "_meta": {"title": "Extract: negative prompt"}
  }
```

Then wire `["15", 0]` → `KlingImage2Video.prompt` and `["16", 0]` → `KlingImage2Video.negative_prompt`.

---

## 6. Scene Data & Batch Processing

### 6.1 Scene Manifest Format

A JSON manifest file lists all 15 scenes for batch iteration:

```json
[
  {
    "id": "day1_morning_commute",
    "scene": "morning_commute",
    "day": 1, "weekday": "Monday",
    "time": "07:48", "weather": "cloudy", "temp_c": 7.3,
    "location": "Departing from home (Paris)",
    "actions": [
      "Sets AC to 23°C", "Seat heating level 2",
      "Plays podcast 'Tech Daily'", "Navigates to work (fastest route)",
      "Eco driving mode", "ACC distance: short"
    ],
    "output_path": "mary/day1/morning_commute"
  },
  {
    "id": "day1_arriving_home",
    "scene": "arriving_home",
    "day": 1, "weekday": "Monday",
    "time": "18:42", "weather": "cloudy",
    "location": "Arriving at home (Paris)",
    "actions": [
      "Media off", "AC power off",
      "Window closed", "Keyless entry disabled",
      "Engine off", "Door locked"
    ],
    "output_path": "mary/day1/arriving_home"
  }
]
```

### 6.2 Batch Iteration Strategy

**Option A — ComfyUI Job Iterator Node** (preferred):

```
[Job Iterator] reads scenes_manifest.json
  → For each entry:
      [Text Node: format user prompt from scene fields]
      → [OpenAI Compatible LLM]
      → [Parse JSON]
      → [Kling Image to Video] (start_frame = shared ref_image)
      → [Save Video] (filename = entry.output_path)
```

**Option B — Python Headless Runner** (fallback):

```python
# Load workflow JSON template
# For each scene in manifest:
#   1. Patch node "10" inputs with scene-specific prompt
#   2. Patch node "30" filename_prefix with output_path
#   3. POST to ComfyUI /api/prompt endpoint
#   4. Poll until complete
```

### 6.3 Scene Extractor Script

Pre-processes mockup data into the manifest format:

```python
# scripts/video_gen/scene_extractor.py
from scenarios.mock_data_generator import generate_full_dataset

def build_scene_manifest() -> list[dict]:
    """Convert mockup data → scene manifest for ComfyUI batch processing."""
    dataset = generate_full_dataset()
    manifest = []

    for scene_type, events in dataset["scenes"].items():
        if scene_type == "noise":
            continue  # Skip non-habitual events
        for event in events:
            actions = _extract_actions(event["signals"])
            manifest.append({
                "id": f"day{event['day']}_{scene_type}",
                "scene": scene_type,
                "day": event["day"],
                "weekday": event.get("weekday", ""),
                "time": _extract_time(event["signals"]),
                "weather": event.get("weather", ""),
                "temp_c": event.get("outside_temp_c", ""),
                "location": _derive_location(event),
                "actions": actions,
                "output_path": f"mary/day{event['day']}/{scene_type}",
            })

    return manifest
```

---

## 7. Scene Storyboard — Mary's 5-Day Diary

### 7.1 Morning Commute (x5 days)

| Day | Time | Weather | Temp | Key Visual Moments |
|-----|------|---------|------|--------------------|
| Mon | ~08:00 | Varies | 3-18°C | Mary enters car, adjusts AC, starts podcast, pulls onto Paris street |
| Tue | ~08:00 | Varies | 3-18°C | Similar routine, subtle clothing change, different light |
| Wed | ~08:00 | Varies | 3-18°C | Rainy variant: cozier cabin, warmer AC setting |
| Thu | ~08:00 | Varies | 3-18°C | Sunny morning variant, brighter palette |
| Fri | ~08:00 | Varies | 3-18°C | End-of-week energy, maybe music instead of podcast |

**Camera**: Interior cabin `tracking` → exterior establishing shot of Paris morning.

**Key actions to visualize**: AC temperature adjustment (hand on controls), podcast UI on dashboard, eco mode indicator, seat heating glow, pulling away from residential street.

### 7.2 Arriving Home (x5 days)

| Day | Time | Key Visual Moments |
|-----|------|--------------------|
| Mon | ~18:30 | Evening Paris, apartment in background, Mary parking, turning off systems |
| Tue-Fri | ~18:30 | Variations: sunset vs dusk, closing window, walking to apartment |

**Camera**: Exterior `tracking` → interior close-up → exterior wide `zoom_out` as she exits.

**Key actions to visualize**: Media fading out, AC powering off (dashboard dimming), window closing, engine off, door opening, walking toward apartment entrance.

### 7.3 Toll/Parking Entry (x5 events)

| Event | POI | Speed | Key Visual Moments |
|-------|-----|-------|--------------------|
| 1-5 | toll_A6 / parking_mall | 1.5-4.8 kph | Slow approach, window rolling down 70-85%, hand gesture |

**Camera**: Exterior side `pan_right` → interior close-up `static` → back to exterior.

**Key actions to visualize**: Car decelerating near toll booth/barrier, window sliding down, hand reaching out, French road scenery or parking structure.

---

## 8. Character Consistency Strategy

### 8.1 Reference Image as Anchor

```
Subgraph 1: Generate once
  └─ [Kling Text to Image] → Mary reference (1:1, front 3/4 view)
  └─ Save as output/videos/mary/reference.png
  └─ IMAGE output wired to ALL Kling I2V nodes as start_frame

Subgraphs 2-4: Per-scene generation
  └─ start_frame = ALWAYS the same reference image
  └─ prompt = ONLY environment + actions (never re-describe Mary's appearance)
  └─ Kling I2V preserves character identity from start_frame
```

### 8.2 Prompt Discipline

| Rule | Rationale |
|------|-----------|
| **DO** describe environment, lighting, actions, mood | These vary per scene |
| **DO** mention "Mary" by name | Anchors the character reference |
| **DON'T** re-describe appearance in video prompts | Let `start_frame` carry identity |
| **DO** include style keywords in every prompt | Prevents style drift |
| **DO** keep `cfg_scale` low (0.5) | More faithful to reference image |

### 8.3 Kling 3.0 Subject Locking

Kling 3.0 natively supports **locked subject consistency** across multi-shot generation. When using the I2V workflow with the same `start_frame`, the model maintains character/object identity. This is the primary mechanism for consistency — no IP-Adapter or ControlNet needed.

---

## 9. File Structure

```
scripts/video_gen/
├── __init__.py
├── config.py                  # API keys, base URLs, style constants
├── scene_extractor.py         # Mockup data → scenes_manifest.json
├── build_manifest.py          # CLI: python -m scripts.video_gen.build_manifest
└── README.md                  # Usage instructions

comfyui_workflows/
├── habit_video_full.json      # Complete workflow (API format) — all 4 subgraphs
├── habit_video_single.json    # Single-scene workflow (for testing)
├── ref_image_only.json        # Subgraph 1 only — generate character reference
└── scenes_manifest.json       # Generated by scene_extractor.py

output/videos/
└── mary/
    ├── reference.png           # Character reference sheet
    ├── day1/
    │   ├── morning_commute.mp4
    │   ├── arriving_home.mp4
    │   └── toll_entry.mp4
    ├── day2/ ...
    └── day5/ ...
```

---

## 10. Execution Plan

| Phase | Task | Deliverable |
|-------|------|-------------|
| 1 | Install ComfyUI + required custom nodes | Working ComfyUI environment |
| 2 | Build `scene_extractor.py` + generate `scenes_manifest.json` | 15-entry manifest from mockup data |
| 3 | Build `ref_image_only.json` workflow | Subgraph 1: character reference generation |
| 4 | Generate Mary reference image | `output/videos/mary/reference.png` |
| 5 | Build `habit_video_single.json` workflow | Subgraphs 2-4: single scene end-to-end |
| 6 | Test Day 1 Morning Commute | Validate LLM → Kling → Save pipeline |
| 7 | Review & iterate prompt/style | Adjust LLM system prompt, cfg_scale, style keywords |
| 8 | Build `habit_video_full.json` with Job Iterator | Batch workflow for all 15 scenes |
| 9 | Batch generate all 15 scenes | Full 5-day diary video set |
| 10 | Quality review & re-generation | Fix any consistency/style failures |

### 10.1 ComfyUI Environment (already installed)

```
Location: ~/comfy/ComfyUI/
All required nodes are built-in Partner Nodes (Kling, OpenAI, Gemini, SaveVideo).
VideoHelperSuite already installed in custom_nodes/.

# Start server:
cd ~/comfy/ComfyUI && python main.py --listen 0.0.0.0 --port 8188

# Kling/OpenAI auth is handled via ComfyUI's Partner Node auth system.
# Log in at the ComfyUI web UI to activate Partner Node credentials.

# For headless pipeline, API keys in environment:
#   OPENAI_API_KEY (set)
#   GEMINI_API_KEY (set)
```

---

## 11. Headless / Programmatic Execution

For CI/CD or automated batch runs without the ComfyUI GUI:

### 11.1 ComfyUI API Endpoint

```bash
# Start ComfyUI server
python main.py --listen 0.0.0.0 --port 8188

# Submit workflow
curl -X POST http://localhost:8188/api/prompt \
  -H "Content-Type: application/json" \
  -d @comfyui_workflows/habit_video_single.json
```

### 11.2 Python Headless Runner

```python
import json, requests

COMFYUI_URL = "http://localhost:8188"

def run_workflow(workflow_path: str, overrides: dict = None):
    """Submit a ComfyUI workflow for execution."""
    with open(workflow_path) as f:
        workflow = json.load(f)

    # Apply per-scene overrides
    if overrides:
        for node_id, params in overrides.items():
            workflow[node_id]["inputs"].update(params)

    resp = requests.post(
        f"{COMFYUI_URL}/api/prompt",
        json={"prompt": workflow}
    )
    return resp.json()

# Example: batch run all scenes
manifest = json.load(open("comfyui_workflows/scenes_manifest.json"))
for scene in manifest:
    run_workflow("comfyui_workflows/habit_video_single.json", {
        "10": {"prompt": format_scene_prompt(scene)},
        "30": {"filename_prefix": scene["output_path"]},
    })
```

---

## 12. Cost & Rate Limits

| Resource | Estimate |
|----------|----------|
| LLM prompt generation | ~15 calls x ~500 tokens = ~7,500 tokens total (negligible) |
| Kling Text-to-Image | 1 reference image |
| Kling Image-to-Video | 15 scene videos (5s each, pro mode) |
| Total generation time | ~30-60 min (async, parallelizable 3 concurrent) |
| Output storage | ~15 x 10MB = ~150MB |

### Rate Limit Handling

- Kling API: respect response headers, exponential backoff on 429
- Max concurrent Kling tasks: 3 (conservative)
- LLM calls: sequential, ~2s each, no rate limit concern

---

## 13. Quality Checklist

- [ ] ComfyUI environment running with all required custom nodes
- [ ] Character reference image clearly shows Mary's face and build
- [ ] LLM prompts correctly incorporate scene-specific data
- [ ] All 15 videos maintain consistent character appearance
- [ ] Time-of-day lighting matches palette specification
- [ ] Weather conditions visible when applicable
- [ ] Key driving actions visually represented per scene
- [ ] French animation style maintained (no photorealistic drift)
- [ ] No text/watermark artifacts in generated videos
- [ ] Videos are 5s each, smooth motion, no flickering
- [ ] All files saved to correct output paths
- [ ] Workflow JSON exports cleanly for headless re-execution
