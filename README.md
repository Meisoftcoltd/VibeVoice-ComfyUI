# VibeVoice ComfyUI Nodes

A comprehensive ComfyUI integration for Microsoft's **VibeVoice** text-to-speech model. This custom node pack allows you to generate high-quality, natural-sounding speech for single or multiple speakers, perform voice cloning, and train custom Voice LoRA models—all directly within your ComfyUI workflows.

---

## ✨ Features

### Core Functionality
- 🎤 **Single Speaker TTS:** Generate natural speech with optional zero-shot voice cloning using reference audio.
- 👥 **Multi-Speaker Conversations:** Support for generating complex scripts with up to 4 distinct speakers and corresponding cloned voices.
- 🎯 **Voice Cloning:** Accurately mimic characteristics of an input audio file.
- 🏋️ **LoRA Training (NEW):** End-to-end dataset preparation and QLoRA training pipelines within the ComfyUI interface.
- 💾 **Smart Checkpointing:** Evaluates checkpoints using a "True Mean" dual-loss metric (text + audio) and implements Auto-Resume to prevent data loss.
- 🛡️ **OOM Protection:** Automatic memory monitoring during training that gracefully cuts the batch size and retries without crashing the UI.
- 🎚️ **Voice Speed Control:** Dynamically adjust the speaking rate of generated audio without altering pitch.
- ⏸️ **Custom Pauses:** Insert deliberate silences into your generation via `[pause]` or `[pause:ms]` syntax.
- 🔄 **Node Chaining & Text Loading:** Load long scripts natively via text files, leveraging automatic chunking for unlimited generation length.

### Performance & Stability
- ⚡ **Optimized Attention:** Native support for memory-efficient `sdpa` and `sage` attention implementations.
- 🔒 **Stable Precision:** Hardcoded safety mechanisms in `bfloat16` to prevent `NaN` crashes during multi-nomial sampling.
- 🧹 **Memory Management:** Auto-unload toggles and a dedicated `Free Memory` node to flush CUDA caches between intensive generation tasks.
- 🍎 **Apple Silicon:** Native GPU acceleration on macOS (M1/M2/M3) via MPS backend.

---

## 📦 Installation

### Automatic Installation (Recommended)
1. Navigate to your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Enemyx-net/VibeVoice-ComfyUI
```
2. Restart ComfyUI. The wrapper will automatically install the necessary Python requirements upon the first run.

---

## 📥 Model Installation

Starting from v1.6.0, models and tokenizers must be downloaded manually and placed in the correct directories. Automatic downloading is bypassed to ensure stability.

*(Note: Highly experimental 0.5B and pre-quantized community models have been deprecated to ensure robust node stability).*

### Download Links
| Model                 | Size    | Download Link |
|------------------------|---------|--------------------|
| **VibeVoice-1.5B**     | ~5.4GB  | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| **VibeVoice-Large**    | ~18.7GB | [aoi-ot/VibeVoice-Large](https://huggingface.co/aoi-ot/VibeVoice-Large) |

### Tokenizer Requirement
VibeVoice explicitly requires the Qwen2.5-1.5B tokenizer.
- **Download:** [Qwen2.5-1.5B Tokenizer](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main)
- **Required Files:** `tokenizer_config.json`, `vocab.json`, `merges.txt`, `tokenizer.json`

### Folder Structure
Place the downloaded files exactly as follows:
```
ComfyUI/models/vibevoice/
├── tokenizer/                 # Qwen tokenizer files go here
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── tokenizer.json
├── VibeVoice-1.5B/           # 1.5B Base Model
│   ├── config.json
│   ├── model-00001-of-00003.safetensors
│   └── ...
└── VibeVoice-Large/          # Large Base Model
    └── ...
```

---

## 🔧 Node Documentation & Technical Details

### 1. VibeVoice Single Speaker
The core TTS generation node.
- **Inputs:** Text string, reference audio (optional), base model selection, LoRA path.
- **Parameters:**
  - `attention_type`: Recommended to leave on `auto` or `sdpa`.
  - `diffusion_steps`: Number of denoising steps. Higher means better quality, slower speed (default: 20).
  - `temperature` & `top_p`: Controls generation variance.
- **Output:** Raw Audio format compatible with ComfyUI SaveAudio nodes.

### 2. VibeVoice Multiple Speakers
Advanced TTS for dialogue.
- **Usage:** Input text using the format `[1]: Hello! [2]: Hi there!`.
- **Inputs:** Up to 4 optional reference audio tracks, mapped to `Speaker 1` through `Speaker 4`.
- **Recommendation:** Use `VibeVoice-Large` for robust multi-character semantic differentiation.

### 3. 🎙️ VibeVoice Dataset Preparator
Automatically processes raw audio into a valid training dataset.
- **Inputs:** A directory containing raw audio files (`.wav`, `.mp3`, etc.).
- **Functionality:** Utilizes the Whisper model to transcribe the audio, normalizes the tracks to 24kHz Mono, and applies "Smart Slicing" to clip tracks into optimal 20-second segments while preserving natural pauses.
- **Output:** An absolute directory path containing a ready-to-train `prompts.jsonl`.

### 4. 🚀 VibeVoice LoRA Trainer
An isolated, advanced QLoRA training pipeline.
- **Advanced Architecture:**
  - **OOM Protector:** If memory fails, automatically halves batch size, scales gradient accumulation, and resumes from the last checkpoint.
  - **Smart Saver:** Monitors dual losses (Acoustic + Text) to calculate a "True Mean". Discards underperforming checkpoints dynamically while keeping the absolute best models.
  - **Auto-Resume:** Scans the target output directory and seamlessly continues training if the process was interrupted.
- **⭐ Official Training Recommendations:**
  - **Learning Rate:** Recommended to set strictly to `2e-6` to avoid catastrophic forgetting and NaN generation.
  - **Early Stopping Patience:**
    - For **Small Datasets** (e.g., a few minutes of audio): Set patience to `20`.
    - For **Large Datasets** (e.g., hours of audio): Set patience to `5` to prevent overfitting.
  - **Batch Size:** `8` with Gradient Accumulation `16` (fallback to batch `4` if using 12GB VRAM or less).

### 5. VibeVoice LoRA Node
Connects a trained LoRA adapter to the main generation nodes.
- **Functionality:** Automatically detects nested `lora/` output folders generated by the VibeVoice Trainer. Applies the weights dynamically to the `language_model`, `prediction_head`, and `connectors` before sampling.

### 6. Utils: Load Text & Free Memory
- **Load Text From File:** Reads a `.txt` file into a string node input, perfect for massive audiobooks.
- **Free Memory:** Instantly unloads the VibeVoice model and flushes the CUDA cache, useful for freeing up VRAM for subsequent image/video generation nodes in mixed workflows.

---

## 📄 License
This ComfyUI wrapper is released under the MIT License.
*Note: The core Microsoft VibeVoice architecture is subject to its original Non-Commercial/Research-Only Microsoft License.*