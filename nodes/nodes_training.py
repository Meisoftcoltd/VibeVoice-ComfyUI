import os
import json
import glob
import torch
import librosa
import soundfile as sf
import subprocess
import threading
import sys
import shutil
import numpy as np
from pydub import AudioSegment
from transformers import pipeline
import folder_paths
import comfy.model_management as mm
import time

# Configure logging
import logging
logger = logging.getLogger("VibeVoice")

class VibeVoice_Dataset_Preparator:
    """
    Prepara un dataset para VibeVoice usando Whisper con inyección de prompt para capturar paralingüística.
    Utiliza una estrategia de corte inteligente para maximizar la longitud de los chunks (hasta 20s).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (["openai/whisper-large-v3-turbo", "openai/whisper-large-v3", "openai/whisper-medium", "openai/whisper-small", "openai/whisper-base", "openai/whisper-tiny"], {"default": "openai/whisper-large-v3-turbo"}),
            "raw_audio_dir": ("STRING", {"default": "./raw_audios"}),
            "output_dataset_dir": ("STRING", {"default": "./vibevoice_dataset"}),
            "language": (["es", "en", "auto"], {"default": "es"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_path",)
    FUNCTION = "prepare_dataset"
    CATEGORY = "VibeVoice/Training"

    def _smart_slice(self, intervals, total_samples, sr, min_len_sec, max_len_sec):
        """
        Slices audio based on non-silent intervals, accumulating them into chunks <= max_len_sec.
        Preserves silence within a chunk, but discards silence between chunks.
        """
        if len(intervals) == 0:
            return []

        chunks = []

        # Current candidate chunk
        current_start = intervals[0][0]
        current_end = intervals[0][1]

        for i in range(1, len(intervals)):
            next_interval = intervals[i]
            next_start = next_interval[0]
            next_end = next_interval[1]

            # Calculate duration if we were to merge the next interval
            # This includes the silence between current_end and next_start
            proposed_end = next_end
            proposed_duration = (proposed_end - current_start) / sr

            if proposed_duration <= max_len_sec:
                # It fits! Extend the current chunk.
                current_end = next_end
            else:
                # It doesn't fit. Finalize current.

                # Check if current chunk is valid (length check handled downstream or logic here)
                # If current chunk is > max_len_sec (e.g. first interval was huge), force split.
                chunk_dur = (current_end - current_start) / sr
                if chunk_dur > max_len_sec:
                    num_splits = int(np.ceil(chunk_dur / max_len_sec))
                    for j in range(num_splits):
                        split_start = current_start + j * int(max_len_sec * sr)
                        split_end = min(current_end, current_start + (j + 1) * int(max_len_sec * sr))
                        if split_end > split_start:
                            chunks.append([split_start, split_end])
                else:
                    chunks.append([current_start, current_end])

                # Start a new chunk with the next interval
                current_start = next_start
                current_end = next_end

        # Handle the final pending chunk
        chunk_dur = (current_end - current_start) / sr
        if chunk_dur > max_len_sec:
            num_splits = int(np.ceil(chunk_dur / max_len_sec))
            for j in range(num_splits):
                split_start = current_start + j * int(max_len_sec * sr)
                split_end = min(current_end, current_start + (j + 1) * int(max_len_sec * sr))
                if split_end > split_start:
                    chunks.append([split_start, split_end])
        else:
            chunks.append([current_start, current_end])

        return chunks

    def prepare_dataset(self, model, raw_audio_dir, output_dataset_dir, language):
        import os, glob, json, traceback
        import numpy as np
        import librosa
        import soundfile as sf
        from pydub import AudioSegment
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        os.makedirs(output_dataset_dir, exist_ok=True)
        wavs_dir = os.path.join(output_dataset_dir, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)
        prompts_path = os.path.join(output_dataset_dir, "prompts.jsonl")

        # Configuraciones acústicas
        TARGET_SR = 24000
        MIN_LEN_SEC = 2.0
        MAX_LEN_SEC = 20.0

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[VibeVoice] Cargando modelo Whisper ({model}) puro (sin pipelines) en {device} con {torch_dtype}...")
        try:
            # CARGA DIRECTA SIN PIPELINES (Inmune a torchcodec)
            processor = WhisperProcessor.from_pretrained(model)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(model, torch_dtype=torch_dtype).to(device)
        except Exception as e:
            print(f"[Error] Failed to load pure Whisper: {e}")
            raise e

        audio_files = glob.glob(os.path.join(raw_audio_dir, "**/*.*"), recursive=True)
        valid_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.mp4')
        
        # PREVENIR BUCLE INFINITO
        out_dir_abs = os.path.abspath(output_dataset_dir)
        audio_files = [
            f for f in audio_files 
            if f.lower().endswith(valid_extensions) and not os.path.abspath(f).startswith(out_dir_abs)
        ]

        print(f"[VibeVoice] Encontrados {len(audio_files)} archivos de audio seguros. Iniciando procesamiento...")

        jsonl_entries = []
        chunk_counter = 0

        for audio_path in audio_files:
            try:
                # 1. Extracción con Pydub (soporta OGG y MP4)
                try:
                    audio_segment = AudioSegment.from_file(audio_path)
                except Exception as e:
                    print(f"[Warning] Pydub failed on {audio_path}: {e}")
                    continue

                # Forzar 16-bit, mono, 24kHz
                audio_segment = audio_segment.set_channels(1).set_frame_rate(24000).set_sample_width(2)
                samples = np.array(audio_segment.get_array_of_samples())
                y = samples.astype(np.float32) / 32768.0
                sr = 24000
                total_duration = len(y) / sr

                # 2. Smart Slicing
                chunks_to_process = []
                if total_duration <= MAX_LEN_SEC:
                     chunks_to_process.append((y, 0, len(y)))
                else:
                    intervals = librosa.effects.split(y, top_db=40)
                    sliced_intervals = self._smart_slice(intervals, len(y), sr, MIN_LEN_SEC, MAX_LEN_SEC)
                    for start, end in sliced_intervals:
                        chunks_to_process.append((y[start:end], start, end))

                for chunk, start_sample, end_sample in chunks_to_process:
                    duration = len(chunk) / sr
                    if duration < MIN_LEN_SEC:
                        continue

                    # 3. Guardado del chunk
                    chunk_24k = chunk
                    rms = librosa.feature.rms(y=chunk_24k)[0]
                    target_rms = 10 ** (-20 / 20)
                    mean_rms = rms.mean() + 1e-9
                    chunk_24k = chunk_24k * (target_rms / mean_rms)

                    chunk_filename = f"chunk_{chunk_counter:05d}.wav"
                    chunk_filepath = os.path.join(wavs_dir, chunk_filename)
                    sf.write(chunk_filepath, chunk_24k, TARGET_SR, subtype='PCM_16')

                    # 4. Transcripción ASR directa (Nivel Dios)
                    chunk_16k = librosa.resample(chunk_24k, orig_sr=TARGET_SR, target_sr=16000)

                    # Convertir el array a espectrograma directamente (sin usar torchaudio)
                    input_features = processor(
                        chunk_16k, sampling_rate=16000, return_tensors="pt"
                    ).input_features.to(device, dtype=torch_dtype)

                    # Inyección de Prompt para capturar risas y suspiros
                    prompt_text = "Uhm, ah, [risa], [suspiro], [tos], eh, mhm, bueno..."
                    
                    # Preparamos las instrucciones exactas
                    gen_kwargs = {
                        "language": language if language != "auto" else "es", # Forzamos español si no es auto
                        "task": "transcribe"
                    }
                    
                    try:
                        # Generación matemática pura
                        prompt_ids = processor.get_prompt_ids(prompt_text, return_tensors="pt").to(device)
                        predicted_ids = whisper_model.generate(
                            input_features, 
                            prompt_ids=prompt_ids, 
                            **gen_kwargs
                        )
                    except AttributeError:
                        # Fallback si get_prompt_ids no está soportado en esta versión de transformers
                        predicted_ids = whisper_model.generate(
                            input_features, 
                            **gen_kwargs
                        )

                    # Decodificamos los tokens a texto legible
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                    transcription = transcription.replace('\n', ' ').replace('|', '')

                    if transcription:
                        # VibeVoice's parser is hyper-strict. Force exact block formatting.
                        formatted_text = f"Speaker 0: {transcription}\n"

                        entry = {
                            "text": formatted_text,
                            "audio": os.path.abspath(chunk_filepath)
                        }
                        jsonl_entries.append(entry)
                        chunk_counter += 1
                        print(f"Procesado: {chunk_filename} -> {formatted_text[:50]}...")

            except Exception as e:
                print(f"[Warning] Error procesando archivo completo {audio_path}: {e}")
                traceback.print_exc()
                continue

        with open(prompts_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        print(f"[VibeVoice] Dataset completado: {chunk_counter} fragmentos válidos generados en {output_dataset_dir}")

        del whisper_model
        del processor
        torch.cuda.empty_cache()

        return (os.path.abspath(output_dataset_dir),)
        
        
class VibeVoice_LoRA_Trainer:
    """
    Ejecuta el entrenamiento del modelo usando un entorno virtual aislado para evitar conflictos de dependencias.
    Clona automáticamente el repositorio de entrenamiento y gestiona el venv.
    """
    @classmethod
    def INPUT_TYPES(cls):
        base_models = [
            "microsoft/VibeVoice-1.5B",
            "aoi-ot/VibeVoice-Large",
            "microsoft/VibeVoice-7B",
            "marksverdhai/vibevoice-7b-bnb-8bit",
            "marksverdhai/vibevoice-7b-bnb-4bit",
            "custom_local_path"
        ]
        return {
            "required": {
                "dataset_path": ("STRING", {"forceInput": True}),
                "base_model_path": (base_models, {"default": "microsoft/VibeVoice-1.5B"}),
                "output_lora_name": ("STRING", {"default": "vibevoice_lora_out"}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 32}),
                "gradient_accum_steps": ("INT", {"default": 16, "min": 1, "max": 128}), # Default 16 as requested
                "epochs": ("INT", {"default": 150, "min": 10, "max": 1000}),
                "learning_rate": ("FLOAT", {"default": 2e-4, "min": 1e-6, "max": 0.1, "step": 1e-6}),
                "mixed_precision": (["bf16", "fp16", "no"], {"default": "bf16"}),
                "lora_rank": ("INT", {"default": 32, "min": 4, "max": 128}),
                "lora_alpha": ("INT", {"default": 64, "min": 8, "max": 256}),
                "early_stopping_patience": ("INT", {"default": 25, "min": 1, "max": 100, "tooltip": "Steps without improvement before stopping."}),
                "early_stopping_threshold": ("FLOAT", {"default": 0.002, "min": 0.0001, "max": 0.1, "step": 0.001}),
                "save_total_limit": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Maximum number of BEST models to keep. Worse ones will be deleted."}),
                "validation_split": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05, "tooltip": "Percentage of data reserved for validation to prevent overfitting. Set to 0 to disable."}),
                "transformers_version": ("STRING", {"default": "4.51.3", "multiline": False}), # Providing flexibility
            },
            "optional": {
                "custom_model_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_output_dir",)
    FUNCTION = "train_lora"
    CATEGORY = "VibeVoice/Training"

    def _get_or_download_model(self, model_id):
        if model_id == "custom_local_path" or os.path.isdir(model_id):
            return model_id

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise Exception("huggingface_hub not installed")

        import folder_paths

        # Determine target directory in ComfyUI models folder
        # Handle full repo IDs by taking just the model name
        safe_name = model_id.split('/')[-1]

        # If we passed a folder name (like VibeVoice-1.5B), map it to repo ID if needed
        # Or if we passed a full repo ID, map to folder name
        repo_map = {
            "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
            "VibeVoice-Large": "aoi-ot/VibeVoice-Large",
            "VibeVoice-Large-Q8": "FabioSarracino/VibeVoice-Large-Q8",
            "VibeVoice-Large-Q4": "DevParker/VibeVoice7b-low-vram",
            "vibevoice-7b-bnb-8bit": "marksverdhai/vibevoice-7b-bnb-8bit",
            "vibevoice-7b-bnb-4bit": "marksverdhai/vibevoice-7b-bnb-4bit"
        }

        repo_id = repo_map.get(model_id, model_id) # Default to model_id if not in map (assuming it's a repo id)

        target_dir = os.path.join(folder_paths.models_dir, "vibevoice", safe_name)

        # Check if already downloaded (looking for main weight files)
        needs_download = True
        if os.path.exists(target_dir):
            files = os.listdir(target_dir)
            if any(f.endswith('.safetensors') or f.endswith('.bin') for f in files):
                needs_download = False

        if needs_download:
            print(f"\n[VibeVoice] ⬇️ Model {model_id} (Repo: {repo_id}) not found locally. Downloading to {target_dir}...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=target_dir,
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.md"], # Ignore unnecessary heavy files
                    local_dir_use_symlinks=False
                )
                print(f"[VibeVoice] ✅ Download complete!")
            except Exception as e:
                print(f"[VibeVoice] ❌ Download failed: {e}")
                raise e
        else:
            print(f"[VibeVoice] 🔎 Model {safe_name} found locally. Skipping download.")

        return target_dir

    def _read_subprocess_output(self, process, output_log):
        """Reads subprocess output, updates ComfyUI progress bar, and prints cleanly line-by-line."""
        import re
        import comfy.utils

        comfy_pbar = None

        for line in iter(process.stdout.readline, b''):
            # Split by \r to get the latest chunk, but print normally to avoid console mess
            parts = line.decode('utf-8', errors='replace').split('\r')
            decoded_line = parts[-1].rstrip('\n').strip()

            if not decoded_line or decoded_line == "[VibeVoice Train]":
                continue

            # --- SUPER SPAM FILTER ---
            if "{'" in decoded_line and "':" in decoded_line and "}" in decoded_line:
                continue
            if "UserWarning: Could not find a config file" in decoded_line or "warnings.warn(" in decoded_line:
                continue

            # Detect tqdm progress bar format
            is_progress_bar = "%|" in decoded_line and ("it/s]" in decoded_line or "s/it]" in decoded_line or "00:" in decoded_line)

            if is_progress_bar:
                # Update ComfyUI Web UI Progress Bar via regex
                match = re.search(r"(\d+)/(\d+) \[", decoded_line)
                if match:
                    current_step = int(match.group(1))
                    total_steps = int(match.group(2))

                    if comfy_pbar is None or comfy_pbar.total != total_steps:
                        comfy_pbar = comfy.utils.ProgressBar(total_steps)

                    comfy_pbar.update_absolute(current_step, total_steps)

            # Print everything line-by-line normally to avoid the overlapping bug
            print(f"[VibeVoice Train] {decoded_line}")
            output_log.append(decoded_line)

        process.stdout.close()

    def _patch_early_stopping(self, repo_dir, patience, threshold, save_total_limit, validation_split):
        target_file = os.path.join(repo_dir, "src", "finetune_vibevoice_lora.py")
        if not os.path.exists(target_file):
            return False

        import subprocess
        try:
            subprocess.check_call(["git", "checkout", "--", "src/finetune_vibevoice_lora.py"], cwd=repo_dir)
        except Exception:
            pass

        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        import re

        # Fix UnboundLocalError: import torch at the top of main()
        # Find the def main line and insert import torch right after it
        content = re.sub(
            r"(def main\s*\([^)]*\)\s*(->\s*None)?\s*:)",
            r"\1\n    import torch",
            content,
            count=1
        )

        # --- SUPPRESS SPAMMY INTRA-EPOCH LOGS ---
        content = re.sub(r"logger\.info\(\{.*?\}\)", "pass  # Suppressed by ComfyUI", content)
        content = re.sub(r"print\(\{.*?\}\)", "pass  # Suppressed by ComfyUI", content)

        callback_code = f"""
# --- VIBEVOICE CUSTOM CALLBACK START ---
import os
import shutil
from transformers import TrainerCallback

class SmartEarlyStoppingAndSaveCallback(TrainerCallback):
    def __init__(self, patience={patience}, threshold={threshold}, keep_best_n={save_total_limit}):
        self.patience = patience
        self.threshold = threshold
        self.keep_best_n = keep_best_n
        self.best_diff_loss = float('inf')
        self.best_ce_loss = float('inf')
        self.counter = 0
        self.best_checkpoints = []

        self.current_diff_loss = None
        self.current_ce_loss = None
        self.current_total_loss = float('inf')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            is_eval = any(k.startswith("eval") for k in logs.keys())
            if is_eval:
                self.current_diff_loss = logs.get("eval/diffusion_loss", logs.get("eval_diffusion_loss", self.current_diff_loss))
                self.current_ce_loss = logs.get("eval/ce_loss", logs.get("eval_ce_loss", self.current_ce_loss))
                self.current_total_loss = logs.get("eval/loss", logs.get("eval_loss", self.current_total_loss))
            else:
                # Fallback if eval is disabled
                self.current_diff_loss = logs.get("train/diffusion_loss", logs.get("diffusion_loss", self.current_diff_loss))
                self.current_ce_loss = logs.get("train/ce_loss", logs.get("ce_loss", self.current_ce_loss))
                self.current_total_loss = logs.get("loss", self.current_total_loss)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.current_diff_loss is not None and self.current_ce_loss is not None:
            current_epoch = int(round(state.epoch or 0))

            diff_improved = self.current_diff_loss < (self.best_diff_loss - self.threshold)
            ce_improved = self.current_ce_loss < (self.best_ce_loss - self.threshold)

            status = "🟢 IMPROVED" if (diff_improved or ce_improved) else "🔴 NO CHANGE"
            print("")
            print(f"📊 [EPOCH {{current_epoch}} VALIDATION] {{status}} | Text Loss: {{self.current_ce_loss:.4f}} | Audio Loss: {{self.current_diff_loss:.4f}} | Total: {{self.current_total_loss:.4f}}")

            if diff_improved or ce_improved:
                if diff_improved: self.best_diff_loss = self.current_diff_loss
                if ce_improved: self.best_ce_loss = self.current_ce_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"\\\\n[VibeVoice Smart Stop] 🛑 AUTO-STOP: Validation loss stagnated for {{self.patience}} epochs.\\\\n")
                    control.should_training_stop = True

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{{state.global_step}}")
        if os.path.exists(ckpt_dir):
            self.best_checkpoints.append((self.current_total_loss, ckpt_dir))
            self.best_checkpoints.sort(key=lambda x: x[0])

            while len(self.best_checkpoints) > self.keep_best_n:
                worst_loss, worst_ckpt = self.best_checkpoints.pop(-1)
                if os.path.exists(worst_ckpt):
                    try:
                        shutil.rmtree(worst_ckpt)
                        print(f"[VibeVoice Smart Saver] 🗑️ Deleted worse checkpoint: {{os.path.basename(worst_ckpt)}}")
                    except Exception:
                        pass

    def on_train_end(self, args, state, control, **kwargs):
        if not self.best_checkpoints: return
        best_loss, best_ckpt = self.best_checkpoints[0]
        best_lora_dir = os.path.join(best_ckpt, "lora")
        final_lora_dir = os.path.join(args.output_dir, "lora")

        print(f"\\\\n[VibeVoice Smart Saver] 🏆 Training complete! Restoring BEST model from {{os.path.basename(best_ckpt)}} (Val Loss: {{best_loss:.4f}})...")
        if os.path.exists(best_lora_dir):
            try:
                if os.path.exists(final_lora_dir): shutil.rmtree(final_lora_dir)
                shutil.copytree(best_lora_dir, final_lora_dir)
                print("[VibeVoice Smart Saver] ✅ Best model set as final output.\\\\n")
            except Exception:
                pass
# --- VIBEVOICE CUSTOM CALLBACK END ---
"""

        # Inject callback class
        content = re.sub(r"(def main\s*\([^)]*\)\s*(->\s*None)?\s*:)", callback_code + r"\n\g<1>", content, count=1)

        #  Build elegant injection for Evaluation Split (Left-aligned to prevent IndentationError)
        # Build elegant injection for Evaluation Split (Left-aligned to prevent IndentationError)
        split_code = f"""    # --- AUTO EVAL SPLIT PATCH ---
    eval_dataset = None
    if {validation_split} > 0.0:
        if hasattr(training_args, 'eval_strategy'): training_args.eval_strategy = "epoch"
        if hasattr(training_args, 'evaluation_strategy'): training_args.evaluation_strategy = "epoch"
        training_args.do_eval = True
        try:
            _eval_size = max(1, int(len(train_dataset) * {validation_split}))
            _train_size = len(train_dataset) - _eval_size
            train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [_train_size, _eval_size], generator=torch.Generator().manual_seed(42))
            print(f"\\\\n[VibeVoice Setup] 🗂️ Split dataset: {{_train_size}} for Training, {{_eval_size}} for Validation.\\\\n")
        except Exception as e:
            print(f"[VibeVoice Setup] ⚠️ Could not split dataset: {{e}}")

    trainer = VibeVoiceTrainer("""

        # 1. Replace the VibeVoiceTrainer initialization
        # Ensure we are replacing the exact 4-space indented original line
        content = content.replace("    trainer = VibeVoiceTrainer(", split_code)

        # 2. Inject our callback into the existing list to PRESERVE the EmaCallback
        my_callback = f"SmartEarlyStoppingAndSaveCallback(patience={patience}, threshold={threshold}, keep_best_n={save_total_limit}), "
        content = content.replace("callbacks=[", f"callbacks=[{my_callback}", 1)

        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)

        print("[VibeVoice Patch] Successfully applied Validation Split and PRESERVED EmaCallback.")
        return True

    def _patch_flash_attention_import(self, repo_dir):
        """Patches modeling_vibevoice.py to fix FlashAttentionKwargs import error."""
        target_file = os.path.join(repo_dir, "src", "vibevoice", "modular", "modeling_vibevoice.py")
        if not os.path.exists(target_file):
            return False

        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if already patched
        if "except ImportError:" in content and "FlashAttentionKwargs = dict" in content:
            return True

        print("[VibeVoice Patch] Patching FlashAttentionKwargs import...")

        search_line = "from transformers.modeling_flash_attention_utils import FlashAttentionKwargs"

        replacement_block = (
            "try:\n"
            "    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs\n"
            "except ImportError:\n"
            "    FlashAttentionKwargs = dict  # Fallback type alias"
        )

        if search_line in content:
            content = content.replace(search_line, replacement_block)
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(content)
            print("[VibeVoice Patch] FlashAttentionKwargs patch applied successfully.")
            return True
        else:
            print("[VibeVoice Patch] Warning: Could not find FlashAttentionKwargs import line to patch.")
            return False

    def _patch_peft_task_type(self, repo_dir):
        target_file = os.path.join(repo_dir, "src", "finetune_vibevoice_lora.py")
        if not os.path.exists(target_file):
            return False

        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        search_str = "task_type=TaskType.CAUSAL_LM,"
        replacement = "task_type=TaskType.FEATURE_EXTRACTION,"

        if search_str in content:
            content = content.replace(search_str, replacement)
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(content)
            print("[VibeVoice Patch] Successfully patched PEFT TaskType to prevent labels kwarg injection.")
            return True
        return False

    def _patch_quantization_loading(self, repo_dir):
        target_file = os.path.join(repo_dir, "src", "finetune_vibevoice_lora.py")
        if not os.path.exists(target_file):
            return False

        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if already patched
        if "from transformers import BitsAndBytesConfig" in content:
            return True

        print("[VibeVoice Patch] Patching model loading for dynamic quantization...")

        quant_loading_injection = """
    # --- DYNAMIC QUANTIZATION LOADING ---
    import torch
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    is_4bit = "4bit" in model_args.model_name_or_path.lower()
    is_8bit = "8bit" in model_args.model_name_or_path.lower()

    if is_4bit or is_8bit:
        print(f"[VibeVoice Loader] 🧊 Detected Quantized Model. Applying BitsAndBytesConfig...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=is_4bit,
            load_in_8bit=is_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["acoustic_tokenizer", "semantic_tokenizer", "prediction_head", "acoustic_connector", "semantic_connector", "lm_head"]
        )
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
        print(f"[VibeVoice Loader] ✅ Model prepared for k-bit training.")
    else:
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
"""
        import re
        # Use a robust regex that consumes the closing parenthesis
        new_content = re.sub(
            r"model\s*=\s*VibeVoiceForConditionalGeneration\.from_pretrained\s*\([^)]+\)",
            quant_loading_injection.strip(),
            content,
            flags=re.DOTALL
        )

        if new_content != content:
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("[VibeVoice Patch] Model loading patch applied successfully.")
            return True
        else:
            print("[VibeVoice Patch] Warning: Could not find model loading line to patch.")
            return False

    def _setup_environment(self, repo_dir, venv_dir, transformers_version, patience, threshold, save_total_limit, validation_split):
        """Sets up the training repository and virtual environment."""

        # 1. Clone Repo if missing
        if not os.path.exists(repo_dir):
            print(f"[VibeVoice Setup] Cloning training repository to {repo_dir}...")
            try:
                subprocess.check_call(["git", "clone", "https://github.com/voicepowered-ai/VibeVoice-finetuning", repo_dir])
            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to clone repository: {e}")
                return False

        # Patch script
        self._patch_flash_attention_import(repo_dir)
        self._patch_early_stopping(repo_dir, patience, threshold, save_total_limit, validation_split)
        self._patch_peft_task_type(repo_dir)  # <--- New PEFT patch
        self._patch_quantization_loading(repo_dir)  # <--- New Quantization patch

        # 2. Create Venv if missing
        if not os.path.exists(venv_dir):
            print(f"[VibeVoice Setup] Creating virtual environment at {venv_dir}...")
            try:
                subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to create venv: {e}")
                return False

        # Determine pip path in venv
        if os.name == 'nt':
            pip_cmd = os.path.join(venv_dir, "Scripts", "pip")
            python_cmd = os.path.join(venv_dir, "Scripts", "python")
        else:
            pip_cmd = os.path.join(venv_dir, "bin", "pip")
            python_cmd = os.path.join(venv_dir, "bin", "python")

        if not os.path.exists(pip_cmd):
            print(f"[Error] Pip not found at {pip_cmd}")
            return False

        # 3. Install requirements
        # We check a marker file to avoid reinstalling every time
        marker_file = os.path.join(venv_dir, ".installed_requirements")
        if not os.path.exists(marker_file):
            print("[VibeVoice Setup] Installing dependencies in venv...")
            try:
                # Install repo in editable mode
                subprocess.check_call([pip_cmd, "install", "-e", "."], cwd=repo_dir)

                # Install specific transformers version
                print(f"[VibeVoice Setup] Installing transformers=={transformers_version}...")
                subprocess.check_call([pip_cmd, "uninstall", "-y", "transformers"], cwd=repo_dir)
                subprocess.check_call([pip_cmd, "install", f"transformers=={transformers_version}"], cwd=repo_dir)

                # Install other potentially missing deps
                subprocess.check_call([pip_cmd, "install", "accelerate", "peft", "soundfile", "librosa", "bitsandbytes"], cwd=repo_dir)

                # Create marker
                with open(marker_file, "w") as f:
                    f.write(f"Installed transformers=={transformers_version}")

            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to install dependencies: {e}")
                return False

        return python_cmd

    def train_lora(self, dataset_path, base_model_path, output_lora_name, batch_size,
                   gradient_accum_steps, epochs, learning_rate, mixed_precision,
                   lora_rank, lora_alpha, early_stopping_patience, early_stopping_threshold,
                   save_total_limit, validation_split, transformers_version, custom_model_path=""):

        # Resolve model path
        model_path_to_use = base_model_path
        if base_model_path == "custom_local_path":
            if not custom_model_path.strip():
                raise ValueError("You selected 'custom_local_path' but left the text field empty. Please provide a valid model path.")
            model_path_to_use = custom_model_path.strip()
        else:
            # TRIGGER JIT DOWNLOAD HERE
            model_path_to_use = self._get_or_download_model(base_model_path)

        # Paths
        current_dir = os.path.dirname(os.path.realpath(__file__))
        repo_dir = os.path.join(current_dir, "vibevoice_finetuning_repo")
        venv_dir = os.path.join(current_dir, "vibevoice_venv")

        # Setup Environment
        python_cmd = self._setup_environment(repo_dir, venv_dir, transformers_version, early_stopping_patience, early_stopping_threshold, save_total_limit, validation_split)
        if not python_cmd:
            return ("Error during setup",)

        # Output directory: ComfyUI models/vibevoice/loras/
        try:
            output_dir = os.path.join(folder_paths.models_dir, "vibevoice", "loras", output_lora_name)
        except AttributeError:
            # Fallback if folder_paths is not available (e.g. testing), though user said it is.
            # Using a sensible default if not in ComfyUI environment
            output_dir = os.path.join(os.getcwd(), "models", "vibevoice", "loras", output_lora_name)

        os.makedirs(output_dir, exist_ok=True)

        prompts_jsonl = os.path.join(dataset_path, "prompts.jsonl")
        if not os.path.exists(prompts_jsonl):
             print(f"[Error] prompts.jsonl not found in {dataset_path}")
             return (output_dir,)

        # Bust Hugging Face cache by creating a unique temporary copy of the dataset
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        run_prompts_jsonl = os.path.join(dataset_path, f"prompts_run_{unique_id}.jsonl")
        shutil.copy(prompts_jsonl, run_prompts_jsonl)

        # Auto-Retry Loop for OOM Protection
        current_batch_size = batch_size
        current_grad_accum = gradient_accum_steps
        max_retries = 5
        training_success = False

        try:
            for attempt in range(max_retries):
                # Construct Command dynamically with current batch/accum
                command = [
                    python_cmd, "-m", "src.finetune_vibevoice_lora",
                    "--model_name_or_path", model_path_to_use,
                    "--train_jsonl", run_prompts_jsonl,
                    "--text_column_name", "text",
                    "--audio_column_name", "audio",
                    "--output_dir", output_dir,
                    "--per_device_train_batch_size", str(current_batch_size),
                    "--gradient_accumulation_steps", str(current_grad_accum),
                    "--num_train_epochs", str(epochs),
                    "--learning_rate", str(learning_rate),
                    "--bf16", "True" if mixed_precision == "bf16" else "False",
                    "--fp16", "True" if mixed_precision == "fp16" else "False",
                    "--lora_r", str(lora_rank),
                    "--lora_alpha", str(lora_alpha),
                    "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    "--train_diffusion_head", "True",
                    "--ddpm_batch_mul", "4",
                    "--diffusion_loss_weight", "1.4",
                    "--ce_loss_weight", "0.04",
                    "--voice_prompt_drop_rate", "0.2",
                    "--logging_strategy", "epoch",
                    "--save_strategy", "epoch",
                    "--save_total_limit", str(save_total_limit + 5),  # Dynamic buffer above callback limit
                    "--remove_unused_columns", "False",
                    "--do_train"
                ]

                import multiprocessing
                # Safely calculate workers: leave some CPUs free, cap at 4 to prevent overhead
                safe_workers = max(1, min(4, multiprocessing.cpu_count() - 2))

                # Append data loading optimizations to the command
                command.extend([
                    "--dataloader_num_workers", str(safe_workers),
                    "--dataloader_prefetch_factor", "2"
                ])

                print(f"\n[VibeVoice] Iniciando entrenamiento (Intento {attempt+1}/{max_retries}) | Batch: {current_batch_size} | GradAccum: {current_grad_accum}")

                output_log = []
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=repo_dir)

                thread = threading.Thread(target=self._read_subprocess_output, args=(process, output_log))
                thread.start()

                while process.poll() is None:
                    if mm.processing_interrupted():
                        print("\n[VibeVoice] Interrupción manual detectada desde ComfyUI. Cancelando entrenamiento...\n")
                        process.terminate()
                        process.wait()
                        return ("Entrenamiento cancelado manualmente",)
                    time.sleep(1)

                thread.join()

                if process.returncode == 0:
                    print(f"[VibeVoice] Entrenamiento finalizado con éxito. LoRA guardado en {output_dir}")
                    training_success = True
                    break # Success, exit retry loop
                else:
                    full_log = "\n".join(output_log)
                    if "OutOfMemoryError" in full_log or "out of memory" in full_log.lower():
                        if current_batch_size > 1:
                            print(f"\n[VibeVoice OOM Protector] ⚠️ Out of Memory detectado! Reduciendo Batch Size...")

                            new_batch = max(1, current_batch_size // 2)
                            # Maintain effective batch size (roughly) by increasing grad accum
                            factor = current_batch_size // new_batch
                            current_grad_accum = current_grad_accum * factor
                            current_batch_size = new_batch

                            print(f"[VibeVoice OOM Protector] Reiniciando automáticamente con Batch Size: {current_batch_size} y Grad Accum: {current_grad_accum}...\n")
                            torch.cuda.empty_cache()
                            continue # Retry the loop
                        else:
                            print(f"\n[VibeVoice OOM Protector] ❌ Out of Memory fatal. El Batch Size ya es 1 y no se puede reducir más.\n")
                            break # Fail completely
                    else:
                        print(f"[VibeVoice] Entrenamiento falló con código {process.returncode}.")
                        break # Failed for a non-OOM reason

        finally:
            # Cleanup the unique dataset copy to save disk space
            if os.path.exists(run_prompts_jsonl):
                os.remove(run_prompts_jsonl)

        if not training_success:
            return ("Fallo en el entrenamiento",)

        # --- SAVE CLEAN TRAINING LOG ---
        try:
            log_path = os.path.join(output_dir, "training_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(output_log))
            print(f"[VibeVoice] 📝 Log del entrenamiento guardado en: {log_path}")
        except Exception as e:
            print(f"[VibeVoice] ⚠️ No se pudo guardar el log: {e}")

        return (os.path.abspath(output_dir),)
