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
                "learning_rate": ("FLOAT", {"default": 2e-4, "step": 1e-5}),
                "mixed_precision": (["bf16", "fp16", "no"], {"default": "bf16"}),
                "lora_rank": ("INT", {"default": 32, "min": 4, "max": 128}),
                "lora_alpha": ("INT", {"default": 64, "min": 8, "max": 256}),
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

    def _read_subprocess_output(self, process):
        """Lee la salida del subproceso asíncronamente para la consola de ComfyUI."""
        for line in iter(process.stdout.readline, b''):
            print(f"[VibeVoice Train] {line.decode('utf-8', errors='replace').rstrip()}")
        process.stdout.close()

    def _patch_early_stopping(self, repo_dir):
        target_file = os.path.join(repo_dir, "src", "finetune_vibevoice_lora.py")
        if not os.path.exists(target_file):
            return False

        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        if "TrainLossEarlyStoppingCallback" in content:
            return True

        print("[VibeVoice Patch] Injecting custom Early Stopping autopilot...")

        callback_code = """
from transformers import TrainerCallback
class TrainLossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, threshold=0.005):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            current_loss = logs["loss"]
            if current_loss < self.best_loss - self.threshold:
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"\\n[VibeVoice] AUTO-STOP TRIGGERED: Loss hasn't improved by {self.threshold} for {self.patience} logging steps.\\n")
                    control.should_training_stop = True
"""
        # Insert callback code before main()
        if "def main():" in content:
            content = content.replace("def main():", callback_code + "\ndef main():")

        # Inject callback into Trainer instantiation
        trainer_init = "trainer = Trainer("
        trainer_replacement = "trainer = Trainer(\n        callbacks=[TrainLossEarlyStoppingCallback(patience=5)],"
        if trainer_init in content:
            content = content.replace(trainer_init, trainer_replacement)

        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)
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

    def _setup_environment(self, repo_dir, venv_dir, transformers_version):
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
        self._patch_early_stopping(repo_dir)
        self._patch_peft_task_type(repo_dir)  # <--- New PEFT patch

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
                subprocess.check_call([pip_cmd, "install", "accelerate", "peft", "soundfile", "librosa"], cwd=repo_dir)

                # Create marker
                with open(marker_file, "w") as f:
                    f.write(f"Installed transformers=={transformers_version}")

            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to install dependencies: {e}")
                return False

        return python_cmd

    def train_lora(self, dataset_path, base_model_path, output_lora_name, batch_size,
                   gradient_accum_steps, epochs, learning_rate, mixed_precision,
                   lora_rank, lora_alpha, transformers_version, custom_model_path=""):

        # Resolve model path
        model_path_to_use = base_model_path
        if base_model_path == "custom_local_path":
            if not custom_model_path.strip():
                raise ValueError("You selected 'custom_local_path' but left the text field empty. Please provide a valid model path.")
            model_path_to_use = custom_model_path.strip()

        # Paths
        current_dir = os.path.dirname(os.path.realpath(__file__))
        repo_dir = os.path.join(current_dir, "vibevoice_finetuning_repo")
        venv_dir = os.path.join(current_dir, "vibevoice_venv")

        # Setup Environment
        python_cmd = self._setup_environment(repo_dir, venv_dir, transformers_version)
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

        # Construct Command
        command = [
            python_cmd, "-m", "src.finetune_vibevoice_lora",
            "--model_name_or_path", model_path_to_use,
            "--train_jsonl", run_prompts_jsonl,
            "--text_column_name", "text",
            "--audio_column_name", "audio",
            "--output_dir", output_dir,
            "--per_device_train_batch_size", str(batch_size),
            "--gradient_accumulation_steps", str(gradient_accum_steps),
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
            "--logging_steps", "10",
            "--save_strategy", "steps",
            "--save_steps", "200",
            "--save_total_limit", "2",
            "--remove_unused_columns", "False",
            "--do_train"
        ]

        print("[VibeVoice] Iniciando subproceso de entrenamiento...")
        print("Comando:", " ".join(command))

        try:
            # Popen
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=repo_dir)

            # Read output
            thread = threading.Thread(target=self._read_subprocess_output, args=(process,))
            thread.start()

            # Instead of process.wait(), use a polling loop to catch ComfyUI interrupts instantly
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
            else:
                print(f"[VibeVoice] Entrenamiento falló con código {process.returncode}.")

        finally:
            # Cleanup the unique dataset copy to save disk space
            if os.path.exists(run_prompts_jsonl):
                os.remove(run_prompts_jsonl)

        return (os.path.abspath(output_dir),)
