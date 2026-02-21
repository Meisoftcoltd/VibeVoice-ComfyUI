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
import uuid
import numpy as np
from transformers import pipeline
import folder_paths

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
        os.makedirs(output_dataset_dir, exist_ok=True)
        wavs_dir = os.path.join(output_dataset_dir, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)
        prompts_path = os.path.join(output_dataset_dir, "prompts.jsonl")

        # Configuraciones acústicas
        TARGET_SR = 24000
        MIN_LEN_SEC = 2.0
        MAX_LEN_SEC = 20.0

        # Cargar Whisper (Pipeline)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VibeVoice] Cargando modelo Whisper ({model}) en {device} con float16...")
        try:
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        except Exception as e:
            print(f"[Error] Failed to load Whisper pipeline: {e}")
            raise e

        audio_files = glob.glob(os.path.join(raw_audio_dir, "**/*.*"), recursive=True)
        valid_extensions = ('.wav', '.mp3', '.flac', '.ogg')
        audio_files = [f for f in audio_files if f.lower().endswith(valid_extensions)]

        print(f"[VibeVoice] Encontrados {len(audio_files)} archivos de audio. Iniciando procesamiento...")

        jsonl_entries = []
        chunk_counter = 0

        for audio_path in audio_files:
            temp_wav_path = None
            try:
                # 1. Preemptive FFmpeg Normalization
                # Generate unique temp filename to avoid collisions
                temp_filename = f"temp_{uuid.uuid4().hex}.wav"
                temp_wav_path = os.path.join(output_dataset_dir, temp_filename)

                # FFmpeg command: convert to 24kHz mono WAV
                # -y: overwrite
                # -ac 1: mono
                # -ar 24000: 24kHz sample rate
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", audio_path,
                    "-ac", "1",
                    "-ar", "24000",
                    temp_wav_path
                ]

                # Run subprocess
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"[Warning] FFmpeg failed for {audio_path}. Stderr: {result.stderr}")
                    continue

                # 2. Cargar audio safely from normalized wav
                y, sr = librosa.load(temp_wav_path, sr=24000, mono=True)
                total_duration = len(y) / sr

                chunks_to_process = []

                if total_duration <= MAX_LEN_SEC:
                     chunks_to_process.append((y, 0, len(y)))
                else:
                    # 2. Smart Slicing
                    intervals = librosa.effects.split(y, top_db=40)
                    sliced_intervals = self._smart_slice(intervals, len(y), sr, MIN_LEN_SEC, MAX_LEN_SEC)
                    for start, end in sliced_intervals:
                        chunks_to_process.append((y[start:end], start, end))

                for chunk, start_sample, end_sample in chunks_to_process:
                    duration = len(chunk) / sr

                    # 3. Filtro de longitud
                    if duration < MIN_LEN_SEC:
                        continue

                    # Remuestrear a 24kHz
                    if sr != TARGET_SR:
                        chunk_24k = librosa.resample(chunk, orig_sr=sr, target_sr=TARGET_SR)
                    else:
                        chunk_24k = chunk

                    # Normalizar RMS a -20 dBFS
                    rms = librosa.feature.rms(y=chunk_24k)[0]
                    target_rms = 10 ** (-20 / 20)
                    mean_rms = rms.mean() + 1e-9
                    chunk_24k = chunk_24k * (target_rms / mean_rms)

                    # Guardar chunk
                    chunk_filename = f"chunk_{chunk_counter:05d}.wav"
                    chunk_filepath = os.path.join(wavs_dir, chunk_filename)
                    sf.write(chunk_filepath, chunk_24k, TARGET_SR, subtype='PCM_16')

                    # 4. Transcripción ASR con Prompt Injection
                    generate_kwargs = {
                        "language": language if language != "auto" else None,
                        "prompt": "Uhm, ah, [risa], [suspiro], [tos], eh, mhm, bueno..."
                    }
                    # Remove None values
                    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

                    # Whisper natively expects 16kHz audio.
                    # We create a 16kHz copy in RAM specifically for the pipeline to bypass torchaudio file loading.
                    chunk_16k = librosa.resample(chunk_24k, orig_sr=TARGET_SR, target_sr=16000)

                    # Pass the raw numpy array instead of the chunk_filepath string
                    transcription = asr_pipeline(chunk_16k, generate_kwargs=generate_kwargs)["text"].strip()

                    # Limpiar saltos de línea
                    transcription = transcription.replace('\n', ' ').replace('|', '')

                    if transcription:
                        # Formato JSONL
                        entry = {
                            "text": transcription,
                            "audio": os.path.abspath(chunk_filepath)
                        }
                        jsonl_entries.append(entry)
                        chunk_counter += 1
                        print(f"Procesado: {chunk_filename} -> {transcription[:50]}...")

            except Exception as e:
                print(f"[Warning] Error procesando {audio_path}: {e}. Saltando archivo...")
                continue
            finally:
                # Cleanup temporary file
                if temp_wav_path and os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                    except Exception as cleanup_error:
                        print(f"[Warning] Failed to remove temp file {temp_wav_path}: {cleanup_error}")

        # 5. Guardar prompts.jsonl
        with open(prompts_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        print(f"[VibeVoice] Dataset completado: {chunk_counter} fragmentos válidos generados en {output_dataset_dir}")

        # Clean up pipeline
        del asr_pipeline
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
                "quantization": (["none (bf16)", "8-bit", "4-bit"], {"default": "none (bf16)"}),
                "output_lora_name": ("STRING", {"default": "vibevoice_lora_out"}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 32}),
                "gradient_accum_steps": ("INT", {"default": 16, "min": 1, "max": 128}), # Default 16 as requested
                "epochs": ("INT", {"default": 150, "min": 10, "max": 1000}),
                "learning_rate": ("FLOAT", {"default": 2e-4, "step": 1e-5}),
                "mixed_precision": (["bf16", "fp16", "no"], {"default": "bf16"}),
                "lora_rank": ("INT", {"default": 32, "min": 4, "max": 128}),
                "lora_alpha": ("INT", {"default": 64, "min": 8, "max": 256}),
                "transformers_version": ("STRING", {"default": "4.44.2", "multiline": False}), # Providing flexibility
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

    def _patch_training_script(self, repo_dir):
        """Patches the training script to support 4-bit/8-bit quantization."""
        script_path = os.path.join(repo_dir, "src", "finetune_vibevoice_lora.py")
        if not os.path.exists(script_path):
            print(f"[VibeVoice Patch] Script not found at {script_path}")
            return False

        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if already patched
        if "load_in_4bit" in content and "BitsAndBytesConfig" in content:
            return True

        print("[VibeVoice Patch] Patching training script for quantization support...")

        # 1. Add Import
        if "from transformers import (" in content:
            content = content.replace(
                "from transformers import (",
                "from transformers import ( BitsAndBytesConfig,"
            )
        else:
             content = "from transformers import BitsAndBytesConfig\n" + content

        # 2. Add Arguments to ModelArguments
        search_str = "class ModelArguments:\n"
        if search_str in content:
            # Insert fields
            fields = (
                "    load_in_4bit: bool = field(default=False, metadata={'help': 'Load in 4-bit mode'})\n"
                "    load_in_8bit: bool = field(default=False, metadata={'help': 'Load in 8-bit mode'})\n"
            )
            content = content.replace(search_str, search_str + fields)

        # 3. Modify Model Loading
        target_block = (
            "    model = VibeVoiceForConditionalGeneration.from_pretrained(\n"
            "        model_args.model_name_or_path,\n"
            "        torch_dtype=dtype,\n"
            "    )"
        )

        replacement_block = (
            "    quantization_config = None\n"
            "    if getattr(model_args, 'load_in_4bit', False):\n"
            "        quantization_config = BitsAndBytesConfig(\n"
            "            load_in_4bit=True,\n"
            "            bnb_4bit_compute_dtype=dtype,\n"
            "            bnb_4bit_use_double_quant=True,\n"
            "            bnb_4bit_quant_type='nf4',\n"
            "        )\n"
            "    elif getattr(model_args, 'load_in_8bit', False):\n"
            "        quantization_config = BitsAndBytesConfig(\n"
            "            load_in_8bit=True,\n"
            "        )\n\n"
            "    model = VibeVoiceForConditionalGeneration.from_pretrained(\n"
            "        model_args.model_name_or_path,\n"
            "        torch_dtype=dtype,\n"
            "        quantization_config=quantization_config,\n"
            "    )"
        )

        if target_block in content:
             content = content.replace(target_block, replacement_block)
        else:
             print("[VibeVoice Patch] Warning: Could not find exact model loading block to patch. Quantization might not work.")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("[VibeVoice Patch] Patch applied successfully.")
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
        self._patch_training_script(repo_dir)
        self._patch_flash_attention_import(repo_dir)

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
                subprocess.check_call([pip_cmd, "install", "accelerate", "peft", "bitsandbytes", "soundfile", "librosa"], cwd=repo_dir)

                # Create marker
                with open(marker_file, "w") as f:
                    f.write(f"Installed transformers=={transformers_version}")

            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to install dependencies: {e}")
                return False

        return python_cmd

    def train_lora(self, dataset_path, base_model_path, quantization, output_lora_name, batch_size,
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

        # Construct Command
        # python -m src.finetune_vibevoice_lora ...
        command = [
            python_cmd, "-m", "src.finetune_vibevoice_lora",
            "--model_name_or_path", model_path_to_use,
            "--train_jsonl", prompts_jsonl,
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
            "--save_strategy", "epoch"
        ]

        # Append Quantization Flags
        if quantization == "8-bit":
            command.extend(["--load_in_8bit", "True"])
        elif quantization == "4-bit":
             command.extend(["--load_in_4bit", "True"])

        print("[VibeVoice] Iniciando subproceso de entrenamiento...")
        print("Comando:", " ".join(command))

        # Popen
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=repo_dir)

        # Read output
        thread = threading.Thread(target=self._read_subprocess_output, args=(process,))
        thread.start()

        process.wait()
        thread.join()

        if process.returncode == 0:
            print(f"[VibeVoice] Entrenamiento finalizado con éxito. LoRA guardado en {output_dir}")
        else:
            print(f"[VibeVoice] Entrenamiento falló con código {process.returncode}.")

        return (os.path.abspath(output_dir),)
