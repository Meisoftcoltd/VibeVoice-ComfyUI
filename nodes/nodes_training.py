import os
import json
import glob
import torch
import torchaudio
import librosa
import soundfile as sf
import subprocess
import threading
import sys
import shutil
from transformers import pipeline

# Configure logging
import logging
logger = logging.getLogger("VibeVoice")

class VibeVoice_Dataset_Preparator:
    """
    Prepara un dataset para VibeVoice generando un archivo prompts.jsonl.
    Corta silencios, remuestrea a 24kHz, normaliza el audio y transcribe con Whisper base.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "raw_audio_dir": ("STRING", {"default": "./raw_audios"}),
            "output_dataset_dir": ("STRING", {"default": "./vibevoice_dataset"}),
            "language": (["es", "en", "auto"], {"default": "es"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_path",)
    FUNCTION = "prepare_dataset"
    CATEGORY = "VibeVoice/Training"

    def prepare_dataset(self, raw_audio_dir, output_dataset_dir, language):
        os.makedirs(output_dataset_dir, exist_ok=True)
        wavs_dir = os.path.join(output_dataset_dir, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)
        prompts_path = os.path.join(output_dataset_dir, "prompts.jsonl")

        # Configuraciones acústicas estrictas
        TARGET_SR = 24000
        MIN_LEN_SEC = 3.0
        MAX_LEN_SEC = 14.0

        # Cargar Whisper estándar (HuggingFace)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VibeVoice] Cargando modelo Whisper en {device}...")
        try:
            asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=device
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
            try:
                # 1. Cargar audio
                y, sr = librosa.load(audio_path, sr=None, mono=True)

                # 2. VAD (Detección de silencios y corte usando librosa effects por estabilidad)
                # Separa el audio donde hay silencios menores a 30 decibelios
                non_mute_intervals = librosa.effects.split(y, top_db=30)

                for interval in non_mute_intervals:
                    start_sample, end_sample = interval
                    chunk = y[start_sample:end_sample]
                    duration = len(chunk) / sr

                    # 3. Filtro de longitud estricto
                    if MIN_LEN_SEC <= duration <= MAX_LEN_SEC:
                        # Remuestrear a 24kHz si es necesario
                        if sr != TARGET_SR:
                            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=TARGET_SR)

                        # Normalizar RMS a -20 dBFS
                        rms = librosa.feature.rms(y=chunk)[0]
                        target_rms = 10 ** (-20 / 20)
                        mean_rms = rms.mean() + 1e-9
                        chunk = chunk * (target_rms / mean_rms)

                        # Guardar chunk
                        chunk_filename = f"chunk_{chunk_counter:05d}.wav"
                        chunk_filepath = os.path.join(wavs_dir, chunk_filename)
                        sf.write(chunk_filepath, chunk, TARGET_SR, subtype='PCM_16')

                        # 4. Transcripción ASR
                        generate_kwargs = {"language": language} if language != "auto" else {}
                        transcription_result = asr_pipeline(chunk_filepath, generate_kwargs=generate_kwargs)
                        transcription = transcription_result["text"].strip()

                        # Limpiar saltos de línea y caracteres raros
                        transcription = transcription.replace('\n', ' ').replace('|', '')

                        if transcription:
                            # Formato JSONL: {"text": "...", "audio": "/absolute/path/to/chunk.wav"}
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

        # 5. Empaquetado: Guardar prompts.jsonl
        with open(prompts_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        print(f"[VibeVoice] Dataset completado: {chunk_counter} fragmentos válidos generados en {output_dataset_dir}")
        return (os.path.abspath(output_dataset_dir),)


class VibeVoice_LoRA_Trainer:
    """
    Ejecuta el entrenamiento del modelo usando un entorno virtual aislado para evitar conflictos de dependencias.
    Clona automáticamente el repositorio de entrenamiento y gestiona el venv.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dataset_path": ("STRING", {"forceInput": True}),
            "base_model_path": ("STRING", {"default": "microsoft/VibeVoice-1.5B"}),
            "output_lora_name": ("STRING", {"default": "vibevoice_lora_out"}),
            "batch_size": ("INT", {"default": 8, "min": 1, "max": 32}),
            "gradient_accum_steps": ("INT", {"default": 16, "min": 1, "max": 128}), # Default 16 as requested
            "epochs": ("INT", {"default": 150, "min": 10, "max": 1000}),
            "learning_rate": ("FLOAT", {"default": 2e-4, "step": 1e-5}),
            "mixed_precision": (["bf16", "fp16", "no"], {"default": "bf16"}),
            # "optimizer": (["adamw_8bit", "adamw_torch"], {"default": "adamw_8bit"}), # Implicit in script
            "lora_rank": ("INT", {"default": 32, "min": 4, "max": 128}),
            "lora_alpha": ("INT", {"default": 64, "min": 8, "max": 256}),
            "transformers_version": ("STRING", {"default": "4.44.2", "multiline": False}), # Providing flexibility
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_output_dir",)
    FUNCTION = "train_lora"
    CATEGORY = "VibeVoice/Training"

    def _read_subprocess_output(self, process):
        """Lee la salida del subproceso asíncronamente para la consola de ComfyUI."""
        for line in iter(process.stdout.readline, b''):
            print(f"[VibeVoice Train] {line.decode('utf-8', errors='replace').rstrip()}")
        process.stdout.close()

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

    def train_lora(self, dataset_path, base_model_path, output_lora_name, batch_size,
                   gradient_accum_steps, epochs, learning_rate, mixed_precision,
                   lora_rank, lora_alpha, transformers_version):

        # Paths
        current_dir = os.path.dirname(os.path.realpath(__file__))
        repo_dir = os.path.join(current_dir, "vibevoice_finetuning_repo")
        venv_dir = os.path.join(current_dir, "vibevoice_venv")

        # Setup Environment
        python_cmd = self._setup_environment(repo_dir, venv_dir, transformers_version)
        if not python_cmd:
            return ("Error during setup",)

        # Output directory
        output_dir = os.path.join(dataset_path, output_lora_name)
        os.makedirs(output_dir, exist_ok=True)

        prompts_jsonl = os.path.join(dataset_path, "prompts.jsonl")
        if not os.path.exists(prompts_jsonl):
             print(f"[Error] prompts.jsonl not found in {dataset_path}")
             return (output_dir,)

        # Construct Command
        # python -m src.finetune_vibevoice_lora ...
        command = [
            python_cmd, "-m", "src.finetune_vibevoice_lora",
            "--model_name_or_path", base_model_path,
            "--train_jsonl", prompts_jsonl, # Updated flag
            "--output_dir", output_dir,
            "--per_device_train_batch_size", str(batch_size),
            "--gradient_accumulation_steps", str(gradient_accum_steps),
            "--num_train_epochs", str(epochs),
            "--learning_rate", str(learning_rate),
            "--bf16", "True" if mixed_precision == "bf16" else "False", # Assuming script uses True/False string or bool
            "--fp16", "True" if mixed_precision == "fp16" else "False",
            "--lora_r", str(lora_rank),
            "--lora_alpha", str(lora_alpha),
            "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", # As requested
            "--train_diffusion_head", "True", # As requested
            "--ddpm_batch_mul", "4", # As requested
            "--diffusion_loss_weight", "1.4", # As requested
            "--ce_loss_weight", "0.04", # As requested
            "--voice_prompt_drop_rate", "0.2", # As requested
            "--logging_steps", "10",
            "--save_strategy", "epoch"
        ]

        # Handle "dataset_name" or similar if required by script, but --train_jsonl seems sufficient based on description.
        # The snippet had "--dataset_name your/dataset", but user said "Instead... use --train_jsonl {dataset_path}/prompts.jsonl"
        # I will assume --train_jsonl replaces --dataset_name or works alongside.

        # Also need to run from repo_dir so python -m src... works

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
