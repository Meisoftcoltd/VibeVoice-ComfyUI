import os
import json
import glob
import torch
import librosa
import soundfile as sf
import gc
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import logging

# Configure logging
logger = logging.getLogger("VibeVoice")

class Qwen2Audio_Dataset_Preparator:
    """
    Prepares a dataset for VibeVoice using Qwen2-Audio-7B-Instruct for transcription.
    Slices audio by silence, resamples to 24kHz, and generates a prompts.jsonl file.
    Designed for 24GB VRAM GPUs (RTX 3090/4090) with strict memory cleanup.
    """
    @classmethod
    def INPUT_TYPES(cls):
        default_prompt = (
            "Transcribe the following Spanish audio exactly. "
            "If you hear non-verbal sounds like laughing, sighing, taking a breath, or clearing the throat, "
            "insert descriptive tags in brackets within the transcription "
            "(e.g., [risa], [suspiro], [respira], [aclara la garganta])."
        )
        return {"required": {
            "raw_audio_dir": ("STRING", {"default": "./raw_audios"}),
            "output_dataset_dir": ("STRING", {"default": "./vibevoice_dataset_qwen"}),
            "transcription_prompt": ("STRING", {"default": default_prompt, "multiline": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_path",)
    FUNCTION = "prepare_dataset"
    CATEGORY = "VibeVoice/Training"

    def prepare_dataset(self, raw_audio_dir, output_dataset_dir, transcription_prompt):
        os.makedirs(output_dataset_dir, exist_ok=True)
        wavs_dir = os.path.join(output_dataset_dir, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)
        prompts_path = os.path.join(output_dataset_dir, "prompts.jsonl")

        # Acoustic configurations
        TARGET_SR = 24000
        MIN_LEN_SEC = 3.0
        MAX_LEN_SEC = 14.0

        # Load Qwen2-Audio Model
        print("[Qwen2Audio] Loading Qwen2-Audio-7B-Instruct (bfloat16)...")
        try:
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct",
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        except Exception as e:
            print(f"[Error] Failed to load Qwen2-Audio model: {e}")
            raise e

        audio_files = glob.glob(os.path.join(raw_audio_dir, "**/*.*"), recursive=True)
        valid_extensions = ('.wav', '.mp3', '.flac', '.ogg')
        audio_files = [f for f in audio_files if f.lower().endswith(valid_extensions)]

        print(f"[Qwen2Audio] Found {len(audio_files)} audio files. Starting processing...")

        jsonl_entries = []
        chunk_counter = 0

        try:
            for audio_path in audio_files:
                try:
                    # 1. Load audio
                    y, sr = librosa.load(audio_path, sr=None, mono=True)

                    # 2. VAD (Silence detection and splitting)
                    non_mute_intervals = librosa.effects.split(y, top_db=30)

                    for interval in non_mute_intervals:
                        start_sample, end_sample = interval
                        chunk = y[start_sample:end_sample]
                        duration = len(chunk) / sr

                        # 3. Strict length filter
                        if MIN_LEN_SEC <= duration <= MAX_LEN_SEC:
                            # Resample to 24kHz for VibeVoice Dataset
                            if sr != TARGET_SR:
                                chunk_24k = librosa.resample(chunk, orig_sr=sr, target_sr=TARGET_SR)
                            else:
                                chunk_24k = chunk

                            # Normalize RMS to -20 dBFS
                            rms = librosa.feature.rms(y=chunk_24k)[0]
                            target_rms = 10 ** (-20 / 20)
                            mean_rms = rms.mean() + 1e-9
                            chunk_24k = chunk_24k * (target_rms / mean_rms)

                            # Save chunk to disk
                            chunk_filename = f"chunk_{chunk_counter:05d}.wav"
                            chunk_filepath = os.path.join(wavs_dir, chunk_filename)
                            sf.write(chunk_filepath, chunk_24k, TARGET_SR, subtype='PCM_16')

                            # 4. Transcription with Qwen2-Audio
                            # We need to process the audio for Qwen
                            # Qwen processor expects audio at specific sampling rate
                            qwen_sr = processor.feature_extractor.sampling_rate
                            if sr != qwen_sr:
                                # Resample original chunk to Qwen SR
                                chunk_qwen = librosa.resample(chunk, orig_sr=sr, target_sr=qwen_sr)
                            else:
                                chunk_qwen = chunk

                            # Prepare input
                            conversation = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "audio", "audio_url": chunk_filepath}, # Placeholder, we pass raw audio
                                        {"type": "text", "text": transcription_prompt}
                                    ]
                                }
                            ]

                            # Note: Qwen processor takes 'audios' as list of numpy arrays
                            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                            inputs = processor(
                                text=text,
                                audios=[chunk_qwen],
                                return_tensors="pt",
                                padding=True
                            )
                            inputs = inputs.to(model.device)

                            # Generate
                            with torch.no_grad():
                                generate_ids = model.generate(**inputs, max_new_tokens=256)

                            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
                            transcription = processor.batch_decode(
                                generate_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False
                            )[0]

                            # Clean up transcription
                            transcription = transcription.strip()
                            transcription = transcription.replace('\n', ' ').replace('|', '')

                            if transcription:
                                # JSONL format: {"text": "...", "audio": "/absolute/path/to/chunk.wav"}
                                entry = {
                                    "text": transcription,
                                    "audio": os.path.abspath(chunk_filepath)
                                }
                                jsonl_entries.append(entry)
                                chunk_counter += 1
                                print(f"Processed: {chunk_filename} -> {transcription[:50]}...")

                except Exception as e:
                    print(f"[Warning] Error processing {audio_path}: {e}. Skipping file...")
                    continue

            # 5. Save prompts.jsonl
            with open(prompts_path, 'w', encoding='utf-8') as f:
                for entry in jsonl_entries:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')

            print(f"[Qwen2Audio] Dataset completed: {chunk_counter} valid chunks generated in {output_dataset_dir}")

        finally:
            # CRITICAL: VRAM Management (Teardown)
            print("[Qwen2Audio] Cleaning up memory...")
            if 'model' in locals():
                del model
            if 'processor' in locals():
                del processor
            if 'inputs' in locals():
                del inputs
            if 'generate_ids' in locals():
                del generate_ids

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except AttributeError:
                    pass # Not always available

            print("[Qwen2Audio] Memory cleanup complete.")

        return (os.path.abspath(output_dataset_dir),)
