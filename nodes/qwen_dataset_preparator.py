import os
import json
import glob
import re
import torch
import librosa
import soundfile as sf
import gc
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import logging

# Configure logging
logger = logging.getLogger("VibeVoice")

class Qwen2Audio_Dataset_Preparator:
    """
    Prepares a dataset for VibeVoice using Qwen2-Audio-7B-Instruct for transcription.
    Slices audio by silence using a smart accumulative strategy, resamples to 24kHz, and generates a prompts.jsonl file.
    Designed for 24GB VRAM GPUs (RTX 3090/4090) with strict memory cleanup.
    """
    @classmethod
    def INPUT_TYPES(cls):
        default_prompt = (
            "You are a strict transcription API. Transcribe the Spanish audio perfectly. "
            "Insert tags like [risa] or [suspiro] if you hear them. "
            "Do NOT output any conversational text, introductions, or quotes. Output ONLY the raw transcript."
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

    def prepare_dataset(self, raw_audio_dir, output_dataset_dir, transcription_prompt):
        os.makedirs(output_dataset_dir, exist_ok=True)
        wavs_dir = os.path.join(output_dataset_dir, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)
        prompts_path = os.path.join(output_dataset_dir, "prompts.jsonl")

        # Acoustic configurations
        TARGET_SR = 24000
        MIN_LEN_SEC = 2.0
        MAX_LEN_SEC = 20.0

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
                    total_duration = len(y) / sr

                    chunks_to_process = [] # list of (audio_chunk, start_sample, end_sample)

                    if total_duration <= MAX_LEN_SEC:
                        # Process whole file if it's short enough
                        chunks_to_process.append((y, 0, len(y)))
                    else:
                        # Smart Slicing
                        # Detect silence boundaries using librosa.effects.split(y, top_db=40)
                        intervals = librosa.effects.split(y, top_db=40)
                        sliced_intervals = self._smart_slice(intervals, len(y), sr, MIN_LEN_SEC, MAX_LEN_SEC)

                        for start, end in sliced_intervals:
                             chunks_to_process.append((y[start:end], start, end))

                    for chunk, start_sample, end_sample in chunks_to_process:
                        duration = len(chunk) / sr

                        # Filter by MIN_LEN_SEC (MAX_LEN_SEC is handled by slicing logic, but safe to check)
                        if duration < MIN_LEN_SEC:
                            continue

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
                            # Resample original chunk to Qwen SR (using original sr for better quality before downsampling)
                            # Actually, we should resample the *original chunk* (y[start:end]) not the 24k one
                            # to avoid double resampling artifacts if possible, but keeping it simple is fine.
                            # But wait, chunk is numpy array.
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

                        # Strict cleaning: Remove conversational wrappers and quotes
                        # Remove "La grabación dice: ", "El audio dice: ", etc.
                        transcription = re.sub(r"^(?:La grabación dice|El audio dice)[:;]?\s*", "", transcription, flags=re.IGNORECASE)
                        # Remove surrounding quotes
                        transcription = transcription.strip('"\'')

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
