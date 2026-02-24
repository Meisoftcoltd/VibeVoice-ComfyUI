# Created by Fabio Sarracino
# Original LoRa code implementation by jpgallegoar-vpai user via PR #127 
# LoRA configuration node for VibeVoice

import logging
import os
from typing import Dict, Any, List

# Setup logging
logger = logging.getLogger("VibeVoice")

# Cache for LoRA scanning to avoid repeated logs
_lora_cache = {
    "first_load_logged": False
}

def get_available_loras() -> List[str]:
    """Get list of available LoRA folders in ComfyUI/models/vibevoice/loras recursively"""
    try:
        import folder_paths

        # Get the ComfyUI models directory
        models_dir = folder_paths.get_folder_paths("checkpoints")[0]
        loras_dir = os.path.join(os.path.dirname(models_dir), "vibevoice", "loras")

        os.makedirs(loras_dir, exist_ok=True)

        lora_folders = []
        if os.path.exists(loras_dir):
            # Deep scan all subdirectories
            for root, dirs, files in os.walk(loras_dir):
                has_config = "adapter_config.json" in files
                has_model = "adapter_model.safetensors" in files or "adapter_model.bin" in files

                if has_config and has_model:
                    # Get path relative to the loras base directory
                    rel_path = os.path.relpath(root, loras_dir)
                    # Normalize slashes for the UI dropdown
                    rel_path = rel_path.replace("\\", "/")
                    lora_folders.append(rel_path)

        if not lora_folders:
            lora_folders = ["None"]
        else:
            # Sort alphabetically to keep checkpoints organized, then prepend "None"
            lora_folders.sort()
            lora_folders.insert(0, "None")

        return lora_folders

    except ImportError:
        logger.error("Could not import folder_paths from ComfyUI")
        return ["None"]
    except Exception as e:
        logger.error(f"Error listing LoRAs: {e}")
        return ["None"]

class VibeVoiceLoRANode:
    """Node for configuring LoRA adapters for VibeVoice models"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Get available LoRA folders dynamically
        available_loras = get_available_loras()

        return {
            "required": {
                "lora_name": (available_loras, {
                    "default": "None",
                    "tooltip": "Select a LoRA adapter from ComfyUI/models/vibevoice/loras folder"
                }),
                "llm_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Strength of the LLM LoRA adapter. Controls how much the LoRA affects the language model"
                }),
                "use_llm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply LLM (language model) LoRA component when available"
                }),
                "use_diffusion_head": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply diffusion head LoRA/replacement when available"
                }),
                "use_acoustic_connector": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply acoustic connector LoRA component when available"
                }),
                "use_semantic_connector": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply semantic connector LoRA component when available"
                }),
            }
        }

    RETURN_TYPES = ("LORA_CONFIG",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "configure_lora"
    CATEGORY = "VibeVoiceWrapper"
    DESCRIPTION = "Configure LoRA adapters for fine-tuned VibeVoice models. Place LoRA folders in ComfyUI/models/vibevoice/loras/"

    def configure_lora(self, lora_name: str, llm_strength: float, use_llm: bool,
                       use_diffusion_head: bool, use_acoustic_connector: bool,
                       use_semantic_connector: bool) -> tuple:
        if lora_name == "None":
            if not _lora_cache.get("first_load_logged"):
                logger.info("No LoRA selected")
                _lora_cache["first_load_logged"] = True
            return (None,)

        try:
            import folder_paths
            models_dir = folder_paths.get_folder_paths("checkpoints")[0]
            loras_dir = os.path.join(os.path.dirname(models_dir), "vibevoice", "loras")

            # Since lora_name is now the exact relative path, we just join it safely
            lora_path = os.path.normpath(os.path.join(loras_dir, lora_name))

            if not os.path.isdir(lora_path):
                logger.error(f"LoRA path is not a directory: {lora_path}")
                raise Exception(f"LoRA path must be a directory: {lora_name}")

            # Check for required files directly in the selected path
            adapter_config = os.path.join(lora_path, "adapter_config.json")
            adapter_model_st = os.path.join(lora_path, "adapter_model.safetensors")
            adapter_model_bin = os.path.join(lora_path, "adapter_model.bin")

            if not os.path.exists(adapter_config):
                raise Exception(f"adapter_config.json not found in {lora_path}")
            if not (os.path.exists(adapter_model_st) or os.path.exists(adapter_model_bin)):
                raise Exception(f"No adapter_model found in {lora_path}")

            logger.info(f"LoRA configured: {os.path.basename(lora_path)} ({lora_path})")

            # Check for additional components inside the selected folder
            diffusion_head = os.path.join(lora_path, "diffusion_head")
            acoustic_connector = os.path.join(lora_path, "acoustic_connector")
            semantic_connector = os.path.join(lora_path, "semantic_connector")

            additional_components = []
            if os.path.exists(diffusion_head): additional_components.append("diffusion_head")
            if os.path.exists(acoustic_connector): additional_components.append("acoustic_connector")
            if os.path.exists(semantic_connector): additional_components.append("semantic_connector")

            if additional_components:
                logger.info(f"Additional LoRA components found: {', '.join(additional_components)}")

            # Create the configuration dictionary
            lora_config = {
                "path": lora_path,
                "llm_strength": llm_strength,
                "use_llm": use_llm,
                "use_diffusion_head": use_diffusion_head,
                "use_acoustic_connector": use_acoustic_connector,
                "use_semantic_connector": use_semantic_connector
            }

            # Log configuration
            enabled_components = []
            if use_llm:
                enabled_components.append(f"LLM (strength: {llm_strength})")
            if use_diffusion_head:
                enabled_components.append("Diffusion Head")
            if use_acoustic_connector:
                enabled_components.append("Acoustic Connector")
            if use_semantic_connector:
                enabled_components.append("Semantic Connector")

            if enabled_components:
                logger.info(f"LoRA components enabled: {', '.join(enabled_components)}")
            else:
                logger.warning("All LoRA components are disabled")

            return (lora_config,)

        except ImportError:
            logger.error("Could not import folder_paths from ComfyUI")
            raise Exception("Failed to access ComfyUI folders")
        except Exception as e:
            logger.error(f"Error configuring LoRA: {e}")
            raise

    @classmethod
    def IS_CHANGED(cls, lora_name: str = "None", **kwargs):
        """Cache key for ComfyUI - includes all parameters"""
        return f"{lora_name}_{kwargs.get('llm_strength', 1.0)}_{kwargs.get('use_llm', True)}_{kwargs.get('use_diffusion_head', True)}_{kwargs.get('use_acoustic_connector', True)}_{kwargs.get('use_semantic_connector', True)}"