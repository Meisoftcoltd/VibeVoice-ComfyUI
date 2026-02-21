import unittest
import sys
import os
from unittest.mock import MagicMock

# Add repo root to path
sys.path.append(os.getcwd())

# Mock all heavy dependencies
sys.modules["torch"] = MagicMock()
sys.modules["torchaudio"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["soundfile"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["folder_paths"] = MagicMock()
sys.modules["folder_paths"].models_dir = "/tmp/models"

# Mock specific attributes needed
sys.modules["torch"].float16 = "float16"
sys.modules["torch"].float32 = "float32"
sys.modules["torch"].cuda.is_available.return_value = True

try:
    from nodes.nodes_training import VibeVoice_Dataset_Preparator
except ImportError as e:
    print(f"Failed to import VibeVoice_Dataset_Preparator: {e}")
    raise e

class TestVibeVoiceNode(unittest.TestCase):
    def test_class_structure(self):
        node = VibeVoice_Dataset_Preparator()
        self.assertTrue(hasattr(node, "prepare_dataset"))
        self.assertTrue(hasattr(node, "INPUT_TYPES"))

        # Verify INPUT_TYPES
        input_types = VibeVoice_Dataset_Preparator.INPUT_TYPES()
        self.assertIn("required", input_types)
        required = input_types["required"]
        self.assertIn("raw_audio_dir", required)
        self.assertIn("output_dataset_dir", required)
        self.assertIn("model", required)
        self.assertIn("language", required)

        # Check model options
        model_options = required["model"][0]
        self.assertIn("openai/whisper-large-v3-turbo", model_options)
        self.assertEqual(required["model"][1]["default"], "openai/whisper-large-v3-turbo")

if __name__ == '__main__':
    unittest.main()
