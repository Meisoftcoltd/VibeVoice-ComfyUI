import unittest
import sys
import os
from unittest.mock import MagicMock

# Add repo root to path
sys.path.append(os.getcwd())

# Mock ComfyUI modules
sys.modules["folder_paths"] = MagicMock()
sys.modules["folder_paths"].get_input_directory.return_value = "/tmp/input"
sys.modules["folder_paths"].get_output_directory.return_value = "/tmp/output"
sys.modules["folder_paths"].get_temp_directory.return_value = "/tmp/temp"

try:
    from nodes.qwen_dataset_preparator import Qwen2Audio_Dataset_Preparator
except ImportError as e:
    print(f"Failed to import Qwen2Audio_Dataset_Preparator: {e}")
    sys.exit(1)

class TestQwenNode(unittest.TestCase):
    def test_class_structure(self):
        node = Qwen2Audio_Dataset_Preparator()
        self.assertTrue(hasattr(node, "prepare_dataset"))
        self.assertTrue(hasattr(node, "INPUT_TYPES"))

        input_types = node.INPUT_TYPES()
        self.assertIn("required", input_types)
        self.assertIn("raw_audio_dir", input_types["required"])
        self.assertIn("output_dataset_dir", input_types["required"])
        self.assertIn("transcription_prompt", input_types["required"])

        # Check defaults
        self.assertEqual(input_types["required"]["transcription_prompt"][1]["default"],
                         "Transcribe the following Spanish audio exactly. If you hear non-verbal sounds like laughing, sighing, taking a breath, or clearing the throat, insert descriptive tags in brackets within the transcription (e.g., [risa], [suspiro], [respira], [aclara la garganta]).")

if __name__ == '__main__':
    unittest.main()
