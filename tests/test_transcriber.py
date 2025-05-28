import unittest
import os
from src.utils.transcriber import transcribe_audio

class TestTranscriber(unittest.TestCase):
    def test_transcribe_audio(self):
        result = transcribe_audio("data/input/audio/audio.wav", "data/output/transcripts")
        self.assertIn("text", result)
        self.assertTrue(os.path.exists(result["txt_path"]))

if __name__ == "__main__":
    unittest.main()