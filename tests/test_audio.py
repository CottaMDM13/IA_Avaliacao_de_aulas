import unittest
from src.analysis.audio import extract_audio_features

class TestAudioAnalysis(unittest.TestCase):
    def test_extract_audio_features(self):
        features = extract_audio_features("data/input/audio/audio.wav")
        self.assertIn("duration_sec", features)
        self.assertGreater(features["duration_sec"], 0)

if __name__ == "__main__":
    unittest.main()