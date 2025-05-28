import unittest
from src.analysis.text import analyze_transcription

class TestTextAnalysis(unittest.TestCase):
    def test_analyze_transcription(self):
        config = {"thresholds": {"posture_score": 0.85, "face_visibility": 0.8, "audio_score": 60, "overall_score": 0.7}}
        result = analyze_transcription("Exemplo de texto", config)
        self.assertIn("overall_score", result)

if __name__ == "__main__":
    unittest.main()