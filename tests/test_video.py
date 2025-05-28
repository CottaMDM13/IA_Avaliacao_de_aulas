import unittest
from src.analysis.video import analyze_video

class TestVideoAnalysis(unittest.TestCase):
    def test_analyze_video(self):
        result = analyze_video("data/input/videos/aula_exemplo.mp4")
        self.assertIn("gestures", result)
        self.assertIn("facial_expressions", result)
        self.assertGreater(result["gestures"]["total_frames"], 0)

if __name__ == "__main__":
    unittest.main()