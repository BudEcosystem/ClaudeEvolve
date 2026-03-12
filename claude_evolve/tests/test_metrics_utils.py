"""Tests for claude_evolve.utils.metrics_utils"""

import math
import unittest

from claude_evolve.utils.metrics_utils import (
    format_feature_coordinates,
    get_fitness_score,
    safe_numeric_average,
    safe_numeric_sum,
)


class TestSafeNumericAverage(unittest.TestCase):
    def test_all_numeric(self):
        result = safe_numeric_average({"a": 1.0, "b": 2.0, "c": 3.0})
        self.assertAlmostEqual(result, 2.0)

    def test_mixed_types(self):
        result = safe_numeric_average({"a": 1.0, "b": "string", "c": 3.0})
        self.assertAlmostEqual(result, 2.0)

    def test_empty(self):
        self.assertEqual(safe_numeric_average({}), 0.0)

    def test_booleans_excluded(self):
        # booleans are int subclass but isinstance(True, (int, float)) is True
        # In the OpenEvolve implementation, booleans pass the isinstance check
        # but we should verify the behavior
        result = safe_numeric_average({"a": True, "b": False, "c": 3.0})
        # True=1.0, False=0.0, so average = (1+0+3)/3 = 1.333...
        self.assertAlmostEqual(result, 4.0 / 3.0)

    def test_all_strings(self):
        result = safe_numeric_average({"a": "hello", "b": "world"})
        self.assertEqual(result, 0.0)

    def test_nan_excluded(self):
        result = safe_numeric_average({"a": float("nan"), "b": 4.0})
        self.assertAlmostEqual(result, 4.0)

    def test_integers(self):
        result = safe_numeric_average({"x": 10, "y": 20})
        self.assertAlmostEqual(result, 15.0)


class TestSafeNumericSum(unittest.TestCase):
    def test_all_numeric(self):
        result = safe_numeric_sum({"a": 1.0, "b": 2.0, "c": 3.0})
        self.assertAlmostEqual(result, 6.0)

    def test_mixed_types(self):
        result = safe_numeric_sum({"a": 5.0, "b": "skip", "c": 10.0})
        self.assertAlmostEqual(result, 15.0)

    def test_empty(self):
        self.assertEqual(safe_numeric_sum({}), 0.0)

    def test_all_non_numeric(self):
        result = safe_numeric_sum({"a": "x", "b": [1, 2]})
        self.assertEqual(result, 0.0)

    def test_nan_excluded(self):
        result = safe_numeric_sum({"a": float("nan"), "b": 7.0})
        self.assertAlmostEqual(result, 7.0)


class TestGetFitnessScore(unittest.TestCase):
    def test_combined_score_preferred(self):
        metrics = {"combined_score": 0.85, "accuracy": 0.9, "speed": 0.7}
        result = get_fitness_score(metrics)
        self.assertAlmostEqual(result, 0.85)

    def test_excludes_features(self):
        metrics = {"accuracy": 0.9, "code_length": 50.0, "speed": 0.7}
        result = get_fitness_score(metrics, feature_dimensions=["code_length"])
        # Should average accuracy and speed only: (0.9 + 0.7) / 2 = 0.8
        self.assertAlmostEqual(result, 0.8)

    def test_fallback_to_average(self):
        metrics = {"accuracy": 0.9, "speed": 0.7}
        result = get_fitness_score(metrics)
        self.assertAlmostEqual(result, 0.8)

    def test_empty(self):
        self.assertEqual(get_fitness_score({}), 0.0)

    def test_combined_score_invalid_type(self):
        # combined_score is present but not convertible
        metrics = {"combined_score": "invalid", "accuracy": 0.9}
        result = get_fitness_score(metrics)
        # Falls through to average of numeric values
        self.assertAlmostEqual(result, 0.9)

    def test_all_features_excluded_falls_back(self):
        # All numeric metrics are feature dims -> falls back to safe_numeric_average of all
        metrics = {"code_length": 50.0, "complexity": 10.0}
        result = get_fitness_score(
            metrics, feature_dimensions=["code_length", "complexity"]
        )
        # fitness_metrics is empty, so falls back to safe_numeric_average(metrics)
        self.assertAlmostEqual(result, 30.0)

    def test_none_feature_dimensions(self):
        metrics = {"accuracy": 0.5, "speed": 0.5}
        result = get_fitness_score(metrics, feature_dimensions=None)
        self.assertAlmostEqual(result, 0.5)


class TestFormatFeatureCoordinates(unittest.TestCase):
    def test_with_features(self):
        metrics = {"code_length": 42.0, "speed": 0.8, "accuracy": 0.9}
        result = format_feature_coordinates(
            metrics, feature_dimensions=["code_length", "speed"]
        )
        self.assertIn("code_length=42.00", result)
        self.assertIn("speed=0.80", result)
        self.assertNotIn("accuracy", result)

    def test_no_features(self):
        metrics = {"accuracy": 0.9}
        result = format_feature_coordinates(metrics, feature_dimensions=[])
        self.assertEqual(result, "")

    def test_missing_feature_dimension(self):
        metrics = {"accuracy": 0.9}
        result = format_feature_coordinates(
            metrics, feature_dimensions=["nonexistent"]
        )
        self.assertEqual(result, "")

    def test_string_feature_value(self):
        metrics = {"category": "fast", "speed": 0.9}
        result = format_feature_coordinates(
            metrics, feature_dimensions=["category", "speed"]
        )
        self.assertIn("category=fast", result)
        self.assertIn("speed=0.90", result)

    def test_nan_feature_value(self):
        metrics = {"speed": float("nan")}
        result = format_feature_coordinates(
            metrics, feature_dimensions=["speed"]
        )
        # NaN should still appear but with raw format since the NaN check catches it
        self.assertIn("speed=nan", result)


if __name__ == "__main__":
    unittest.main()
