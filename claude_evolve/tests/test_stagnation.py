"""
Tests for claude_evolve.core.stagnation.

Covers StagnationLevel enum, StagnationReport dataclass, and
StagnationEngine analysis with various score history patterns including
improving scores, plateaus at each severity level, edge cases (empty/single),
exploration boosts, strategy suggestions, failed approaches, and
recommendation generation.
"""

import unittest

from claude_evolve.core.stagnation import (
    StagnationConfig,
    StagnationEngine,
    StagnationLevel,
    StagnationReport,
)


# ---------------------------------------------------------------------------
# StagnationLevel enum
# ---------------------------------------------------------------------------


class TestStagnationLevel(unittest.TestCase):
    """Tests for StagnationLevel enum values."""

    def test_enum_values(self):
        self.assertEqual(StagnationLevel.NONE.value, "none")
        self.assertEqual(StagnationLevel.MILD.value, "mild")
        self.assertEqual(StagnationLevel.MODERATE.value, "moderate")
        self.assertEqual(StagnationLevel.SEVERE.value, "severe")
        self.assertEqual(StagnationLevel.CRITICAL.value, "critical")

    def test_enum_from_value(self):
        self.assertIs(StagnationLevel("none"), StagnationLevel.NONE)
        self.assertIs(StagnationLevel("mild"), StagnationLevel.MILD)
        self.assertIs(StagnationLevel("moderate"), StagnationLevel.MODERATE)
        self.assertIs(StagnationLevel("severe"), StagnationLevel.SEVERE)
        self.assertIs(StagnationLevel("critical"), StagnationLevel.CRITICAL)

    def test_enum_count(self):
        self.assertEqual(len(StagnationLevel), 5)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            StagnationLevel("nonexistent")


# ---------------------------------------------------------------------------
# StagnationReport dataclass
# ---------------------------------------------------------------------------


class TestStagnationReport(unittest.TestCase):
    """Tests for StagnationReport creation and field access."""

    def test_creation_with_all_fields(self):
        report = StagnationReport(
            level=StagnationLevel.MILD,
            iterations_stagnant=4,
            best_score=0.85,
            score_history=[0.5, 0.7, 0.85, 0.85, 0.85, 0.85, 0.85],
            exploration_ratio_boost=0.1,
            suggested_strategy="diversify",
            failed_approaches=["approach_a"],
            diagnosis="Mild stagnation",
            recommendations=["Try something different."],
        )
        self.assertEqual(report.level, StagnationLevel.MILD)
        self.assertEqual(report.iterations_stagnant, 4)
        self.assertAlmostEqual(report.best_score, 0.85)
        self.assertEqual(len(report.score_history), 7)
        self.assertAlmostEqual(report.exploration_ratio_boost, 0.1)
        self.assertEqual(report.suggested_strategy, "diversify")
        self.assertEqual(report.failed_approaches, ["approach_a"])
        self.assertIn("stagnation", report.diagnosis.lower())
        self.assertEqual(len(report.recommendations), 1)

    def test_empty_lists(self):
        report = StagnationReport(
            level=StagnationLevel.NONE,
            iterations_stagnant=0,
            best_score=0.0,
            score_history=[],
            exploration_ratio_boost=0.0,
            suggested_strategy="standard",
            failed_approaches=[],
            diagnosis="OK",
            recommendations=[],
        )
        self.assertEqual(report.failed_approaches, [])
        self.assertEqual(report.recommendations, [])
        self.assertEqual(report.score_history, [])


# ---------------------------------------------------------------------------
# StagnationConfig
# ---------------------------------------------------------------------------


class TestStagnationConfig(unittest.TestCase):
    """Tests for StagnationConfig defaults and overrides."""

    def test_defaults(self):
        cfg = StagnationConfig()
        self.assertEqual(cfg.mild_threshold, 3)
        self.assertEqual(cfg.moderate_threshold, 6)
        self.assertEqual(cfg.severe_threshold, 11)
        self.assertEqual(cfg.critical_threshold, 20)
        self.assertAlmostEqual(cfg.score_tolerance, 0.001)
        self.assertAlmostEqual(cfg.exploration_boost_mild, 0.1)
        self.assertAlmostEqual(cfg.exploration_boost_moderate, 0.2)
        self.assertAlmostEqual(cfg.exploration_boost_severe, 0.3)
        self.assertAlmostEqual(cfg.exploration_boost_critical, 0.5)

    def test_custom_thresholds(self):
        cfg = StagnationConfig(mild_threshold=5, critical_threshold=30)
        self.assertEqual(cfg.mild_threshold, 5)
        self.assertEqual(cfg.critical_threshold, 30)


# ---------------------------------------------------------------------------
# StagnationEngine -- construction
# ---------------------------------------------------------------------------


class TestStagnationEngineInit(unittest.TestCase):
    """Tests for StagnationEngine initialization."""

    def test_default_config(self):
        engine = StagnationEngine()
        self.assertEqual(engine.mild_threshold, 3)
        self.assertAlmostEqual(engine.score_tolerance, 0.001)

    def test_custom_config(self):
        cfg = StagnationConfig(mild_threshold=5, score_tolerance=0.01)
        engine = StagnationEngine(config=cfg)
        self.assertEqual(engine.mild_threshold, 5)
        self.assertAlmostEqual(engine.score_tolerance, 0.01)


# ---------------------------------------------------------------------------
# StagnationEngine.analyze -- level classification
# ---------------------------------------------------------------------------


class TestStagnationEngineAnalyzeLevel(unittest.TestCase):
    """Tests for StagnationEngine.analyze() with various score histories."""

    def setUp(self):
        self.engine = StagnationEngine()

    # -- NONE level --

    def test_empty_history_returns_none_level(self):
        report = self.engine.analyze([])
        self.assertEqual(report.level, StagnationLevel.NONE)
        self.assertEqual(report.iterations_stagnant, 0)
        self.assertAlmostEqual(report.best_score, 0.0)
        self.assertEqual(report.score_history, [])

    def test_single_score_returns_none_level(self):
        report = self.engine.analyze([0.5])
        self.assertEqual(report.level, StagnationLevel.NONE)
        self.assertEqual(report.iterations_stagnant, 0)

    def test_two_improving_scores(self):
        report = self.engine.analyze([0.3, 0.5])
        self.assertEqual(report.level, StagnationLevel.NONE)
        self.assertEqual(report.iterations_stagnant, 0)

    def test_steadily_improving_scores(self):
        history = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        report = self.engine.analyze(history)
        self.assertEqual(report.level, StagnationLevel.NONE)
        self.assertEqual(report.iterations_stagnant, 0)
        self.assertAlmostEqual(report.best_score, 0.9)

    def test_best_at_end_is_none(self):
        """Last score is the best -- zero stagnant iterations."""
        history = [0.5, 0.4, 0.3, 0.2, 0.8]
        report = self.engine.analyze(history)
        self.assertEqual(report.level, StagnationLevel.NONE)
        self.assertEqual(report.iterations_stagnant, 0)

    def test_two_stagnant_still_none(self):
        """Only 2 iterations stagnant (below mild_threshold=3)."""
        history = [0.5, 0.8, 0.7, 0.6]
        report = self.engine.analyze(history)
        # best=0.8 at index 1, stagnant = 3-1 = 2
        self.assertEqual(report.level, StagnationLevel.NONE)
        self.assertEqual(report.iterations_stagnant, 2)

    # -- MILD level (3-5 stagnant) --

    def test_exactly_three_stagnant_is_mild(self):
        # best at index 0, then 3 non-improving iterations
        history = [0.9, 0.5, 0.6, 0.7]
        report = self.engine.analyze(history)
        self.assertEqual(report.iterations_stagnant, 3)
        self.assertEqual(report.level, StagnationLevel.MILD)

    def test_five_stagnant_is_mild(self):
        history = [0.9, 0.5, 0.6, 0.5, 0.4, 0.3]
        report = self.engine.analyze(history)
        self.assertEqual(report.iterations_stagnant, 5)
        self.assertEqual(report.level, StagnationLevel.MILD)

    def test_plateau_three_same_scores_is_mild(self):
        """Plateau: best achieved once then 3 lower scores."""
        history = [0.3, 0.5, 0.8, 0.6, 0.7, 0.6]
        report = self.engine.analyze(history)
        # best=0.8 at index 2, stagnant = 5-2 = 3
        self.assertEqual(report.iterations_stagnant, 3)
        self.assertEqual(report.level, StagnationLevel.MILD)

    # -- MODERATE level (6-10 stagnant) --

    def test_six_stagnant_is_moderate(self):
        history = [0.9] + [0.5] * 6
        report = self.engine.analyze(history)
        self.assertEqual(report.iterations_stagnant, 6)
        self.assertEqual(report.level, StagnationLevel.MODERATE)

    def test_ten_stagnant_is_moderate(self):
        history = [0.9] + [0.5] * 10
        report = self.engine.analyze(history)
        self.assertEqual(report.iterations_stagnant, 10)
        self.assertEqual(report.level, StagnationLevel.MODERATE)

    def test_moderate_with_varied_but_non_improving_scores(self):
        history = [0.9, 0.4, 0.5, 0.6, 0.7, 0.5, 0.3, 0.6]
        report = self.engine.analyze(history)
        # best=0.9 at index 0, stagnant = 7-0 = 7
        self.assertEqual(report.iterations_stagnant, 7)
        self.assertEqual(report.level, StagnationLevel.MODERATE)

    # -- SEVERE level (11-20 stagnant) --

    def test_eleven_stagnant_is_severe(self):
        history = [0.9] + [0.5] * 11
        report = self.engine.analyze(history)
        self.assertEqual(report.iterations_stagnant, 11)
        self.assertEqual(report.level, StagnationLevel.SEVERE)

    def test_twenty_stagnant_is_severe(self):
        """Exactly 20 should still be CRITICAL since threshold is 20."""
        history = [0.9] + [0.5] * 20
        report = self.engine.analyze(history)
        self.assertEqual(report.iterations_stagnant, 20)
        self.assertEqual(report.level, StagnationLevel.CRITICAL)

    def test_nineteen_stagnant_is_severe(self):
        history = [0.9] + [0.5] * 19
        report = self.engine.analyze(history)
        self.assertEqual(report.iterations_stagnant, 19)
        self.assertEqual(report.level, StagnationLevel.SEVERE)

    # -- CRITICAL level (20+ stagnant) --

    def test_twenty_stagnant_is_critical(self):
        history = [0.9] + [0.5] * 20
        report = self.engine.analyze(history)
        self.assertEqual(report.iterations_stagnant, 20)
        self.assertEqual(report.level, StagnationLevel.CRITICAL)

    def test_thirty_stagnant_is_critical(self):
        history = [0.9] + [0.5] * 30
        report = self.engine.analyze(history)
        self.assertEqual(report.iterations_stagnant, 30)
        self.assertEqual(report.level, StagnationLevel.CRITICAL)

    # -- Complex patterns --

    def test_improved_then_plateaued(self):
        """Score improves for a while, then hits a plateau."""
        history = [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        report = self.engine.analyze(history)
        # best=0.9 at index 4, stagnant = 11-4 = 7
        self.assertEqual(report.iterations_stagnant, 7)
        self.assertEqual(report.level, StagnationLevel.MODERATE)

    def test_multiple_plateaus_latest_counts(self):
        """Best is re-achieved partway through -- only count from last best."""
        history = [0.5, 0.9, 0.5, 0.5, 0.9, 0.5, 0.5]
        report = self.engine.analyze(history)
        # best=0.9 last at index 4, stagnant = 6-4 = 2
        self.assertEqual(report.iterations_stagnant, 2)
        self.assertEqual(report.level, StagnationLevel.NONE)

    def test_within_tolerance_counts_as_best(self):
        """Scores within tolerance of best should reset stagnation."""
        engine = StagnationEngine(config=StagnationConfig(score_tolerance=0.01))
        history = [0.9, 0.5, 0.5, 0.5, 0.895]  # 0.895 is within 0.01 of 0.9
        report = engine.analyze(history)
        # 0.895 is within tolerance of 0.9, so last_best_idx = 4, stagnant = 0
        self.assertEqual(report.iterations_stagnant, 0)
        self.assertEqual(report.level, StagnationLevel.NONE)

    def test_all_identical_scores(self):
        """All scores the same -- best is at the last one within tolerance."""
        history = [0.5, 0.5, 0.5, 0.5, 0.5]
        report = self.engine.analyze(history)
        # best=0.5, last occurrence at index 4, stagnant=0
        self.assertEqual(report.iterations_stagnant, 0)
        self.assertEqual(report.level, StagnationLevel.NONE)

    def test_all_zero_scores(self):
        history = [0.0, 0.0, 0.0, 0.0]
        report = self.engine.analyze(history)
        # All 0.0 within tolerance, last at index 3, stagnant = 0
        self.assertEqual(report.iterations_stagnant, 0)
        self.assertEqual(report.level, StagnationLevel.NONE)

    def test_decreasing_scores(self):
        """Scores only decrease: best is at the start."""
        history = [1.0, 0.9, 0.8, 0.7, 0.6]
        report = self.engine.analyze(history)
        # best=1.0 at index 0, stagnant = 4-0 = 4
        self.assertEqual(report.iterations_stagnant, 4)
        self.assertEqual(report.level, StagnationLevel.MILD)

    def test_negative_scores(self):
        """Negative scores should work correctly."""
        history = [-0.5, -0.3, -0.1, -0.4, -0.5, -0.6]
        report = self.engine.analyze(history)
        # best=-0.1 at index 2, stagnant = 5-2 = 3
        self.assertEqual(report.iterations_stagnant, 3)
        self.assertEqual(report.level, StagnationLevel.MILD)
        self.assertAlmostEqual(report.best_score, -0.1)


# ---------------------------------------------------------------------------
# StagnationEngine -- exploration boost
# ---------------------------------------------------------------------------


class TestExplorationBoost(unittest.TestCase):
    """Tests that exploration boost values match stagnation levels."""

    def setUp(self):
        self.engine = StagnationEngine()

    def test_none_boost_is_zero(self):
        report = self.engine.analyze([0.5, 0.6, 0.7])
        self.assertAlmostEqual(report.exploration_ratio_boost, 0.0)

    def test_mild_boost(self):
        history = [0.9, 0.5, 0.6, 0.7]  # 3 stagnant
        report = self.engine.analyze(history)
        self.assertAlmostEqual(report.exploration_ratio_boost, 0.1)

    def test_moderate_boost(self):
        history = [0.9] + [0.5] * 7  # 7 stagnant
        report = self.engine.analyze(history)
        self.assertAlmostEqual(report.exploration_ratio_boost, 0.2)

    def test_severe_boost(self):
        history = [0.9] + [0.5] * 15  # 15 stagnant
        report = self.engine.analyze(history)
        self.assertAlmostEqual(report.exploration_ratio_boost, 0.3)

    def test_critical_boost(self):
        history = [0.9] + [0.5] * 25  # 25 stagnant
        report = self.engine.analyze(history)
        self.assertAlmostEqual(report.exploration_ratio_boost, 0.5)

    def test_custom_boost_values(self):
        cfg = StagnationConfig(
            exploration_boost_mild=0.15,
            exploration_boost_critical=0.45,
        )
        engine = StagnationEngine(config=cfg)
        report = engine.analyze([0.9, 0.5, 0.5, 0.5])  # 3 stagnant -> mild
        self.assertAlmostEqual(report.exploration_ratio_boost, 0.15)


# ---------------------------------------------------------------------------
# StagnationEngine -- strategy suggestions
# ---------------------------------------------------------------------------


class TestStrategySuggestion(unittest.TestCase):
    """Tests that strategy suggestions match stagnation levels."""

    def setUp(self):
        self.engine = StagnationEngine()

    def test_none_strategy(self):
        report = self.engine.analyze([0.5, 0.6, 0.7])
        self.assertEqual(report.suggested_strategy, "standard")

    def test_mild_strategy(self):
        report = self.engine.analyze([0.9, 0.5, 0.5, 0.5])
        self.assertEqual(report.suggested_strategy, "diversify")

    def test_moderate_strategy(self):
        report = self.engine.analyze([0.9] + [0.5] * 8)
        self.assertEqual(report.suggested_strategy, "paradigm_shift")

    def test_severe_strategy(self):
        report = self.engine.analyze([0.9] + [0.5] * 15)
        self.assertEqual(report.suggested_strategy, "radical_departure")

    def test_critical_strategy(self):
        report = self.engine.analyze([0.9] + [0.5] * 25)
        self.assertEqual(report.suggested_strategy, "full_restart")


# ---------------------------------------------------------------------------
# StagnationEngine -- failed approaches
# ---------------------------------------------------------------------------


class TestFailedApproaches(unittest.TestCase):
    """Tests that failed approaches are passed through in the report."""

    def setUp(self):
        self.engine = StagnationEngine()

    def test_no_failed_approaches(self):
        report = self.engine.analyze([0.5])
        self.assertEqual(report.failed_approaches, [])

    def test_failed_approaches_included(self):
        failed = ["brute force", "greedy algorithm"]
        report = self.engine.analyze([0.9, 0.5, 0.5, 0.5], failed_approaches=failed)
        self.assertEqual(report.failed_approaches, failed)

    def test_failed_approaches_none_becomes_empty(self):
        report = self.engine.analyze([0.5], failed_approaches=None)
        self.assertEqual(report.failed_approaches, [])

    def test_failed_approaches_not_mutated(self):
        """The engine should not mutate the caller's list."""
        failed = ["approach_a"]
        report = self.engine.analyze([0.9, 0.5, 0.5, 0.5], failed_approaches=failed)
        # Modify the report's list
        report.failed_approaches.append("approach_b")
        # Original should be unchanged
        self.assertEqual(failed, ["approach_a"])


# ---------------------------------------------------------------------------
# StagnationEngine -- recommendations
# ---------------------------------------------------------------------------


class TestRecommendations(unittest.TestCase):
    """Tests that recommendations differ by level and incorporate failed approaches."""

    def setUp(self):
        self.engine = StagnationEngine()

    def test_none_has_recommendations(self):
        report = self.engine.analyze([0.5, 0.6])
        self.assertGreater(len(report.recommendations), 0)

    def test_mild_has_recommendations(self):
        report = self.engine.analyze([0.9, 0.5, 0.5, 0.5])
        self.assertGreater(len(report.recommendations), 0)

    def test_critical_has_more_recommendations_than_none(self):
        report_none = self.engine.analyze([0.5, 0.6])
        report_critical = self.engine.analyze([0.9] + [0.5] * 25)
        self.assertGreater(
            len(report_critical.recommendations),
            len(report_none.recommendations),
        )

    def test_recommendations_mention_failed_approaches_few(self):
        """When there are few failed approaches, they are listed by name."""
        failed = ["brute force", "DP"]
        report = self.engine.analyze(
            [0.9, 0.5, 0.5, 0.5], failed_approaches=failed
        )
        # At least one recommendation should mention the failed approaches
        combined = " ".join(report.recommendations)
        self.assertIn("brute force", combined)
        self.assertIn("DP", combined)

    def test_recommendations_mention_failed_approaches_many(self):
        """When there are many failed approaches, a count is used instead."""
        failed = ["a", "b", "c", "d", "e"]
        report = self.engine.analyze(
            [0.9, 0.5, 0.5, 0.5], failed_approaches=failed
        )
        combined = " ".join(report.recommendations)
        self.assertIn("5", combined)

    def test_severe_with_failed_approaches_mentions_analysis(self):
        """Severe/critical levels should recommend analyzing failed patterns."""
        failed = ["approach_a"]
        report = self.engine.analyze(
            [0.9] + [0.5] * 15, failed_approaches=failed
        )
        combined = " ".join(report.recommendations).lower()
        self.assertIn("pattern", combined)

    def test_different_levels_have_different_recommendations(self):
        """Each level should produce distinct recommendation sets."""
        report_none = self.engine.analyze([0.5, 0.6])
        report_mild = self.engine.analyze([0.9, 0.5, 0.5, 0.5])
        report_moderate = self.engine.analyze([0.9] + [0.5] * 8)
        report_severe = self.engine.analyze([0.9] + [0.5] * 15)
        report_critical = self.engine.analyze([0.9] + [0.5] * 25)

        # Each should have at least one unique recommendation
        sets = [
            set(report_none.recommendations),
            set(report_mild.recommendations),
            set(report_moderate.recommendations),
            set(report_severe.recommendations),
            set(report_critical.recommendations),
        ]
        # No two sets should be identical
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertNotEqual(sets[i], sets[j])


# ---------------------------------------------------------------------------
# StagnationEngine -- diagnosis text
# ---------------------------------------------------------------------------


class TestDiagnosis(unittest.TestCase):
    """Tests for human-readable diagnosis text."""

    def setUp(self):
        self.engine = StagnationEngine()

    def test_none_diagnosis_mentions_progressing(self):
        report = self.engine.analyze([0.5, 0.6])
        self.assertIn("progressing", report.diagnosis.lower())

    def test_mild_diagnosis_mentions_mild(self):
        report = self.engine.analyze([0.9, 0.5, 0.5, 0.5])
        self.assertIn("mild", report.diagnosis.lower())

    def test_moderate_diagnosis_mentions_moderate(self):
        report = self.engine.analyze([0.9] + [0.5] * 8)
        self.assertIn("moderate", report.diagnosis.lower())

    def test_severe_diagnosis_mentions_severe(self):
        report = self.engine.analyze([0.9] + [0.5] * 15)
        self.assertIn("severe", report.diagnosis.lower())

    def test_critical_diagnosis_mentions_critical(self):
        report = self.engine.analyze([0.9] + [0.5] * 25)
        self.assertIn("critical", report.diagnosis.lower())

    def test_diagnosis_includes_best_score(self):
        report = self.engine.analyze([0.9, 0.5, 0.5, 0.5])
        self.assertIn("0.9", report.diagnosis)

    def test_diagnosis_includes_iteration_count(self):
        report = self.engine.analyze([0.9] + [0.5] * 15)
        self.assertIn("15", report.diagnosis)

    def test_empty_history_diagnosis(self):
        report = self.engine.analyze([])
        # Should not crash, should mention progressing or similar
        self.assertIn("progressing", report.diagnosis.lower())


# ---------------------------------------------------------------------------
# StagnationEngine -- custom thresholds
# ---------------------------------------------------------------------------


class TestCustomThresholds(unittest.TestCase):
    """Tests that custom thresholds change level classification."""

    def test_raised_mild_threshold(self):
        cfg = StagnationConfig(mild_threshold=5)
        engine = StagnationEngine(config=cfg)
        # 3 stagnant would normally be MILD, but now threshold is 5
        report = engine.analyze([0.9, 0.5, 0.5, 0.5])
        self.assertEqual(report.level, StagnationLevel.NONE)

    def test_lowered_mild_threshold(self):
        cfg = StagnationConfig(mild_threshold=1)
        engine = StagnationEngine(config=cfg)
        report = engine.analyze([0.9, 0.5])
        self.assertEqual(report.level, StagnationLevel.MILD)

    def test_large_tolerance_collapses_stagnation(self):
        """With a very large tolerance, small differences are ignored."""
        cfg = StagnationConfig(score_tolerance=0.5)
        engine = StagnationEngine(config=cfg)
        # All scores within 0.5 of best=0.9, so last "best" is at the end
        history = [0.9, 0.8, 0.7, 0.6, 0.5]
        report = engine.analyze(history)
        self.assertEqual(report.iterations_stagnant, 0)
        self.assertEqual(report.level, StagnationLevel.NONE)

    def test_zero_tolerance_strict_matching(self):
        """With zero tolerance, only exact matches count."""
        cfg = StagnationConfig(score_tolerance=0.0)
        engine = StagnationEngine(config=cfg)
        history = [0.9, 0.8999999]  # Very close but not equal
        report = engine.analyze(history)
        # best=0.9 at index 0, 0.8999999 != 0.9, stagnant=1
        self.assertEqual(report.iterations_stagnant, 1)


# ---------------------------------------------------------------------------
# StagnationEngine -- score_history preserved in report
# ---------------------------------------------------------------------------


class TestScoreHistoryInReport(unittest.TestCase):
    """Tests that the report preserves the input score history."""

    def setUp(self):
        self.engine = StagnationEngine()

    def test_score_history_preserved(self):
        history = [0.1, 0.3, 0.5, 0.7]
        report = self.engine.analyze(history)
        self.assertEqual(report.score_history, history)

    def test_score_history_is_copy(self):
        """Modifying the report's score_history should not affect the original."""
        history = [0.1, 0.3, 0.5]
        report = self.engine.analyze(history)
        report.score_history.append(0.9)
        self.assertEqual(len(history), 3)

    def test_empty_history_preserved(self):
        report = self.engine.analyze([])
        self.assertEqual(report.score_history, [])


# ---------------------------------------------------------------------------
# StagnationEngine -- internal method tests
# ---------------------------------------------------------------------------


class TestInternalMethods(unittest.TestCase):
    """Direct tests for internal helper methods."""

    def setUp(self):
        self.engine = StagnationEngine()

    def test_count_stagnant_iterations_empty(self):
        self.assertEqual(self.engine._count_stagnant_iterations([]), 0)

    def test_count_stagnant_iterations_single(self):
        self.assertEqual(self.engine._count_stagnant_iterations([0.5]), 0)

    def test_count_stagnant_iterations_improving(self):
        self.assertEqual(
            self.engine._count_stagnant_iterations([0.1, 0.2, 0.3]), 0
        )

    def test_count_stagnant_iterations_plateau(self):
        self.assertEqual(
            self.engine._count_stagnant_iterations([0.9, 0.5, 0.5, 0.5]), 3
        )

    def test_classify_level_boundaries(self):
        self.assertEqual(self.engine._classify_level(0), StagnationLevel.NONE)
        self.assertEqual(self.engine._classify_level(2), StagnationLevel.NONE)
        self.assertEqual(self.engine._classify_level(3), StagnationLevel.MILD)
        self.assertEqual(self.engine._classify_level(5), StagnationLevel.MILD)
        self.assertEqual(self.engine._classify_level(6), StagnationLevel.MODERATE)
        self.assertEqual(self.engine._classify_level(10), StagnationLevel.MODERATE)
        self.assertEqual(self.engine._classify_level(11), StagnationLevel.SEVERE)
        self.assertEqual(self.engine._classify_level(19), StagnationLevel.SEVERE)
        self.assertEqual(self.engine._classify_level(20), StagnationLevel.CRITICAL)
        self.assertEqual(self.engine._classify_level(100), StagnationLevel.CRITICAL)

    def test_get_exploration_boost_all_levels(self):
        self.assertAlmostEqual(
            self.engine._get_exploration_boost(StagnationLevel.NONE), 0.0
        )
        self.assertAlmostEqual(
            self.engine._get_exploration_boost(StagnationLevel.MILD), 0.1
        )
        self.assertAlmostEqual(
            self.engine._get_exploration_boost(StagnationLevel.MODERATE), 0.2
        )
        self.assertAlmostEqual(
            self.engine._get_exploration_boost(StagnationLevel.SEVERE), 0.3
        )
        self.assertAlmostEqual(
            self.engine._get_exploration_boost(StagnationLevel.CRITICAL), 0.5
        )

    def test_suggest_strategy_all_levels(self):
        self.assertEqual(
            self.engine._suggest_strategy(StagnationLevel.NONE), "standard"
        )
        self.assertEqual(
            self.engine._suggest_strategy(StagnationLevel.MILD), "diversify"
        )
        self.assertEqual(
            self.engine._suggest_strategy(StagnationLevel.MODERATE),
            "paradigm_shift",
        )
        self.assertEqual(
            self.engine._suggest_strategy(StagnationLevel.SEVERE),
            "radical_departure",
        )
        self.assertEqual(
            self.engine._suggest_strategy(StagnationLevel.CRITICAL),
            "full_restart",
        )


if __name__ == "__main__":
    unittest.main()
