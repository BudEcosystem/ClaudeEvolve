import json
import re
import time
import unittest
from claude_evolve.core.artifact import Artifact


class TestArtifact(unittest.TestCase):
    def test_create_artifact_with_defaults(self):
        a = Artifact(id="test-1", content="print('hello')")
        self.assertEqual(a.id, "test-1")
        self.assertEqual(a.content, "print('hello')")
        self.assertEqual(a.artifact_type, "python")
        self.assertIsNone(a.parent_id)
        self.assertEqual(a.generation, 0)
        self.assertEqual(a.metrics, {})
        self.assertAlmostEqual(a.timestamp, time.time(), delta=2)

    def test_create_artifact_with_all_fields(self):
        a = Artifact(
            id="test-2",
            content="# prompt text",
            artifact_type="markdown",
            parent_id="test-1",
            generation=3,
            iteration_found=7,
            metrics={"score": 0.85, "clarity": 0.9},
            complexity=42.0,
            diversity=0.7,
            metadata={"island": 2, "changes": "improved clarity"},
            changes_description="Rewrote introduction paragraph",
            eval_artifacts={"stderr": "warning: unused var"},
        )
        self.assertEqual(a.artifact_type, "markdown")
        self.assertEqual(a.parent_id, "test-1")
        self.assertEqual(a.generation, 3)
        self.assertEqual(a.metrics["score"], 0.85)

    def test_to_dict_roundtrip(self):
        a = Artifact(
            id="rt-1",
            content="code here",
            artifact_type="python",
            metrics={"combined_score": 0.5},
        )
        d = a.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["id"], "rt-1")
        self.assertEqual(d["content"], "code here")
        b = Artifact.from_dict(d)
        self.assertEqual(b.id, a.id)
        self.assertEqual(b.content, a.content)
        self.assertEqual(b.metrics, a.metrics)

    def test_from_dict_ignores_unknown_fields(self):
        d = {"id": "u-1", "content": "x", "unknown_field": 42}
        a = Artifact.from_dict(d)
        self.assertEqual(a.id, "u-1")

    def test_from_dict_backward_compat_code_field(self):
        """OpenEvolve uses 'code' field; we accept both 'code' and 'content'"""
        d = {"id": "bc-1", "code": "print(1)", "language": "python"}
        a = Artifact.from_dict(d)
        self.assertEqual(a.content, "print(1)")
        self.assertEqual(a.artifact_type, "python")

    def test_json_serialization(self):
        a = Artifact(id="j-1", content="test", metrics={"s": 1.0})
        j = json.dumps(a.to_dict())
        d = json.loads(j)
        b = Artifact.from_dict(d)
        self.assertEqual(b.id, "j-1")


# ---------------------------------------------------------------------------
# Thought-Code Coevolution: rationale and offspring_count fields
# ---------------------------------------------------------------------------
class TestArtifactRationaleAndOffspring(unittest.TestCase):
    """Tests for the rationale and offspring_count fields added for
    thought-code coevolution (CG2) and power-law selection (CG5)."""

    def test_artifact_has_rationale_and_offspring(self):
        a = Artifact(
            id="r-1",
            content="test",
            artifact_type="python",
            rationale="Use SA because local optima",
            offspring_count=3,
        )
        self.assertEqual(a.rationale, "Use SA because local optima")
        self.assertEqual(a.offspring_count, 3)

    def test_artifact_default_rationale_is_none(self):
        a = Artifact(id="r-2", content="test", artifact_type="python")
        self.assertIsNone(a.rationale)
        self.assertEqual(a.offspring_count, 0)

    def test_artifact_roundtrip_with_rationale(self):
        a = Artifact(
            id="r-3",
            content="test",
            artifact_type="python",
            rationale="my reason",
            offspring_count=7,
        )
        d = a.to_dict()
        self.assertEqual(d["rationale"], "my reason")
        self.assertEqual(d["offspring_count"], 7)
        loaded = Artifact.from_dict(d)
        self.assertEqual(loaded.rationale, "my reason")
        self.assertEqual(loaded.offspring_count, 7)

    def test_from_dict_missing_rationale_uses_default(self):
        """from_dict with data lacking rationale/offspring_count should use defaults."""
        d = {"id": "r-4", "content": "x"}
        a = Artifact.from_dict(d)
        self.assertIsNone(a.rationale)
        self.assertEqual(a.offspring_count, 0)

    def test_json_roundtrip_with_rationale(self):
        a = Artifact(
            id="r-5",
            content="code",
            rationale="greedy heuristic",
            offspring_count=2,
        )
        j = json.dumps(a.to_dict())
        d = json.loads(j)
        loaded = Artifact.from_dict(d)
        self.assertEqual(loaded.rationale, "greedy heuristic")
        self.assertEqual(loaded.offspring_count, 2)

    def test_extract_rationale_from_content(self):
        """RATIONALE-START/END markers can be extracted from content."""
        content = (
            "RATIONALE-START\n"
            "Use simulated annealing with adaptive cooling.\n"
            "RATIONALE-END\n"
            "def solve(): pass"
        )
        match = re.search(
            r"RATIONALE-START\s*\n(.*?)\nRATIONALE-END", content, re.DOTALL
        )
        self.assertIsNotNone(match)
        self.assertIn("simulated annealing", match.group(1))

    def test_extract_rationale_multiline(self):
        """Multi-line rationale between markers can be extracted."""
        content = (
            "RATIONALE-START\n"
            "First, use greedy approach.\n"
            "Then refine with local search.\n"
            "RATIONALE-END\n"
            "def solve(): pass"
        )
        match = re.search(
            r"RATIONALE-START\s*\n(.*?)\nRATIONALE-END", content, re.DOTALL
        )
        self.assertIsNotNone(match)
        self.assertIn("greedy approach", match.group(1))
        self.assertIn("local search", match.group(1))


if __name__ == "__main__":
    unittest.main()
