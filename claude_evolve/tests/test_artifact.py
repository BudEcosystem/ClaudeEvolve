import json
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


if __name__ == "__main__":
    unittest.main()
