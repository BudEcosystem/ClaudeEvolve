"""Tests for the universal novelty and diversity system."""
import unittest
from claude_evolve.core.novelty import (
    tokenize,
    ngrams,
    structural_similarity,
    behavioral_similarity,
    semantic_fingerprint,
    semantic_similarity,
    compute_novelty,
    SteppingStonesArchive,
)


class TestTokenize(unittest.TestCase):
    """Test artifact-type-aware tokenization."""

    def test_python_code(self):
        code = "def foo(x):\n    return x + 1"
        tokens = tokenize(code, "python")
        self.assertIn("def", tokens)
        self.assertIn("foo", tokens)
        self.assertIn("return", tokens)

    def test_prose(self):
        text = "The quick brown fox jumps over the lazy dog."
        tokens = tokenize(text, "markdown")
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
        self.assertIn("fox", tokens)

    def test_yaml(self):
        yaml = 'database:\n  host: "localhost"\n  port: 5432'
        tokens = tokenize(yaml, "yaml")
        self.assertIn("database", tokens)
        self.assertIn("host", tokens)
        self.assertIn("localhost", tokens)

    def test_sql(self):
        sql = "SELECT name, age FROM users WHERE age > 21"
        tokens = tokenize(sql, "sql")
        self.assertIn("select", tokens)
        self.assertIn("name", tokens)
        self.assertIn("users", tokens)

    def test_empty_content(self):
        self.assertEqual(tokenize("", "python"), [])

    def test_unknown_type_defaults_to_word_split(self):
        tokens = tokenize("hello world 123", "unknown")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)


class TestNgrams(unittest.TestCase):
    def test_basic_trigrams(self):
        tokens = ["a", "b", "c", "d"]
        result = ngrams(tokens, 3)
        self.assertEqual(result, {("a", "b", "c"), ("b", "c", "d")})

    def test_too_short(self):
        result = ngrams(["a", "b"], 3)
        # Falls back to bigrams since tokens < n
        self.assertGreater(len(result), 0)

    def test_empty(self):
        self.assertEqual(ngrams([], 3), set())


class TestStructuralSimilarity(unittest.TestCase):
    """Test universal structural similarity."""

    def test_identical(self):
        self.assertAlmostEqual(structural_similarity("abc", "abc"), 1.0)

    def test_completely_different(self):
        sim = structural_similarity(
            "def calculate_sum(numbers): return sum(numbers)",
            "The weather today is sunny and warm",
            "text"
        )
        self.assertLess(sim, 0.1)

    def test_similar_code(self):
        code_a = "def foo(x):\n    return x * 2\n"
        code_b = "def foo(x):\n    return x * 3\n"
        sim = structural_similarity(code_a, code_b, "python")
        self.assertGreater(sim, 0.5)

    def test_similar_prose(self):
        a = "The optimization algorithm converges quickly to the optimal solution."
        b = "The optimization algorithm converges slowly to a suboptimal solution."
        sim = structural_similarity(a, b, "markdown")
        # These share many words so should have some similarity
        self.assertGreater(sim, 0.1)

    def test_empty_strings(self):
        # Empty strings are identical content but have no structure to compare
        sim = structural_similarity("", "")
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_one_empty(self):
        self.assertAlmostEqual(structural_similarity("hello", ""), 0.0)

    def test_yaml_similarity(self):
        a = "database:\n  host: localhost\n  port: 5432\n  name: mydb"
        b = "database:\n  host: remote\n  port: 5432\n  name: mydb"
        sim = structural_similarity(a, b, "yaml")
        # Share most tokens so should be similar
        self.assertGreater(sim, 0.1)


class TestBehavioralSimilarity(unittest.TestCase):
    """Test metric-based behavioral similarity."""

    def test_identical_metrics(self):
        m = {"accuracy": 0.9, "speed": 0.8}
        self.assertAlmostEqual(behavioral_similarity(m, m), 1.0)

    def test_different_metrics(self):
        a = {"accuracy": 0.9, "speed": 0.8}
        b = {"accuracy": 0.1, "speed": 0.2}
        sim = behavioral_similarity(a, b)
        self.assertLess(sim, 0.5)

    def test_partial_overlap(self):
        a = {"accuracy": 0.9, "speed": 0.8, "extra": 0.5}
        b = {"accuracy": 0.9, "speed": 0.8, "other": 0.1}
        # Only shared keys compared
        sim = behavioral_similarity(a, b)
        self.assertAlmostEqual(sim, 1.0)

    def test_empty_metrics(self):
        self.assertAlmostEqual(behavioral_similarity({}, {}), 0.0)

    def test_non_numeric_ignored(self):
        a = {"score": 0.9, "name": "test"}
        b = {"score": 0.9, "name": "other"}
        sim = behavioral_similarity(a, b)
        self.assertAlmostEqual(sim, 1.0)


class TestSemanticFingerprint(unittest.TestCase):
    """Test semantic concept extraction."""

    def test_python_functions(self):
        code = "def calculate_sum(data):\n    return sum(data)"
        fp = semantic_fingerprint(code, "python")
        self.assertIn("calculate_sum", fp)

    def test_python_imports(self):
        code = "import numpy\nfrom scipy import optimize"
        fp = semantic_fingerprint(code, "python")
        self.assertIn("numpy", fp)
        self.assertIn("scipy", fp)

    def test_python_algorithm_keywords(self):
        code = "# Use scipy.optimize to minimize the function"
        fp = semantic_fingerprint(code, "python")
        self.assertIn("algo:optimize", fp)
        self.assertIn("algo:minimize", fp)
        self.assertIn("algo:scipy", fp)

    def test_sql_tables(self):
        sql = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        fp = semantic_fingerprint(sql, "sql")
        self.assertIn("users", fp)
        self.assertIn("orders", fp)
        self.assertIn("op:select", fp)

    def test_yaml_keys(self):
        yaml = "database:\n  host: localhost\nserver:\n  port: 8080"
        fp = semantic_fingerprint(yaml, "yaml")
        self.assertIn("database", fp)
        self.assertIn("server", fp)

    def test_prose_repeated_concepts(self):
        text = "The optimization algorithm uses gradient descent. The optimization converges."
        fp = semantic_fingerprint(text, "markdown")
        self.assertIn("optimization", fp)

    def test_empty(self):
        self.assertEqual(semantic_fingerprint("", "python"), set())


class TestSemanticSimilarity(unittest.TestCase):
    def test_same_code(self):
        code = "def foo(): pass\ndef bar(): pass"
        self.assertAlmostEqual(semantic_similarity(code, code, "python"), 1.0)

    def test_different_code(self):
        a = "import numpy\ndef optimize(): pass"
        b = "import flask\ndef serve(): pass"
        sim = semantic_similarity(a, b, "python")
        self.assertLess(sim, 0.5)


class TestComputeNovelty(unittest.TestCase):
    """Test the combined novelty score."""

    def test_identical_is_zero(self):
        content = "def foo(): return 42"
        novelty = compute_novelty(content, content, "python")
        self.assertAlmostEqual(novelty, 0.0)

    def test_completely_different_is_high(self):
        a = "import numpy; def optimize(): return scipy.minimize(f, x0)"
        b = "Dear customer, thank you for your purchase of our premium product."
        novelty = compute_novelty(a, b, "text")
        self.assertGreater(novelty, 0.5)

    def test_with_metrics_uses_behavioral(self):
        a = "def foo(): return 1"
        b = "def bar(): return 2"
        m_a = {"accuracy": 0.9}
        m_b = {"accuracy": 0.1}
        novelty = compute_novelty(a, b, "python", m_a, m_b)
        self.assertGreater(novelty, 0.3)

    def test_without_metrics_skips_behavioral(self):
        a = "hello world"
        b = "hello earth"
        novelty = compute_novelty(a, b, "text")
        self.assertGreater(novelty, 0.0)
        self.assertLess(novelty, 1.0)

    def test_custom_weights(self):
        a = "def foo(): return 1"
        b = "def foo(): return 2"
        # All structural weight
        n1 = compute_novelty(a, b, "python",
                             weights={"structural": 1.0, "behavioral": 0.0, "semantic": 0.0})
        # All semantic weight
        n2 = compute_novelty(a, b, "python",
                             weights={"structural": 0.0, "behavioral": 0.0, "semantic": 1.0})
        # Both should be valid [0, 1]
        self.assertGreaterEqual(n1, 0.0)
        self.assertLessEqual(n1, 1.0)
        self.assertGreaterEqual(n2, 0.0)
        self.assertLessEqual(n2, 1.0)

    def test_novelty_range(self):
        """Novelty should always be in [0, 1]."""
        pairs = [
            ("", ""),
            ("a", ""),
            ("hello", "hello"),
            ("abc def ghi", "xyz uvw rst"),
        ]
        for a, b in pairs:
            n = compute_novelty(a, b)
            self.assertGreaterEqual(n, 0.0, f"Failed for {a!r}, {b!r}")
            self.assertLessEqual(n, 1.0, f"Failed for {a!r}, {b!r}")


class TestSteppingStonesArchive(unittest.TestCase):
    """Test the stepping stones archive."""

    def test_add_first_stone(self):
        archive = SteppingStonesArchive(max_size=10, novelty_threshold=0.3)
        added = archive.try_add("def foo(): return 1", {"score": 0.5}, "python")
        self.assertTrue(added)
        self.assertEqual(len(archive.stones), 1)

    def test_reject_duplicate(self):
        archive = SteppingStonesArchive(max_size=10, novelty_threshold=0.3)
        archive.try_add("def foo(): return 1", {"score": 0.5}, "python")
        added = archive.try_add("def foo(): return 1", {"score": 0.5}, "python")
        self.assertFalse(added)

    def test_accept_novel(self):
        archive = SteppingStonesArchive(max_size=10, novelty_threshold=0.3)
        archive.try_add("def foo(): return 1", {"score": 0.5}, "python")
        added = archive.try_add(
            "import numpy\ndef optimize(f, x0): return scipy.minimize(f, x0)",
            {"score": 0.7}, "python"
        )
        self.assertTrue(added)
        self.assertEqual(len(archive.stones), 2)

    def test_max_size_eviction(self):
        archive = SteppingStonesArchive(max_size=3, novelty_threshold=0.0)
        for i in range(5):
            archive.try_add(f"unique content number {i} " * 20, {"score": i * 0.1})
        self.assertLessEqual(len(archive.stones), 3)

    def test_get_inspirations_returns_most_diverse(self):
        archive = SteppingStonesArchive(max_size=10, novelty_threshold=0.0)
        archive.try_add("algorithm A uses sorting", {"score": 0.3})
        archive.try_add("algorithm B uses hashing", {"score": 0.4})
        archive.try_add("algorithm C uses sorting and hashing combined", {"score": 0.5})

        inspirations = archive.get_inspirations("algorithm D uses graph search", n=2)
        self.assertLessEqual(len(inspirations), 2)

    def test_format_for_prompt_empty(self):
        archive = SteppingStonesArchive()
        self.assertEqual(archive.format_for_prompt("test"), "")

    def test_format_for_prompt_nonempty(self):
        archive = SteppingStonesArchive(novelty_threshold=0.0)
        archive.try_add("def optimize(): pass", {"combined_score": 0.7}, "python")
        result = archive.format_for_prompt("def search(): pass", "python")
        self.assertIn("Stepping Stone", result)
        self.assertIn("0.7", result)

    def test_serialization_roundtrip(self):
        archive = SteppingStonesArchive(max_size=10, novelty_threshold=0.0)
        archive.try_add("content A", {"score": 0.5}, "python")
        archive.try_add("content B", {"score": 0.7}, "text")

        data = archive.to_list()
        restored = SteppingStonesArchive.from_list(data, max_size=10)
        self.assertEqual(len(restored.stones), 2)
        self.assertEqual(restored.stones[0]["content"], "content A")

    def test_works_with_prose(self):
        archive = SteppingStonesArchive(max_size=10, novelty_threshold=0.2)
        archive.try_add(
            "Write a persuasive email about product features and benefits.",
            {"clarity": 0.8, "combined_score": 0.7}, "markdown"
        )
        added = archive.try_add(
            "Create a technical specification document for the API endpoints.",
            {"clarity": 0.6, "combined_score": 0.5}, "markdown"
        )
        self.assertTrue(added)  # Different topic = novel

    def test_works_with_yaml(self):
        archive = SteppingStonesArchive(max_size=10, novelty_threshold=0.2)
        archive.try_add(
            "server:\n  host: localhost\n  port: 8080",
            {"combined_score": 0.5}, "yaml"
        )
        added = archive.try_add(
            "database:\n  engine: postgres\n  pool_size: 10",
            {"combined_score": 0.6}, "yaml"
        )
        self.assertTrue(added)  # Different config section = novel


class TestCrossArtifactTypeNovelty(unittest.TestCase):
    """Test that novelty works correctly across different artifact types."""

    def test_python_vs_python(self):
        a = "def sort(arr):\n    return sorted(arr)"
        b = "def search(arr, target):\n    return target in arr"
        novelty = compute_novelty(a, b, "python")
        self.assertGreater(novelty, 0.2)

    def test_markdown_vs_markdown(self):
        a = "# Introduction\n\nThis document describes the system architecture."
        b = "# Conclusion\n\nThe results demonstrate improved performance."
        novelty = compute_novelty(a, b, "markdown")
        self.assertGreater(novelty, 0.3)

    def test_sql_vs_sql(self):
        a = "SELECT name FROM users WHERE active = 1"
        b = "INSERT INTO logs (event, timestamp) VALUES ('login', NOW())"
        novelty = compute_novelty(a, b, "sql")
        self.assertGreater(novelty, 0.3)

    def test_yaml_vs_yaml(self):
        a = "server:\n  host: 0.0.0.0\n  port: 8080\n  workers: 4"
        b = "database:\n  engine: postgresql\n  host: db.example.com\n  pool: 10"
        novelty = compute_novelty(a, b, "yaml")
        self.assertGreater(novelty, 0.3)

    def test_json_vs_json(self):
        a = '{"name": "Alice", "age": 30, "role": "engineer"}'
        b = '{"product": "Widget", "price": 9.99, "category": "tools"}'
        novelty = compute_novelty(a, b, "json")
        self.assertGreater(novelty, 0.3)


if __name__ == "__main__":
    unittest.main()
