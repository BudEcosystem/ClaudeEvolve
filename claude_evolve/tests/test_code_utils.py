"""Tests for claude_evolve.utils.code_utils"""

import unittest

from claude_evolve.utils.code_utils import (
    apply_diff,
    apply_diff_blocks,
    calculate_edit_distance,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_evolve_blocks,
    parse_full_rewrite,
    split_diffs_by_target,
    _can_apply_linewise,
    _format_block_lines,
)


class TestExtractDiffs(unittest.TestCase):
    def test_single_block(self):
        diff_text = (
            "<<<<<<< SEARCH\nold line\n=======\nnew line\n>>>>>>> REPLACE"
        )
        result = extract_diffs(diff_text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("old line", "new line"))

    def test_multiple_blocks(self):
        diff_text = (
            "<<<<<<< SEARCH\nfirst old\n=======\nfirst new\n>>>>>>> REPLACE\n"
            "some text in between\n"
            "<<<<<<< SEARCH\nsecond old\n=======\nsecond new\n>>>>>>> REPLACE"
        )
        result = extract_diffs(diff_text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ("first old", "first new"))
        self.assertEqual(result[1], ("second old", "second new"))

    def test_no_diffs(self):
        result = extract_diffs("no diffs here")
        self.assertEqual(result, [])

    def test_multiline_search_replace(self):
        diff_text = (
            "<<<<<<< SEARCH\nline1\nline2\n=======\nrepl1\nrepl2\nrepl3\n>>>>>>> REPLACE"
        )
        result = extract_diffs(diff_text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "line1\nline2")
        self.assertEqual(result[0][1], "repl1\nrepl2\nrepl3")


class TestApplyDiff(unittest.TestCase):
    def test_single_diff(self):
        original = "line1\nline2\nline3"
        diff_text = "<<<<<<< SEARCH\nline2\n=======\nreplaced\n>>>>>>> REPLACE"
        result = apply_diff(original, diff_text)
        self.assertEqual(result, "line1\nreplaced\nline3")

    def test_no_match(self):
        original = "line1\nline2\nline3"
        diff_text = "<<<<<<< SEARCH\nnonexistent\n=======\nreplaced\n>>>>>>> REPLACE"
        result = apply_diff(original, diff_text)
        self.assertEqual(result, original)

    def test_no_diff_blocks(self):
        original = "line1\nline2"
        result = apply_diff(original, "no diffs")
        self.assertEqual(result, original)


class TestApplyDiffBlocks(unittest.TestCase):
    def test_multiple_blocks(self):
        original = "aaa\nbbb\nccc"
        diff_blocks = [("aaa", "AAA"), ("ccc", "CCC")]
        result, applied = apply_diff_blocks(original, diff_blocks)
        self.assertEqual(result, "AAA\nbbb\nCCC")
        self.assertEqual(applied, 2)

    def test_no_matching_blocks(self):
        original = "aaa\nbbb\nccc"
        diff_blocks = [("zzz", "ZZZ")]
        result, applied = apply_diff_blocks(original, diff_blocks)
        self.assertEqual(result, original)
        self.assertEqual(applied, 0)

    def test_empty_blocks(self):
        original = "hello\nworld"
        result, applied = apply_diff_blocks(original, [])
        self.assertEqual(result, original)
        self.assertEqual(applied, 0)


class TestParseFullRewrite(unittest.TestCase):
    def test_extract_python_code(self):
        response = "Here is the code:\n```python\ndef foo():\n    return 42\n```\nDone."
        result = parse_full_rewrite(response, language="python")
        self.assertEqual(result, "def foo():\n    return 42")

    def test_extract_generic_code_block(self):
        response = "Here:\n```\ngeneric code\n```"
        result = parse_full_rewrite(response, language="python")
        self.assertEqual(result, "generic code")

    def test_no_code_block(self):
        response = "Just plain text with no code blocks"
        result = parse_full_rewrite(response)
        self.assertEqual(result, response)

    def test_language_specific(self):
        response = "```rust\nfn main() {}\n```"
        result = parse_full_rewrite(response, language="rust")
        self.assertEqual(result, "fn main() {}")


class TestCalculateEditDistance(unittest.TestCase):
    def test_identical_strings(self):
        self.assertEqual(calculate_edit_distance("abc", "abc"), 0)

    def test_different_strings(self):
        self.assertEqual(calculate_edit_distance("kitten", "sitting"), 3)

    def test_empty_strings(self):
        self.assertEqual(calculate_edit_distance("", ""), 0)

    def test_one_empty(self):
        self.assertEqual(calculate_edit_distance("abc", ""), 3)
        self.assertEqual(calculate_edit_distance("", "abc"), 3)

    def test_single_char_diff(self):
        self.assertEqual(calculate_edit_distance("a", "b"), 1)


class TestExtractCodeLanguage(unittest.TestCase):
    def test_python(self):
        code = "import os\nprint('hello')"
        self.assertEqual(extract_code_language(code), "python")

    def test_python_def(self):
        code = "def foo():\n    pass"
        self.assertEqual(extract_code_language(code), "python")

    def test_rust(self):
        code = "fn main() {\n    println!(\"hello\");\n}"
        self.assertEqual(extract_code_language(code), "rust")

    def test_javascript(self):
        code = "const x = 5;\nconsole.log(x);"
        self.assertEqual(extract_code_language(code), "javascript")

    def test_unknown(self):
        code = "some random text"
        self.assertEqual(extract_code_language(code), "unknown")

    def test_cpp(self):
        code = "#include <stdio.h>\nint main() { return 0; }"
        self.assertEqual(extract_code_language(code), "cpp")


class TestFormatDiffSummary(unittest.TestCase):
    def test_basic_format_single_line(self):
        blocks = [("old", "new")]
        result = format_diff_summary(blocks)
        self.assertIn("Change 1", result)
        self.assertIn("old", result)
        self.assertIn("new", result)

    def test_multiline_format(self):
        blocks = [("line1\nline2", "repl1\nrepl2")]
        result = format_diff_summary(blocks)
        self.assertIn("Change 1", result)
        self.assertIn("Replace:", result)
        self.assertIn("with:", result)

    def test_empty_blocks(self):
        result = format_diff_summary([])
        self.assertEqual(result, "")

    def test_multiple_changes(self):
        blocks = [("a", "b"), ("c", "d")]
        result = format_diff_summary(blocks)
        self.assertIn("Change 1", result)
        self.assertIn("Change 2", result)


class TestParseEvolveBlocks(unittest.TestCase):
    def test_single_block(self):
        code = "before\n# EVOLVE-BLOCK-START\nevolving\n# EVOLVE-BLOCK-END\nafter"
        blocks = parse_evolve_blocks(code)
        self.assertEqual(len(blocks), 1)
        start, end, content = blocks[0]
        self.assertEqual(content, "evolving")

    def test_no_blocks(self):
        code = "just regular code\nno blocks here"
        blocks = parse_evolve_blocks(code)
        self.assertEqual(blocks, [])

    def test_multiple_blocks(self):
        code = (
            "# EVOLVE-BLOCK-START\nblock1\n# EVOLVE-BLOCK-END\n"
            "gap\n"
            "# EVOLVE-BLOCK-START\nblock2\n# EVOLVE-BLOCK-END"
        )
        blocks = parse_evolve_blocks(code)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0][2], "block1")
        self.assertEqual(blocks[1][2], "block2")


class TestSplitDiffsByTarget(unittest.TestCase):
    def test_routes_to_code(self):
        code_text = "aaa\nbbb\nccc"
        desc_text = "xxx\nyyy\nzzz"
        diff_blocks = [("bbb", "BBB")]
        code_b, desc_b, unmatched = split_diffs_by_target(
            diff_blocks, code_text=code_text, changes_description_text=desc_text
        )
        self.assertEqual(len(code_b), 1)
        self.assertEqual(len(desc_b), 0)
        self.assertEqual(len(unmatched), 0)

    def test_routes_to_desc(self):
        code_text = "aaa\nbbb\nccc"
        desc_text = "xxx\nyyy\nzzz"
        diff_blocks = [("yyy", "YYY")]
        code_b, desc_b, unmatched = split_diffs_by_target(
            diff_blocks, code_text=code_text, changes_description_text=desc_text
        )
        self.assertEqual(len(code_b), 0)
        self.assertEqual(len(desc_b), 1)

    def test_unmatched(self):
        diff_blocks = [("nomatch", "NM")]
        code_b, desc_b, unmatched = split_diffs_by_target(
            diff_blocks, code_text="aaa", changes_description_text="bbb"
        )
        self.assertEqual(len(unmatched), 1)

    def test_ambiguous_raises(self):
        # Same text in both targets
        same = "shared\nline"
        diff_blocks = [("shared", "NEW")]
        with self.assertRaises(ValueError):
            split_diffs_by_target(
                diff_blocks, code_text=same, changes_description_text=same
            )


class TestCanApplyLinewise(unittest.TestCase):
    def test_match(self):
        self.assertTrue(_can_apply_linewise(["a", "b", "c"], ["b"]))

    def test_no_match(self):
        self.assertFalse(_can_apply_linewise(["a", "b", "c"], ["z"]))

    def test_empty_needle(self):
        self.assertFalse(_can_apply_linewise(["a", "b"], []))


class TestFormatBlockLines(unittest.TestCase):
    def test_basic(self):
        result = _format_block_lines(["hello", "world"])
        self.assertIn("hello", result)
        self.assertIn("world", result)

    def test_empty(self):
        result = _format_block_lines([])
        self.assertIn("(empty)", result)

    def test_truncation(self):
        long_line = "x" * 200
        result = _format_block_lines([long_line], max_line_len=50)
        self.assertIn("...", result)
        # The result line should be capped
        for line in result.split("\n"):
            # 2 spaces indent + up to max_line_len
            self.assertLessEqual(len(line), 50 + 2)


if __name__ == "__main__":
    unittest.main()
