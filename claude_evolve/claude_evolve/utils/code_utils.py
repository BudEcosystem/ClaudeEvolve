"""
Utilities for code parsing, diffing, and manipulation.

Extracted and adapted from OpenEvolve's code_utils module for the
claude_evolve artifact-evolution pipeline.
"""

import re
from typing import Dict, List, Optional, Tuple, Union

# Default diff pattern used throughout the system for SEARCH/REPLACE blocks.
DEFAULT_DIFF_PATTERN = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code.

    Scans for lines containing ``# EVOLVE-BLOCK-START`` and
    ``# EVOLVE-BLOCK-END`` markers and returns the content between them.

    Args:
        code: Source code with evolve block markers.

    Returns:
        List of tuples ``(start_line, end_line, block_content)`` where
        *start_line* / *end_line* are 0-based line indices of the markers
        and *block_content* is the text between them (newline-joined).
    """
    lines = code.split("\n")
    blocks: List[Tuple[int, int, str]] = []

    in_block = False
    start_line = -1
    block_content: List[str] = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def extract_diffs(
    diff_text: str,
    diff_pattern: str = DEFAULT_DIFF_PATTERN,
) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text.

    Args:
        diff_text: Diff in the SEARCH/REPLACE format.
        diff_pattern: Regex pattern for the SEARCH/REPLACE format.

    Returns:
        List of tuples ``(search_text, replace_text)`` with trailing
        whitespace stripped from each side.
    """
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def apply_diff(
    original_code: str,
    diff_text: str,
    diff_pattern: str = DEFAULT_DIFF_PATTERN,
) -> str:
    """
    Apply a diff to the original code.

    Each SEARCH block is matched line-by-line in *original_code* and replaced
    with the corresponding REPLACE block.

    Args:
        original_code: Original source code.
        diff_text: Diff in the SEARCH/REPLACE format.
        diff_pattern: Regex pattern for the SEARCH/REPLACE format.

    Returns:
        Modified code with all applicable diffs applied.
    """
    result_lines = original_code.split("\n")

    diff_blocks = extract_diffs(diff_text, diff_pattern)

    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response.

    Tries, in order:
      1. A fenced code block tagged with *language* (e.g. ``python``).
      2. Any fenced code block.
      3. The entire response as plain text.

    Args:
        llm_response: Response from the LLM.
        language: Programming language to look for.

    Returns:
        Extracted code, or the raw response if no code block is found.
    """
    code_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def _format_block_lines(
    lines: List[str], max_line_len: int = 100, max_lines: int = 30
) -> str:
    """Format a block of lines for diff summary display.

    Each line is indented by two spaces. Lines longer than *max_line_len* are
    truncated with ``...``.  If there are more than *max_lines*, only the first
    *max_lines* are shown with a summary of how many were omitted.
    """
    truncated: List[str] = []
    for line in lines[:max_lines]:
        s = line.rstrip()
        if len(s) > max_line_len:
            s = s[: max_line_len - 3] + "..."
        truncated.append("  " + s)
    if len(lines) > max_lines:
        truncated.append(f"  ... ({len(lines) - max_lines} more lines)")
    return "\n".join(truncated) if truncated else "  (empty)"


def format_diff_summary(
    diff_blocks: List[Tuple[str, str]],
    max_line_len: int = 100,
    max_lines: int = 30,
) -> str:
    """
    Create a human-readable summary of the diff.

    Single-line changes are rendered inline; multi-line changes show the full
    SEARCH and REPLACE blocks.

    Args:
        diff_blocks: List of ``(search_text, replace_text)`` tuples.
        max_line_len: Maximum characters per line before truncation.
        max_lines: Maximum lines per SEARCH/REPLACE block.

    Returns:
        Summary string.
    """
    summary: List[str] = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_block = _format_block_lines(search_lines, max_line_len, max_lines)
            replace_block = _format_block_lines(replace_lines, max_line_len, max_lines)
            summary.append(
                f"Change {i+1}: Replace:\n{search_block}\nwith:\n{replace_block}"
            )

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two strings.

    Uses classic dynamic-programming approach. Suitable for short-to-medium
    length strings; for very large inputs consider a more memory-efficient
    variant.

    Args:
        code1: First string.
        code2: Second string.

    Returns:
        Edit distance (number of single-character insertions, deletions, or
        substitutions needed to transform *code1* into *code2*).
    """
    if code1 == code2:
        return 0

    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


def extract_code_language(code: str) -> str:
    """
    Try to determine the programming language of a code snippet.

    Uses simple heuristics based on common language-specific keywords at the
    start of lines.

    Args:
        code: Code snippet.

    Returns:
        Detected language name or ``"unknown"``.
    """
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"


def _can_apply_linewise(haystack_lines: List[str], needle_lines: List[str]) -> bool:
    """Return True if *needle_lines* appear as a contiguous subsequence in *haystack_lines*."""
    if not needle_lines:
        return False

    for i in range(len(haystack_lines) - len(needle_lines) + 1):
        if haystack_lines[i : i + len(needle_lines)] == needle_lines:
            return True

    return False


def apply_diff_blocks(
    original_text: str, diff_blocks: List[Tuple[str, str]]
) -> Tuple[str, int]:
    """
    Apply diff blocks line-wise and return the new text with an applied count.

    Each ``(search_text, replace_text)`` pair is matched against the current
    state of the text (allowing earlier blocks to affect later matches).

    Args:
        original_text: The text to patch.
        diff_blocks: List of ``(search_text, replace_text)`` tuples.

    Returns:
        Tuple of ``(new_text, applied_count)`` where *applied_count* is how
        many blocks were successfully matched and applied.
    """
    lines = original_text.split("\n")
    applied = 0

    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        for i in range(len(lines) - len(search_lines) + 1):
            if lines[i : i + len(search_lines)] == search_lines:
                lines[i : i + len(search_lines)] = replace_lines
                applied += 1
                break

    return "\n".join(lines), applied


def split_diffs_by_target(
    diff_blocks: List[Tuple[str, str]],
    *,
    code_text: str,
    changes_description_text: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Route diff blocks to either code or changes_description based on exact
    line-wise match of the SEARCH text.

    Args:
        diff_blocks: List of ``(search_text, replace_text)`` tuples.
        code_text: The current code text to match against.
        changes_description_text: The current changes description to match against.

    Returns:
        Tuple of ``(code_blocks, changes_desc_blocks, unmatched_blocks)``.

    Raises:
        ValueError: If a SEARCH block matches both targets (ambiguous).
    """
    code_lines = code_text.split("\n")
    desc_lines = changes_description_text.split("\n")

    code_blocks: List[Tuple[str, str]] = []
    desc_blocks: List[Tuple[str, str]] = []
    unmatched: List[Tuple[str, str]] = []

    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")

        matches_code = _can_apply_linewise(code_lines, search_lines)
        matches_desc = _can_apply_linewise(desc_lines, search_lines)

        if matches_code and matches_desc:
            raise ValueError(
                "Ambiguous diff block: SEARCH matches both code and changes_description"
            )
        if matches_code:
            code_blocks.append((search_text, replace_text))
        elif matches_desc:
            desc_blocks.append((search_text, replace_text))
        else:
            unmatched.append((search_text, replace_text))

    return code_blocks, desc_blocks, unmatched
