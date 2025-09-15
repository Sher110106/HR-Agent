from __future__ import annotations

from typing import Optional
import re


def build_condensed_summary(full_reasoning: str, max_bullets: int = 5, max_chars: int = 700) -> str:
    """Create a concise, business-facing summary from full reasoning.

    Strategy:
    - Extract the section under headings like "Key Insights", "Actionable Recommendations", or leading bullets
    - Keep top N bullets and trim to a character budget
    - Fallback: take the first paragraph trimmed to the budget
    """
    if not full_reasoning:
        return ""

    text = full_reasoning.strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Try to extract bullets after common headings
    sections = []
    heading_patterns = [
        r"key insights\s*:?",
        r"actionable recommendations\s*:?",
        r"insights\s*:?",
        r"recommendations\s*:?",
        r"summary\s*:?",
    ]

    lowered = full_reasoning.lower()
    for pat in heading_patterns:
        m = re.search(pat, lowered)
        if m:
            start = m.end()
            sections.append(full_reasoning[start:])

    candidate = sections[0] if sections else full_reasoning

    # Extract bullet lines (markdown style) from candidate text
    bullet_lines = re.findall(r"(?:^|\n)\s*[-*•]\s+(.+?)(?=\n|$)", candidate)
    bullet_lines = [b.strip() for b in bullet_lines if b.strip()]

    if bullet_lines:
        bullets = bullet_lines[:max_bullets]
        summary = "\n".join(f"- {b}" for b in bullets)
    else:
        # Fallback: first sentence or paragraph
        paragraph = candidate.strip()
        # Split on sentence terminators
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        summary = " ".join(sentences[:3])  # up to 3 sentences

    # Enforce character budget
    summary = summary.strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"

    # Ensure non-empty
    return summary


