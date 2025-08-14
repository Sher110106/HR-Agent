"""Color palette utilities

This module centralises colour palette definitions so that other chart
modules (matplotlib or plotly) import colours from a single place.
Keeping palettes in one file avoids the duplication that existed in
`plot_helpers.py`.
"""
from __future__ import annotations

from typing import Dict, List

# Base palette dictionary ----------------------------------------------------
# Each palette returns a list of hex colour strings. The first element of each
# dictionary is considered the default/fallback single colour for that palette.

_PALETTES: Dict[str, Dict[str, List[str] | str]] = {
    "primary": {
        "colors": [
            "#2E86C1",
            "#28B463",
            "#F39C12",
            "#E74C3C",
            "#8E44AD",
            "#17A2B8",
            "#FFC107",
            "#6C757D",
            "#20C997",
            "#FD7E14",
        ],
        "single": "#2E86C1",
    },
    "secondary": {
        "colors": [
            "#5D6D7E",
            "#85929E",
            "#AEB6BF",
            "#D5DBDB",
            "#F8F9FA",
            "#E8EAED",
            "#CED4DA",
            "#6C757D",
            "#495057",
            "#343A40",
        ],
        "single": "#5D6D7E",
    },
    "categorical": {
        "colors": [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
            "#FFB6C1",
            "#98FB98",
            "#F0E68C",
            "#DEB887",
        ],
        "single": "#FF6B6B",
    },
    "attrition": {
        "colors": ["#e74c3c", "#c0392b", "#a93226", "#922b21", "#7b241c"],
        "single": "#e74c3c",
    },
    "retention": {
        "colors": ["#27ae60", "#229954", "#1e8449", "#196f3d", "#145a32"],
        "single": "#27ae60",
    },
    "performance": {
        "colors": ["#f39c12", "#e67e22", "#d35400", "#ba4a00", "#a04000"],
        "single": "#f39c12",
    },
    "neutral": {
        "colors": ["#3498db", "#2980b9", "#1f618d", "#154360", "#0e2f44"],
        "single": "#3498db",
    },
}


def get_palette(name: str = "primary", n: int | None = None) -> List[str]:
    """Return a colour list for *name* palette.

    Parameters
    ----------
    name:
        Palette name.  Falls back to *primary* if the name is unknown.
    n:
        Optional number of colours required.  If *n* is larger than the palette
        length, the colours are cycled.  If *n* is smaller, the list is
        truncated.
    """
    palette_def = _PALETTES.get(name, _PALETTES["primary"])
    colours = palette_def["colors"]
    if n is None:
        return colours
    # Repeat or slice as needed
    repeats = (n + len(colours) - 1) // len(colours)
    full = (colours * repeats)[:n]
    return full


__all__ = ["get_palette"]