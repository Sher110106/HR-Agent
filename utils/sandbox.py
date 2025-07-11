"""Sandbox utilities for secure code execution.

This module exposes a curated list of *safe* built-in functions that can be
exposed to dynamically executed user / LLM generated code.  The goal is to
prevent access to powerful primitives such as `__import__`, `open`, or
`eval`, which could otherwise lead to arbitrary file-system access or remote
code execution when `exec()` is employed inside the application.

Usage
-----
>>> from utils.sandbox import SAFE_BUILTINS, build_sandbox_globals
>>> exec(user_code, build_sandbox_globals({"pd": pandas}), {})
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Public: SAFE_BUILTINS – immutable mapping of allowed built-ins
# ---------------------------------------------------------------------------
# Only pure-Python, side-effect-free utilities are exposed.  Any function that
# may interact with I/O, import additional modules, spawn subprocesses, or
# introspect the interpreter is intentionally *omitted*.

_SAFE_BUILTINS: Dict[str, Any] = {
    # Iterable helpers
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,

    # Math / numeric
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,

    # Type constructors
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "str": str,
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,

    # Logic helpers
    "all": all,
    "any": any,

    # Basic output – harmless, useful for debugging
    "print": print,
}

# Expose a *read-only* view so that runtime code cannot mutate the mapping.
SAFE_BUILTINS = MappingProxyType(_SAFE_BUILTINS)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def build_sandbox_globals(additional: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a new *globals* dict suitable for passing to ``exec``.

    Parameters
    ----------
    additional:
        Variables (e.g. pre-imported libraries) that should be visible inside the
        sandboxed execution environment.
    """
    sandbox_globals: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS}

    if additional:
        # User-provided env takes precedence, but we guard against accidental
        # override of __builtins__.
        for k, v in additional.items():
            if k == "__builtins__":
                continue
            sandbox_globals[k] = v

    return sandbox_globals 