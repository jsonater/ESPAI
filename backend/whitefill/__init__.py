"""Whitefill: photos of pure-white cubes/triangles -> one printable STL per piece.

Only pure white is used. Other colours, mixed colours, other shapes and big objects are ignored.
"""

from .pipeline import analyze

__all__ = ["analyze"]
