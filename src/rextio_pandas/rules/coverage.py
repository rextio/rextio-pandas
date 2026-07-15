"""Declared pandas package and symbol coverage."""

from rextio.plugins.api import CoverageDecl

COVERAGE = CoverageDecl(
    packages=("pandas",),
    modules=("pandas",),
    symbols=("pandas.Series.map",),
)

__all__ = ["COVERAGE"]
