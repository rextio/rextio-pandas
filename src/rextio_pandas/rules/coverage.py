"""Declared pandas package and symbol coverage."""

from rextio.plugins.api import CoverageDecl

COVERAGE = CoverageDecl(
    packages=("pandas",),
    modules=("pandas",),
    symbols=("pandas.Series.map", "pandas.DataFrame.apply"),
)

__all__ = ["COVERAGE"]
