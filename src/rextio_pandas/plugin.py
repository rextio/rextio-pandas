"""Rextio API 1.3 plugin facade for audited pandas Series.map lowering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rextio_pandas.__about__ import __version__

if TYPE_CHECKING:
    from rextio.config.schema import RextioConfig
    from rextio.plugins.api import (
        ClaimResult,
        ClaimSite,
        CoverageDecl,
        CrateDependency,
        LoweredExpr,
        LoweringContext,
        PluginType,
        RuleRecord,
    )
    from rextio.plugins.models import RextioPlugin

PLUGIN_ID = "rextio-pandas"
# Public dependency floor: rextio>=0.1.3,<0.2 with plugin API 1.3.
REQUIRED_PLUGIN_API = "1.3"


def _require_api_13() -> None:
    from rextio.plugins.api import PLUGIN_API_VERSION

    if PLUGIN_API_VERSION != REQUIRED_PLUGIN_API:
        raise RuntimeError(
            "rextio-pandas requires Rextio plugin API 1.3 "
            f"(rextio>=0.1.3,<0.2); this environment advertises "
            f"PLUGIN_API_VERSION={PLUGIN_API_VERSION!r}"
        )


class RextioPandasPlugin:
    """Lower only audited pandas operations under plugin API 1.3."""

    plugin_id = PLUGIN_ID
    api_version = REQUIRED_PLUGIN_API

    def to_rextio_plugin(self) -> RextioPlugin:
        """Return the metadata object registered by Rextio core."""
        _require_api_13()
        from rextio.plugins.models import RextioPlugin

        from rextio_pandas.rules import COVERAGE

        return RextioPlugin(
            id=PLUGIN_ID,
            name=f"pandas Series.map (rextio-pandas {__version__})",
            source_language="python",
            target_language="rust",
            packages=COVERAGE.packages,
        )

    def covers(self) -> CoverageDecl:
        """Describe the pandas package and symbols owned by this provider."""
        _require_api_13()
        from rextio_pandas.rules import COVERAGE

        return COVERAGE

    def describe(self, config: RextioConfig) -> tuple[RuleRecord, ...]:
        """Return the deterministic machine-readable rule records."""
        _require_api_13()
        from rextio_pandas.rules import pandas_rule_records

        del config
        return pandas_rule_records()

    def type_vocabulary(self) -> tuple[PluginType, ...]:
        """Return materialized pandas boundary types."""
        _require_api_13()
        from rextio_pandas.plugin_types import plugin_types

        return plugin_types()

    def claim(self, site: ClaimSite, config: RextioConfig) -> ClaimResult:
        """Claim an exact audited pandas call site."""
        _require_api_13()
        from rextio_pandas.claim import claim

        return claim(site, config)

    def lower(self, claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
        """Lower a previously accepted site after defensive revalidation."""
        _require_api_13()
        from rextio_pandas.lower import lower

        return lower(claimed, ctx)

    def crate_dependencies(self) -> tuple[CrateDependency, ...]:
        """Return the exact rust-numpy crate pin used for owned conversion.

        The SHA-256 used by the method-identity guard comes from the core-managed
        ``sha2`` crate already present in the generated manifest, so it is not
        re-declared here (core reserves core crate names).
        """
        _require_api_13()
        from rextio.plugins.api import CrateDependency

        return (CrateDependency(name="numpy", version="=0.29.0"),)


def plugin() -> RextioPandasPlugin:
    """Return the entry-point provider object."""
    return RextioPandasPlugin()


__all__ = ["PLUGIN_ID", "REQUIRED_PLUGIN_API", "RextioPandasPlugin", "plugin"]
