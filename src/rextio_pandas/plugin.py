"""Rextio API 1.3 plugin facade for the private pandas incubator."""

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
CORE_COMMIT = "2bd1d1da0cf59e97d1659606bcb1ec12491e032c"


def _require_api_13() -> None:
    from rextio.plugins.api import PLUGIN_API_VERSION

    if PLUGIN_API_VERSION != "1.3":
        raise RuntimeError(
            "rextio-pandas is a private plugin API 1.3 incubator built against "
            f"Rextio core commit {CORE_COMMIT}; released rextio 0.1.2 is incompatible"
        )


class RextioPandasPlugin:
    """Lower only audited pandas operations under the integrated API 1.3 core."""

    plugin_id = PLUGIN_ID
    api_version = "1.3"

    def to_rextio_plugin(self) -> RextioPlugin:
        """Return the metadata object registered by Rextio core."""
        _require_api_13()
        from rextio.plugins.models import RextioPlugin

        from rextio_pandas.rules import COVERAGE

        return RextioPlugin(
            id=PLUGIN_ID,
            name=f"pandas Series.map incubator (rextio-pandas {__version__})",
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
        """Return the deterministic machine-readable incubator records."""
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


__all__ = ["CORE_COMMIT", "PLUGIN_ID", "RextioPandasPlugin", "plugin"]
