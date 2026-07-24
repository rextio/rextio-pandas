"""Machine-readable rule records for rextio-pandas."""

from rextio.plugins.api import RuleRecord, RuleScope

RULE_RECORDS: tuple[RuleRecord, ...] = (
    RuleRecord(
        id="rextio-pandas/series-where-mask-no-go",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="series.where / series.mask"),
        constraint=(
            "Core can represent a narrow SeriesBool-condition and same-dtype scalar "
            "replacement, but the plugin has not frozen the complete pandas where/mask "
            "alignment, casting, manager, and global-binding authority graph. The shared "
            "Series boundary also rejects empty values because type-changing empty "
            "Series.map calls preserve their input dtype in pandas."
        ),
        outcome="fallback",
        diagnostic_code=None,
        guidance=(
            "Keep Series.where and Series.mask on ordinary pandas fallback until their "
            "reachable executable authority and empty/null behavior are certified."
        ),
        stability="experimental",
        verified=False,
    ),
    RuleRecord(
        id="rextio-pandas/dataframe-apply-prototype-no-go",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="pandas.DataFrame.apply(axis=1) prototype"),
        constraint=(
            "Research code characterized a homogeneous-float64 row loop, but the complete "
            "pandas executable authority behind DataFrame.apply cannot be bounded by the "
            "frozen partial class/global digest. No DataFrame apply product route or hot loop is "
            "registered, claimed, lowered, benchmarked, or headline-eligible. Shared Series "
            "boundary support may still emit unused prototype frame definitions."
        ),
        outcome="fallback",
        diagnostic_code=None,
        guidance=(
            "Keep DataFrame.apply on ordinary pandas fallback. Treat the retained helpers "
            "and characterization tests as a private prototype, not a product route."
        ),
        stability="experimental",
        verified=False,
    ),
    RuleRecord(
        id="rextio-pandas/series-map",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="series.map(udf)"),
        constraint=(
            "Exact pandas 2.3.3 Series with NumPy 2.3.5 float64/int64/bool input storage, "
            "an unnamed canonical RangeIndex, default metadata, and one statically "
            "resolved audited scalar UDF. Empty inputs and runtime contract misses "
            "raise deterministic TypeError. The UDF executes in one GIL-detached "
            "Rust loop with one extraction and one pandas materialization."
        ),
        outcome="native",
        diagnostic_code=None,
        guidance=(
            "Annotate the receiver with rextio_pandas.types.SeriesF64, SeriesI64, or "
            "SeriesBool. Pass one bare project function and keep its complete body "
            "inside the documented closed subset."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-pandas/series-map-body",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="series.map(udf) callable body"),
        constraint="Every callable body node and scalar type must be in the audited subset.",
        outcome="fallback",
        diagnostic_code="RXTP-PANDAS-003",
        guidance="Remove calls and unaudited operators; use the documented closed UDF grammar.",
        stability="experimental",
    ),
    RuleRecord(
        id="rextio-pandas/series-map-shape",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="series.map call shape"),
        constraint=(
            "Exactly one positional bare project-function reference, with na_action "
            "omitted or exactly literal None."
        ),
        outcome="fallback",
        diagnostic_code="RXTP-PANDAS-001",
        guidance=(
            "Write series.map(udf) or series.map(udf, na_action=None) on a plain "
            "annotated receiver name."
        ),
        stability="experimental",
    ),
    RuleRecord(
        id="rextio-pandas/series-map-signature",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="series.map scalar UDF signature"),
        constraint="One exactly typed scalar parameter and one supported scalar return.",
        outcome="fallback",
        diagnostic_code="RXTP-PANDAS-002",
        guidance=(
            "Annotate the mapper as float->float/int/bool, int->int/float/bool, "
            "or bool->bool/int/float; numeric results from float/bool inputs must "
            "remain literal-safe within the documented closed body grammar."
        ),
        stability="experimental",
    ),
)


def pandas_rule_records() -> tuple[RuleRecord, ...]:
    """Return stable ordered rule records."""
    return RULE_RECORDS


__all__ = ["RULE_RECORDS", "pandas_rule_records"]
