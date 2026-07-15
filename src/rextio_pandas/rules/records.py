"""Machine-readable incubator rule records."""

from rextio.plugins.api import RuleRecord, RuleScope

RULE_RECORDS: tuple[RuleRecord, ...] = (
    RuleRecord(
        id="rextio-pandas/series-map",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="series.map(udf)"),
        constraint=(
            "Exact pandas 2.3.3 Series with NumPy 2.3.5 float64/int64 storage, "
            "an unnamed canonical RangeIndex, default metadata, and one statically "
            "resolved audited scalar UDF. Empty inputs and runtime contract misses "
            "raise deterministic TypeError. The UDF executes in one GIL-detached "
            "Rust loop with one extraction and one pandas materialization."
        ),
        outcome="native",
        diagnostic_code=None,
        guidance=(
            "Annotate the receiver and result with rextio_pandas.types.SeriesF64 "
            "or SeriesI64, pass one bare project function, and keep its complete "
            "body inside the documented comparison/conditional numeric subset."
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
        constraint="Exactly one positional bare project-function reference and no keywords.",
        outcome="fallback",
        diagnostic_code="RXTP-PANDAS-001",
        guidance="Write exactly series.map(udf) on a plain annotated receiver name.",
        stability="experimental",
    ),
    RuleRecord(
        id="rextio-pandas/series-map-signature",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="series.map scalar UDF signature"),
        constraint="One exactly typed scalar parameter and one supported scalar return.",
        outcome="fallback",
        diagnostic_code="RXTP-PANDAS-002",
        guidance="Annotate the mapper as float->float, int->int, or int->float.",
        stability="experimental",
    ),
)


def pandas_rule_records() -> tuple[RuleRecord, ...]:
    """Return stable ordered rule records."""
    return RULE_RECORDS


__all__ = ["RULE_RECORDS", "pandas_rule_records"]
