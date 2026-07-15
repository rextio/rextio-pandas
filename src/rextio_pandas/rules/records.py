"""Machine-readable incubator rule records."""

from rextio.plugins.api import RuleRecord, RuleScope

RULE_RECORDS: tuple[RuleRecord, ...] = (
    RuleRecord(
        id="rextio-pandas/dataframe-apply-axis1",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="frame.apply(udf, axis=1)"),
        constraint=(
            "Exact pandas 2.3.3 homogeneous NumPy float64 DataFrame with a valid "
            "ordered all-float declared schema, canonical RangeIndex, default metadata, "
            "and one closed row UDF using only literal row subscripts, finite float "
            "literals, unary negation, comparisons, boolean composition, and a float "
            "conditional. Empty/zero-column and runtime contract misses raise TypeError."
        ),
        outcome="native",
        diagnostic_code=None,
        guidance=(
            "Annotate the receiver DataFrameF64[Row], declare one or more float fields "
            "in exact column order, and write frame.apply(udf, axis=1) with literal "
            'row["field"] reads.'
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-pandas/dataframe-apply-body",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="frame.apply row UDF body"),
        constraint="Every row body node must be in the stricter audited float64 subset.",
        outcome="fallback",
        diagnostic_code="RXTP-PANDAS-013",
        guidance="Remove all binops, calls, attributes, and dynamic field accesses.",
        stability="experimental",
    ),
    RuleRecord(
        id="rextio-pandas/dataframe-apply-schema",
        provider="rextio-pandas",
        scope=RuleScope(kind="type", pattern="DataFrameF64[Schema]"),
        constraint="The schema is nonempty, ordered, unique, and contains only float fields.",
        outcome="fallback",
        diagnostic_code="RXTP-PANDAS-012",
        guidance="Declare a simple schema class with one or more float-only fields.",
        stability="experimental",
    ),
    RuleRecord(
        id="rextio-pandas/dataframe-apply-shape",
        provider="rextio-pandas",
        scope=RuleScope(kind="call", pattern="frame.apply call shape"),
        constraint="Exactly one positional row UDF and the sole literal keyword axis=1.",
        outcome="fallback",
        diagnostic_code="RXTP-PANDAS-011",
        guidance="Write exactly frame.apply(udf, axis=1).",
        stability="experimental",
    ),
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
