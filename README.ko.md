# rextio-pandas

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio 아이콘">
</p>

<p align="center"><strong>의도적으로 좁고 감사된 `pandas.Series.map` 범위를 위한 네이티브 Rust lowering.</strong></p>

<p align="center">
  <a href="README.md">English</a> · 한국어 · <a href="README.zh-hans.md">简体中文</a> · <a href="README.zh-hant.md">繁體中文</a> · <a href="README.ja.md">日本語</a>
</p>

`rextio-pandas`는 receiver, mapper, dtype, index, pandas runtime authority가 감사된 계약과 정확히 일치할 때 숫자/bool `Series.map` 호출을 GIL-detached Rust loop로 컴파일합니다. 분석 시 claim되지 않은 호출 형식은 일반 pandas fallback에 남지만, 네이티브 route가 선택된 뒤 런타임 계약을 충족하지 못하면 조용히 deopt하지 않고 안정적인 `TypeError`를 발생시킵니다.

> **공개 Alpha 0.1.2** (2026-07-26 릴리스). **CPython 3.11 전용**이며 packaging이 3.12+를 거부합니다. `rextio>=0.1.3,<0.2`, `pandas==2.3.3`, `numpy==2.3.5`가 필요합니다.
>
> **명확한 경계:** `DataFrame.apply(axis=1)`, `Series.where`, `Series.mask`, 더 넓은 DataFrame 연산은 **네이티브 route가 아닙니다**. Python fallback에 남습니다.

## 측정된 근거

보존된 historical benchmark는 실제 generated wrapper를 통해 단일 단계 `SeriesF64 -> SeriesF64` mapper 하나를 측정합니다.

| 원소 수 | Native vs pandas fallback |
| ---: | ---: |
| 1 | 9.80× 느림 |
| 10 | 9.18× 느림 |
| 100 | 5.91× 느림 |
| 1,000 | 1.35× 느림 |
| 10,000 | 6.15× 빠름 |
| 100,000 | 26.86× 빠름 |

Sustained measured break-even은 **10,000개 원소**이며 해당 측정 크기와 정확한 single-stage F64 case에 한정됩니다. Bool lane, multi-stage pipeline, 다른 workload, 측정하지 않은 크기의 근거가 아닙니다. Vectorized NumPy/pandas와 warm Numba는 모든 측정 크기에서 더 빨랐으며 Rextio target 주장이 아닌 context-only 비교입니다.

## 동작 방식

1. Rextio가 정확한 플러그인 annotation의 `Series.map` 호출을 인식하고 project mapper의 닫힌 expression grammar를 검사합니다.
2. 각 공개 네이티브 호출은 입력을 추출하면서 도달 가능한 전체 pandas method/runtime authority를 한 번 검증합니다.
3. Python callback 없이 하나의 GIL-detached Rust loop가 실행됩니다. 지원되는 2단계/4단계 pipeline은 intermediate를 Rust에 유지하고 마지막에 pandas를 한 번만 materialize합니다.

선택된 네이티브 호출 안에서 계약 불일치는 조용히 deopt하지 않고 안정적인 `TypeError`로 fail closed합니다. 분석 시 claim되지 않은 호출은 일반 Python fallback입니다.

## 빠른 시작

```bash
python3.11 -m pip install "rextio-pandas==0.1.2"
```

```toml
# rextio.toml
[rust]
build_tool = "cargo"

[plugins]
enabled = ["rextio-pandas"]
```

```python
from rextio_pandas.types import SeriesF64

def absolute(value: float) -> float:
    return value if value >= 0.0 else -value

def normalize(series: SeriesF64) -> SeriesF64:
    return series.map(absolute)
```

```bash
rextio build .
```

Mapper는 positional bare project-function reference 하나여야 합니다. 생략된 `na_action`과 literal `na_action=None`은 지원하지만 `na_action="ignore"`, 동적 값, keyword mapper 형식, lambda는 fallback입니다.

## 지원되는 `Series.map` 범위

| Receiver | 지원 result | Mapper grammar 요약 |
| --- | --- | --- |
| `SeriesF64` | `SeriesF64`, `SeriesBool`, `SeriesI64` | finite float literal, 단항 negation, same-type 비교, bool composition/conditional, 감사된 `+ - *`; int result는 literal-safe conditional만 |
| `SeriesI64` | `SeriesI64`, `SeriesBool`, `SeriesF64` | full-domain-safe identity/literal/comparison/boolean/conditional body; float result는 지원 conditional만 |
| `SeriesBool` | `SeriesBool`, `SeriesI64`, `SeriesF64` | bool parameter/literal, `not`, `and`/`or`, equality/inequality, conditional; 숫자 result는 정확한 literal 선택 |

정수 literal은 signed int64 범위, float literal은 finite여야 합니다. float-to-int coercion은 없습니다. 호출, division, floor/mod/power/matmul, bit/shift, identity/membership operator, side effect, closure, 미감사 body는 네이티브 grammar 밖입니다.

## 런타임 경계

허용되는 입력은 다음을 모두 만족해야 합니다.

- 정확한 NumPy `float64`, `int64`, `bool` storage를 가진 exact nonempty `pandas.Series`;
- 이름 없는 `RangeIndex(0, len, 1)`;
- `attrs == {}`, `flags.allows_duplicate_labels is True`, `name`은 `None` 또는 `str`;
- 고정된 수정되지 않은 reachable pandas authority graph.

Nullable/extension/object/categorical/Arrow storage, subclass, empty Series, 임의 index/MultiIndex, 비표준 metadata는 미지원입니다. Strided array는 logical indexing으로 소유 Rust storage에 복사됩니다. 결과는 exact Series class, dtype, 값(NaN/Inf/signed zero 포함), 순서, RangeIndex, name, default metadata를 보존합니다.

CPython `-O`는 별도의 frozen supported authority digest를 가집니다. `-OO`는 의도적으로 미지원이며 fail closed합니다. 이 provider는 standalone artifact capability를 광고하지 않습니다.

## 명시적인 fallback 범위

- `DataFrame.apply(axis=1)`는 제품 route로 등록, claim, lower, benchmark되지 않습니다. import 가능한 research marker는 지원을 뜻하지 않습니다.
- `Series.where`와 `Series.mask`는 alignment/casting authority graph와 empty-Series semantics가 인증되지 않아 fallback입니다.
- Binary-Series 연산, nullable data, 임의 pandas call option, 미지원 mapper, 목록 밖 pandas/DataFrame 연산은 fallback입니다.

## Benchmark 근거

[Benchmark 방법](benchmarks/README.md)과 [공식 schema-4 결과](benchmarks/results/latest.json)를 참고하세요. 실행에는 validation, copy, conversion, 결과 construction/destruction, paired sample, correctness digest, route/build evidence, provenance, null-call floor가 포함됩니다. 작은 입력의 loss도 숨기지 않습니다.

## 개발

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests benchmarks
.venv/bin/mypy src
```

릴리스 이력은 [CHANGELOG.md](CHANGELOG.md)를 참고하세요.

## 라이선스

MIT
