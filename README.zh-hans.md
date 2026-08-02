# rextio-pandas

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio 图标">
</p>

<p align="center"><strong>针对刻意收窄、已审计 `pandas.Series.map` 范围的原生 Rust 降级。</strong></p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.ko.md">한국어</a> · 简体中文 · <a href="README.zh-hant.md">繁體中文</a> · <a href="README.ja.md">日本語</a>
</p>

当 receiver、mapper、dtype、index 和 pandas runtime authority 都与已审计契约严格匹配时，`rextio-pandas` 把数值/布尔 `Series.map` 调用编译为脱离 GIL 的 Rust 循环。分析时未认领的调用形式保留普通 pandas fallback；原生路由一旦选定，若运行时契约不匹配，则会抛出稳定 `TypeError`，而不会静默 deopt。

> **公开 Alpha 0.1.2**（2026-07-26 发布）。**仅支持 CPython 3.11**，packaging 会拒绝 3.12+。需要 `rextio>=0.1.3,<0.2`、`pandas==2.3.3` 和 `numpy==2.3.5`。
>
> **清晰边界：**`DataFrame.apply(axis=1)`、`Series.where`、`Series.mask` 和更广泛的 DataFrame 操作**不是原生路由**，都保留 Python fallback。

## 测量证据

保留的 historical benchmark 通过真实 generated wrapper 测量一个单阶段 `SeriesF64 -> SeriesF64` mapper：

| 元素数 | Native vs pandas fallback |
| ---: | ---: |
| 1 | 慢 9.80× |
| 10 | 慢 9.18× |
| 100 | 慢 5.91× |
| 1,000 | 慢 1.35× |
| 10,000 | 快 6.15× |
| 100,000 | 快 26.86× |

持续测得的 break-even 是 **10,000 个元素**，仅适用于这些测量尺寸和该精确单阶段 F64 case。它不是布尔 lane、多阶段 pipeline、其他 workload 或未测尺寸的证据。Vectorized NumPy/pandas 和 warm Numba 在每个测量尺寸都更快，但只是 context-only 比较，不是 Rextio target 声明。

## 工作原理

1. Rextio 识别带精确插件 annotation 的 `Series.map` 调用，并审计项目 mapper 的封闭表达式 grammar。
2. 每次公开原生调用在提取输入时完整验证可达的 pandas method/runtime authority 一次。
3. 一个脱离 GIL 的 Rust 循环运行且不调用 Python callback。受支持的两阶段或四阶段 pipeline 把中间值留在 Rust，最后只物化一次 pandas。

选中的原生调用内若契约不匹配，不会静默 deopt，而会以稳定 `TypeError` fail closed。分析时未 claim 的调用保留普通 Python fallback。

## 快速开始

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

Mapper 必须是一个位置参数形式的裸项目函数引用。支持省略 `na_action` 和字面量 `na_action=None`；`na_action="ignore"`、动态值、keyword mapper 形式和 lambda 均走 fallback。

## 支持的 `Series.map` 范围

| Receiver | 支持的结果 | Mapper grammar 摘要 |
| --- | --- | --- |
| `SeriesF64` | `SeriesF64`, `SeriesBool`, `SeriesI64` | finite float literal、一元 negation、同类型比较、布尔组合/条件、已审计 `+ - *`；int 结果仅通过 literal-safe conditional |
| `SeriesI64` | `SeriesI64`, `SeriesBool`, `SeriesF64` | full-domain-safe identity/literal/comparison/boolean/conditional body；float 结果仅通过受支持 conditional |
| `SeriesBool` | `SeriesBool`, `SeriesI64`, `SeriesF64` | bool 参数/literal、`not`、`and`/`or`、equality/inequality、conditional；数值结果选择精确 literal |

整数 literal 必须适合 signed int64，float literal 必须有限。没有 float-to-int coercion。调用、division、floor/mod/power/matmul、bit/shift、identity/membership operator、副作用、closure 和未审计 body 都不在原生 grammar 内。

## 运行时边界

接受的输入必须同时满足：

- 精确、非空 `pandas.Series`，使用精确 NumPy `float64`、`int64` 或 `bool` storage；
- 未命名 `RangeIndex(0, len, 1)`；
- `attrs == {}`、`flags.allows_duplicate_labels is True`，且 `name` 为 `None` 或 `str`；
- 使用固定且未修改的可达 pandas authority graph。

Nullable/extension/object/categorical/Arrow storage、subclass、empty Series、任意 index/MultiIndex 和非标准 metadata 均不支持。Strided array 按逻辑 indexing 复制到 Rust 自有 storage。结果保留精确 Series class、dtype、值（包括 NaN/Inf/signed zero）、顺序、RangeIndex、name 和默认 metadata。

CPython `-O` 有单独冻结的受支持 authority digest。`-OO` 被刻意设为不支持并 fail closed。本 provider 不声明 standalone artifact capability。

## 明确的 fallback 范围

- `DataFrame.apply(axis=1)` 没有作为产品路由注册、claim、lower 或 benchmark。可导入的 research marker 不代表支持。
- `Series.where` 和 `Series.mask` 因 alignment/casting authority graph 与 empty-Series semantics 未认证而保留 fallback。
- Binary-Series 操作、nullable data、任意 pandas call option、不支持的 mapper 及所有未列 pandas/DataFrame 操作均走 fallback。

## Benchmark 证据

参见 [benchmark 方法](benchmarks/README.md)和[权威 schema-4 结果](benchmarks/results/latest.json)。运行包含 validation、copy、conversion、结果 construction/destruction、paired sample、correctness digest、route/build evidence、provenance 和 null-call floor。小输入 loss 不会被隐藏。

## 开发

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests benchmarks
.venv/bin/mypy src
```

发布历史见 [CHANGELOG.md](CHANGELOG.md)。

## 许可证

MIT
