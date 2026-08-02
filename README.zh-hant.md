# rextio-pandas

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio 圖示">
</p>

<p align="center"><strong>針對刻意縮小、已稽核 `pandas.Series.map` 範圍的原生 Rust lowering。</strong></p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.ko.md">한국어</a> · <a href="README.zh-hans.md">简体中文</a> · 繁體中文 · <a href="README.ja.md">日本語</a>
</p>

當 receiver、mapper、dtype、index 與 pandas runtime authority 都與已稽核契約嚴格相符時，`rextio-pandas` 把數值/布林 `Series.map` 呼叫編譯為脫離 GIL 的 Rust 迴圈。分析時未 claim 的呼叫形式保留普通 pandas fallback；原生路由一旦選定，若執行時契約不符，則會拋出穩定 `TypeError`，而不會靜默 deopt。

> **公開 Alpha 0.1.2**（2026-07-26 發布）。**僅支援 CPython 3.11**，packaging 會拒絕 3.12+。需要 `rextio>=0.1.3,<0.2`、`pandas==2.3.3` 與 `numpy==2.3.5`。
>
> **清楚邊界：**`DataFrame.apply(axis=1)`、`Series.where`、`Series.mask` 和更廣泛的 DataFrame 操作**不是原生路由**，都保留 Python fallback。

## 測量證據

保留的 historical benchmark 透過真實 generated wrapper 測量一個單階段 `SeriesF64 -> SeriesF64` mapper：

| 元素數 | Native vs pandas fallback |
| ---: | ---: |
| 1 | 慢 9.80× |
| 10 | 慢 9.18× |
| 100 | 慢 5.91× |
| 1,000 | 慢 1.35× |
| 10,000 | 快 6.15× |
| 100,000 | 快 26.86× |

持續測得的 break-even 是 **10,000 個元素**，僅適用於這些測量尺寸與該精確單階段 F64 case。它不是布林 lane、多階段 pipeline、其他 workload 或未測尺寸的證據。Vectorized NumPy/pandas 與 warm Numba 在每個測量尺寸都更快，但只是 context-only 比較，不是 Rextio target 聲明。

## 運作方式

1. Rextio 識別帶精確外掛 annotation 的 `Series.map` 呼叫，並稽核專案 mapper 的封閉運算式 grammar。
2. 每次公開原生呼叫在擷取輸入時完整驗證可達的 pandas method/runtime authority 一次。
3. 一個脫離 GIL 的 Rust 迴圈執行且不呼叫 Python callback。受支援的兩階段或四階段 pipeline 把中間值留在 Rust，最後只實體化一次 pandas。

選取的原生呼叫內若契約不符，不會靜默 deopt，而會以穩定 `TypeError` fail closed。分析時未 claim 的呼叫保留普通 Python fallback。

## 快速開始

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

Mapper 必須是一個 positional 形式的 bare project-function reference。支援省略 `na_action` 與字面值 `na_action=None`；`na_action="ignore"`、動態值、keyword mapper 形式與 lambda 均走 fallback。

## 支援的 `Series.map` 範圍

| Receiver | 支援結果 | Mapper grammar 摘要 |
| --- | --- | --- |
| `SeriesF64` | `SeriesF64`, `SeriesBool`, `SeriesI64` | finite float literal、一元 negation、同型別比較、布林組合/條件、已稽核 `+ - *`；int 結果僅透過 literal-safe conditional |
| `SeriesI64` | `SeriesI64`, `SeriesBool`, `SeriesF64` | full-domain-safe identity/literal/comparison/boolean/conditional body；float 結果僅透過受支援 conditional |
| `SeriesBool` | `SeriesBool`, `SeriesI64`, `SeriesF64` | bool 參數/literal、`not`、`and`/`or`、equality/inequality、conditional；數值結果選擇精確 literal |

整數 literal 必須符合 signed int64，float literal 必須有限。沒有 float-to-int coercion。呼叫、division、floor/mod/power/matmul、bit/shift、identity/membership operator、副作用、closure 與未稽核 body 都不在原生 grammar 內。

## 執行時邊界

接受的輸入必須同時滿足：

- 精確、非空 `pandas.Series`，使用精確 NumPy `float64`、`int64` 或 `bool` storage；
- 未命名 `RangeIndex(0, len, 1)`；
- `attrs == {}`、`flags.allows_duplicate_labels is True`，且 `name` 為 `None` 或 `str`；
- 使用固定且未修改的可達 pandas authority graph。

Nullable/extension/object/categorical/Arrow storage、subclass、empty Series、任意 index/MultiIndex 與非標準 metadata 均不支援。Strided array 按邏輯 indexing 複製到 Rust 自有 storage。結果保留精確 Series class、dtype、值（包含 NaN/Inf/signed zero）、順序、RangeIndex、name 與預設 metadata。

CPython `-O` 有單獨凍結的受支援 authority digest。`-OO` 被刻意設為不支援並 fail closed。本 provider 不宣告 standalone artifact capability。

## 明確的 fallback 範圍

- `DataFrame.apply(axis=1)` 沒有作為產品路由註冊、claim、lower 或 benchmark。可匯入的 research marker 不代表支援。
- `Series.where` 和 `Series.mask` 因 alignment/casting authority graph 與 empty-Series semantics 未認證而保留 fallback。
- Binary-Series 操作、nullable data、任意 pandas call option、不支援的 mapper 與所有未列 pandas/DataFrame 操作均走 fallback。

## Benchmark 證據

參見 [benchmark 方法](benchmarks/README.md)與[權威 schema-4 結果](benchmarks/results/latest.json)。執行包含 validation、copy、conversion、結果 construction/destruction、paired sample、correctness digest、route/build evidence、provenance 與 null-call floor。小輸入 loss 不會被隱藏。

## 開發

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests benchmarks
.venv/bin/mypy src
```

發布歷史見 [CHANGELOG.md](CHANGELOG.md)。

## 授權條款

MIT
