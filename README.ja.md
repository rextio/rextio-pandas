# rextio-pandas

<p align="center">
  <img src="./assets/readme/rextio-icon.png" width="96" alt="Rextio アイコン">
</p>

<p align="center"><strong>意図的に狭く監査された `pandas.Series.map` 範囲のためのネイティブ Rust lowering。</strong></p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.ko.md">한국어</a> · <a href="README.zh-hans.md">简体中文</a> · <a href="README.zh-hant.md">繁體中文</a> · 日本語
</p>

`rextio-pandas` は receiver、mapper、dtype、index、pandas runtime authority が監査済み契約に厳密に一致する場合、数値/bool `Series.map` 呼び出しを GIL-detached Rust loop にコンパイルします。解析時に claim されない呼び出し形式は通常の pandas fallback に残りますが、ネイティブ route の選択後に実行時契約を満たさなければ、静かに deopt せず安定した `TypeError` を送出します。

> **Public Alpha 0.1.2**（2026-07-26 リリース）。**CPython 3.11 専用**で、packaging は 3.12+ を拒否します。`rextio>=0.1.3,<0.2`、`pandas==2.3.3`、`numpy==2.3.5` が必要です。
>
> **明確な境界：**`DataFrame.apply(axis=1)`、`Series.where`、`Series.mask`、より広い DataFrame 操作は**ネイティブ route ではありません**。Python fallback に残ります。

## 測定済みの根拠

保存済み historical benchmark は、実 generated wrapper を通じて単一段 `SeriesF64 -> SeriesF64` mapper 1 つを測定します。

| 要素数 | Native vs pandas fallback |
| ---: | ---: |
| 1 | 9.80× 遅い |
| 10 | 9.18× 遅い |
| 100 | 5.91× 遅い |
| 1,000 | 1.35× 遅い |
| 10,000 | 6.15× 速い |
| 100,000 | 26.86× 速い |

Sustained measured break-even は **10,000 要素**で、これらの測定サイズと厳密な single-stage F64 case に限定されます。Bool lane、multi-stage pipeline、他 workload、未測定サイズの根拠ではありません。Vectorized NumPy/pandas と warm Numba は全測定サイズで高速でしたが、Rextio target 主張ではなく context-only 比較です。

## 仕組み

1. Rextio が厳密なプラグイン annotation の `Series.map` 呼び出しを認識し、project mapper の閉じた expression grammar を監査します。
2. 各公開ネイティブ呼び出しは入力抽出時に、到達可能な pandas method/runtime authority 全体を 1 回検証します。
3. Python callback なしに 1 つの GIL-detached Rust loop が動きます。サポート対象の 2 段/4 段 pipeline は intermediate を Rust に保持し、最後に pandas を 1 回だけ materialize します。

選択済みネイティブ呼び出し内の契約不一致は静かに deopt せず、安定した `TypeError` で fail closed します。解析時に claim されない呼び出しは通常の Python fallback です。

## クイックスタート

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

Mapper は 1 個の positional bare project-function reference でなければなりません。省略 `na_action` とリテラル `na_action=None` はサポートされます。`na_action="ignore"`、動的値、keyword mapper 形式、lambda は fallback です。

## サポート対象 `Series.map` 範囲

| Receiver | サポート結果 | Mapper grammar 概要 |
| --- | --- | --- |
| `SeriesF64` | `SeriesF64`, `SeriesBool`, `SeriesI64` | finite float literal、単項 negation、same-type 比較、bool composition/conditional、監査済み `+ - *`。int 結果は literal-safe conditional のみ |
| `SeriesI64` | `SeriesI64`, `SeriesBool`, `SeriesF64` | full-domain-safe identity/literal/comparison/boolean/conditional body。float 結果はサポート conditional のみ |
| `SeriesBool` | `SeriesBool`, `SeriesI64`, `SeriesF64` | bool parameter/literal、`not`、`and`/`or`、equality/inequality、conditional。数値結果は厳密な literal を選択 |

整数 literal は signed int64 範囲内、float literal は finite でなければなりません。float-to-int coercion はありません。呼び出し、division、floor/mod/power/matmul、bit/shift、identity/membership operator、副作用、closure、未監査 body はネイティブ grammar 外です。

## 実行時境界

受理入力は以下をすべて満たす必要があります。

- 厳密な NumPy `float64`、`int64`、`bool` storage を持つ exact nonempty `pandas.Series`。
- 名前なし `RangeIndex(0, len, 1)`。
- `attrs == {}`、`flags.allows_duplicate_labels is True`、`name` は `None` または `str`。
- 固定済みで変更されていない reachable pandas authority graph。

Nullable/extension/object/categorical/Arrow storage、subclass、empty Series、任意 index/MultiIndex、非標準 metadata は未対応です。Strided array は logical indexing で所有 Rust storage にコピーされます。結果は exact Series class、dtype、値（NaN/Inf/signed zero 含む）、順序、RangeIndex、name、default metadata を保持します。

CPython `-O` は個別に frozen されたサポート authority digest を持ちます。`-OO` は意図的に未対応で fail closed します。この provider は standalone artifact capability を宣伝しません。

## 明示的な fallback 範囲

- `DataFrame.apply(axis=1)` は製品 route として登録、claim、lower、benchmark されません。import 可能な research marker はサポートを意味しません。
- `Series.where` と `Series.mask` は alignment/casting authority graph と empty-Series semantics が認証されていないため fallback です。
- Binary-Series 操作、nullable data、任意 pandas call option、未対応 mapper、未掲載の pandas/DataFrame 操作は fallback です。

## Benchmark 根拠

[benchmark 方法](benchmarks/README.md)と[権威ある schema-4 結果](benchmarks/results/latest.json)を参照してください。実行には validation、copy、conversion、結果 construction/destruction、paired sample、correctness digest、route/build evidence、provenance、null-call floor が含まれます。小入力 loss も隠しません。

## 開発

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests benchmarks
.venv/bin/mypy src
```

リリース履歴は [CHANGELOG.md](CHANGELOG.md) を参照してください。

## ライセンス

MIT
