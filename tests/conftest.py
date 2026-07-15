from __future__ import annotations

from typing import Any

from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_pandas.plugin import plugin


class FakeEntryPoint:
    name = "rextio-pandas"
    dist = None

    def load(self) -> Any:
        return plugin


def pandas_registry():
    return load_plugin_registry(
        PluginConfig(enabled=("rextio-pandas",)),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )
