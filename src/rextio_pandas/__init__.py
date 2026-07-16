"""Public alpha plugin for audited pandas Series.map lowering under Rextio API 1.3."""

from rextio_pandas.__about__ import __version__
from rextio_pandas.plugin import RextioPandasPlugin, plugin

__all__ = ["RextioPandasPlugin", "__version__", "plugin"]
