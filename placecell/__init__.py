"""pcell - Package for analyzing place cells."""

try:
    from placecell._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

from placecell.dataset import PlaceCellDataset, UnitResult, unique_bundle_path

__all__ = ["PlaceCellDataset", "UnitResult", "unique_bundle_path", "__version__"]
