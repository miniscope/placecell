"""pcell - Package for analyzing place cells."""

try:
    from placecell._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

from placecell.dataset import PlaceCellDataset

__all__ = ["PlaceCellDataset", "__version__"]
