from .environment import Environment
from .farming import Farming
from .population import Population


def default_systems():
    """The systems of a standard run, in the order they run each year."""
    return [Environment(), Farming(), Population()]
