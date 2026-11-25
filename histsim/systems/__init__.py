from .culture import Cultures
from .environment import Environment
from .farming import Farming
from .population import Population
from .settlements import Settlements
from .states import States
from .technology import Technology
from .trade import Trade


def default_systems():
    """The systems of a standard run, in the order they run each year."""
    return [Environment(), Farming(), Population(), Cultures(), Settlements(), States(), Technology(), Trade()]
