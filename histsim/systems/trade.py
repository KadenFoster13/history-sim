"""Trade: routes between towns, the goods they carry, and what trade does (phase 4 step 4).

Every `trade_every` years, once towns have grown:

  routes     each town (of `trade_town_min` people or more) gets the land it reaches most
             cheaply. Travel is cheapest on open plains, costlier in forest, desert and
             mountains; cheaper with the wheel, on roads inside a state, along rivers where
             people sail, and along coasts by boat (paddling there without sailing is slow).
             Towns whose lands meet are linked, along the cheapest path between them.
  flows      every pair of towns within reach trades by a gravity rule (bigger places and
             nearer ones trade more: size x size / cost^2), carried along the chain of links
             between them, so busy roads and crossroads emerge.
  goods      copper, tin and gold or silver reach towns that lack them from towns that have
             them within `goods_reach` (bronze can be made far from tin, as on Earth).
  knowledge  each link carries what one end knows to the other (much faster than
             overland: systems/technology.py reads `offer`).
  towns      towns on busy routes grow (crossroads, river ports, coasts) and can be fed more.

The `trade` layer is the volume passing through each cell (for the Trade map mode).
"""
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from worldgen.generator import shift
from worldgen.hydrology import cell_river_flow

from ..core.simulation import System
from .geo import DIRECTIONS4, cell_scale, great_circle_km, touches
from .population import TRAVEL, _by_biome

GOODS = ("copper_near", "tin_near", "precious_near")


class Trade(System):
    name = "trade"

    def __init__(self):
        self.every = None

    # ----------------------------------------------------------------- setup
    def setup(self, state):
        cfg, world = state.config, state.world
        self.every = cfg.trade_every
        self.scale = cell_scale(world, cfg)
        self.world = world
        h, w = world.ocean.shape
        self.n = h * w
        self.xyz = np.stack(world.grid.xyz, axis=-1).reshape(-1, 3)
        land = ~world.ocean & ~world.lake
        self.land = land
        sea = world.ocean.copy()
        coast_sea = sea & touches(land)
        for _ in range(max(0, int(round(1 / self.scale)) - 1)):    # the same width in km on fine maps
            coast_sea |= sea & touches(coast_sea)
        self.coast_sea = coast_sea
        self.mask = land | coast_sea
        self.base = np.where(land, _by_biome(world.biome, TRAVEL, 1.5) * (1 + world.slope_m_per_km / 40), 1.0
                             ).astype(np.float32)
        self.river = land & (cell_river_flow(world) > 1000)
        kpp = world.grid.km_per_px
        cos_lat = np.maximum(np.cos(np.radians(world.grid.lat)), 0.05)
        idx = np.arange(self.n).reshape(h, w)
        east = self.mask & np.roll(self.mask, -1, axis=1)
        south = self.mask[:-1] & self.mask[1:]
        self.ea = np.concatenate([idx[east], idx[:-1][south]])
        self.eb = np.concatenate([np.roll(idx, -1, axis=1)[east], idx[1:][south]])
        self.ekm = np.concatenate([(kpp * cos_lat)[east], np.full(int(south.sum()), kpp)]).astype(np.float64)
        state.add_layer("trade", np.float32, record="snapshot")
        self.offer = np.zeros(self.n, np.uint64)        # knowledge arriving by trade, per cell
        self.offer_weight = np.zeros(self.n, np.float32)
        self.goods = {g: np.zeros(self.n, bool) for g in GOODS}    # goods arriving by trade
        state.tables["trade"] = self
        self.links = []
        self._stats(state, 0, 0.0, 0.0)

    # ------------------------------------------------------------------ step
    def step(self, state):
        cfg = state.config
        size = state.cells.get("settlement")
        if size is None:
            return
        towns = np.flatnonzero((size.ravel() >= cfg.trade_town_min) & self.mask.ravel())
        if len(towns) < 2:
            return
        tech = state.cells.get("tech")
        cost = self._costs(state, tech)
        wt = self.ekm * 0.5 * (cost[self.ea] + cost[self.eb])
        graph = coo_matrix((np.concatenate([wt, wt]), (np.concatenate([self.ea, self.eb]),
                                                      np.concatenate([self.eb, self.ea]))),
                           shape=(self.n, self.n)).tocsr()
        dist, pred, src = dijkstra(graph, directed=False, indices=towns, min_only=True,
                                   return_predecessors=True, limit=cfg.trade_link_cost)
        # Links: where two towns' lands meet, the cheapest crossing.
        sa, sb = src[self.ea], src[self.eb]
        ok = (sa >= 0) & (sb >= 0) & (sa != sb)
        if not ok.any():
            self._clear(state)
            return
        node_of = np.full(self.n, -1, np.int64)
        node_of[towns] = np.arange(len(towns))
        na, nb = node_of[sa[ok]], node_of[sb[ok]]
        lc = dist[self.ea[ok]] + wt[ok] + dist[self.eb[ok]]
        lo, hi = np.minimum(na, nb), np.maximum(na, nb)
        key = lo.astype(np.int64) * len(towns) + hi
        order = np.lexsort((lc, key))
        first = np.ones(len(order), bool)
        first[1:] = key[order][1:] != key[order][:-1]
        pick = order[first]
        link_a, link_b, link_cost = lo[pick], hi[pick], lc[pick]
        cross_a, cross_b = self.ea[ok][pick], self.eb[ok][pick]
        swap = na[pick] != link_a                        # crossing cells belong to (a, b) in order
        cross_a, cross_b = np.where(swap, cross_b, cross_a), np.where(swap, cross_a, cross_b)
        nt, nl = len(towns), len(link_a)
        # Flows between all towns within reach, carried along the cheapest chains of links.
        tgraph = coo_matrix((np.concatenate([link_cost, link_cost]),
                             (np.concatenate([link_a, link_b]), np.concatenate([link_b, link_a]))),
                            shape=(nt, nt)).tocsr()
        tdist, tpred = dijkstra(tgraph, directed=False, return_predecessors=True, limit=cfg.trade_reach)
        people = size.ravel()[towns].astype(np.float64) / 1e4
        d = np.maximum(tdist, cfg.trade_near) / 100.0
        flow = np.where(np.isfinite(tdist), people[:, None] * people[None, :] / d ** 2, 0.0)
        np.fill_diagonal(flow, 0.0)
        link_id = np.full((nt, nt), -1, np.int32)
        link_id[link_a, link_b] = np.arange(nl)
        link_id[link_b, link_a] = np.arange(nl)
        far = np.where(np.isfinite(tdist), tdist, np.inf)
        order = np.argsort(far, axis=1).astype(np.int32)
        volume = _accumulate(tpred.astype(np.int32), flow, link_id, order, nl)
        # Volume on the map: along each link's path, from each end back to its town.
        trade = np.zeros(self.n, np.float32)
        _paint(pred.astype(np.int64), cross_a.astype(np.int64), cross_b.astype(np.int64),
               volume.astype(np.float32), trade)
        state.cells["trade"][:] = trade.reshape(state.shape)
        # The longest distance goods travel (pairs that trade more than a trickle).
        xyz_t = self.xyz[towns]
        strong = flow >= cfg.trade_min_flow
        longest = 0.0
        if strong.any():
            i, j = np.nonzero(strong)
            longest = float(great_circle_km(xyz_t[i], xyz_t[j]).max())
        self._goods(state, towns, tdist, src, dist)
        self._knowledge(state, tech, towns, link_a, link_b, volume)
        self.links = [(int(towns[a]), int(towns[b]), float(v)) for a, b, v in zip(link_a, link_b, volume)]
        self._stats(state, nl, float(volume.sum()), longest)

    def _costs(self, state, tech):
        """Travel cost per cell (x km) with what each place knows."""
        cfg = state.config
        cost = self.base.ravel().copy()
        land = self.land.ravel()
        if tech is not None:
            from ..tech import bit
            t = tech.ravel()
            has = lambda key: (t & np.uint64(bit(key))) != 0
            cost = np.where(land & has("wheel"), cost * (1 - cfg.wheel_saving), cost)
            own = state.cells.get("state")
            if own is not None:
                cost = np.where(land & has("roads") & (own.ravel() > 0), cost * (1 - cfg.roads_saving), cost)
            sail = has("sailing")
            cost = np.where(self.river.ravel() & sail, np.minimum(cost, cfg.river_cost), cost)
            # Coastal waters: by sail where the shore's people sail, else paddling.
            sail_grid = sail.reshape(state.shape)
            shore = np.zeros(state.shape, bool)
            for dy, dx in DIRECTIONS4:
                shore |= shift(sail_grid, dy, dx)
            sea = self.coast_sea.ravel()
            cost = np.where(sea, np.where(shore.ravel(), cfg.sea_cost_sail, cfg.sea_cost), cost)
        else:
            cost = np.where(self.coast_sea.ravel(), cfg.sea_cost, cost)
        return cost.astype(np.float64)

    def _goods(self, state, towns, tdist, src, dist):
        """Copper, tin, gold and silver reach towns within `goods_reach` of a town that has
        them, and the lands around those towns."""
        cfg = state.config
        system = state.tables.get("technology")
        if system is None:
            return
        region = (src >= 0) & np.isfinite(dist)
        node_of = np.full(self.n, -1, np.int64)
        node_of[towns] = np.arange(len(towns))
        cell_node = np.where(region, node_of[np.maximum(src, 0)], -1)
        for g in GOODS:
            have = system.static[g][towns]
            if not have.any():
                self.goods[g][:] = False
                continue
            reach = tdist[have].min(axis=0) <= cfg.goods_reach          # towns that get it by trade
            self.goods[g] = (cell_node >= 0) & reach[np.maximum(cell_node, 0)]

    def _knowledge(self, state, tech, towns, link_a, link_b, volume):
        """Each link offers what one end knows to the other, more strongly the busier it is."""
        cfg = state.config
        self.offer[:] = 0
        self.offer_weight[:] = 0
        if tech is None:
            return
        t = tech.ravel()
        weight = (volume / (volume + cfg.trade_half)).astype(np.float32)
        ca, cb = towns[link_a], towns[link_b]
        np.bitwise_or.at(self.offer, cb, t[ca])
        np.bitwise_or.at(self.offer, ca, t[cb])
        np.maximum.at(self.offer_weight, cb, weight)
        np.maximum.at(self.offer_weight, ca, weight)

    def _clear(self, state):
        state.cells["trade"][:] = 0
        self.offer[:] = 0
        self.offer_weight[:] = 0
        for g in GOODS:
            self.goods[g][:] = False
        self.links = []
        self._stats(state, 0, 0.0, 0.0)

    def busy(self, state):
        """0..1 per cell: how busy a place's trade is (half at `trade_half`)."""
        t = state.cells["trade"]
        return t / (t + state.config.trade_half)

    def _stats(self, state, links, volume, longest):
        state.stats.update({"trade_links": links, "trade_volume": volume, "longest_trade_km": longest})


@njit(cache=True)
def _accumulate(pred, flow, link_id, order, nl):
    """Volume on each link: every source's flows pushed back along its shortest-path tree."""
    nt = pred.shape[0]
    volume = np.zeros(nl)
    acc = np.zeros(nt)
    for s in range(nt):
        for j in range(nt):
            acc[j] = flow[s, j]
        for k in range(nt - 1, -1, -1):
            j = order[s, k]
            p = pred[s, j]
            if j == s or p < 0:
                continue
            l = link_id[p, j]
            if l >= 0:
                volume[l] += acc[j]
            acc[p] += acc[j]
    return volume


@njit(cache=True)
def _paint(pred, cross_a, cross_b, volume, out):
    """Add each link's volume along its path: from each crossing cell back to its town."""
    for l in range(len(cross_a)):
        v = volume[l]
        for start in (cross_a[l], cross_b[l]):
            c = start
            steps = 0
            while c >= 0 and steps < 100000:
                out[c] += v
                c = pred[c]
                steps += 1
