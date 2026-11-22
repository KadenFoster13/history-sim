"""The history viewer: watch a history live as it is simulated, or replay a saved run.

    python -m histsim watch --world 31 --seed 1          simulate and watch (add --save to keep it)
    python -m histsim replay runs/world31_seed1          replay a saved run

Controls
  Play / Pause (or Space)         run history forward
  Slower / Faster (or - / +)      years per frame
  Step (or Right arrow)           one jump forward while paused
  Year slider (replay)            jump to any year
  Map buttons on the right        people, density, cultures, language families, background only, arrival
  Background buttons              draw maps over terrain, biomes or plain white (remembered)
  Scroll to zoom around the cursor, drag to move, R to reset the view
  Dark / Light (or D)             the same remembered setting as the world viewer
  S                               save the current view as a PNG in renders/
  Hover over the map              what is there: people, way of life, culture, language, town
"""
import os
import textwrap
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from ..analysis.maps import (BACKGROUNDS, MODE_COLOURS, TOWN_MIN, arrival_map, base_map, culture_colours,
                             culture_map, density_map, family_map, people_map)
from ..core.calendar import label, real_year
from ..systems.population import FARMING, FORAGING, HERDING

MODES = {"People": people_map, "Density": density_map, "Cultures": culture_map,
         "Language families": family_map, "Background only": None, "Arrival": arrival_map}
BACKGROUND_LABELS = {"Terrain": "terrain", "Biomes": "biomes", "White": "white"}
SPEEDS = [1, 5, 10, 25, 50, 100, 250, 500]          # years per frame
WAYS = {0: "nobody", FORAGING: "foragers", HERDING: "herders", FARMING: "farmers"}
FRAME_SECONDS = 0.35                                 # most time spent simulating per frame (live)
FIG_W, FIG_H = 15.5, 8.6                             # designed window size, inches (text scales from it)


def _theme_tools():
    """The world viewer's themes and remembered preference (one setting for both viewers)."""
    try:
        from worldgen.viewer import THEMES, _load_preferences, _save_preferences
        return THEMES, _load_preferences, _save_preferences
    except ImportError:   # an older worldgen without dark mode
        themes = {"light": {"bg": "white", "fg": "black", "button": "0.85", "hover": "0.95",
                            "active": "#9cc3e6", "edge": "0.25"}}
        themes["dark"] = themes["light"]
        return themes, lambda: {}, lambda prefs: None


# ------------------------------------------------------------------ sources
class LiveSource:
    """A simulation being run as it is watched."""

    live = True

    def __init__(self, sim, recorder=None):
        self.sim = sim
        self.recorder = recorder
        self.recent = deque(maxlen=60)
        sim.add_observer(lambda state: self.recent.extend(state.events))
        if not sim._ready:
            sim._setup()

    @property
    def year(self):
        return self.sim.state.year

    first_year = property(lambda self: self.sim.start_year)
    last_year = None

    def layers(self):
        return self.sim.state.cells

    def tables(self):
        return self.sim.state.tables

    def stats(self):
        return self.sim.state.stats

    def events(self):
        return list(self.recent)

    def advance(self, years, budget=FRAME_SECONDS):
        """Simulate up to `years` years, stopping early when the frame's time is used up."""
        target = self.sim.state.year + years
        t0 = time.perf_counter()
        while self.sim.state.year < target and time.perf_counter() - t0 < budget:
            step = max(1, getattr(self.sim.systems[-1], "every", 1) or 1)
            self.sim.run(min(target, self.sim.state.year + step))
        if self.recorder is not None and self.sim.state.year % 500 < years:
            self.recorder.checkpoint()

    def close(self):
        if self.recorder is not None:
            self.recorder.close()


class ReplaySource:
    """A saved run, played back."""

    live = False

    def __init__(self, run):
        self.run = run
        self.first_year, self.last_year = run.first_year, run.last_year
        self._stats = run.stats()
        self._events = run.events()
        self._event_years = np.array([e["year"] for e in self._events])
        self._tables = run.tables()
        self.year = self.first_year
        self._layers = None
        self.seek(self.first_year)

    def seek(self, year):
        self.year = int(np.clip(year, self.first_year, self.last_year))
        self._layers = self.run.layers_at(self.year)

    def advance(self, years, budget=None):
        self.seek(self.year + years)

    def layers(self):
        return self._layers

    def tables(self):
        return self._tables

    def stats(self):
        years = self._stats["year"]
        i = int(np.clip(np.searchsorted(years, self.year, side="right") - 1, 0, len(years) - 1))
        return {k: v[i] for k, v in self._stats.items()}

    def events(self):
        n = int(np.searchsorted(self._event_years, self.year, side="right"))
        return self._events[max(0, n - 60):n]

    def close(self):
        pass


# ------------------------------------------------------------------ viewer
class HistoryViewer:
    def __init__(self, world, source, title=""):
        self.world = world
        self.source = source
        self.title = title
        self.mode = "People"
        self.playing = False
        self.speed_i = SPEEDS.index(25)
        self.view = None
        self.drag = None
        self.themes, self._load_prefs, self._save_prefs = _theme_tools()
        self.dark = bool(self._load_prefs().get("dark", False))
        self.background = self._load_prefs().get("history_background", "terrain")
        if self.background not in BACKGROUNDS:
            self.background = "terrain"
        self._labels = []
        self._panel = []           # texts and patches of the info panel and legend (redrawn)
        self.scale = 1.0           # text size factor, follows the window size

        self.fig = plt.figure(figsize=(FIG_W, FIG_H))
        self.fig.canvas.manager.set_window_title("History viewer")
        self.ax = self.fig.add_axes([0.015, 0.16, 0.74, 0.76])
        self.ax.set_axis_off()
        self.ax.format_coord = self._describe
        self.image = self.ax.imshow(base_map(world), interpolation="nearest")
        self.towns = self.ax.scatter([], [], s=[], c="white", edgecolors="black", linewidths=0.6, zorder=3)
        h, w = world.ocean.shape
        self.full_view = (-0.5, w - 0.5, h - 0.5, -0.5)
        self._build_controls()
        self._build_side_panel()
        self._capture_sizes()
        self.apply_theme()

        c = self.fig.canvas
        c.mpl_connect("key_press_event", self._on_key)
        c.mpl_connect("scroll_event", self._on_scroll)
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("close_event", lambda _: self.source.close())
        c.mpl_connect("resize_event", self._on_resize)
        self.timer = c.new_timer(interval=30)
        self.timer.add_callback(self._tick)
        self.timer.start()
        self.redraw()

    # ---------------------------------------------------------------- layout
    def _build_controls(self):
        y, h = 0.05, 0.045
        self.btn_play = Button(self.fig.add_axes([0.02, y, 0.07, h]), "Play")
        self.btn_slow = Button(self.fig.add_axes([0.10, y, 0.06, h]), "Slower")
        self.btn_fast = Button(self.fig.add_axes([0.165, y, 0.06, h]), "Faster")
        self.btn_step = Button(self.fig.add_axes([0.23, y, 0.06, h]), "Step")
        self.speed_text = self.fig.text(0.30, y + 0.015, "", fontsize=10)
        self.btn_play.on_clicked(lambda _: self.toggle_play())
        self.btn_slow.on_clicked(lambda _: self.change_speed(-1))
        self.btn_fast.on_clicked(lambda _: self.change_speed(+1))
        self.btn_step.on_clicked(lambda _: self.step())
        self.slider = None
        if not self.source.live:
            ax = self.fig.add_axes([0.42, y + 0.01, 0.30, 0.025])
            self.slider = Slider(ax, "Year", self.source.first_year, self.source.last_year,
                                 valinit=self.source.year, valstep=1)
            self.slider.valtext.set_visible(False)
            self.slider.on_changed(self._on_slider)
            self._slider_busy = False
        self.btn_theme = Button(self.fig.add_axes([0.90, 0.94, 0.08, 0.04]), "Light" if self.dark else "Dark")
        self.btn_theme.on_clicked(lambda _: self.toggle_dark())

    def _build_side_panel(self):
        x = 0.775
        radio_ax = self.fig.add_axes([x, 0.71, 0.115, 0.2])
        radio_ax.set_title("Map", fontsize=11, loc="left", weight="bold")
        self.radio = RadioButtons(radio_ax, list(MODES), active=0, label_props={"fontsize": [9.5]},
                                  radio_props={"s": [60]})
        self.radio.on_clicked(self._on_mode)
        bg_ax = self.fig.add_axes([x + 0.125, 0.79, 0.075, 0.12])
        bg_ax.set_title("Background", fontsize=11, loc="left", weight="bold")
        labels = list(BACKGROUND_LABELS)
        active = list(BACKGROUND_LABELS.values()).index(self.background)
        self.bg_radio = RadioButtons(bg_ax, labels, active=active, label_props={"fontsize": [9.5]},
                                     radio_props={"s": [60]})
        self.bg_radio.on_clicked(self._on_background)
        self.info_ax = self.fig.add_axes([x, 0.30, 0.205, 0.39])     # a two-column table
        self.info_ax.set_axis_off()
        self.events_text = self.fig.text(x, 0.275, "", fontsize=8, va="top", wrap=False)
        self.legend_ax = self.fig.add_axes([0.015, 0.105, 0.74, 0.045])
        self.legend_ax.set_axis_off()

    # ------------------------------------------------------------ window size
    def fs(self, size):
        """A font size scaled to the window."""
        return size * self.scale

    def _capture_sizes(self):
        """Remember the designed size of the fixed text and radio markers, to scale them later."""
        from matplotlib.text import Text
        self._fixed_texts = [(t, t.get_fontsize()) for t in self.fig.findobj(Text) if t.get_text()]
        self._markers = []
        for radio in (self.radio, self.bg_radio):
            coll = getattr(radio, "_buttons", None)
            if coll is not None and hasattr(coll, "get_sizes"):
                self._markers.append((coll, np.array(coll.get_sizes(), float)))

    def _on_resize(self, event):
        w_in, h_in = self.fig.get_size_inches()
        scale = float(np.clip(min(w_in / FIG_W, h_in / FIG_H), 0.5, 3))
        if abs(scale - self.scale) < 0.01:
            return
        self.scale = scale
        for text, size in self._fixed_texts:
            text.set_fontsize(size * scale)
        for coll, sizes in self._markers:
            coll.set_sizes(sizes * scale ** 2)    # marker sizes are areas
        self.redraw()

    # ---------------------------------------------------------------- drawing
    def redraw(self):
        src = self.source
        layers = src.layers()
        tables = src.tables()
        draw = MODES[self.mode]
        img = (base_map(self.world, self.background) if draw is None
               else draw(self.world, layers, tables, background=self.background))
        self.image.set_data(img)
        self._draw_towns(layers, tables)
        self._draw_culture_labels(layers, tables)
        stats = src.stats()
        pop = float(stats.get("world_population", 0))
        self.ax.set_title(f"{self.title}   |   {label(src.year)}   |   {_count(pop)} people",
                          fontsize=self.fs(12), color=self.theme["fg"])
        self.speed_text.set_text(f"{SPEEDS[self.speed_i]} years per frame"
                                 + ("   (live)" if src.live else ""))
        self.speed_text.set_fontsize(self.fs(10))
        for artist in self._panel:
            artist.remove()
        self._panel = []
        self._draw_info(layers, tables, stats)
        self._draw_legend(layers, tables)
        self.events_text.set_text(self._events_text())
        self.events_text.set_fontsize(self.fs(8))
        if self.slider is not None and int(self.slider.val) != src.year:
            self._slider_busy = True
            self.slider.set_val(src.year)
            self._slider_busy = False
        self.fig.canvas.draw_idle()

    def _draw_towns(self, layers, tables):
        for t in self._labels:
            t.remove()
        self._labels = []
        self._placed = []
        size = layers.get("settlement")
        if size is None:
            self.towns.set_offsets(np.empty((0, 2)))
            return
        rows, cols = np.nonzero(size >= TOWN_MIN)
        s = size[rows, cols]
        self.towns.set_offsets(np.column_stack([cols, rows]) if len(rows) else np.empty((0, 2)))
        self.towns.set_sizes(6 + 14 * np.log10(s / TOWN_MIN + 1) ** 2)
        ids = layers.get("settlement_id")
        places = tables.get("settlements") or []
        if ids is None or not len(rows):
            return
        self._placed = []
        for k in np.argsort(-s)[:10]:
            r, c = rows[k], cols[k]
            if not self._room(c, r):
                continue           # keep labels from piling up
            sid = int(ids[r, c])
            if 0 < sid < len(places) and places[sid]:
                self._placed.append((c, r))
                self._labels.append(self.ax.annotate(
                    places[sid]["name"], (c, r), xytext=(4, 3), textcoords="offset points", fontsize=self.fs(8),
                    color="black", zorder=5,
                    bbox={"boxstyle": "round,pad=0.15", "fc": (1, 1, 1, 0.65), "ec": "none"}))

    def _draw_culture_labels(self, layers, tables):
        if self.mode not in ("Cultures", "Language families"):
            return
        cult = layers.get("culture")
        records = tables.get("cultures")
        if cult is None or not records:
            return
        pop = layers.get("population")
        flat = cult.ravel()
        n = len(records)
        weight = (pop.ravel() if pop is not None else np.ones(flat.size)) + 1e-6
        people = np.bincount(flat, weights=weight, minlength=n)
        h, w = cult.shape
        rows, cols = np.divmod(np.arange(flat.size), w)
        angle = cols / w * 2 * np.pi
        cx = np.bincount(flat, weights=weight * np.cos(angle), minlength=n)
        cy = np.bincount(flat, weights=weight * np.sin(angle), minlength=n)
        cr = np.bincount(flat, weights=weight * rows, minlength=n) / np.maximum(people, 1e-9)
        ccol = (np.arctan2(cy, cx) % (2 * np.pi)) / (2 * np.pi) * w
        if self.mode == "Language families":
            fam = {}
            for cid in np.flatnonzero(people[1:] > 1) + 1:
                f = records[cid]["family"]
                fam.setdefault(f, []).append(cid)
            groups = sorted(fam.values(), key=lambda ids: -sum(people[i] for i in ids))[:10]
            langs = tables.get("languages") or []
            for ids in groups:
                lead = max(ids, key=lambda i: people[i])
                f = records[lead]["family"]
                name = langs[f].get("family_name") if f < len(langs) and langs[f] else None
                self._culture_label(ccol[lead], cr[lead], name or records[lead]["name"])
            return
        for cid in np.argsort(-people[1:])[:12] + 1:
            if people[cid] > 1:
                self._culture_label(ccol[cid], cr[cid], records[cid]["name"])

    def _room(self, col, row):
        """Is there space for a label here (none too close on screen)?"""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        dx, dy = abs(x1 - x0) * 0.075, abs(y0 - y1) * 0.045
        return all(abs(col - pc) > dx or abs(row - pr) > dy for pc, pr in self._placed)

    def _culture_label(self, col, row, text):
        if not self._room(col, row):
            return
        self._placed.append((col, row))
        self._labels.append(self.ax.text(col, row, text, fontsize=self.fs(8.5), ha="center", va="center",
                                         weight="bold", color="black", zorder=4,
                                         bbox={"boxstyle": "round,pad=0.2", "fc": (1, 1, 1, 0.55), "ec": "none"}))

    def _draw_info(self, layers, tables, stats):
        """The side table: names on the left, numbers lined up on the right."""
        pop = float(stats.get("world_population", 0))
        g = lambda k: stats.get(k, 0) or 0
        rows = [("People", _count(pop), "row"),
                ("farmers", _share(g("farmers"), pop), "sub"),
                ("herders", _share(g("herders"), pop), "sub"),
                ("foragers", _share(g("foragers"), pop), "sub"),
                ("Cultures", f"{int(g('cultures')):,}", "row"),
                ("Language families", f"{int(g('language_families')):,}", "row"),
                ("Towns (2,000+)", f"{int(g('towns')):,}", "row"),
                ("Cities (10,000+)", f"{int(g('cities')):,}", "row"),
                ("Living in towns", _share(g("urban_share") * pop, pop), "row")]
        cult = layers.get("culture")
        records = tables.get("cultures")
        if cult is not None and records:
            p = layers.get("population")
            people = np.bincount(cult.ravel(), weights=p.ravel(), minlength=len(records))
            top = [cid for cid in np.argsort(-people[1:])[:5] + 1 if people[cid] > 0]
            if top:
                rows += [("", "", "gap"), ("Largest peoples", "", "head")]
                rows += [(records[cid]["name"], _count(people[cid]), "sub") for cid in top]
        size = layers.get("settlement")
        ids = layers.get("settlement_id")
        places = tables.get("settlements") or []
        if size is not None and ids is not None and places:
            top = np.argsort(-size.ravel())[:5]
            towns = [(places[int(ids.flat[i])]["name"], _count(size.flat[i]), "sub") for i in top
                     if size.flat[i] >= TOWN_MIN and 0 < int(ids.flat[i]) < len(places) and places[int(ids.flat[i])]]
            if towns:
                rows += [("", "", "gap"), ("Largest towns", "", "head")] + towns
        ax, fg = self.info_ax, self.theme["fg"]
        step = 1 / 23
        y = 1.0
        for name, value, kind in rows:
            if kind == "gap":
                y -= step * 0.5
                continue
            weight = "bold" if kind == "head" else "normal"
            x = 0.05 if kind == "sub" else 0.0
            self._panel.append(ax.text(x, y, name, transform=ax.transAxes, fontsize=self.fs(8.5), va="top",
                                       ha="left", color=fg, weight=weight, clip_on=False))
            if value:
                self._panel.append(ax.text(1.0, y, value, transform=ax.transAxes, fontsize=self.fs(8.5),
                                           va="top", ha="right", color=fg, clip_on=False))
            y -= step

    # ---------------------------------------------------------------- legends
    def _draw_legend(self, layers, tables):
        """Colour keys under the map: swatches for kinds of things, a scale for amounts."""
        ax, fg = self.legend_ax, self.theme["fg"]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        town = ("dot", "Town (bigger = more people)")
        if self.mode == "People":
            names = {1: "Foragers", 2: "Herders", 3: "Farmers"}
            self._swatches([(MODE_COLOURS[k], names[k]) for k in (1, 2, 3)] + [town],
                           note="stronger colour = more people")
        elif self.mode == "Density":
            ticks = [(0.0, "0.001"), (0.2, "0.01"), (0.4, "0.1"), (0.6, "1"), (0.8, "10"), (1.0, "100")]
            self._scale_bar("inferno", ticks, "people per km\u00b2")
        elif self.mode == "Arrival":
            arrival = layers.get("arrival_year")
            reached = arrival[np.isfinite(arrival)] if arrival is not None else np.array([])
            if reached.size:
                lo, hi = float(reached.min()), min(float(reached.max()), 0.0)
                ticks = [(f, _bce(lo + (hi - lo) * f)) for f in (0, 0.25, 0.5, 0.75, 1)]
                self._scale_bar("viridis", ticks, "when people first arrived",
                                extra=[((0.6, 0.6, 0.6), "Not reached yet")])
        elif self.mode in ("Cultures", "Language families"):
            self._swatches(self._group_swatches(layers, tables) + [town],
                           note="related peoples have similar colours" if self.mode == "Cultures" else "")
        else:
            items = self._biome_swatches() if self.background == "biomes" else []
            self._swatches([town] + items)

    def _swatches(self, items, note=""):
        """A row of coloured squares with names (and the town dot)."""
        ax, fg = self.legend_ax, self.theme["fg"]
        char = 0.0068                    # width of a character, as a share of the legend's width
        x = 0.0
        for colour, name in items:
            width = 0.022 + char * len(name) + 0.03
            if x + width > 1.0:
                break                    # no room for more
            if colour == "dot":
                self._panel.append(ax.scatter([x + 0.008], [0.5], s=40 * self.scale ** 2, c="white",
                                              edgecolors="black", linewidths=0.6, transform=ax.transAxes,
                                              clip_on=False))
            else:
                from matplotlib.patches import Rectangle
                self._panel.append(ax.add_patch(Rectangle((x, 0.2), 0.016, 0.6, transform=ax.transAxes,
                                                          facecolor=colour, edgecolor=self.theme["edge"],
                                                          linewidth=0.5, clip_on=False)))
            self._panel.append(ax.text(x + 0.022, 0.5, name, transform=ax.transAxes, va="center",
                                       fontsize=self.fs(9), color=fg, clip_on=False))
            x += width
        if note and x + char * len(note) <= 1.0:
            self._panel.append(ax.text(x, 0.5, note, transform=ax.transAxes, va="center", ha="left",
                                       fontsize=self.fs(8.5), color=fg, style="italic", clip_on=False))

    def _scale_bar(self, cmap, ticks, title, extra=()):
        """A colour scale from left to right, with labels under it."""
        from matplotlib import colormaps
        ax, fg = self.legend_ax, self.theme["fg"]
        x0, x1 = 0.0, 0.55
        gradient = colormaps[cmap](np.linspace(0, 1, 256))[None, :, :3]
        self._panel.append(ax.imshow(gradient, extent=(x0, x1, 0.45, 1.0), aspect="auto",
                                     transform=ax.transAxes, clip_on=False))
        for f, text in ticks:
            align = "left" if f == 0 else "right" if f == 1 else "center"     # stay inside the bar's ends
            self._panel.append(ax.text(x0 + (x1 - x0) * f, 0.3, text, transform=ax.transAxes, va="top",
                                       ha=align, fontsize=self.fs(8.5), color=fg, clip_on=False))
        self._panel.append(ax.text(x1 + 0.02, 0.72, title, transform=ax.transAxes, va="center",
                                   fontsize=self.fs(9), color=fg, clip_on=False))
        from matplotlib.patches import Rectangle
        for i, (colour, name) in enumerate(extra):
            x = 0.8 + i * 0.12
            self._panel.append(ax.add_patch(Rectangle((x, 0.45), 0.016, 0.55, transform=ax.transAxes,
                                                      facecolor=colour, edgecolor=self.theme["edge"],
                                                      linewidth=0.5, clip_on=False)))
            self._panel.append(ax.text(x + 0.022, 0.72, name, transform=ax.transAxes, va="center",
                                       fontsize=self.fs(9), color=fg, clip_on=False))

    def _group_swatches(self, layers, tables, n=6):
        """The largest peoples (or language families) with their map colours."""
        cult = layers.get("culture")
        records = tables.get("cultures")
        if cult is None or not records:
            return []
        p = layers.get("population")
        people = np.bincount(cult.ravel(), weights=p.ravel(), minlength=len(records))
        if self.mode == "Cultures":
            colours = culture_colours(tables)
            return [(tuple(colours[c]), records[c]["name"]) for c in np.argsort(-people[1:])[:n] + 1
                    if people[c] > 0]
        colours = culture_colours(tables, by_family=True)
        langs = tables.get("languages") or []
        totals = {}
        for c in np.flatnonzero(people[1:] > 0) + 1:
            f = records[c]["family"]
            best = totals.get(f, (0.0, c))
            totals[f] = (best[0] + people[c], c if people[c] > people[best[1]] else best[1])
        out = []
        for f, (_, lead) in sorted(totals.items(), key=lambda kv: -kv[1][0])[:n]:
            name = langs[f].get("family_name") if f < len(langs) and langs[f] else None
            out.append((tuple(colours[lead]), (name or records[lead]["name"]).replace(" languages", "")))
        return out

    def _biome_swatches(self, n=7):
        """The commonest biomes, in the (muted) colours of the biomes background."""
        from matplotlib.colors import to_rgb
        from worldgen.generator import BIOMES
        land = ~self.world.ocean & ~self.world.lake
        counts = np.bincount(self.world.biome[land].ravel(), minlength=len(BIOMES))
        out = []
        for i in np.argsort(-counts)[:n]:
            if counts[i] > 0:
                name, colour = BIOMES[i]
                out.append((tuple(np.clip(np.array(to_rgb(colour)) * 0.55 + 0.42, 0, 1)), name))
        return out

    def _events_text(self):
        events = [e for e in self.source.events() if e.get("kind") != "environment"][-6:]
        if not events:
            return ""
        lines = ["Recent events"]
        for e in reversed(events):
            wrapped = textwrap.wrap(f"{label(e['year'])}: {e['text']}", 50)
            if len(lines) + len(wrapped) > 14:
                break
            lines += wrapped
        return "\n".join(lines)

    # ---------------------------------------------------------------- playing
    def _tick(self):
        if not self.playing:
            return
        if not self.source.live and self.source.year >= self.source.last_year:
            self.toggle_play()
            return
        self.source.advance(SPEEDS[self.speed_i])
        self.redraw()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def change_speed(self, d):
        self.speed_i = int(np.clip(self.speed_i + d, 0, len(SPEEDS) - 1))
        self.redraw()

    def step(self):
        self.source.advance(SPEEDS[self.speed_i], budget=30)
        self.redraw()

    def _on_slider(self, value):
        if getattr(self, "_slider_busy", False):
            return
        self.source.seek(int(value))
        self.redraw()

    def _on_background(self, label_):
        if self.bg_radio.value_selected != label_:
            self.bg_radio.set_active(list(BACKGROUND_LABELS).index(label_))   # calls back here
            return
        self.background = BACKGROUND_LABELS[label_]
        self._save_prefs({**self._load_prefs(), "history_background": self.background})
        self.redraw()

    def _on_mode(self, label_):
        self.mode = label_
        if self.radio.value_selected != label_:
            self.radio.set_active(list(MODES).index(label_))   # calls back here
            return
        self.redraw()

    # ------------------------------------------------------------------ theme
    @property
    def theme(self):
        return self.themes["dark" if self.dark else "light"]

    def toggle_dark(self):
        self.dark = not self.dark
        self.btn_theme.label.set_text("Light" if self.dark else "Dark")
        self._save_prefs({**self._load_prefs(), "dark": self.dark})
        self.apply_theme()
        self.redraw()

    def apply_theme(self):
        from matplotlib.text import Text
        t = self.theme
        self.fig.patch.set_facecolor(t["bg"])
        for text in self.fig.findobj(Text):
            if text not in self._labels:
                text.set_color(t["fg"])
        for ax in self.fig.axes:
            ax.set_facecolor(t["bg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(t["edge"])
        for b in (self.btn_play, self.btn_slow, self.btn_fast, self.btn_step, self.btn_theme):
            b.color, b.hovercolor = t["button"], t["hover"]
            b.ax.set_facecolor(t["button"])
        self.radio.set_radio_props({"edgecolor": t["fg"], "facecolor": t["fg"]})
        self.bg_radio.set_radio_props({"edgecolor": t["fg"], "facecolor": t["fg"]})
        if self.slider is not None:
            self.slider.poly.set_facecolor(t["active"])
            self.slider.ax.set_facecolor(t["button"])

    # ------------------------------------------------------------ zoom & pan
    def _set_view(self, x0, x1, y0, y1):
        fx0, fx1, fy0, fy1 = self.full_view
        wd, ht = min(x1 - x0, fx1 - fx0), min(y0 - y1, fy0 - fy1)
        x0 = float(np.clip(x0, fx0, fx1 - wd))
        y1 = float(np.clip(y1, fy1, fy0 - ht))
        self.ax.set_xlim(x0, x0 + wd)
        self.ax.set_ylim(y1 + ht, y1)
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes is not self.ax:
            return
        f = 1 / 1.3 if event.button == "up" else 1.3
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        fx0, fx1, fy0, fy1 = self.full_view
        wd = float(np.clip((x1 - x0) * f, (fx1 - fx0) / 32, fx1 - fx0))
        ht = wd / 2
        rx = (cx - x0) / (x1 - x0)
        ry = (cy - y1) / (y0 - y1)
        nx0 = cx - rx * wd
        ny1 = cy - ry * ht
        self._set_view(nx0, nx0 + wd, ny1 + ht, ny1)

    def _on_press(self, event):
        if event.inaxes is self.ax and event.button == 1:
            self.drag = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def _on_release(self, event):
        self.drag = None

    def _on_motion(self, event):
        if self.drag is None:
            return
        x, y, (x0, x1), (y0, y1) = self.drag
        bbox = self.ax.get_window_extent()
        dx = (event.x - x) / bbox.width * (x1 - x0)
        dy = (event.y - y) / bbox.height * (y0 - y1)
        self._set_view(x0 - dx, x1 - dx, y0 + dy, y1 + dy)

    def reset_view(self):
        self._set_view(*self.full_view)

    # ------------------------------------------------------------------- keys
    def _on_key(self, event):
        k = event.key
        if k == " ":
            self.toggle_play()
        elif k == "right":
            self.step()
        elif k in ("+", "="):
            self.change_speed(+1)
        elif k == "-":
            self.change_speed(-1)
        elif k in ("r", "R"):
            self.reset_view()
        elif k in ("d", "D"):
            self.toggle_dark()
        elif k in ("s", "S"):
            print(f"saved {self.save()}")

    def save(self, path=None):
        os.makedirs("renders", exist_ok=True)
        path = path or os.path.join("renders", f"history_{self.mode.lower().replace(' ', '_')}_{self.source.year}.png")
        self.fig.savefig(path, dpi=110, facecolor=self.fig.get_facecolor())
        return path

    # ------------------------------------------------------------------ hover
    def _describe(self, x, y):
        h, w = self.world.ocean.shape
        c, r = int(round(x)) % w, int(round(y))
        if not 0 <= r < h:
            return ""
        lat, lon = float(self.world.grid.lat[r, c]), float(self.world.grid.lon[r, c])
        parts = [f"{abs(lat):.1f}{'N' if lat >= 0 else 'S'} {abs(lon):.1f}{'E' if lon >= 0 else 'W'}"]
        layers = self.source.layers()
        tables = self.source.tables()
        pop = layers.get("population")
        if pop is not None and pop[r, c] > 0:
            area = self.world.grid.km_per_px ** 2 * max(np.cos(np.radians(lat)), 0.01)
            way = WAYS.get(int(layers.get("subsistence", np.zeros_like(pop))[r, c]), "")
            parts.append(f"{_count(pop[r, c])} {way} ({pop[r, c] / area:.2f}/km2)")
            cult = layers.get("culture")
            records = tables.get("cultures")
            if cult is not None and records and 0 < cult[r, c] < len(records) and records[cult[r, c]]:
                rec = records[cult[r, c]]
                langs = tables.get("languages") or []
                fam = langs[rec["family"]].get("family_name", "") if rec["family"] < len(langs) else ""
                parts.append(f"the {rec['name']} ({fam})" if fam else f"the {rec['name']}")
        elif not (self.world.ocean[r, c] or self.world.lake[r, c]):
            parts.append("nobody lives here")
        size = layers.get("settlement")
        ids = layers.get("settlement_id")
        places = tables.get("settlements") or []
        if size is not None and ids is not None and size[r, c] >= TOWN_MIN and 0 < ids[r, c] < len(places):
            parts.append(f"town: {places[ids[r, c]]['name']} ({_count(size[r, c])})")
        return "   |   ".join(parts)


# ------------------------------------------------------------------ helpers
def _count(n):
    n = float(n)
    if n >= 1e9:
        return f"{n / 1e9:.2f} billion"
    if n >= 1e6:
        return f"{n / 1e6:.1f} million"
    if n >= 1e4:
        return f"{n / 1e3:.0f} thousand"
    return f"{n:,.0f}"


def _share(part, total):
    return f"{part / total:.0%}" if total > 0 else "-"


def _bce(sim_year):
    """A simulation year before year 0 as a round 'N BCE'."""
    years_bce = 1 - real_year(sim_year)
    step = 1000 if years_bce >= 10_000 else 100
    return f"{round(years_bce / step) * step:,.0f} BCE"


def _legend(mode):
    return {
        "People": "gold = foragers, orange = herders, crimson = farmers (stronger = more people); dots = towns",
        "Density": "people per km2, log scale (dark = few, bright = many); dots = towns",
        "Cultures": "each culture in its own colour, related cultures in similar colours; dots = towns",
        "Language families": "cultures coloured by language family; dots = towns",
        "Background only": "the land alone (choose terrain, biomes or white on the right); dots = towns",
        "Arrival": "when people first arrived (dark = earliest, light = latest, grey = not yet)",
    }[mode]


def watch(world, sim, title="", recorder=None):
    """Open the viewer on a live simulation."""
    viewer = HistoryViewer(world, LiveSource(sim, recorder), title)
    plt.show()
    return viewer


def replay(world, run, title=""):
    """Open the viewer on a saved run."""
    viewer = HistoryViewer(world, ReplaySource(run), title)
    plt.show()
    return viewer
