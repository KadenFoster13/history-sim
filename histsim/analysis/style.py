"""Light and dark styles for graphs and maps (dark mode follows the viewers' setting)."""

LIGHT = {"bg": "white", "fg": "black", "grid": "0.75", "line": "#1f5fa8", "target": "0.55"}
DARK = {"bg": "#1d1f23", "fg": "#e4e6ea", "grid": "#4a4f57", "line": "#7fb6ff", "target": "0.6"}


def prefers_dark():
    """The dark/light choice remembered by the viewers (~/.worldgen_viewer.json)."""
    try:
        from worldgen.viewer import _load_preferences
        return bool(_load_preferences().get("dark", False))
    except ImportError:
        return False


def preferred_background():
    """The map background last chosen in the history viewer (terrain, biomes or white)."""
    try:
        from worldgen.viewer import _load_preferences
        value = _load_preferences().get("history_background", "terrain")
    except ImportError:
        value = "terrain"
    return value if value in ("terrain", "biomes", "white") else "terrain"


def palette(dark=None):
    if dark is None:
        dark = prefers_dark()
    return DARK if dark else LIGHT


def style_figure(fig, axes, colours):
    """Colour a figure's background, text, ticks and frames."""
    fig.patch.set_facecolor(colours["bg"])
    for ax in axes:
        ax.set_facecolor(colours["bg"])
        ax.tick_params(colors=colours["fg"], which="both")
        ax.title.set_color(colours["fg"])
        ax.xaxis.label.set_color(colours["fg"])
        ax.yaxis.label.set_color(colours["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(colours["grid"])
    if fig._suptitle is not None:
        fig._suptitle.set_color(colours["fg"])
