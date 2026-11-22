"""Loading the world a history runs on."""
import worldgen

EXPECTED_DATA_VERSION = 2   # the worldgen data layout this simulator is written for (see DATA.md)


def load_world(seed, width=800, cache_dir="world_cache"):
    """Generate (or load from the cache) the worldgen world for `seed`."""
    if worldgen.DATA_VERSION != EXPECTED_DATA_VERSION:
        raise RuntimeError(f"worldgen data version {worldgen.DATA_VERSION}, but this simulator expects "
                           f"{EXPECTED_DATA_VERSION}: check worldgen's DATA.md for what changed")
    return worldgen.load_or_generate(worldgen.WorldConfig(seed=seed, width=width), cache_dir)
