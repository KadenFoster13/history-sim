"""The simulation calendar.

Year 0 of the simulation is 10,000 BCE. Counting goes straight on from there, so (as
there is no year 0 between BCE and CE) year 9,999 is 1 BCE, year 10,000 is 1 CE, and
2025 CE is year 12,024.

Internally, real-world years use astronomical numbering (0 = 1 BCE, -1 = 2 BCE ...):
`astronomical = sim_year - 9,999`.
"""

YEAR_ZERO_REAL = -9_999   # astronomical year of simulation year 0 (= 10,000 BCE)


def real_year(sim_year):
    """Astronomical real-world year matching a simulation year (0 = 1 BCE)."""
    return sim_year + YEAR_ZERO_REAL


def sim_year(real):
    """Simulation year matching an astronomical real-world year."""
    return real - YEAR_ZERO_REAL


def real_label(sim_year_value):
    """'4500 BCE' / '1066 CE' for a simulation year."""
    astro = int(round(real_year(sim_year_value)))
    return f"{1 - astro} BCE" if astro <= 0 else f"{astro} CE"


def label(sim_year_value):
    """'year 5,500 (4500 BCE)': the simulation year with its real-world equivalent."""
    return f"year {int(sim_year_value):,} ({real_label(sim_year_value)})"
