"""Shortcut for the history simulator's command line (same as `python -m histsim`).

    python main.py run --world 31 --seed 1          simulate and save a history
    python main.py graphs runs/world31_seed1        draw its graphs
    python main.py info runs/world31_seed1          summarise it
"""
from histsim.__main__ import main

if __name__ == "__main__":
    main()
