Contains two files that do random simulations on the closed loop acasxu system with dubins car dynamics: `acasxu_dubins.py` and `parallel_acasxu_dubins.py`

On my system, I can run 10000 simulations in about 12 seconds single-threaded (1.2 ms per sim), and about 2 seconds multi-threaded (0.2 ms per sim). This also uses numba to speed things up using jit decortors.