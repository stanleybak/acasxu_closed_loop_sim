'''
ACASXu neural networks closed loop simulation with dubin's car dynamics

Used for falsification, where the opponent is allowed to maneuver over time

This version uses multiprocessing pool for more simulations
'''

import time
import multiprocessing

import numpy as np

from acasxu_dubins import State, make_random_input, plot, state7_to_state5, run_network

def sim_single(seed):
    """run single simulation and return min_dist, seed"""

    rv = (np.inf, seed)

    if seed % 1000 == 0:
        print(f"{(seed//1000) % 10}", end='', flush=True)
    elif seed % 100 == 0:
        print(".", end='', flush=True)

    init_vec, cmd_list = make_random_input(seed)

    # reject start states where initial command is not clear-of-conflict
    state5 = state7_to_state5(init_vec, State.v_own, State.v_int)
    res = run_network(State.nets[0], state5)
    command = np.argmin(res)

    if command == 0:
        # run the simulation
        s = State(init_vec, save_states=True)
        s.simulate(cmd_list)

        # reject simulations where the minimum distance was near the start
        #if s.vec[-1] >= 3.0:
        rv = (s.min_dist, seed)

    return rv

def main():
    'main entry point'

    interesting_seed = -1

    num_sims = 10000
    start = time.perf_counter()

    with multiprocessing.Pool() as pool:
        results = pool.map(sim_single, range(num_sims))

    min_dist, interesting_seed = min(results)

    diff = time.perf_counter() - start
    ms_per_sim = round(1000 * diff / num_sims, 3)
    print(f"\nDid {num_sims} parallel sims in {round(diff, 1)} secs ({ms_per_sim}ms per sim)")

    d = round(min_dist, 1)
    print(f"\nplotting most interesting state with seed {interesting_seed} and min_dist {d}ft")

    # optional: do plot
    init_vec, cmd_list = make_random_input(interesting_seed)
    s = State(init_vec, save_states=True)
    s.simulate(cmd_list)
    plot(s, save_mp4=False)

if __name__ == "__main__":
    main()
