'''
ACASXu neural networks closed loop simulation with dubin's car dynamics

Used for falsification, where the opponent is allowed to maneuver over time

This version uses multiprocessing pool for more simulations
'''

import time
import multiprocessing
import argparse

import numpy as np

from acasxu_dubins import State, make_random_input, plot, state7_to_state5, run_network

def sim_single(seed):
    """run single simulation and return min_dist, seed"""

    rv = (np.inf, seed)

    if seed % 1000 == 0:
        print(f"{(seed//1000) % 10}", end='', flush=True)
    elif seed % 100 == 0:
        print(".", end='', flush=True)

    init_vec, cmd_list, init_velo = make_random_input(seed)

    v_own = init_velo[0]
    v_int = init_velo[1]

    # reject start states where initial command is not clear-of-conflict
    state5 = state7_to_state5(init_vec, v_own, v_int)
    res = run_network(State.nets[0], state5)
    command = np.argmin(res)

    if command == 0:
        # run the simulation
        s = State(init_vec, save_states=False)
        s.simulate(cmd_list)

        # reject simulations where the minimum distance was near the start
        #if s.vec[-1] >= 3.0:
        rv = (s.min_dist, seed)

    return rv

def main():
    'main entry point'

    'parse arguments'
    parser = argparse.ArgumentParser(description='Run ACASXU Dublins model simulator.')
    parser.add_argument("--save-mp4", action='store_true', default=False, help="Save plotted mp4 files to disk.")
    args = parser.parse_args()

    save_mp4 = args.save_mp4
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
    init_vec, cmd_list, init_velo = make_random_input(interesting_seed)
    s = State(init_vec, init_velo[0], init_velo[1], save_states=True)
    s.simulate(cmd_list)
    plot(s, save_mp4)

if __name__ == "__main__":
    main()
