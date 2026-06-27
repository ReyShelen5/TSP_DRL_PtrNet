import torch
from time import time
from env import Env_tsp
from config import Config, load_pkl, pkl_parser
from search import sampling, active_search


def search_tour(cfg, env):
    test_input = env.get_nodes(cfg.seed)

    print("\n===================================")
    print("INPUT CITIES")
    print("===================================")

    for i in range(cfg.city_t):
        print(
            f"City {i:2d}: "
            f"({test_input[i][0].item():.4f}, "
            f"{test_input[i][1].item():.4f})"
        )

    # ---------------- Random Tour ----------------
    print("\nGenerating Random Tour...")
    random_tour = env.get_random_tour()
    env.show(test_input, random_tour)

    # ---------------- Sampling ----------------
    print("\nSampling...")

    t1 = time()
    pred_tour = sampling(cfg, env, test_input)
    t2 = time()

    print(f"\nExecution Time: {(t2-t1):.2f} sec")

    env.show(test_input, pred_tour)

    # ---------------- Active Search ----------------
    print("\nActive Search...")

    t1 = time()
    pred_tour = active_search(cfg, env, test_input)
    t2 = time()

    print(f"\nExecution Time: {(t2-t1):.2f} sec")

    env.show(test_input, pred_tour)

    """
    # Uncomment if you want the exact optimal tour
    print("\nOptimal Tour...")

    t1 = time()
    optimal_tour = env.get_optimal_tour(test_input)
    t2 = time()

    print(f"\nExecution Time: {(t2-t1):.2f} sec")

    env.show(test_input, optimal_tour)
    """


if __name__ == "__main__":

    cfg = load_pkl(pkl_parser().path)

    print("\n===================================")
    print("CUSTOM TSP INPUT")
    print("===================================")

    n = int(input("Enter number of cities: "))

    coords = []

    print("\nEnter coordinates as:")
    print("x y\n")

    for i in range(n):
        x, y = map(float, input(f"City {i}: ").split())
        coords.append([x, y])

    # Update configuration
    cfg.city_t = n

    # Create environment using custom coordinates
    env = Env_tsp(cfg, custom_nodes=coords)

    print("\n===================================")
    print("ENTERED COORDINATES")
    print("===================================")

    for i, (x, y) in enumerate(coords):
        print(f"City {i:2d}: ({x:.4f}, {y:.4f})")

    if cfg.mode == "test":
        search_tour(cfg, env)

    else:
        raise NotImplementedError(
            "Please use a TEST configuration file."
        )