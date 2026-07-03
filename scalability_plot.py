import time
import csv
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from config import Config
from env import Env_tsp
from search_defined_rt import (
    sampling_benchmark,
    active_search_benchmark
)
from actor import PtrNet1

# -----------------------------------------------------
# SETTINGS
# -----------------------------------------------------

MODEL_PATH = "./Pt/train15_0629_13_10_step9999_act.pt"

BATCH = 512
EMBED = 128
HIDDEN = 128
STEPS = 10
SEED = 123

NUM_INSTANCES = 10

# -----------------------------------------------------

cfg = Config(
    mode="test",
    batch=BATCH,
    city_t=20,
    steps=STEPS,

    embed=EMBED,
    hidden=HIDDEN,
    clip_logits=10,
    softmax_T=1.0,

    optim="Adam",

    init_min=-0.08,
    init_max=0.08,

    n_glimpse=1,
    n_process=3,

    decode_type="sampling",

    lr=1e-3,
    is_lr_decay=True,
    lr_decay=0.96,
    lr_decay_step=5000,

    act_model_path=MODEL_PATH,

    seed=SEED,
    alpha=0.99,

    islogger=False,
    issaver=False,

    log_step=10,

    log_dir="./Csv/",
    model_dir="./Pt/",
    pkl_dir="./Pkl/",

    cuda_dv="0"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################
# LOAD SAMPLING MODEL
#########################################################

sample_model = PtrNet1(cfg)
sample_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
sample_model = sample_model.to(device)
sample_model.eval()

#########################################################
# LOAD ACTIVE SEARCH MODEL
#########################################################

act_model = PtrNet1(cfg)
act_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
act_model = act_model.to(device)
act_model.train()

act_optim = optim.Adam(act_model.parameters(), lr=cfg.lr)

#########################################################

results = []

sampling_plot = []
active_plot = []

print("=" * 60)
print("RL SCALABILITY BENCHMARK")
print("=" * 60)
csv_file = "cities.csv"
for n in range(10, 101, 10):

    print(f"\nTesting {n} Cities")

    cfg.city_t = n

    env = Env_tsp(cfg,custom_nodes=csv_file)

    sampling_times = []
    active_times = []

    for instance in range(NUM_INSTANCES):

        print(f"Instance {instance+1}")

        test_input = env.get_nodes(seed=SEED + instance)

        #################################################
        # SAMPLING
        #################################################

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        sample_tour, sample_distance = sampling_benchmark(
            cfg,
            env,
            test_input,
            sample_model
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sample_runtime = time.perf_counter() - start

        sampling_times.append(sample_runtime)

        results.append([
            n,
            instance + 1,
            "Sampling",
            sample_runtime,
            sample_distance,
            "-".join(map(str, sample_tour.cpu().tolist()))
        ])

        #################################################
        # ACTIVE SEARCH
        #################################################

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        active_tour, active_distance = active_search_benchmark(
            cfg,
            env,
            test_input,
            act_model,
            act_optim
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        active_runtime = time.perf_counter() - start

        active_times.append(active_runtime)

        results.append([
            n,
            instance + 1,
            "Active Search",
            active_runtime,
            active_distance,
            "-".join(map(str, active_tour.cpu().tolist()))
        ])

    #################################################
    # Average runtime for plotting
    #################################################

    sampling_plot.append([n, sum(sampling_times) / NUM_INSTANCES])
    active_plot.append([n, sum(active_times) / NUM_INSTANCES])

#########################################################
# SAVE CSV
#########################################################

with open("rl_scalability_results.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Cities",
        "Instance",
        "Method",
        "Runtime(sec)",
        "Tour_Length",
        "Tour_Sequence"
    ])

    writer.writerows(results)

#########################################################
# PLOT
#########################################################

plt.figure(figsize=(8,5))

cities = [x[0] for x in sampling_plot]

sampling_runtime = [x[1] for x in sampling_plot]

active_runtime = [x[1] for x in active_plot]

plt.plot(
    cities,
    sampling_runtime,
    marker='o',
    linewidth=2,
    label="Sampling"
)

plt.plot(
    cities,
    active_runtime,
    marker='s',
    linewidth=2,
    label="Active Search"
)

plt.xlabel("Number of Cities")
plt.ylabel("Average Runtime (seconds)")
plt.title("Sampling vs Active Search Runtime")
plt.grid(True)
plt.legend()

plt.savefig("rl_scalability.png", dpi=300)

plt.show()

print("\nCSV saved as rl_scalability_results.csv")
print("Plot saved as rl_scalability.png")