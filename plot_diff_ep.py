import pickle
path = './gridworld_ex_fm_po/ml1_p0.8_epsilon0.15_sto[0.1, 0.2]contSZ'
with open(f"{path}/V0.pkl", "rb") as file_v1:
    V0_iter = pickle.load(file_v1)
print(V0_iter[-1])

with open(f"{path}/V1.pkl", "rb") as file_v1:
    V1_iter = pickle.load(file_v1)
print(V1_iter[-1])

with open(f"{path}/V0_attack.pkl", "rb") as file_v1:
    V0_attack_iter = pickle.load(file_v1)
print(V0_attack_iter[-1])


# V0=[17.20148400370468, 17.014499130549282, 16.827806735015706, 16.641032976806486, 16.453990412608363,16.266903802685754, 16.079833022827874]
# V0_attack = [16.894043375239363, 16.71686694022086, 16.55008301886246, 16.3760508724809, 16.213087253440527, 16.047447531957342, 15.881491560849081]


import os
import pickle
import matplotlib.pyplot as plt

import os
import pickle
import matplotlib.pyplot as plt


import os
import pickle
import matplotlib.pyplot as plt

import os
import pickle
import matplotlib.pyplot as plt

def plot_v0_and_attack_subplots(path, epsilons, episodes):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot: V0_iter for all epsilons
    for eps in epsilons:
        eps_str = f"{eps:.2f}"
        folder = f"ml1_p0.8_epsilon{eps_str}_sto[0.1, 0.2]contSZ"
        full_path = os.path.join(path, folder)

        try:
            with open(os.path.join(full_path, "V0.pkl"), "rb") as file_v0:
                V0_iter = pickle.load(file_v0)
            axs[0].plot(range(episodes), V0_iter, label=fr'$\epsilon$ = {eps_str}')
        except FileNotFoundError:
            print(f"Missing V0.pkl for ε = {eps_str}, skipping.")

    axs[0].set_title(r"$V_0$ over Episodes for Different $\epsilon$")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Value")
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot: V0_attack for all epsilons
    for eps in epsilons:
        eps_str = f"{eps:.2f}"
        folder = f"ml1_p0.8_epsilon{eps_str}_sto[0.1, 0.2]contSZ"
        full_path = os.path.join(path, folder)

        try:
            with open(os.path.join(full_path, "V0_attack.pkl"), "rb") as file_v0_attack:
                V0_attack_iter = pickle.load(file_v0_attack)
            axs[1].plot(range(episodes), V0_attack_iter, label=fr'$\epsilon$ = {eps_str}')
        except FileNotFoundError:
            print(f"Missing V0_attack.pkl for ε = {eps_str}, skipping.")

    axs[1].set_title(r"$V_0$ under Attack over Episodes for Different $\epsilon$")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{path}/V0_V0attack_subplots.png")
    # plt.show()


path = "./gridworld_ex_fm_po"
epsilons = [round(0.08 + 0.01 * i, 2) for i in range(8)]  # [0.08, ..., 0.15]
episodes = 10000
plot_v0_and_attack_subplots(path, epsilons, episodes)

