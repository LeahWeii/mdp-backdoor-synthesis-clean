# import os
import pickle
# path = './gridworld_ex_fm_po/ml1_p0.8_epsilon0.1_sto[0.1, 0.2]'
# with open(f"{path}/V0.pkl", "rb") as file_v1:
#     V0_iter = pickle.load(file_v1)
# print(V0_iter[-1])
#
# with open(f"{path}/V1.pkl", "rb") as file_v1:
#     V1_iter = pickle.load(file_v1)
# print(V1_iter[-1])


import matplotlib.pyplot as plt
import numpy as np

V0_iter_list = []
V1_iter_list = []
V0_iter_attack_list = []

base_path = './gridworld_ex_fm_po/ml1_p0.8_epsilon0.1_sto[0.1, 0.2]'

for x in range(1, 6):
    path = base_path + str(x)

    with open(f"{path}/V0.pkl", "rb") as file_v0:
        V0_iter = pickle.load(file_v0)
        V0_iter_list.append(V0_iter)

    with open(f"{path}/V1.pkl", "rb") as file_v1:
        V1_iter = pickle.load(file_v1)
        V1_iter_list.append(V1_iter)

    with open(f"{path}/V0_attack.pkl", "rb") as file_v0_attack:
        V0_iter_attack = pickle.load(file_v0_attack)
        V0_iter_attack_list.append(V0_iter_attack)



import matplotlib.pyplot as plt
import numpy as np

def plot_results(path, V0_iter_list, V1_iter_list, V0_iter_attack_list, episodes, lb):
    def compute_stats(data_list):
        data_array = np.array(data_list)  # shape: (5, episodes)
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        return mean, std

    def add_subplot_with_inset(ax, mean, std, title, ylabel, color, label):
        x = np.arange(episodes)
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=color)
        ax.set_title(title)
        ax.set_xlabel('Episodes')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

        # Inset: last 10% episodes
        start = int(0.9 * episodes)
        inset_ax = ax.inset_axes([0.5, 0.2, 0.47, 0.47])
        inset_ax.plot(x[start:], mean[start:], color=color)
        inset_ax.fill_between(x[start:], mean[start:] - std[start:], mean[start:] + std[start:], alpha=0.3, color=color)
        inset_ax.set_title("Last 10% Episodes")
        inset_ax.grid(True)

    # Compute statistics
    V0_mean, V0_std = compute_stats(V0_iter_list)
    V0a_mean, V0a_std = compute_stats(V0_iter_attack_list)

    fig, axs = plt.subplots(2, 1, figsize=(9, 8))

    # V0 subplot with lower bound
    axs[0].axhline(y=lb, color='r', linestyle='--', linewidth=2, label=f'Lower bound: {lb:.2f}')
    add_subplot_with_inset(axs[0], V0_mean, V0_std, 'V0 Iteration Results', 'Value', 'blue', 'V0')

    # V0 under attack subplot
    add_subplot_with_inset(axs[1], V0a_mean, V0a_std, 'V0 under Attack Iteration Results', 'Value', 'green', 'V0 under attack')

    plt.tight_layout()
    plt.savefig(f"{path}/plot_v_iter_stat.png")
    # plt.show()



plot_results('./gridworld_ex_fm_po/ml1_p0.8_epsilon0.1_sto[0.1, 0.2]1', V0_iter_list, V1_iter_list, V0_iter_attack_list, 10000, 16.83)