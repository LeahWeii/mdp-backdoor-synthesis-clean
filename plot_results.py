path_list = ['./gridworld_ex_fm_po/p0.8_epsilon0.1/ablation_pi1_decaylr']
import pickle

with open(f"{path_list[0]}/V0.pkl", "rb") as file_v0:  # "rb" means read in binary mode
    V0_iter = pickle.load(file_v0)
with open(f"{path_list[0]}/V1.pkl", "rb") as file_v1:  # "rb" means read in binary mode
    V1_iter = pickle.load(file_v1)

with open(f"{path_list[0]}/V0_attack.pkl", "rb") as file_or:  # "rb" means read in binary mode
    V0_attack = pickle.load(file_or)

print(V1_iter[-1])

import matplotlib.pyplot as plt

plt.plot(V0_iter, label=r'$V_0(\pi_0^t, M )$')
plt.plot(V1_iter, label=r'$V_1(\pi_0^t, \pi_1^t, \mathcal{M})$')
# Optional: Add labels, title, legend
lb = 91*0.95
plt.axhline(y=lb, color='r', linestyle='--', linewidth=2, label='lower bound:'+str(lb))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Iterations")
plt.legend()
plt.ylabel("Value for Original Reward/Attack Reward")
plt.title("")
plt.show()
plt.savefig(f"{path}/plot_v_iter.png")