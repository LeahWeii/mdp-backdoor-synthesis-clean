import pickle
path = './gridworld_ex_fm_po/ml1_p0.8_epsilon0.10_sto[0.1, 0.2]contSZ/ablation_pi1_decaylr'
# with open(f"{path}/V0.pkl", "rb") as file_v1:
#     V0_iter = pickle.load(file_v1)
# print(V0_iter[-1])

with open(f"{path}/V1.pkl", "rb") as file_v1:
    V1_iter = pickle.load(file_v1)
print(V1_iter[-1])

with open(f"{path}/V0_attack.pkl", "rb") as file_v1:
    V0_attack_iter = pickle.load(file_v1)
print(V0_attack_iter[-1])