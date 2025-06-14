import numpy as np

DATASET = 1 # 選擇資料集
COMP_MODE = 4
ID = [186, 519, 563, 1, 165, 60, 544]
LOCATION = f"SHAPSampling\\result_data\\{ID[DATASET]}"
lossPath = f"{LOCATION}\\LOSS"
with open(lossPath + f"\\loss_mode{COMP_MODE}.npy", 'rb') as file:
    LOSS_LIMIT = np.load(file, allow_pickle=True).item()

mode4all = 0
count = 0
for i in LOSS_LIMIT.values():
    count += 1
    mode4all += i
print(LOSS_LIMIT)
print(mode4all/count)
