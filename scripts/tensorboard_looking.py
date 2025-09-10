import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

# 定義三個 TensorBoard 日誌檔案的路徑
log_files = [
    # 'tensorboard/transfer_hopper_torch_ver1_lr_5e-5_traj_1_20_50_adaptive_decay_pow1_seed',
    # 'tensorboard/transfer_ant_torch_ver1_lr_1e-4_traj_1_1_100_adaptive_decay_pow1_seed',
    # 'tensorboard/transfer_cheetah_torch_ver1_lr_1e-4_traj_1_1_100_adaptive_decay_pow1_seed',
    # 'tensorboard/transfer_lift_torch_ver1_lr_1e-4_traj_1_1_100_adaptive_decay_pow1_seed',
    'tensorboard/transfer_door_torch_ver1_lr_1e-4_traj_1_1_100_adaptive_decay_pow1_seed',
    # 'tensorboard/transfer_wipe_torch_ver1_lr_1e-4_traj_1_10_50_adaptive_decay_pow1_seed'
]

# 標籤名稱
tag_name = 'Test average return'
num_seed = 5

for log_file in log_files:
    # 用於儲存每個檔案的最後 5 個資料點
    all_values = [[] for _ in range(5)]
    # 用於儲存每個檔案的 step（用於檢查對齊）
    all_steps = [[] for _ in range(5)]
    for i in range(5):
        if not os.path.exists(f'{log_file}{i}'):
            print(f"日誌檔案 {log_file}{i} 不存在，跳過")
            continue
        
        # 初始化 EventAccumulator
        event_acc = EventAccumulator(f'{log_file}{i}')
        event_acc.Reload()  # 載入日誌檔案

        # 檢查是否有指定標籤
        if tag_name not in event_acc.Tags()['scalars']:
            print(f"標籤 {tag_name} 在 {log_file}{i} 中不存在，跳過")
            continue

        # 提取 Test average return 的資料和 step
        events = event_acc.Scalars(tag_name)
        values = [event.value for event in events]
        steps = [event.step for event in events]

        # 確保有至少 5 個資料點
        if len(values) < 5:
            print(f"{log_file} 中資料點數量不足 5 個，只有 {len(values)} 個，跳過")
            continue

        # 提取最後 5 個資料點和對應的 step
        all_values[i] = values[-5:]
        all_steps[i] = steps[-5:]
        # print(f"{log_file} 的最後 5 個資料點 (step={all_steps[i]}): {all_values[i]}")

    # 檢查是否有有效的資料
    if not all(all_values):
        print("沒有足夠的資料點可以計算")
    else:
        # 檢查 step 是否對齊
        steps_first = all_steps[0]
        if not all(np.array_equal(steps_first, steps) for steps in all_steps[1:]):
            print("警告：不同檔案的 step 不完全對齊，假設最後 5 個資料點按順序對應相同時間點")

        # 將資料轉置，按時間點組織（假設每個檔案的最後 5 個資料點對應相同時間點）
        values_by_time = np.array(all_values)  # 形狀為 (3, 5)，3 個檔案，5 個時間點
        # values_by_time = values_by_time.T      # 轉置為 (5, 3)，5 個時間點，每個時間點 3 個值

        # 計算每個時間點的平均值
        mean_values = np.mean(values_by_time, axis=1)
        # print(f"\n每個時間點的平均值：{mean_values}")

        # 計算最大平均值對應時間點的三個值的標準差
        mean_list = []
        for i in range(num_seed):
            mean_list.append(np.mean(values_by_time[i]))
        mean_list = np.array(mean_list)
        print('mean list: ', mean_list)
        print('mean std: ', np.std(mean_list))
        print('mean return: ', np.mean(mean_list))