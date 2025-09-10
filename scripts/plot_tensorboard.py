import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import pandas as pd  # For EMA calculation

def exponential_moving_average(data, alpha):
    return pd.Series(data).ewm(alpha=alpha, adjust=True).mean().values

labels = ['DemoDICE', 'Avatar+DemoDICE', 'SMODICE', 'GWIL', 'IGDF+IQ-Learn']
groups = [
          './tensorboard/Joint_target_wipe_torch_ver1_lr_1e-4_1e-4_traj_1_10_50_seed', 
          './tensorboard/transfer_wipe_torch_ver1_lr_1e-4_1e-4_traj_1_10_50_adaptive_decay_pow1_woflow_seed', 
          './tensorboard/smodice_wipe_traj_1_10_50_ver1_seed', 
          './tensorboard/gwil_wipe_src_traj_10_seed'
          ]

num_files = 5
label_id = 0
# Define the tag to extract
tag = 'Test average return'

expert_score = 100
smoothing_factor = 0.6
# Create figure
plt.figure(figsize=(10, 6))
# Iterate through each group of log files
for group in groups:
    all_steps = []
    all_values = []
    for i in range(num_files):
        log_path = f'{group}{i}'
        if not os.path.exists(log_path):
            print(f"Log file {log_path} does not exist!")
            continue
        
        # Initialize EventAccumulator
        event_acc = EventAccumulator(log_path)
        event_acc.Reload()
        
        # Check if the specified tag exists
        if tag not in event_acc.Tags()['scalars']:
            print(f"Tag '{tag}' not found in {log_path}")
            continue
        
        # Extract data
        scalars = event_acc.Scalars(tag)
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]
        
        all_steps.append(steps)
        all_values.append(values)

    # Ensure there is data
    if not all_steps or not all_values:
        print(f"No data found in any of the log files for group {group}!")
        continue

    # Find common steps across all runs (use the shortest step count as reference)
    min_steps = min(len(steps) for steps in all_steps)
    common_steps = all_steps[0][:min_steps]
    for steps in all_steps[1:]:
        common_steps = [s for s in common_steps if s in steps[:min_steps]]

    # Calculate mean and std for each common step
    avg_values = []
    std_values = []
    for step in common_steps:
        step_values = []
        for steps, values in zip(all_steps, all_values):
            idx = steps.index(step)
            step_values.append(values[idx])
        avg_values.append(np.mean(step_values))
        std_values.append(np.std(step_values))

    # Apply EMA to mean and std
    smoothed_values = exponential_moving_average(avg_values, smoothing_factor)
    smoothed_std = exponential_moving_average(std_values, smoothing_factor)
    smoothed_steps = common_steps  # EMA preserves the full length

    # Plot mean line and shaded std area
    plt.plot(smoothed_steps, smoothed_values, label=labels[label_id])
    plt.fill_between(smoothed_steps, 
                     smoothed_values - smoothed_std, 
                     smoothed_values + smoothed_std, 
                     alpha=0.2)  # Shaded area for std
    label_id += 1


# Add expert score dashed line
# plt.axhline(y=expert_score, color='black', linestyle='--', label='Expert Score')

plt.xlabel('Steps')
plt.ylabel('Test Average Return')
plt.title('Average Test Return Across Runs with Standard Deviation')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('wipe_critic_lr_1e-4_avg_test_return_with_std.png')
# plt.show()

print("Plot saved as 'hopper_avg_test_return_with_std.png'")