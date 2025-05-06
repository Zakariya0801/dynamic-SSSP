import numpy as np

import matplotlib.pyplot as plt

# Data from the table
update_counts = [13, 50, 100, 1000, 2000, 3000, 4000]
sequential_times = [493, 962, 1571, 7521, 14577, 21572, 28372]  # in ms
hybrid_times_ms = [3262.31, 7444.9, 7249.27, 8097.25, 8125.95, 8107.51, 8125.34]  # in seconds


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Bar graph
x = np.arange(len(update_counts))
width = 0.35

# Plot bars
ax1.bar(x - width/2, sequential_times, width, label='Sequential (ms)')
ax1.bar(x + width/2, hybrid_times_ms, width, label='Hybrid (ms)')

# Add labels, title and legend
ax1.set_xlabel('Number of Updates')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Comparison of Sequential vs Hybrid Processing Time')
ax1.set_xticks(x)
ax1.set_xticklabels(update_counts)
ax1.legend()

# Add value annotations on top of bars
for i, v in enumerate(sequential_times):
    ax1.text(i - width/2, v + 0.05*max(sequential_times), f"{v}", ha='center')
    
for i, v in enumerate(hybrid_times_ms):
    ax1.text(i + width/2, v + 0.05*max(hybrid_times_ms), f"{v:.0f}", ha='center')

# Line graph
ax2.plot(update_counts, sequential_times, marker='o', linestyle='-', label='Sequential (ms)')
ax2.plot(update_counts, hybrid_times_ms, marker='s', linestyle='-', label='Hybrid (ms)')

# Add labels, title and legend for line graph
ax2.set_xlabel('Number of Updates')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Performance Scaling with Increasing Updates')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# Set x-axis to log scale to better visualize the increasing updates
ax2.set_xscale('log')

# Adjust layout and show
plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300)
plt.show()