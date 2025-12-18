import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False

# Single window experiments
days = [0, 1, 2, 3, 5, 7, 14, 30]
aucs = [84.60, 84.95, 85.00, 84.75, 84.71, 84.41, 84.79, 84.65]
labels = ['E0', 'E1', 'E1a', 'E2', 'E2a', 'E3', 'E4', 'E5']

plt.figure(figsize=(10, 6))
plt.plot(days, aucs, marker='o', linewidth=2, markersize=8)

for i in range(len(days)):
    plt.vlines(x=days[i], ymin=plt.ylim()[0], ymax=aucs[i], colors='gray',
               linestyles='dashed', alpha=0.3, linewidth=1)

for i, label in enumerate(labels):
    plt.annotate(f'{label}\n{aucs[i]:.2f}%',
                xy=(days[i], aucs[i]),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=9)

plt.xlabel('windows size (days)', fontsize=12)
plt.ylabel('AUC (%)', fontsize=12)
plt.title('Different Window Sizes AUC Performance', fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim([84.2, 85.2])

plt.xticks(days)

output_dir = './outputs/'
output_path = os.path.join(output_dir, 'temporal_window_auc.png')
os.makedirs(output_dir, exist_ok=True)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"pic saved as {output_path}")