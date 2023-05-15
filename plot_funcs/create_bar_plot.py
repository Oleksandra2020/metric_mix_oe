import matplotlib.pyplot as plt
import numpy as np


def main():
    barWidth = 0.2
    fig = plt.subplots(figsize=(15, 8))

    # TNR coarse conf

    mixoe_10 = [0.993, 0.8634, 0.9517, 0.9151]
    mixoe_10_std = [0.005945, 0.01948, 0.009266, 0.02123]
    label_mix = [0.869, 0.7497, 0.9074, 0.8168]
    label_mix_std = [0.07734, 0.03394, 0.01944, 0.0264]
    baseline = [0.8851, 0.6633, 0.8853, 0.7027]
    baseline_std = [0.05512, 0.01409, 0.005465, 0.02701]

    # TNR fine conf

#     mixoe_10 = [0.6248, 0.2536, 0.3747, 0.2583]
#     mixoe_10_std = [0.03248, 0.01333, 0.02997, 0.07845]
#     label_mix = [0.6173, 0.2508, 0.3757, 0.2566]
#     label_mix_std = [0.03069, 0.008553, 0.04858, 0.07629]
#     baseline = [0.5324, 0.2195, 0.3198, 0.1952]
#     baseline_std = [0.02287, 0.005404, 0.0262, 0.08188]

    # Acc

#     mixoe_10 = [0.9272, 0.8291, 0.8927, 0.9007]
#     mixoe_10_std = [0.003682, 0.00626, 0.01019, 0.004435]
#     label_mix = [0.9267, 0.8398, 0.8932, 0.902]
#     label_mix_std = [0.002258, 0.0047, 0.009159, 0.005579]
#     baseline = [0.9109, 0.8135, 0.886, 0.8882]
#     baseline_std = [0.00278, 0.008194, 0.0102, 0.004445]

    # Set position of bar on X axis
    br1 = np.arange(len(label_mix))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br2, mixoe_10, color='#A3EBB1', width=barWidth,
            edgecolor='grey', label='mixoe', yerr=mixoe_10_std)
    plt.bar(br1, label_mix, color='#E7625F', width=barWidth,
            edgecolor='grey', label='label_mix', yerr=label_mix_std)
    plt.bar(br3, baseline, color='#68BBE3', width=barWidth,
            edgecolor='grey', label='baseline', yerr=baseline_std)

    # Adding Xticks
    plt.xlabel('Dataset', fontsize=30)
    plt.ylabel('TNR95', fontsize=30)
    plt.xticks([r + barWidth for r in range(len(label_mix))],
               ['Car', 'Bird', 'Butterfly', 'Aircraft'], fontsize=25)
    plt.yticks(np.array(list(range(0, 11, 1))) / 10, fontsize=25)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True, fontsize=30)
    plt.savefig("./figures/label_mix_coarse_10.png")
    plt.savefig("./figures/label_mix_coarse_10.pdf", format='pdf')
    plt.show()



if __name__ == "__main__":
    main()
