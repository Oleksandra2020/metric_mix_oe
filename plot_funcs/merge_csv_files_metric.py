import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def return_df(csv_files, compare):
    if csv_files == []:
        return pd.DataFrame([])

    df_concat = pd.concat([pd.read_csv(f)
                          for f in csv_files], ignore_index=True)

    df_concat.rename(columns={'mean_tnr95_fine_across_splits': 'mean_tnr95_coarse_across_splits',
                     'mean_tnr95_coarse_across_splits': 'mean_tnr95_fine_across_splits',
                              'std_tnr95_fine_across_splits': 'std_tnr95_coarse_across_splits',
                              'std_tnr95_coarse_across_splits': 'std_tnr95_fine_across_splits',
                              }, inplace=True)

    less_df = df_concat[['method', 'margin', 'beta', 'mean_acc_across_splits',
                         'mean_tnr95_fine_across_splits', 'mean_tnr95_coarse_across_splits',
                         'std_acc_across_splits', 'std_tnr95_fine_across_splits',
                         'std_tnr95_coarse_across_splits', 'mean_tnr_fine_across_splits',
                         'mean_tnr_coarse_across_splits', 'std_tnr_fine_across_splits',
                         'std_tnr_coarse_across_splits']]

    mixup_exp = less_df.sort_values(compare)

    return mixup_exp


def plot_consts(type, dataset, d, ax, with_std, mx_size):
    if type == 'fine':

        if dataset == 'car' and d == 0:

            ax.plot([0.9109], [0.5324], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.9109, 0.5324), fontsize=fontsize)

            ax.plot([0.9272], [0.6248], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.9272, 0.6248), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.9272], [0.6248], xerr=0.003682,
                            yerr=0.03248, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.9109], [0.5324], xerr=0.00278,
                            yerr=0.02287, fmt='-o', capsize=2, color='red')

        if dataset == 'car' and d == 1:
            ax.plot([0.9109], [0.2542], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.9109, 0.5324), fontsize=fontsize)

            ax.plot([0.9272], [0.2397], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.9272, 0.6248), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.9272], [0.2397], xerr=0.003682,
                            yerr=0.02423, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.9109], [0.2542], xerr=0.00278,
                            yerr=0.00009, fmt='-o', capsize=2, color='red')

        if dataset == 'bird' and d == 0:

            ax.plot([0.8135], [0.2195], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8135, 0.2195), fontsize=fontsize)

            ax.plot([0.8291], [0.2536], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.8291, 0.2536), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.8291], [0.2536], xerr=0.00626,
                            yerr=0.01333, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8135], [0.2195], xerr=0.008194,
                            yerr=0.005404, fmt='-o', capsize=2, color='red')

        if dataset == 'bird' and d == 1:

            ax.plot([0.8135], [0.09912], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8135, 0.2195), fontsize=fontsize)

            ax.plot([0.8291], [0.0936], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.8291, 0.2536), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.8291], [0.0936], xerr=0.00626,
                            yerr=0.01067, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8135], [0.09912], xerr=0.008194,
                            yerr=0.01061, fmt='-o', capsize=2, color='red')

        if dataset == 'butterfly' and d == 0:

            ax.plot([0.8866], [0.3198], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8866, 0.3198), fontsize=fontsize)

            ax.plot([0.8927], [0.3747], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.8927, 0.3747), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.8927], [0.3747], xerr=0.01019,
                            yerr=0.02997, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8866], [0.3198], xerr=0.0102,
                            yerr=0.0262, fmt='-o', capsize=2, color='red')

        if dataset == 'butterfly' and d == 1:
            ax.plot([0.8866], [0.219], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8866, 0.3198), fontsize=fontsize)

            ax.plot([0.8927], [0.2612], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.8927, 0.3747), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.8927], [0.2612], xerr=0.01019,
                            yerr=0.03306, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8866], [0.219], xerr=0.0102,
                            yerr=0.05224, fmt='-o', capsize=2, color='red')

        if dataset == 'aircraft' and d == 0:

            ax.plot([0.8882], [0.1952], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8882, 0.1952), fontsize=fontsize)

            ax.plot([0.9007], [0.2583], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.9007, 0.2583), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.9007], [0.2583], xerr=0.004435,
                            yerr=0.07845, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8882], [0.1952], xerr=0.002724,
                            yerr=0.08188, fmt='-o', capsize=2, color='red')

        if dataset == 'aircraft' and d == 1:
            ax.plot([0.8882], [0.1229], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8882, 0.1952), fontsize=fontsize)

            ax.plot([0.9007], [0.1289], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.9007, 0.2583), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.9007], [0.1289], xerr=0.004435,
                            yerr=0.03448, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8882], [0.1229], xerr=0.002724,
                            yerr=0.05993, fmt='-o', capsize=2, color='red')

    if type == 'coarse':

        if dataset == 'car' and d == 0:
            ax.plot([0.9109], [0.8851], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.9109, 0.8851), fontsize=fontsize)

            ax.plot([0.9272], [0.993], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.9272, 0.993), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.9272], [0.993], xerr=0.005945,
                            yerr=0.005945, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.9109], [0.8851], xerr=0.00278,
                            yerr=0.05512, fmt='-o', capsize=2, color='red')

        if dataset == 'car' and d == 1:
            ax.plot([0.9109], [0.9998], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.9109, 0.8851), fontsize=fontsize)

            ax.plot([0.9272], [0.9945], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.9272, 0.993), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.9272], [0.9945], xerr=0.005945,
                            yerr=0.02118, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.9109], [0.9998], xerr=0.00278,
                            yerr=0.00009, fmt='-o', capsize=2, color='red')

        if dataset == 'bird' and d == 0:
            ax.plot([0.8135], [0.6638], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8135, 0.6638), fontsize=fontsize)

            ax.plot([0.8291], [0.8634], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.86162, 0.8634), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.8291], [0.8634], xerr=0.00626,
                            yerr=0.01948, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8135], [0.6638], xerr=0.008194,
                            yerr=0.01469, fmt='-o', capsize=2, color='red')

        if dataset == 'bird' and d == 1:
            ax.plot([0.8135], [0.7538], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8135, 0.6638), fontsize=fontsize)

            ax.plot([0.8291], [0.8715], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.86162, 0.8634), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.8291], [0.8715], xerr=0.00626,
                            yerr=0.03409, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8135], [0.7538], xerr=0.008194,
                            yerr=0.01362, fmt='-o', capsize=2, color='red')

        if dataset == 'butterfly' and d == 0:
            ax.plot([0.8866], [0.8853], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8866, 0.8853), fontsize=fontsize)

            ax.plot([0.8927], [0.9517], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.8927, 0.9517), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.8927], [0.9517], xerr=0.01019,
                            yerr=0.009266, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8866], [0.8853], xerr=0.0102,
                            yerr=0.005465, fmt='-o', capsize=2, color='red')

        if dataset == 'butterfly' and d == 1:
            ax.plot([0.8866], [0.967], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8866, 0.8853), fontsize=fontsize)

            ax.plot([0.8927], [0.8878], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.8927, 0.9517), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.8927], [0.8878], xerr=0.01019,
                            yerr=0.04311, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8866], [0.967], xerr=0.0102,
                            yerr=0.01234, fmt='-o', capsize=2, color='red')

        if dataset == 'aircraft' and d == 0:
            ax.plot([0.8882], [0.7027], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8882, 0.7027), fontsize=fontsize)

            ax.plot([0.9007], [0.9151], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.9007, 0.9151), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.9007], [0.9151], xerr=0.004435,
                            yerr=0.02123, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8882], [0.7027], xerr=0.002724,
                            yerr=0.02701, fmt='-o', capsize=2, color='red')

        if dataset == 'aircraft' and d == 1:
            ax.plot([0.8882], [0.9842], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.8882, 0.7027), fontsize=fontsize)

            ax.plot([0.9007], [0.9808], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.9007, 0.9151), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.9007], [0.9808], xerr=0.004435,
                            yerr=0.02071, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.8882], [0.9842], xerr=0.002724,
                            yerr=0.007299, fmt='-o', capsize=2, color='red')


def make_comparison_plot2(type='fine', with_std=False, compare='margin', save_dir=f"i1_i1_i2"):

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    datasets = ['car', 'bird', 'butterfly', 'aircraft']
    for a, dataset in enumerate(datasets):

        if a == 0:
            ax = axs[0, 0]
        elif a == 1:
            ax = axs[0, 1]
        elif a == 2:
            ax = axs[1, 0]
        elif a == 3:
            ax = axs[1, 1]

        if "rand" in save_dir:
            inl_df = return_df(
                glob.glob('./metric_csv_i/*{}*mixup=0*{}.csv'.format(dataset, save_dir[:-2])), compare)
        else:
            inl_df = return_df(
                glob.glob('./metric_csv_i/*{}*mixup=0*{}.csv'.format(dataset, save_dir)), compare)

        inl_mx_df = return_df(
            glob.glob('./metric_csv_i/*{}*mixup=2*{}.csv'.format(dataset, save_dir)), compare)

        colours_green = ["#060E02", "#183609", "#295F0F", "#3B8716", "#4F932D", "#629F45", "#76AB5C", "#89B773", "#9DC38B", "#B1CFA2", "#C4DBB9", "#D8E7D0"][::-1]
        colours_blue = ["#162F45", "#254F74", "#346FA2", "#4F9BDD", "#4A9EE7", "#5CA8E9", "#6EB1EC", "#80BBEE", "#92C5F1", "#A5CFF3", "#B7D8F5", "#C9E2F8"][::-1]

        fine_tnrs, names, coarse_tnrs, accs = [], [], [], []

        try:
            margins = list(set(inl_df[compare]))
        except KeyError:
            return

        margins.sort()
        mx_size = 10

        d = 0

        for o in range(2):

            if o == 0:
                cur_df = inl_df
                colors = colours_green
            elif o == 1:
                cur_df = inl_mx_df
                colors = colours_blue

            for i, (index, row) in enumerate(cur_df.iterrows()):
                if i % 3 == 0:
                    color = colors[c]
                    c += 1
                if o == 0:
                    name = 'b+'
                if o == 3:
                    name = 'b+mx+'
                beta = row[2]
                name += f'triplet(m={row[1]}_β={beta})'
                acc = row[3]
                std_acc = row[6]

                if d == 0:
                    fine_tnr95 = row[4]
                    coarse_tnr95 = row[5]
                    std_fine = row[7]
                    std_coarse = row[8]
                else:
                    fine_tnr95 = row[9]
                    coarse_tnr95 = row[10]
                    std_fine = row[11]
                    std_coarse = row[12]

                fine_tnrs.append(fine_tnr95)
                coarse_tnrs.append(coarse_tnr95)
                accs.append(acc)
                names.append(name)

                if beta == 0.1:
                    marker = 's'
                if beta == 0.3:
                    marker = 'o'
                if beta == 0.5:
                    marker = 'x'

                if type == 'fine':
                    if marker == 's' or marker == 'x':
                        ax.plot([acc], [fine_tnr95], color=color, markerfacecolor='none',
                                label=name, marker=marker, markersize=mx_size)
                    else:
                        ax.plot([acc], [fine_tnr95], color=color,
                                label=name, marker=marker, markersize=mx_size)
                    if with_std:
                        ax.errorbar([acc], [fine_tnr95], xerr=std_acc, yerr=std_fine, fmt='-',
                                    capsize=2, color=color)
                    # ax.annotate(name, (acc, fine_tnr95), fontsize=fontsize)

                if type == 'coarse':
                    if marker == 's' or marker == 'x':
                        ax.plot([acc], [coarse_tnr95], color=color, markerfacecolor='none',
                                label=name, marker=marker, markersize=mx_size)
                    else:
                        ax.plot([acc], [coarse_tnr95], color=color,
                                label=name, marker=marker, markersize=mx_size)
                    if with_std:
                        ax.errorbar([acc], [coarse_tnr95], xerr=std_acc, yerr=std_coarse,
                                    fmt='-', capsize=2, color=color)
                    # ax.annotate(name, (acc, coarse_tnr95), fontsize=fontsize)


        step = 2
        ax.set(xticks=np.array(range(60, 100, step)) / 100)

        step = 3
        if type == 'fine':
            ax.set(yticks=np.array(range(5, 70, step)) / 100)

        if type == 'coarse':
            ax.set(yticks=np.array(range(50, 100, step)) / 100)

        plot_consts(type, dataset, d, ax, with_std, mx_size)

        ax.set_title(dataset, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)

        ax.xaxis.get_label().set_fontsize(20)
        ax.yaxis.get_label().set_fontsize(20)
        ax.set_xlabel("Classification accuracy", fontsize=20)
        ax.set_ylabel("TNR95", fontsize=20)
        ax.grid()
        plt.setp(ax.get_xticklabels(), visible=True)

        ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*100))
        ax.xaxis.set_major_formatter(ticks)
        ax.yaxis.set_major_formatter(ticks)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc='upper center', fontsize=15,
               bbox_to_anchor=(0.5, 1.), fancybox=True, shadow=True)

    if type == 'fine':
        plt.savefig(f"../all/fine_all_{save_dir}.png")
        plt.savefig(f"../all/fine_all_{save_dir}.pdf", format='pdf')

    if type == 'coarse':
        plt.savefig(f"../all/coarse_all_{save_dir}.png")
        plt.savefig(f"../all/coarse_all_{save_dir}.pdf", format='pdf')

    # plt.show()
    plt.close()


if __name__ == "__main__":
    compare = 'margin'
    dirs = ["i1_i1_i2", "i1_i1_o", "io_i1_i2_rand_1", "io_i1_o_rand_1"]
    for tp in ['fine', 'coarse']:
        for dr in dirs:
            make_comparison_plot2(type=tp, with_std=True,
                                  compare=compare, save_dir=dr)
