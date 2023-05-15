import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_consts(type, dataset, ax, with_std, mx_size):
    if type == 'fine':

        if dataset == 'car':

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

        if dataset == 'bird':

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

        if dataset == 'butterfly':

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

        if dataset == 'aircraft':

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

    if type == 'coarse':
        if dataset == 'car':

            ax.plot([0.9109], [0.8851], color='red',
                    label='vanilla', marker='o')
            #ax.annotate('vanilla', (0.9109, 0.8851), fontsize=fontsize)

            ax.plot([0.9272], [0.993], color='orange',
                    label='mixoe', marker='o', markersize=mx_size)
            #ax.annotate('mixoe', (0.9272, 0.993), fontsize=fontsize)

            if with_std:
                ax.errorbar([0.9272], [0.993], xerr=0.003682,
                            yerr=0.005945, fmt='-o', capsize=2, color='orange')

                ax.errorbar([0.9109], [0.8851], xerr=0.00278,
                            yerr=0.05512, fmt='-o', capsize=2, color='red')

        if dataset == 'bird':

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

        if dataset == 'butterfly':

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

        if dataset == 'aircraft':

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


def make_comparison_plot2(type='fine', with_std=True, save_dir='vary'):

    fig, axs = plt.subplots(2, 2, figsize=(15, 16))

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

        csv_files = glob.glob('./csv_results_vary/*{}*'.format(dataset, 'csv'))

        print(len(csv_files))
        df_concat = pd.concat([pd.read_csv(f)
                               for f in csv_files], ignore_index=True)

        df_concat['no_outlier_classes'][df_concat['no_outlier_classes'] == 0] = 1000

        df_concat.rename(columns={'mean_tnr95_fine_across_splits': 'mean_tnr95_coarse_across_splits',
                                  'mean_tnr95_coarse_across_splits': 'mean_tnr95_fine_across_splits',
                                  'std_tnr95_fine_across_splits': 'std_tnr95_coarse_across_splits',
                                  'std_tnr95_coarse_across_splits': 'std_tnr95_fine_across_splits'}, inplace=True)

        less_df = df_concat[['method', 'no_outlier_classes', 'no_outliers', 'mean_acc_across_splits',
                             'mean_tnr95_fine_across_splits', 'mean_tnr95_coarse_across_splits',
                             'std_acc_across_splits', 'std_tnr95_fine_across_splits',
                             'std_tnr95_coarse_across_splits']]

        # mixup_exp = less_df[(less_df['method'] == 'mixup')].sort_values('no_outlier_classes')
        outl_df = less_df[less_df.no_outlier_classes == 1000]
        outl_df = outl_df[(outl_df['method'] == 'mixup')
                          ].sort_values('no_outliers')

        cls_df = less_df[less_df.no_outlier_classes != 1000]
        cls_df = cls_df[(cls_df['method'] == 'mixup')
                        ].sort_values('no_outlier_classes')

        colours_green = ["#060E02", "#183609", "#295F0F", "#3B8716", "#4F932D", "#629F45", "#76AB5C", "#89B773", "#9DC38B", "#B1CFA2", "#C4DBB9", "#D8E7D0"][::-1]
        colours_blue = ["#162F45", "#254F74", "#346FA2", "#4F9BDD", "#4A9EE7", "#5CA8E9", "#6EB1EC", "#80BBEE", "#92C5F1", "#A5CFF3", "#B7D8F5", "#C9E2F8"][::-1]


        print(len(colours_blue) + len(colours_green))

        fine_tnrs, names, coarse_tnrs, accs = [], [], [], []
        mx_size = 10

        for i, (index, row) in enumerate(outl_df.iterrows()):
            # name = f'mixup_{row[1]}cls_{row[2]}outl'
            name = f'mixoe_outl={row[2]}'
            # name = f'mixup_cls={row[1]}'
            acc = row[3]
            fine_tnr95 = row[4]
            coarse_tnr95 = row[5]
            std_acc = row[6]
            std_fine = row[7]
            std_coarse = row[8]

            fine_tnrs.append(fine_tnr95)
            coarse_tnrs.append(coarse_tnr95)
            accs.append(acc)
            names.append(name)

            if row[1] == 1000 and row[2] == 1948448:
                name = 'mixoe'
                continue

            if type == 'fine':
                color = colours_green[i]
                ax.plot([acc], [fine_tnr95], color=color,
                        label=name, marker='o', markersize=mx_size)
                if with_std:
                    ax.errorbar([acc], [fine_tnr95], xerr=std_acc, yerr=std_fine, fmt='-o',
                                capsize=2, color=color)
                # #ax.annotate(name,
                #             (acc, fine_tnr95), fontsize=fontsize)

            if type == 'coarse':
                color = colours_green[i]
                ax.plot([acc], [coarse_tnr95], color=color,
                        label=name, marker='o', markersize=mx_size)
                if with_std:
                    ax.errorbar([acc], [coarse_tnr95], xerr=std_acc, yerr=std_coarse,
                                fmt='-o', capsize=2, color=color)

        for i, (index, row) in enumerate(cls_df.iterrows()):
            name = f'mixup_cls={row[1]}'
            acc = row[3]
            fine_tnr95 = row[4]
            coarse_tnr95 = row[5]
            std_acc = row[6]
            std_fine = row[7]
            std_coarse = row[8]

            fine_tnrs.append(fine_tnr95)
            coarse_tnrs.append(coarse_tnr95)
            accs.append(acc)
            names.append(name)

            if row[1] == 1000 and row[2] == 1948448:
                name = 'mixoe'
                continue

            if type == 'fine':
                color = colours_blue[i]
                ax.plot([acc], [fine_tnr95], color=color,
                        label=name, marker='o', markersize=mx_size)
                if with_std:
                    ax.errorbar([acc], [fine_tnr95], xerr=std_acc, yerr=std_fine, fmt='-o',
                                capsize=2, color=color)
                # #ax.annotate(name,
                #             (acc, fine_tnr95), fontsize=fontsize)

            if type == 'coarse':
                color = colours_blue[i]
                ax.plot([acc], [coarse_tnr95], color=color,
                        label=name, marker='o', markersize=mx_size)
                if with_std:
                    ax.errorbar([acc], [coarse_tnr95], xerr=std_acc, yerr=std_coarse,
                                fmt='-o', capsize=2, color=color)

        step = 1
        ax.set(xticks=np.array(range(60, 101, step)) / 100)

        step = 2
        if type == 'fine':
            ax.set(yticks=np.array(range(5, 70, step)) / 100)

        if type == 'coarse':
            ax.set(yticks=np.array(range(50, 101, step)) / 100)

        plot_consts(type, dataset, ax, with_std, mx_size)

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
    fig.legend(handles, labels, ncol=4, loc='upper center', fontsize=10,
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
    for tp in ['fine', 'coarse']:
        make_comparison_plot2(type=tp, with_std=True, save_dir='vary')
