import matplotlib.pyplot as plt
import pandas as pd
import os

if __name__ == '__main__':
    # cut_dict = {}
    # cut_list = []
    # i = 1
    # for file_name in os.listdir("../data_event&cut/sifted/"):
    #     print('--- ' + file_name + ' ---')
    #     df_tmp = pd.read_csv("../data_event&cut/sifted/" + file_name)
    #     cut_dict[i] = df_tmp["cut_length"]
    #     cut_list.append(df_tmp["cut_length"])
    #     i += 1
    # df = pd.DataFrame(cut_dict)
    # df.plot.box(title="Cut_length")
    # plt.grid(linestyle="--", alpha=0.3)
    # # plt.violinplot(df, showmeans=False, showmedians=True)
    # plt.show()

    df = pd.read_csv("../Dataset/Data_dataset_tmp.csv")
    df_df = df[["cut_mean", "speed_mean", "space_max"]]
    df_df.columns = ['Amplitude(mean)', 'Speed(mean)', 'Space(max)']
    # df_df.plot.box(title="Motion Features Box")
    df_df.plot.box(fontsize=14)

    plt.grid(linestyle="--", alpha=0.3)
    plt.savefig("../images/Paper/Motion Features Box.png")
    plt.show()
