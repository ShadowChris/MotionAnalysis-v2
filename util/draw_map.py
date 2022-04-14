import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def draw_heatmap(data):
    ylabels = data.columns.values.tolist()

    ss = StandardScaler()  # 归一化
    data = ss.fit_transform(data)

    df = pd.DataFrame(data)

    dfData = df.corr(method='spearman')
    plt.subplots(figsize=(15, 15))  # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, yticklabels=ylabels, xticklabels=ylabels, cmap="RdBu")
    plt.savefig("../images/paper/Map.png")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("../Dataset/spearman map.csv", index_col=0)
    print(df.describe())
    # df_df = df[["cut_mean", "speed_mean", "space_max"]]
    draw_heatmap(df)
