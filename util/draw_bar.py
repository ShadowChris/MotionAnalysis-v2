import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def draw_bar(title, data, error):
    plt.figure(figsize=(6, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    plt.bar(range(len(data)), data, yerr=error, align='center', color='#5B9BD5', ecolor='#A6A6A6', capsize=10,
            width=0.4)
    plt.title(title)
    plt.xticks(range(4), ['单手身前切割', '单手身旁切割', '单手反手切割', '双手切割'])
    # plt.xticks(range(4), ['Slashing block in front', 'Slashing block beside',
    #                       'Slashing block backhand', 'Slashing blocks with both hands'])
    plt.grid(linestyle="--", alpha=0.3)
    for x, y in zip(range(4), data):
        plt.text(x, y + 0.03, '%.3f' % y, ha='center', va='bottom')
    plt.savefig("../images/paper" + title + ".png")
    plt.show()


def draw_all_bar(title, data, error):
    plt.figure(figsize=(6, 5))
    N = 3
    ind = np.arange(N)
    width = 0.2
    plt.bar(ind, data[0], width, color='#375DA1', yerr=error[0], label='Slashing block in front')
    plt.bar(ind + width, data[1], width, color='#4472C4', yerr=error[1], label='Slashing block beside')
    plt.bar(ind + 2 * width, data[2], width, color='#5B9BD5', yerr=error[2], label='Slashing block backhand')
    plt.bar(ind + 3 * width, data[3], width, color='#ADC6E5', yerr=error[3], label='Slashing blocks with both hands')

    # plt.ylabel('Scores')
    # plt.title('Scores by group and gender')

    plt.xticks(ind + 1.5 * width, ('Amplitude', 'Speed', 'Space'))
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.25), borderaxespad=0.)
    plt.grid(linestyle="--", alpha=0.3)
    plt.savefig("../images/paper/" + title + ".png")
    plt.show()


def draw_all_bar_no_error(title, data):
    plt.figure(figsize=(6, 4))
    plt.ylim(0, 1.8)
    N = 3
    ind = np.arange(N)
    width = 0.25
    plt.bar(ind, data[0], width, color='#375DA1', label='Repetition 1')
    plt.bar(ind + width, data[1], width, color='#5B9BD5', label='Repetition 2')
    plt.bar(ind + 2 * width, data[2], width, color='#ADC6E5', label='Repetition 3')

    for i in range(3):
        for x, y in zip(range(4), data[i]):
            plt.text(x + i * width, y + 0.03, '%.3f' % y, ha='center', va='bottom')
    # plt.ylabel('Scores')
    # plt.title('Scores by group and gender')

    plt.xticks(ind + width, ('Amplitude', 'Speed', 'Space'))
    plt.legend(loc='best')
    plt.grid(linestyle="--", alpha=0.3)
    plt.savefig("../images/paper/" + title + ".png")
    plt.show()


if __name__ == '__main__':
    # df = pd.read_csv("../Dataset/event&motion1.csv", encoding='gbk')
    df = pd.read_csv("../Dataset/trial&motion.csv", encoding='gbk')
    # features = ['平均切割动作幅度', '平均切割动作速度', '最大游戏空间大小']
    features = ['第一次游戏', '第二次游戏', '第三次游戏']
    # features = ['单手身前切割', '单手身旁切割', '单手反手切割', '双手切割']
    data = list()
    error = list()
    for feature in features:
        # data_t = df[feature].tolist()[0:4]
        # error_t = df[feature].tolist()[4:8]
        # draw_bar(feature, data_t, error_t)

        data.append(df[feature].tolist()[0:3])
        # error.append(df[feature].tolist()[3:6])
    # draw_all_bar('all', data, error)
    draw_all_bar_no_error('repetition', data)
