import numpy as np
import pandas as pd
from tqdm import tqdm
from get_VR_event import get_all_event, get_event_by_time
from util.time_subtraction import time_sub_ms
import os


def space(filename):
    df = pd.read_csv("data_csv/" + filename + ".csv")

    space_list = []
    for row1 in range(df.shape[0]):
        time = df.iloc[row1, 0]
        space_lx = abs(df.iloc[row1, 13] - df.iloc[row1, 1])
        space_lz = abs(df.iloc[row1, 15] - df.iloc[row1, 3])
        l_space = np.sqrt(space_lx ** 2 + space_lz ** 2)

        space_rx = abs(df.iloc[row1, 13] - df.iloc[row1, 4])
        space_rz = abs(df.iloc[row1, 15] - df.iloc[row1, 6])
        r_space = np.sqrt(space_rx ** 2 + space_rz ** 2)
        space_list.append([time, l_space, r_space])

    df_space = pd.DataFrame(space_list, columns=['time', 'l_space', 'r_space'])

    # sift left
    q3 = np.percentile(df_space["l_space"], 75)
    q1 = np.percentile(df_space["l_space"], 25)
    max_sift = q3 + 1.5 * (q3 - q1)
    min_sift = q1 - 1.5 * (q3 - q1)
    df_space = df_space[(df_space.l_space <= max_sift)]  # filter
    df_space = df_space[(df_space.l_space >= min_sift)]  # filter

    # sift right
    q3 = np.percentile(df_space["r_space"], 75)
    q1 = np.percentile(df_space["r_space"], 25)
    max_sift = q3 + 1.5 * (q3 - q1)
    min_sift = q1 - 1.5 * (q3 - q1)
    df_space = df_space[(df_space.r_space <= max_sift)]  # filter
    df_space = df_space[(df_space.r_space >= min_sift)]  # filter

    all_event = get_all_event()
    space_event_list = []
    i = 0
    start_time = df_space.iloc[0, 0]
    for row2 in range(0, len(all_event)):
        list_tmp = []
        while True:
            # df_space columns=['time', 'l_space', 'r_space']
            if i < df_space.shape[0] and time_sub_ms(start_time, df_space.iloc[i, 0]) - float(all_event[row2][0]) <= 0:
                list_tmp.append([df_space.iloc[i, 1], df_space.iloc[i, 2]])
                i += 1
            else:
                break
        df_tmp = pd.DataFrame(list_tmp, columns=['l_space', 'r_space'])
        space_event_list.append([all_event[row2][1],
                                 df_tmp['l_space'].min(), df_tmp['l_space'].mean(), df_tmp['l_space'].max(),
                                 df_tmp['r_space'].min(), df_tmp['r_space'].mean(), df_tmp['r_space'].max()])
    df_space_event = pd.DataFrame(space_event_list, columns=['event', 'l_space_min', 'l_space_mean', 'l_space_max',
                                                             'r_space_min', 'r_space_mean', 'r_space_max'])

    df_space_event.to_csv("data_event&space/" + filename + ".csv", index=False)


if __name__ == '__main__':
    for txt_file in tqdm(os.listdir("data_csv")):
        file_name = txt_file.split(".")[0]
        if os.path.exists("data_event&space/" + file_name + ".csv"):
            continue
        space(file_name)
