import numpy as np
import pandas as pd
from tqdm import tqdm
from get_VR_event import get_all_event, get_event_by_time
from util.time_subtraction import time_sub_ms
import os


def speed(filename):
    df = pd.read_csv("data_csv/" + filename + ".csv")

    df_speed = df[['time', 'lc_s_x', 'lc_s_y', 'lc_s_z', 'rc_s_x', 'rc_s_y', 'rc_s_z']]

    all_event = get_all_event()
    space_event_list = []
    i = 0
    start_time = df_speed.iloc[0, 0]
    for row2 in range(0, len(all_event)):
        list_tmp = []
        while True:
            # df_space columns=['time', 'l_space', 'r_space']
            if i < df_speed.shape[0] and time_sub_ms(start_time, df_speed.iloc[i, 0]) - float(all_event[row2][0]) <= 0:
                list_tmp.append([abs(df_speed.iloc[i, 1]), abs(df_speed.iloc[i, 2]), abs(df_speed.iloc[i, 3]),
                                 abs(df_speed.iloc[i, 3]), abs(df_speed.iloc[i, 5]), abs(df_speed.iloc[i, 6])])
                i += 1
            else:
                break
        df_tmp = pd.DataFrame(list_tmp, columns=['lc_s_x', 'lc_s_y', 'lc_s_z', 'rc_s_x', 'rc_s_y', 'rc_s_z'])
        space_event_list.append([all_event[row2][1],
                                 df_tmp['lc_s_x'].min(), df_tmp['lc_s_x'].mean(), df_tmp['lc_s_x'].max(),
                                 df_tmp['lc_s_y'].min(), df_tmp['lc_s_y'].mean(), df_tmp['lc_s_y'].max(),
                                 df_tmp['lc_s_z'].min(), df_tmp['lc_s_z'].mean(), df_tmp['lc_s_z'].max(),
                                 df_tmp['rc_s_x'].min(), df_tmp['rc_s_x'].mean(), df_tmp['rc_s_x'].max(),
                                 df_tmp['rc_s_y'].min(), df_tmp['rc_s_y'].mean(), df_tmp['rc_s_y'].max(),
                                 df_tmp['rc_s_z'].min(), df_tmp['rc_s_z'].mean(), df_tmp['rc_s_z'].max()])
    df_space_event = pd.DataFrame(space_event_list, columns=['event',
                                                             'lc_s_x_min', 'lc_s_x_mean', 'lc_s_x_max',
                                                             'lc_s_y_min', 'lc_s_y_mean', 'lc_s_y_max',
                                                             'lc_s_z_min', 'lc_s_z_mean', 'lc_s_z_max',
                                                             'rc_s_x_min', 'rc_s_x_mean', 'rc_s_x_max',
                                                             'rc_s_y_min', 'rc_s_y_mean', 'rc_s_y_max',
                                                             'rc_s_z_min', 'rc_s_z_mean', 'rc_s_z_max'])
    df_space_event.to_csv("data_event&speed/" + filename + ".csv", index=False)


if __name__ == '__main__':
    for txt_file in tqdm(os.listdir("data_csv")):
        file_name = txt_file.split(".")[0]
        if os.path.exists("data_event&speed/" + file_name + ".csv"):
            continue

        if int(txt_file.split('.')[0].split('-')[0]) <= 20:
            continue
        else:
            speed(file_name)
