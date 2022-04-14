import os
import pandas as pd
from get_user_feature import get_all_user_feature_filtered


# save all motion data from folder:data_event&cut
# + add uid trial
def concat_all_motion():
    df_list = []
    for file in os.listdir("data_event&cut/sifted/"):
        df = pd.read_csv('data_event&cut/sifted/' + file, encoding='gbk', index_col=0)
        # resolve filename and add column to df
        uid = file.split('.')[0].split('-')[0]
        trial = file.split('.')[0].split('-')[1]
        df.insert(0, 'uid', uid)
        df.insert(1, 'trial', trial)

        # add to df_list
        df_list.append(df)

    df_motion = pd.concat(df_list, axis=0, join='outer')
    print('Concat all motion file done.')

    return df_motion


# save all motion data from folder:data_event&space
# + add uid trial
def concat_all_space():
    df_list = []

    for space_file in os.listdir("data_event&space"):
        df = pd.read_csv('data_event&space/' + space_file, encoding='gbk')
        # resolve filename and add column to df
        uid = space_file.split('.')[0].split('-')[0]
        trial = space_file.split('.')[0].split('-')[1]
        df.insert(0, 'uid', uid)
        df.insert(1, 'trial', trial)

        # add to df_list
        df_list.append(df)

    df_space = pd.concat(df_list, axis=0, join='outer')
    print('Concat all space file done.')

    return df_space


# save all data (add user features)
# filter some irrelevant features: eg. ['passed_time', 'name']
def add_user(df_motion, feature_list):
    df_motion["uid"] = df_motion["uid"].astype(str)
    df_user = get_all_user_feature_filtered(feature_list)
    df_result = pd.merge(df_motion, df_user, on='uid')
    print('add_user: Add user done.')
    return df_result


# get mean, min, max, ... data to form a dataset
def get_dataset(feature_list):
    df_dataset = {
        "uid": [],
        "side": [],
        "event": [],
        "trial": [],
        "cut_mean": [],
        "cut_max": [],
        "cut_min": [],
        "cut_std": [],
        "cut_var": [],
        "speed_mean": [],
        "speed_max": [],
        "speed_min": [],
        "speed_std": [],
        "speed_var": [],
        "space_mean": [],
        "space_max": [],
        "space_min": [],
    }

    df_space = pd.read_csv('Dataset/Data_space.csv', encoding='gbk')
    df_motion = pd.read_csv('Dataset/Data_motion.csv', encoding='gbk')

    side_op = ['left', 'right']

    # todo modify range
    for uid_t in range(1, 63):  # uid: 1 ~ 62
        for event_type_t in range(1, 5):  # event_type: 1 2 3 4
            for trial_t in range(1, 4):  # trial: 1 2 3
                for side_t in side_op:  # side: 0 1
                    df_motion_t = df_motion[(df_motion['event_type'] == event_type_t) & (df_motion['uid'] == uid_t) & (
                            df_motion['trial'] == trial_t) & (df_motion['side'] == side_t)]
                    df_space_t = df_space[
                        (df_space.event == event_type_t) & (df_space.uid == uid_t) & (df_space.trial == trial_t)]
                    df_dataset["uid"].append(uid_t)
                    df_dataset["side"].append(side_t)
                    df_dataset["event"].append(event_type_t)
                    df_dataset["trial"].append(trial_t)
                    df_dataset["cut_mean"].append(df_motion_t["cut_length"].mean())
                    df_dataset["cut_min"].append(df_motion_t["cut_length"].min())
                    df_dataset["cut_max"].append(df_motion_t["cut_length"].max())
                    df_dataset["cut_std"].append(df_motion_t["cut_length"].std())
                    df_dataset["cut_var"].append(df_motion_t["cut_length"].var())
                    df_motion_t = df_motion_t[(df_motion_t.speed > 0)]
                    df_dataset["speed_mean"].append(df_motion_t["speed"].mean())
                    df_dataset["speed_min"].append(df_motion_t["speed"].min())
                    df_dataset["speed_max"].append(df_motion_t["speed"].max())
                    df_dataset["speed_std"].append(df_motion_t["speed"].std())
                    df_dataset["speed_var"].append(df_motion_t["speed"].var())
                    if side_t == 0:
                        df_dataset["space_mean"].append(df_space_t["l_space_mean"].mean())
                        df_dataset["space_min"].append(df_space_t["l_space_min"].min())
                        df_dataset["space_max"].append(df_space_t["l_space_max"].max())
                    else:
                        df_dataset["space_mean"].append(df_space_t["r_space_mean"].mean())
                        df_dataset["space_min"].append(df_space_t["r_space_min"].min())
                        df_dataset["space_max"].append(df_space_t["r_space_max"].max())

    df_dataset = pd.DataFrame(df_dataset)
    print('get_dataset: Get dataset done.')

    df_dataset = add_user(df_dataset, feature_list)
    print('get_dataset: Add user done.')

    return df_dataset


# get mean, min, max, ... data to form a dataset
# WITHOUT side & trial
def get_dataset_tmp(feature_list):
    df_dataset = {
        "uid": [],
        "event": [],
        "cut_mean": [],
        "cut_max": [],
        "cut_min": [],
        "cut_std": [],
        "cut_var": [],
        "speed_mean": [],
        "speed_max": [],
        "speed_min": [],
        "speed_std": [],
        "speed_var": [],
        "space_mean": [],
        "space_max": [],
        "space_min": [],
    }

    df_space = pd.read_csv('Dataset/Data_space.csv', encoding='gbk')
    df_motion = pd.read_csv('Dataset/Data_motion.csv', encoding='gbk')

    for uid_t in range(1, 63):  # uid: 1 ~ 62
        for event_type_t in range(1, 5):  # event_type: 1 2 3 4
            df_motion_t = df_motion[(df_motion['event_type'] == event_type_t) & (df_motion['uid'] == uid_t)]
            df_space_t = df_space[
                (df_space.event == event_type_t) & (df_space.uid == uid_t)]
            df_dataset["uid"].append(uid_t)
            df_dataset["event"].append(event_type_t)
            df_dataset["cut_mean"].append(df_motion_t["cut_length"].mean())
            df_dataset["cut_min"].append(df_motion_t["cut_length"].min())
            df_dataset["cut_max"].append(df_motion_t["cut_length"].max())
            df_dataset["cut_std"].append(df_motion_t["cut_length"].std())
            df_dataset["cut_var"].append(df_motion_t["cut_length"].var())
            df_motion_t = df_motion_t[(df_motion_t.speed > 0)]
            df_dataset["speed_mean"].append(df_motion_t["speed"].mean())
            df_dataset["speed_min"].append(df_motion_t["speed"].min())
            df_dataset["speed_max"].append(df_motion_t["speed"].max())
            df_dataset["speed_std"].append(df_motion_t["speed"].std())
            df_dataset["speed_var"].append(df_motion_t["speed"].var())
            df_dataset["space_mean"].append((df_space_t["l_space_mean"].mean() + df_space_t["r_space_mean"].mean()) / 2)
            df_dataset["space_min"].append(min(df_space_t["l_space_min"].min(), df_space_t["r_space_min"].min()))
            df_dataset["space_max"].append(max(df_space_t["l_space_max"].max(), df_space_t["r_space_max"].max()))

    df_dataset = pd.DataFrame(df_dataset)
    print('get_dataset: Get dataset done.')

    df_dataset = add_user(df_dataset, feature_list)
    print('get_dataset: Add user done.')

    return df_dataset


if __name__ == '__main__':
    # # include all motion data
    df_m = concat_all_motion()
    df_m.to_csv('Dataset/Data_motion.csv', encoding='gbk')
    df_s = concat_all_space()
    df_s.to_csv('Dataset/Data_space.csv', encoding='gbk')

    features = ['uid', 'gender', 'age', 'height', 'weight',
                'fre_side', 'VR_exp', 'game_fre', 'sport_fre',
                'difficulty', 'enjoyment', 'fatigue', 'personality', 'familiarity']
    df_d = get_dataset(features)
    df_d = df_d.dropna()
    df_d.to_csv('Dataset/Data_dataset.csv', encoding='gbk', index=None)

    df_d_tmp = get_dataset_tmp(features)
    df_d_tmp = df_d_tmp.dropna()
    df_d_tmp.to_csv('Dataset/Data_dataset_tmp.csv', encoding='gbk', index=None)
