import pandas as pd
from util.draw import draw_scatter, draw_scatter_multi


def analyze_event(df, uid, event_type):
    df_t = df[(df.event_type == event_type) & (df.uid == uid)]
    cut_list.append(df_t['cut_length'].mean())
    speed_list.append(df_t['speed'].mean())
    x_cut.append(str(uid) + '-' + str(event_type))
    print('uid:', uid, ' event_type:', event_type)
    # print(df_t.describe())


def analyze_trial(df, uid, event_type, trial):
    df_t = df[(df.event_type == event_type) & (df.uid == uid) & (df.trial == trial)]
    cut_trial_list.append(df_t['cut_length'].mean())
    speed_trial_list.append(df_t['speed'].mean())
    x_trial.append(str(event_type) + '-' + str(trial))
    print('uid:', uid, ' event_type:', event_type, ' trial:', trial, ' cut_length:', df_t['cut_length'].mean())


if __name__ == '__main__':
    # Analyze Event
    filename = 'Dataset/Data_motion_user.csv'
    df_all = pd.read_csv(filename, encoding='gbk')
    x_cut = []
    cut_list = []
    speed_list = []
    x_trial = []
    cut_trial_list_all = []
    speed_trial_list_all = []
    # todo modify range
    for i in range(1, 13):  # uid
        x_trial = []
        cut_trial_list = []
        speed_trial_list = []
        for j in range(1, 6):  # event_type
            analyze_event(df_all, i, j)

            # Analyze Trial
            for k in range(1, 4):  # trial
                analyze_trial(df_all, i, j, k)
        cut_trial_list_all.append(cut_trial_list)
        speed_trial_list_all.append(speed_trial_list)
        # draw_scatter(x_trial, trial_list, 'event-trial', 'cut-length')
    draw_scatter_multi(x_trial, cut_trial_list_all, 'event-trial', 'cut-length', 'Average Cut Length of Each User')
    draw_scatter_multi(x_trial, speed_trial_list_all, 'event-trial', 'speed', 'Average Speed of Each User')
    draw_scatter(x_cut, cut_list, 'uid-event', 'cut-length', 'Average Cut Length of Each User-Event')
