import csv
import pandas as pd


# return a dataframe given all features of all users
def get_all_users_all_features():
    with open('data_user/user_pre.csv', 'r', encoding='utf-8') as csv_file:
        csv_file = csv.reader(csv_file)
        all_user_data = []
        flag = True
        for line in csv_file:
            # skip the first row
            if flag:
                flag = False
                continue

            # one piece of user data
            user_data = list()
            user_data.append(line[0])  # uid
            for i in range(1, len(line)):  # append other features
                user_data.append(line[i])

            # add the user data to all-list
            all_user_data.append(user_data)
    # todo modify the columns
    all_user_data_df = pd.DataFrame(all_user_data, columns=['uid', 'gender', 'age', 'height', 'weight',
                                                            'fre_side', 'VR_exp', 'game_fre', 'sport_fre',
                                                            'difficulty', 'enjoyment', 'fatigue', 'personality',
                                                            'familiarity'])
    return all_user_data_df


# example featureList = ['uid', 'name', 'gender', 'age', 'height(cm)', 'weight(kg)']
def get_all_user_feature_filtered(feature_list):
    df = get_all_users_all_features()
    return df[feature_list]


def get_user_by_id(all_user_data, uid):
    return all_user_data[uid - 1]


def generate_user_feature():
    df = pd.read_csv("data_user/user_pre.csv", index_col=0)
    # change to 0 and 1
    df.gender = df.gender.apply(lambda x: 0 if x == 1 else 1)
    df.side = df.side.apply(lambda x: 0 if x == 1 else 1)
    df.personality = df.personality.apply(lambda x: 1 if x == 1 else 0)

    # multiple ranges
    df.age = df.age.apply(lambda x: 0 if x <= 20 else 1)
    df.height = df.height.apply(lambda x: 0 if x < 160 else (1 if x < 170 else (2 if x < 180 else 3)))
    df.weight = df.weight.apply(lambda x: 0 if x < 55 else (1 if x < 70 else (2 if x < 85 else 3)))

    # bool values
    df.VR_exp = df.VR_exp.apply(lambda x: 0 if x <= 1 else 1)
    df.game_fre = df.game_fre.apply(lambda x: 0 if x <= 2 else 1)
    df.sport_fre = df.sport_fre.apply(lambda x: 0 if x <= 3 else 1)
    df.difficulty = df.difficulty.apply(lambda x: 0 if x <= 2 else 1)
    df.enjoyment = df.enjoyment.apply(lambda x: 0 if x <= 3 else 1)
    df.fatigue = df.fatigue.apply(lambda x: 0 if x <= 2 else 1)

    df.to_csv("data_user/user.csv", encoding='gbk')


if __name__ == '__main__':
    generate_user_feature()
