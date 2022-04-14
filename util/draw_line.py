import matplotlib.pyplot as plt
import pandas as pd
import os


def draw_train_process(title, iters, acc_matching, acc_proto, acc_maml, label_matching, lable_proto, label_maml):
    # plt.title(title, fontsize=20)
    plt.xlabel("EPOCH_NUM", fontsize=16)
    plt.ylabel("Testing Accuracy", fontsize=16)
    plt.ylim([0, 1])
    plt.plot(iters, acc_matching, color='#4472C4', label=label_matching)
    plt.plot(iters, acc_proto, color='#5B9BD5', label=lable_proto)
    plt.plot(iters, acc_maml, color='#A5A5A5', label=label_maml)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("../images/" + title + ".png")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("../Dataset/all_2.csv")

    iters = range(1, df.shape[0] + 1)
    features = ['cut', 'speed', 'space']
    for feature in features:
        all_matching_accs = df[feature + '_matching_val'].tolist()
        all_proto_accs = df[feature + '_proto_val'].tolist()
        all_maml_accs = df[feature + '_maml_val'].tolist()
        draw_train_process(feature, iters, all_matching_accs, all_proto_accs, all_maml_accs, "Matching Net",
                           "Proto Net", "MAML")
