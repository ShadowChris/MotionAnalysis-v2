import matplotlib.pyplot as plt
import pandas as pd
import os

if __name__ == '__main__':
    df = pd.read_csv("../Dataset/all_2.csv")
    iters = range(1, df.shape[0] + 1)

    cut_matching_val = df['cut_matching_val'].tolist()
    cut_proto_val = df['cut_proto_val'].tolist()
    cut_maml_val = df['cut_maml_val'].tolist()
    speed_matching_val = df['speed_matching_val'].tolist()
    speed_proto_val = df['speed_proto_val'].tolist()
    speed_maml_val = df['speed_maml_val'].tolist()
    space_matching_val = df['space_matching_val'].tolist()
    space_proto_val = df['space_proto_val'].tolist()
    space_maml_val = df['space_maml_val'].tolist()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4), dpi=100)
    plt.figure(1)
    ax[0] = plt.subplot(131)
    ax[0].set_xlabel("EPOCH", fontsize=16)
    ax[0].set_ylabel("Amplitude Accuracy", fontsize=16)
    ax[0].set_ylim([0, 1])
    ax[0].plot(iters, cut_matching_val, color='#4472C4', label="Matching Net")
    ax[0].plot(iters, cut_proto_val, color='#5B9BD5', label="Proto Net")
    ax[0].plot(iters, cut_maml_val, color='#A5A5A5', label="MAML")
    ax[0].legend(loc='lower right')
    ax[0].grid()

    ax[1] = plt.subplot(132)
    ax[1].set_xlabel("EPOCH", fontsize=16)
    ax[1].set_ylabel("Speed Accuracy", fontsize=16)
    ax[1].set_ylim([0, 1])
    ax[1].plot(iters, speed_matching_val, color='#4472C4', label="Matching Net")
    ax[1].plot(iters, speed_proto_val, color='#5B9BD5', label="Proto Net")
    ax[1].plot(iters, speed_maml_val, color='#A5A5A5', label="MAML")
    ax[1].legend(loc='lower right')
    ax[1].grid()

    ax[2] = plt.subplot(133)
    ax[2].set_xlabel("EPOCH", fontsize=16)
    ax[2].set_ylabel("Space Accuracy", fontsize=16)
    ax[2].set_ylim([0, 1])
    ax[2].plot(iters, space_matching_val, color='#4472C4', label="Matching Net")
    ax[2].plot(iters, space_proto_val, color='#5B9BD5', label="Proto Net")
    ax[2].plot(iters, space_maml_val, color='#A5A5A5', label="MAML")
    ax[2].legend(loc='lower right')
    ax[2].grid()

    plt.savefig("../images/paper/all_2.png")
    plt.show()
