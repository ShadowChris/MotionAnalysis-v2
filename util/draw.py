import matplotlib.pyplot as plt


# x_axis: str list for x axis
def draw_scatter(x_axis, y, x_label, y_label, title):
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    x = range(len(y))
    plt.scatter(x, y, c='b', alpha=0.5)
    plt.xlabel(x_label, font2)
    plt.ylabel(y_label, font2)
    plt.xticks(x, x_axis)
    plt.tick_params(labelsize=14)  # 刻度字体大小14
    plt.title(title, fontsize=14)
    plt.savefig("images/" + title + ".png")
    plt.show()


# x_axis: str list for x axis
def draw_scatter_multi(x_axis, y_list, x_label, y_label, title):
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    x = range(len(y_list[0]))
    for i in range(len(y_list)):
        plt.scatter(x, y_list[i], c='b', alpha=float((i + 1) / len(y_list)))
    plt.xlabel(x_label, font2)
    plt.ylabel(y_label, font2)
    plt.xticks(x, x_axis)
    plt.tick_params(labelsize=14)  # 刻度字体大小14
    plt.title(title, fontsize=14)
    plt.savefig("images/" + title + ".png")
    plt.show()
