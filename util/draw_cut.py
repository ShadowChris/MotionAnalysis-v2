import matplotlib.pyplot as plt
import xlrd
import pandas as pd

# 读取文件
df = pd.read_csv("../data_csv/62-1.csv")

# 打开画图窗口1，在三维空间中绘图
fig = plt.figure(1)
ax = fig.gca(projection='3d')
# ax.invert_xaxis()   # x轴反向
# ax.invert_yaxis()   # y轴反向
ax.set_xlabel('x')
ax.set_ylabel("z")
ax.set_zlabel("y")
# plt.title('Motion Track of Controllers and Trackers', fontsize=12)

# left controller
lc_x = list()
lc_y = list()
lc_z = list()
lc_x_t = list()
lc_y_t = list()
lc_z_t = list()
# right controller
rc_x = list()
rc_y = list()
rc_z = list()
rc_x_t = list()
rc_y_t = list()
rc_z_t = list()
# left tracker
lt_x = list()
lt_y = list()
lt_z = list()
# right tracker
rt_x = list()
rt_y = list()
rt_z = list()
# right tracker
h_x = list()
h_y = list()
h_z = list()

for row in range(270, 280):
    lc_x_t.append(float(df.iloc[row, 1]))
    lc_y_t.append(float(df.iloc[row, 2]))
    lc_z_t.append(float(df.iloc[row, 3]))
    rc_x_t.append(float(df.iloc[row, 4]))
    rc_y_t.append(float(df.iloc[row, 5]))
    rc_z_t.append(float(df.iloc[row, 6]))

for row in range(280, 310):
    lc_x.append(float(df.iloc[row, 1]))
    lc_y.append(float(df.iloc[row, 2]))
    lc_z.append(float(df.iloc[row, 3]))
    rc_x.append(float(df.iloc[row, 4]))
    rc_y.append(float(df.iloc[row, 5]))
    rc_z.append(float(df.iloc[row, 6]))
    lt_x.append(float(df.iloc[row, 7]))
    lt_y.append(float(df.iloc[row, 8]))
    lt_z.append(float(df.iloc[row, 9]))
    rt_x.append(float(df.iloc[row, 10]))
    rt_y.append(float(df.iloc[row, 11]))
    rt_z.append(float(df.iloc[row, 12]))
    h_x.append(float(df.iloc[row, 13]))
    h_y.append(float(df.iloc[row, 14]))
    h_z.append(float(df.iloc[row, 15]))

for row in range(310, 330):
    lc_x_t.append(float(df.iloc[row, 1]))
    lc_y_t.append(float(df.iloc[row, 2]))
    lc_z_t.append(float(df.iloc[row, 3]))
    rc_x_t.append(float(df.iloc[row, 4]))
    rc_y_t.append(float(df.iloc[row, 5]))
    rc_z_t.append(float(df.iloc[row, 6]))

color_seg = '#FA8072'
color_norm = '#87CEEB'
color_gray = '#808080'
color_lightgray = '#C0C0C0'

# 绘制散点
ax.scatter3D(lc_x_t, lc_z_t, lc_y_t, c=color_lightgray, cmap='Blues')  # 绘制散点图
ax.scatter3D(rc_x_t, rc_z_t, rc_y_t, c=color_lightgray, cmap='Blues')  # 绘制散点图
ax.scatter3D(lc_x, lc_z, lc_y, c=color_seg, cmap='Blues')  # 绘制散点图
ax.scatter3D(rc_x, rc_z, rc_y, c=color_seg, cmap='Blues')  # 绘制散点图
# 标记散点
ax.text(lc_x_t[0] - 0.16, lc_z_t[0], lc_y_t[0] + 0.1, "Left Controller")
ax.text(rc_x_t[0] - 0.16, rc_z_t[0], rc_y_t[0] + 0.1, "Right Controller")

# tracker
ax.scatter3D(lt_x, lt_z, lt_y, c=color_gray, cmap='Blues')  # 绘制散点图
ax.scatter3D(rt_x, rt_z, rt_y, c=color_gray, cmap='Blues')  # 绘制散点图
ax.text(lt_x[0] - 0.16, lt_z[0], lt_y[0] + 0.1, "Left Tracker")
ax.text(rt_x[0] - 0.16, rt_z[0], rt_y[0] + 0.1, "Right Tracker")

# headset
ax.scatter3D(h_x, h_z, h_y, c=color_gray, cmap='Blues')  # 绘制散点图
ax.text(h_x[0] + 0.1, h_z[0], h_y[0], "Headset")

# 将数组中的前两个点进行连线
# figure = ax.plot(lc_x, lc_z, lc_y, c=color_left, label='left controller')
# figure = ax.plot(rc_x, rc_z, rc_y, c=color_right, label='right controller')
# figure = ax.plot(lt_x, lt_z, lt_y, c='r', label='left tracker')
# figure = ax.plot(rt_x, rt_z, rt_y, c='g', label='right tracker')

# plt.text("Right Controller")
plt.savefig("../images/MotionTrack.png")
plt.show()
