import os
import json
import xlwt


# pid: person id        tid: trial id
# lc_x: left controller x axis
def convert_txt_xls(txt_path, xls_path, pid, tid):
    title = ["time", "lc_x", "lc_y", "lc_z", "rc_x", "rc_y", "rc_z", "lt_x", "lt_y", "lt_z", "rt_x", "rt_y", "rt_z"]
    book = xlwt.Workbook()  # 创建一个excel对象
    sheet = book.add_sheet('Sheet1', cell_overwrite_ok=True)  # 添加一个sheet页

    for i in range(len(title)):  # 循环列
        sheet.write(0, i, title[i])  # 将title数组中的字段写入到0行i列中

    with open(txt_path) as file:
        lines = file.readlines()
        count = 1
        for line in lines:
            line = line.strip("\n")
            string = line.split("  ")

            leftController = string[1].strip(" ")[16:-1].split(",")
            rightController = string[2][17:-1].split(",")
            leftTracker = string[3][13:-1].split(",")
            rightTracker = string[4][14:-1].split(",")

            sheet.write(count, 0, string[0])  # time
            sheet.write(count, 1, leftController[0].strip(" "))  # leftController x
            sheet.write(count, 2, leftController[1].strip(" "))  # leftController y
            sheet.write(count, 3, leftController[2].strip(" "))  # leftController z
            sheet.write(count, 4, rightController[0].strip(" "))  # rightController x
            sheet.write(count, 5, rightController[1].strip(" "))  # rightController y
            sheet.write(count, 6, rightController[2].strip(" "))  # rightController z
            sheet.write(count, 7, leftTracker[0].strip(" "))  # leftTracker x
            sheet.write(count, 8, leftTracker[1].strip(" "))  # leftTracker y
            sheet.write(count, 9, leftTracker[2].strip(" "))  # leftTracker z
            sheet.write(count, 10, rightTracker[0].strip(" "))  # rightTracker x
            sheet.write(count, 11, rightTracker[1].strip(" "))  # rightTracker y
            sheet.write(count, 12, rightTracker[2].strip(" "))  # rightTracker z

            count += 1
    book.save(xls_path)


for txt_file in os.listdir("../data_txt"):
    # print(txt_file)
    xls_file = txt_file.split(".")[0] + ".xls"
    info = txt_file.split(".")[0].split("-")
    convert_txt_xls("../data_txt/" + txt_file, "../data_xls/" + xls_file, info[0], info[1])
