import os
import json


# pid: person id        tid: trial id
def convert_txt_json(txt_path, json_path, pid, tid):
    with open(txt_path) as file:
        lines = file.readlines()
        result = dict()
        result["pid"] = pid
        result["tid"] = tid
        data = dict()
        count = 1
        for line in lines:
            line = line.strip("\n")
            string = line.split("  ")
            one = dict()
            one["time"] = string[0]
            one["leftController"] = string[1].strip(" ")[16:-1]
            one["rightController"] = string[2][17:-1]
            one["leftTracker"] = string[3][13:-1]
            one["rightTracker"] = string[4][14:-1]
            data[count] = one
            count += 1
        result["data"] = data
    # dump into json file
    with open(json_path, "w") as file1:
        jas = json.dumps(result)
        file1.write(jas)


for txt_file in os.listdir("../data_txt"):
    # print(txt_file)
    json_file = txt_file.split(".")[0] + ".json"
    info = txt_file.split(".")[0].split("-")
    convert_txt_json("../data_txt/" + txt_file, "../data_json/" + json_file, info[0], info[1])

# json file looks like this:
# {
#   "pid": 1,
#   "tid": 1,
#   "data": {
#     "1": {
#       "time": "2020.12.24 20:22:15",
#       "leftController": "0.22, 1.08, 0.11",
#       "rightController": "0.47, 0.94, 0.16",
#       "leftTracker": "0.07, 0.13, -0.22",
#       "rightTracker": "0.46, 0.10, -0.24"
#     },
#     "2": {
#       ...
#     },
#     ...
# }
