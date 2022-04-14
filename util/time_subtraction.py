# perform time2-time1
# input time format: 2020.12.28 16:27:45:123
# output result format: xxx.xxx(seconds)
def time_sub_ms(time1, time2):
    time1_h_m_s_ms = get_h_m_s_ms(time1)
    time2_h_m_s_ms = get_h_m_s_ms(time2)
    seconds = float((time2_h_m_s_ms[0] - time1_h_m_s_ms[0]) * 3600 + (time2_h_m_s_ms[1] - time1_h_m_s_ms[1]) * 60 + (
            time2_h_m_s_ms[2] - time1_h_m_s_ms[2]) + float((time2_h_m_s_ms[3] - time1_h_m_s_ms[3]) / 1000))
    return seconds


def get_h_m_s_ms(time):
    time = time.split(" ")[1]
    # print(time)
    time_h_m_s_ms = list()
    time_h_m_s_ms.append(int(time.split(":")[0]))  # hour
    time_h_m_s_ms.append(int(time.split(":")[1]))  # minute
    time_h_m_s_ms.append(int(time.split(":")[2]))  # second
    if len(time.split(":")) == 4:
        time_h_m_s_ms.append(int(time.split(":")[3]))  # millisecond

    else:
        time_h_m_s_ms.append(0)  # millisecond
    return time_h_m_s_ms
