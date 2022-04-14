# MotionAnalysis

FUNCTION
---
已完成：基于Unity开发VR游戏，逐帧获取Controller和Tracker的位置数据。
该仓库完成以下几个功能：
1. get_cut.py 通过对位置数据进行动作分割、并找出相对应的VR事件，写入data_event&cut文件中。
2. get_space.py 对于不同的VR事件下，确定用户的动作范围
3. get_speed.py 对于不同的VR事件下，确定用户的动作速度
4. get_VR_event.py 从data_event获取VR事件数据
5. get_user_feature.py 从data_user获取用户数据
6. get_dataset.py 对data_event&cut中的动作数据集成相对应的用户数据
7. motion_analysis.py 对dataset进行分析
8. apriori.py 数据关联规则学习
9. ripper.py 基于规则的分类器算法


FILE FOLDER
---
三维数据： VR事件、用户特性、用户动作

data_event
存放VR事件数据

data_user
存放用户特性数据

data_txt data_csv
均存放用户在使用VR设备时骨骼关节点的位置数据，以不同的数据格式存储

---

data_event&cut
存放用户的动作幅度数据和其对应的VR事件

data_event&space
存放VR事件下用户的动作范围数据

data_event&speed
存放VR事件下用户的动作速度数据

Dataset
合并用户动作数据、特性数据、VR事件数据

---

images
存放matplotlib画的各种图

util
各种工具文件


PROCESS
---
1. txt_to_csv.py
2. get_cut.py
3. get_space.py
4. get_speed.py
5. generate_user_feature()
6. get_dataset.py
7. apriori/ripper

可视化：draw_cut draw_box motion_analysis