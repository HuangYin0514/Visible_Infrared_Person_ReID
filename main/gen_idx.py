import numpy as np


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
