import glob
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io
from config_plot import *

OUTPUT_DIR = "./analysis/results/ablation"


def plot_parameter_lossWeight():
    # 设置数据 --------------------------------------------------------
    x_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    map_values = [56.32, 57.57, 58.16, 58.68, 58.59, 58.73, 58.76, 59.12, 59.00, 58.62, 58.69, 58.44, 58.45]
    rank1_values = [67.24, 68.55, 69.46, 69.82, 69.86, 70.18, 70.63, 71.18, 70.27, 70.09, 70.50, 70.23, 70.23]

    # 创建画布 --------------------------------------------------------
    LINE_NUM = 1
    ROW_NUM = 1
    FIGSIZE = (4 * ROW_NUM, 3 * LINE_NUM)
    fig, axis = plt.subplots(LINE_NUM, ROW_NUM, figsize=FIGSIZE, dpi=LATAX_DPI)
    axis_1 = axis
    axis_2 = axis_1.twinx()  #  创建双y轴

    # 图参数
    max_values = max([max(m, r) for m, r in zip(map_values, rank1_values)])
    min_values = min([min(m, r) for m, r in zip(map_values, rank1_values)])
    offset = int((max_values - min_values) * 0.4)

    # 绘制左侧轴
    axis_1.set_ylim(min_values - offset, max_values - offset)
    axis_1.set_ylabel("mAP (%)")

    # 绘制右侧轴
    axis_2.set_ylim(min_values + offset, max_values + offset)
    axis_2.set_ylabel("Rank-1 (%)")

    # 绘制横轴
    axis_1.set_xlabel(r"Parameter $\lambda$")

    # 绘制图像
    axis_1.plot(x_values, map_values, label="mAP", color="#FD625E", marker="o", markersize=MARKERSIZE, zorder=2)
    axis_2.plot(x_values, rank1_values, label="Rank-1", color="#71D4EB", marker="o", markersize=MARKERSIZE, zorder=2)

    # 图像调整
    # axis_1.grid(axis="x")
    axis_2.grid(False)
    handles1, labels1 = axis_1.get_legend_handles_labels()
    handles2, labels2 = axis_2.get_legend_handles_labels()
    axis_2.legend(handles1 + handles2, labels1 + labels2).get_frame().set_alpha(1)
    plt.tight_layout()

    # 保存
    # plt.show()
    output_file = os.path.join(OUTPUT_DIR, "parameter_lossWeight.png")
    plt.savefig(output_file, bbox_inches="tight")
    print(f"图像已保存到: {output_file}")


if __name__ == "__main__":
    # Path check
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)  # 删除整个目录及其内容
    OUTPUT_DIR = Path(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # parameter of loss weight
    plot_parameter_lossWeight()
