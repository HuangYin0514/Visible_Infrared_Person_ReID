def compare_by_line(file1, file2):
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        for i, (line1, line2) in enumerate(zip(f1, f2), start=1):
            if line1 != line2:
                print(f"第 {i} 行不同：")
                print(f"文件1: {line1.strip()}")
                print(f"文件2: {line2.strip()}")
                return False
    return True


if __name__ == "__main__":

    # path_a = "/Users/drhy/Documents/projects/Visible_Infrared_Person_ReID/_dataset_processing/RegDB/idx/test_thermal_.txt"
    # path_b = "/Users/drhy/Documents/projects/Visible_Infrared_Person_ReID/_dataset_processing/test/RegDB/idx/test_thermal_1.txt"
    # compare_by_line(path_a, path_b)
    import os

    def get_txt_files(folder_path):
        return [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    # 示例
    folder = r"/Users/drhy/Documents/projects/Visible_Infrared_Person_ReID/_dataset_processing/test/RegDB/idx/"
    txt_files = get_txt_files(folder)
    for txt_file in txt_files:
        path_a = "/Users/drhy/Documents/projects/Visible_Infrared_Person_ReID/_dataset_processing/RegDB/idx/" + txt_file
        path_b = "/Users/drhy/Documents/projects/Visible_Infrared_Person_ReID/_dataset_processing/test/RegDB/idx/" + txt_file
        compare_by_line(path_a, path_b)
        print("All files are the same.")
