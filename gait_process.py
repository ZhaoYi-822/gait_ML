import os

import pandas as pd


def drop_feature():
    path = 'gait_csv'
    obj = os.scandir(path)
    sorted_entries = sorted(obj, key=lambda entry: entry.name)

    for entry in sorted_entries:
        name=entry.name
        data = pd.read_csv(path + '/' + entry.name)

        data = data.drop(['Measure number', 'Timestamp', 'MotionDeg', 'Roll', 'Pitch', 'Yaw'], axis=1)
        data=data.dropna()
        data.to_csv('reduce_feature/'+name,index=False)
    obj.close()

def combine_data():
    path = 'original_data'
    obj = os.scandir(path)
    final_data = pd.DataFrame()
    sorted_entries = sorted(obj, key=lambda entry: entry.name)
    for entry in sorted_entries:
        data = pd.read_csv(path + '/' + entry.name)
        final_data = pd.concat([final_data, data], axis=0)
    # final_data.to_csv('sample_dataset/original_gait_dataset.csv',index=False)
    obj.close()

def data_label():
    path = 'original_data'
    obj = os.scandir(path)
    sorted_entries = sorted(obj, key=lambda entry: entry.name)

    result = []
    # 遍历每个 CSV 文件
    for i, file_name in enumerate(os.listdir(path), start=1):
        # 读取 CSV 文件
        file_path = os.path.join(path, file_name)

        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 获取行数

        num_rows = df.shape[0]

        # 生成相应数量的数字
        numbers = [i] * num_rows

        # 将结果添加到列表中
        result.extend(numbers)
    result = pd.DataFrame(result)
    print(result)
    result.to_csv('sample_dataset/original_gait_label.csv',index=False)


def original_data():
    path = 'gait_csv'
    obj = os.scandir(path)
    sorted_entries = sorted(obj, key=lambda entry: entry.name)

    for entry in sorted_entries:
        name = entry.name
        data = pd.read_csv(path + '/' + entry.name)

        data = data.drop(['Measure number', 'Timestamp'], axis=1)
        data = data.dropna()
        data.to_csv('original_data/' + name, index=False)
    obj.close()

if __name__ == '__main__':

    # drop_feature()
    # combine_data()
    # data_label()
    original_data()