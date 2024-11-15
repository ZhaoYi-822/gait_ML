import pandas as pd
from sklearn.model_selection import train_test_split


def data_random_split():
    data=pd.read_csv('new_gait_dataset/unknow_data.csv')
    x_ts_data=data.iloc[:,:13]
    y_ts_label_data=data.iloc[:,13]

    x_train, x_test, y_train, y_test = train_test_split(x_ts_data, y_ts_label_data, train_size=0.2)
    x_train.to_csv('new_gait_dataset/original_vaild_gait_dataset_2.csv', index=False)
    print(x_train)


import pandas as pd

# 生成一个示例数据集
def data_split():
    df = pd.read_csv('new_dataset/original_gait_dataset.csv')
    # 按照category进行分组，并对每个组进行80%抽样
    sampled_df_60 = df.groupby('0').apply(lambda x: x.iloc[:int(len(x) * 0.6)]).reset_index(drop=True)
    sampled_df_60.to_csv('new_dataset/train_gait_dataset.csv', index=False)
    # 按类别进行分组，并按顺序抽取每个类别的后40%的数据
    sampled_df_40 = df.groupby('0').apply(lambda x: x.iloc[int(len(x) * 0.6):]).reset_index(drop=True)

    # 将剩下的40%数据再划分为两个50%的数据
    sampled_df_40_part1 = sampled_df_40.groupby('0').apply(lambda x: x.iloc[:int(len(x) * 0.5)]).reset_index(drop=True)
    sampled_df_40_part2 = sampled_df_40.groupby('0').apply(lambda x: x.iloc[int(len(x) * 0.5):]).reset_index(drop=True)
    sampled_df_40_part1.to_csv('new_dataset/test_gait_dataset.csv', index=False)
    sampled_df_40_part2.to_csv('new_dataset/valid_gait_dataset.csv', index=False)
    print(sampled_df_40_part2)

def unknown_split():
    df = pd.read_csv('new_dataset/unknow_data.csv')
    sampled_df_p1 =df.groupby('0').apply(lambda x: x.iloc[:int(len(x) * 0.2)]).reset_index(drop=True)
    sampled_df_p1.to_csv('new_dataset/unknow_vaild_data.csv', index=False)

    df=pd.read_csv('new_dataset/feature_unknown_class.csv')
    sampled_df_p2 = df.groupby('0').apply(lambda x: x.iloc[:int(len(x) * 0.2)]).reset_index(drop=True)
    sampled_df_p2.to_csv('new_dataset/feature_unknow_vaild_data.csv', index=False)


def sample_data():

    # data=pd.read_csv('new_dataset/feature_train_gait_dataset.csv')
    # data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    # data.to_csv('new_dataset/sp_feature_train_gait_dataset.csv', index=False)

    data = pd.read_csv('new_dataset/feature_test_gait_dataset.csv')
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data=data.to_csv('new_dataset/sp_feature_test_gait_dataset.csv', index=False)









if __name__ == '__main__':
   # data_split()
   # unknown_split()
   # sample_data()
   data_random_split()