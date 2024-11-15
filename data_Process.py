import os
from collections import Counter

import numpy as np
import pandas as pd
import pywt
from imblearn.under_sampling import EditedNearestNeighbours, ClusterCentroids
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def  gait_construct():

    path = 'gait_csv'
    obj = os.scandir(path)
    final_data=pd.DataFrame()
    sorted_entries = sorted(obj, key=lambda entry: entry.name)
    i=1
    result=[]

    for entry in sorted_entries :



        data = pd.read_csv(path+'/'+entry.name)
        data = data.drop(['Measure number', 'Timestamp','MotionDeg','Roll','Pitch','Yaw'], axis=1)

        # num_rows = data.shape[0]
        # numbers = [i] * num_rows
        # i = i + 1
        # result.extend(numbers)

        # data=data.dropna()

        final_data = pd.concat([final_data, data],axis=0)

    final_data.to_csv('sample_dataset/sp_gait_dataset.csv')
    obj.close()


def gait_number():
    path = 'gait_csv'
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
    result=pd.DataFrame(result)
    result.to_csv('sample_dataset/sp_gait_number.csv')


def butter_highpass(cutoff, fs, order=6):
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def gait_removenoise(data):
    cutoff=6
    fs = 84
    b,a = butter_highpass(cutoff,fs)
    y = filtfilt(b,a,data)
    return y

def noise(data):
    final_data = pd.DataFrame()
    for index, row in data.iterrows():
        data = row
        data = data.dropna()
        data = np.array(data)
        new_data = gait_removenoise(data)
        new_data = pd.DataFrame(new_data)

        scaler_standard = StandardScaler()
        standardized_data = scaler_standard.fit_transform(new_data)

        data = standardized_data

        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        normalized_data=pd.DataFrame(normalized_data)

        final_data = pd.concat([final_data, normalized_data], axis=1)

    final_data = final_data.to_csv('p_gait_dataset.csv')


def gait_rebuild():

    data=pd.read_csv('f_gait_dataset.csv')


    final_data=pd.DataFrame()
    multiples_of_13 = [13 * i for i in range(59)]
    data=np.array(data)
    num_columns = data.shape[1]


    step = 13

    for i in range(0, num_columns, step):
        columns = data[:, i:i + step]
        columns=pd.DataFrame(columns)
        final_data = pd.concat([final_data,  columns], axis=0)
        final_data.to_csv('n_f_gait_dataset.csv')
        print(f"Columns {i} to {i + step - 1}:\n{columns}\n")
    x=1




def gait_process(data):

    data=data.T

    new_data = pd.DataFrame()

    data=data.drop(data.index[0])

    for index, row in data.iterrows():
       y = row
       y=np.array(y)
       x = np.arange(len(y))

       if index != 577:

           mask = ~np.isnan(y)
           x_clean = x[mask]
           y_clean = y[mask]

           cs = CubicSpline( x_clean, y_clean)
           x_new = np.linspace(x_clean.min(), x_clean.max(), 983)

           y_new= cs(x_new)
           y_new=pd.DataFrame(y_new)
           new_data=pd.concat([new_data, y_new], axis=1)
    new_data=new_data.to_csv('f_gait_dataset.csv')
    x=1

def data_sample():

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=10, n_classes=3,
                               weights=[0.7, 0.2, 0.1], flip_y=0, random_state=42)

    data=pd.read_csv('sample_dataset/sp_gait_dataset.csv')
    label=pd.read_csv('sample_dataset/sp_gait_label.csv')




    cc = ClusterCentroids(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = cc.fit_resample( data, label)

    #
    #
    print(f"ClusterCentroids欠采样后数据集类别分布: {Counter(y_resampled)}")

    #
    #
    enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=3, kind_sel='all', n_jobs=-1)
    X_final, y_final = enn.fit_resample(X_resampled, y_resampled)


    print(f"ENN欠采样后数据集类别分布: {Counter(y_final)}")
    x = 1



if __name__ == '__main__':
    # gait_number()
    # gait_construct()
    # data=pd.read_csv('gait_dataset.csv')
    # data=data.T
    # # noise(data)
    # data = pd.read_csv('p_gait_dataset.csv')
    # gait_process(data)
    # gait_rebuild()
    data_sample()
















    # new_data=noise(or_data)








    # mse_before = mean_squared_error(or_data, np.zeros_like(or_data))
    # mse_after = mean_squared_error( new_data, np.zeros_like( new_data))
    # print(mse_before)
    # print(mse_after)
    #
    # fs=84
    # t = np.arange(0, 367, 1)
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(t,data, label='Original Signal')
    # plt.title('Original Signal with Low-Frequency Noise')
    # plt.legend()
    # t = np.arange(0, 367, 1)
    # plt.subplot(2, 1, 2)
    # plt.plot(t,r_data, label='Filtered Signal', color='orange')
    # plt.title('Signal after High-Pass Filtering')
    # plt.legend()
    # plt.show()


    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # data=new_data
    # t = np.arange(0, 367, 1 )
    # freqs = np.fft.fftfreq(len( data), 1 / fs)
    # fft_values = np.fft.fft(data)
    # plt.figure(figsize=(10, 6))
    # plt.plot(t,   data)
    # plt.xlabel('时间 [秒]')
    # plt.ylabel('幅度')
    # plt.title('包含低频噪声的时间序列图')
    # plt.grid()
    # plt.show()
    #
    # # 绘制频谱图
    # plt.figure(figsize=(10, 6))
    # plt.plot(freqs, np.abs(fft_values))
    # plt.xlabel('频率 [Hz]')
    # plt.ylabel('幅度')
    # plt.title('频谱图')
    # plt.grid()
    # plt.show()

