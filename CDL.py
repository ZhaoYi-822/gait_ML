import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 生成模拟IMU步态数据（包含低频噪声）
np.random.seed(0)
t = np.linspace(0, 10, 500)
signal = np.sin(2 * np.pi * 0.75 * t) + np.sin(2 * np.pi * 1.25 * t) + np.random.normal(0, 0.5, t.shape)

# 设计Butterworth高通滤波器
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# 应用高通滤波器
def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 设置参数
fs = 50.0  # 采样频率
cutoff = 1.0  # 截止频率

# 滤波
filtered_signal = highpass_filter(signal, cutoff, fs)

# 绘制原始信号和滤波后信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.title('Original Signal with Low-Frequency Noise')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, label='Filtered Signal', color='orange')
plt.title('Signal after High-Pass Filtering')
plt.legend()

plt.tight_layout()
plt.show()
