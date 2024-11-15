import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC

train_data=pd.read_csv('new_gait_dataset/original_train_gait_dataset.csv')
test_data=pd.read_csv('new_gait_dataset/original_vaild_gait_dataset.csv')

x_train=train_data.iloc[:,:17]
y_train=train_data.iloc[:,17]


x_test=test_data.iloc[:,:17]
y_test=test_data.iloc[:,17]




# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
# ## 创建KNN分类器，选择k=3
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train_scaled, y_train)
#
# # 获取KNN模型对测试集的预测结果
# y_pred_knn = knn.predict(X_test_scaled)
# distances, _ = knn.kneighbors(X_test_scaled)  # 获取每个测试样本的最近邻距离
# # 设置一个距离阈值
# distance_threshold = 1
#
# # 判断样本是否为未知类别
# y_pred_final = np.where(distances[:, 0] > distance_threshold, -1, y_pred_knn)  # -1表示未知类别
#
# # 输出分类报告
# print(classification_report(y_test, y_pred_final))

ovr_svm = OneVsRestClassifier(SVC( C=15, kernel='rbf', decision_function_shape='ovr',probability=True))
ovr_svm.fit(X_train_scaled, y_train)
# 预测概率
y_pred_prob = ovr_svm.predict_proba(X_test_scaled)

# 设置概率阈值
prob_threshold = 0.1  # 可以调整阈值

y_pred_final = []

for sample_probs in y_pred_prob:
    max_prob = np.max(sample_probs)

    # 如果最大概率值低于阈值，则认为是未知类别
    if max_prob < prob_threshold:
        y_pred_final.append(-1)  # -1表示未知类别
    else:
        y_pred_final.append(np.argmax(sample_probs))

# 输出分类报告
print(classification_report(y_test, y_pred_final))
