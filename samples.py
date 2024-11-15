from imblearn.under_sampling import ClusterCentroids, EditedNearestNeighbours
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter

# 生成一个不平衡的多类别数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=10, n_classes=3, weights=[0.7, 0.2, 0.1], flip_y=0, random_state=42)

# 查看原始数据集的类别分布
print(f"原始数据集类别分布: {Counter(y)}")

# 使用ClusterCentroids进行初步欠采样
cc = ClusterCentroids(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = cc.fit_resample(X, y)

# 查看ClusterCentroids欠采样后的数据集类别分布
print(f"ClusterCentroids欠采样后数据集类别分布: {Counter(y_resampled)}")

# 使用ENN进行进一步欠采样
enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=3, kind_sel='all', n_jobs=-1)
X_final, y_final = enn.fit_resample(X_resampled, y_resampled)

# 查看ENN欠采样后的数据集类别分布
print(f"ENN欠采样后数据集类别分布: {Counter(y_final)}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=42)

# 使用随机森林进行训练
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测并评估模型
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
