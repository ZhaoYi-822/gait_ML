from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# x_train=pd.read_csv('new_dataset/sp_feature_train_gait_dataset.csv')
# y_train=pd.read_csv('new_dataset/sp_feature_label.csv')
train_data=pd.read_csv('new_gait_dataset/only_sample_gait_dataset.csv')
test_data=pd.read_csv('new_gait_dataset/original_test_gait_dataset.csv')

# x_test=pd.read_csv('new_dataset/sp_feature_test_gait_dataset.csv')
# y_test=pd.read_csv('new_dataset/sp_feature_test_label.csv')


x_train=train_data.iloc[:,:17]
y_train=train_data.iloc[:,17]


x_test=test_data.iloc[:,:17]
y_test=test_data.iloc[:,17]

# x_train=pd.read_csv('sample_dataset/x_train_2.csv')
# y_train=pd.read_csv('sample_dataset/y_train_2.csv')
# x_test=pd.read_csv('sample_dataset/x_test_2.csv')
# y_test=pd.read_csv('sample_dataset/y_test_2.csv')

#
# x_train=np.array(x_train)
# y_train=np.array(y_train)
# x_test=np.array(x_test)
# y_test=np.array(y_test)



# ukn_data=pd.read_csv('C:\\Users\\zhao\\Desktop\\ukn_gait_dataset.csv')
# ukn_label=pd.read_csv('c:\\Users\\zhao\\Desktop\\ukn_gait_label.csv')
#
# x_ts_data=pd.read_csv('sample_dataset/train_dataset.csv')
# y_ts_label_data=pd.read_csv('sample_dataset/train_dataset_label.csv')

Stand_X = StandardScaler()  # 特征进行标准化
x_train = Stand_X.fit_transform(x_train)
x_test= Stand_X.fit_transform(x_test)

# x_ts_data = Stand_X.fit_transform(x_ts_data)
#
# x_train, x_test, y_train, y_test = train_test_split(x_ts_data, y_ts_label_data, test_size=0.4)

# x_train=pd.read_csv('sample_dataset/x_train_2.csv')
# y_train=pd.read_csv('sample_dataset/y_train_2.csv')
# x_test=pd.read_csv('sample_dataset/x_test_2.csv')
# y_test=pd.read_csv('sample_dataset/y_test_2.csv')

# Stand_X = StandardScaler()  # 特征进行标准化
# x_train = Stand_X.fit_transform(x_train)
# x_test= Stand_X.fit_transform(x_test)


# x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
#
# v_train, v_test, vv_train, vv_test = train_test_split(ukn_data, ukn_label, test_size=0.15)
#
# x_val=pd.DataFrame(x_val)
# y_val=pd.DataFrame(y_val)
#
# v_test=pd.DataFrame(v_test)
# vv_test=pd.DataFrame(vv_test)
#
#
#
# xv_data=pd.concat([x_val,v_test],axis=0)
# xv_labels=pd.concat([y_val,vv_test],axis=0)
# normlize = Normalizer()
# x_train=normlize.fit_transform(x_train)
# x_test=normlize.fit_transform(x_test)




def knn():
    knn = KNeighborsClassifier(n_neighbors=5,weights='distance', metric='manhattan')

    knn.fit(x_train, y_train)  # 放入训练数据进行训练

    y_pred = knn.predict(x_test)

    print(knn.predict(x_test))  # 打印预测内容
    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)
    joblib.dump(knn, 'model/only_knn.pkl')

def svm():

    clf = SVC( C=15, kernel='rbf', decision_function_shape='ovr',probability=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)
    joblib.dump(clf, 'model/only_svm.pkl')

def rf():
    forest = RandomForestClassifier(n_estimators=200, random_state=42)
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    joblib.dump(forest, 'model/only_rf.pkl')




if __name__ == '__main__':
    start_time = datetime.now()
    svm()
    end_time = datetime.now()
    #
    # # 计算训练花费的时间
    # training_time = end_time - start_time

    # print(f"模型训练花费的时间为: {training_time}")
    # start_time = datetime.now()
    # knn()
    # end_time = datetime.now()
    # training_time = end_time - start_time
    #
    # print(f"模型训练花费的时间为: {training_time}")

    # start_time = datetime.now()
    # rf()
    # end_time = datetime.now()
    # training_time = end_time - start_time
    #
    # print(f"模型训练花费的时间为: {training_time}")

    # rf()

