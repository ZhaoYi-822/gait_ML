import time as ti

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, \
    confusion_matrix
import pingouin as pg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def original():
    # 创建已知类数据集
    known_data =pd.read_csv('new_gait_dataset/original_gait_dataset.csv')
    unknown_data = pd.read_csv('new_gait_dataset/unknow_data.csv')

    # 提取已知类数据的特征和标签
    X_known = known_data.iloc[:, :17]
    y_known = known_data.iloc[:, 17]

    # 合并已知类和未知类数据集用于交叉验证
    data = pd.concat([known_data, unknown_data])

    # 提取所有数据的特征和标签
    X = data.iloc[:, :17]
    y = data.iloc[:, 17]

    # 定义模型
    model =  RandomForestClassifier(n_estimators=200, random_state=42)
    # model = KNeighborsClassifier(n_neighbors=9,weights='distance', metric='manhattan')
    # model= SVC( C=15, kernel='rbf', decision_function_shape='ovr',probability=True)

    # 定义 K 折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    unknown_splits = np.array_split(unknown_data, 6)
    # 存储每次交叉验证的得分
    scores = []
    pre=[]
    rec=[]
    f1score=[]
    train_time=0
    t0 = ti.time()

    xx=kf.split(X_known)
    Stand_X = StandardScaler()  # 特征进行标准化

    # 进行 K 折交叉验证
    for i , (train_index, test_index) in enumerate(kf.split(X_known)):
        # 训练集只包含已知类数据


        X_train, y_train = X_known.iloc[train_index], y_known.iloc[train_index]
        X_test_known,  y_test_known= X_known.iloc[test_index], y_known.iloc[test_index]




        X_test=pd.concat([X_test_known,unknown_splits[i].iloc[:, :17]], ignore_index=True)
        y_test=pd.concat([y_test_known, unknown_splits[i].iloc[:, 17]], ignore_index=True)

        X_train=Stand_X.fit_transform(X_train)
        X_test=Stand_X.transform(X_test)

        model.fit(X_train, y_train)

        # knn_thres=0.7
        # rf_thres=0.7
        threshold=0.7
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        max_prob = np.max(y_prob, axis=1)
        unknown_samples = np.max(y_prob, axis=1)
        y_pred[max_prob < threshold] = 51
        #
        # accsore=accuracy_score(y_test, y_pred)
        # print(accsore)

        cm = confusion_matrix(y_test, y_pred)

        diam = np.trace(cm)
        sum = cm.sum()

        TN = cm[-1, -1]
        TP = diam - TN

        FP = cm[-1, :].sum() - TN
        FN = cm[:, -1].sum() - TN
        # FN=sum-TN-TP-FP

        accuracy = diam / sum
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        scores.append(accuracy*100)
        pre.append(precision*100)
        rec.append(recall*100)
        f1score.append(f1)


    t1 = ti.time()
    time = t1 - t0
    train_time = train_time + time

    print("交叉验证得分:", scores)
    print("平均交叉验证得分:", np.mean(scores))

    print("精确率得分:", pre)
    print("平均精确率:", np.mean(pre))

    print("精确率得分:", rec)
    print("平均召回率:", np.mean(rec))

    print("F1值:", f1score)
    print("平均F1值:", np.mean(f1score))

    print( "时间", train_time)

    return scores



def feature_only():

    known_data = pd.read_csv('new_gait_dataset/only_feature.csv')
    unknown_data = pd.read_csv('new_gait_dataset/only_feature_vaild.csv')

    X_known = known_data.iloc[:, :13]
    y_known = known_data.iloc[:, 13]

    data = pd.concat([known_data, unknown_data])

    X = data.iloc[:, :13]
    y = data.iloc[:, 13]

    model =  RandomForestClassifier(n_estimators=200, random_state=42)
    # model = KNeighborsClassifier(n_neighbors=9, weights='distance', metric='manhattan')
    # model = SVC(C=15, kernel='rbf', decision_function_shape='ovr', probability=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    unknown_splits = np.array_split(unknown_data, 6)


    scores = []
    pre=[]
    rec=[]
    f1score=[]
    train_time=0
    t0 = ti.time()
    for i , (train_index, test_index) in enumerate(kf.split(X_known)):
        # 训练集只包含已知类数据
        X_train, y_train = X_known.iloc[train_index], y_known.iloc[train_index]
        X_test_known,  y_test_known= X_known.iloc[test_index], y_known.iloc[test_index]

        X_test=pd.concat([X_test_known, unknown_splits[i].iloc[:, :13]], ignore_index=True)
        y_test=pd.concat([y_test_known, unknown_splits[i].iloc[:, 13]], ignore_index=True)


        # 训练模型
        model.fit(X_train, y_train)

        # knn_threshold = 0.9
        threshold=0.7
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        max_prob = np.max(y_prob, axis=1)
        unknown_samples = np.max(y_prob, axis=1)
        y_pred[max_prob < threshold] = 51

        # 计算准确率并存储
        # score = accuracy_score(y_test,  y_pred)
        # precession=precision_score(y_test, y_pred, average='macro')
        # recall=recall_score(y_test, y_pred, average='micro')
        # f1=f1_score(y_test, y_pred, average='macro')
        #
        # report = classification_report(y_test, y_pred, zero_division=0, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        diam = np.trace(cm)
        sum = cm.sum()

        TN=cm[-1, -1]
        TP=diam-TN


        FP=cm[-1,:].sum()-TN
        FN = cm[:, -1].sum() - TN
        # FN=sum-TN-TP-FP

        accuracy=diam/sum
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        f1=2* precision*recall/( precision+recall)


        scores.append(accuracy)
        pre.append( precision)
        rec.append(recall)
        f1score.append(f1)



    acc=scores
    print("交叉验证得分:", scores*100)
    print("平均交叉验证得分:", np.mean(scores)*100)

    print("精确率得分:", pre*100)
    print("平均精确率:", np.mean(pre)*100)

    print("精确率得分:", rec*100)
    print("平均召回率:", np.mean(rec)*100)

    print("F1值:", f1score*100)
    print("平均F1值:", np.mean(f1score))

    t1 = ti.time()
    time = t1 - t0
    train_time = time + train_time
    print(train_time)

    return acc



def sample_only():

    known_data = pd.read_csv('new_gait_dataset/only_sample_original_gait_dataset.csv')
    unknown_data = pd.read_csv('new_gait_dataset/only_sample_original_vaild_gait_dataset.csv')

    X_known = known_data.iloc[:, :17]
    y_known = known_data.iloc[:, 17]

    data = pd.concat([known_data, unknown_data])

    X = data.iloc[:, :17]
    y = data.iloc[:, 17]

    model =  RandomForestClassifier(n_estimators=211, random_state=42)
    # model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')
    # model = SVC(C=15, kernel='rbf', decision_function_shape='ovr', probability=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    unknown_splits = np.array_split(unknown_data, 6)

    scores = []
    pre = []
    rec = []
    f1score = []
    train_time = 0
    t0 = ti.time()

    for i , (train_index, test_index) in enumerate(kf.split(X_known)):
        # 训练集只包含已知类数据
        X_train, y_train = X_known.iloc[train_index], y_known.iloc[train_index]
        X_test_known,  y_test_known= X_known.iloc[test_index], y_known.iloc[test_index]

        X_test=pd.concat([X_test_known, unknown_splits[i].iloc[:, :17]], ignore_index=True)
        y_test=pd.concat([y_test_known, unknown_splits[i].iloc[:, 17]], ignore_index=True)

        # 训练模型
        model.fit(X_train, y_train)

        # knn_threshold = 0.7
        #rf_threshold=0.7

        threshold=0.7
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        max_prob = np.max(y_prob, axis=1)
        unknown_samples = np.max(y_prob, axis=1)
        y_pred[max_prob < threshold] = 51

        cm = confusion_matrix(y_test, y_pred)

        diam = np.trace(cm)
        sum = cm.sum()

        TN = cm[-1, -1]
        TP = diam - TN

        FP = cm[-1, :].sum() - TN
        FN = cm[:, -1].sum() - TN
        # FN=sum-TN-TP-FP

        accuracy = diam / sum
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        scores.append(accuracy )
        pre.append(precision )
        rec.append(recall )
        f1score.append(f1)

    acc = scores
    print("交叉验证得分:", scores)
    print("平均交叉验证得分:", np.mean(scores)*100)

    print("精确率得分:", pre)
    print("平均精确率:", np.mean(pre) * 100)

    print("精确率得分:", rec)
    print("平均召回率:", np.mean(rec) * 100)

    print("F1值:", f1score )
    print("平均F1值:", np.mean(f1score))

    t1 = ti.time()
    time = t1 - t0
    train_time = time + train_time
    print(train_time)

    return acc


def both_data():

    known_data = pd.read_csv('new_gait_dataset/both_original_dataset.csv')
    unknown_data = pd.read_csv('new_gait_dataset/both_original_vaild_dataset.csv')

    X_known = known_data.iloc[:, :13]
    y_known = known_data.iloc[:, 13]

    data = pd.concat([known_data, unknown_data])

    X = data.iloc[:, :13]
    y = data.iloc[:, 13]

    model =  RandomForestClassifier(n_estimators=211, random_state=42)
    # model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')
    # model = SVC(C=15, kernel='rbf', decision_function_shape='ovr', probability=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    unknown_splits = np.array_split(unknown_data, 8)

    scores = []
    pre = []
    rec = []
    f1score = []
    train_time = 0
    t0 = ti.time()

    for i , (train_index, test_index) in enumerate(kf.split(X_known)):
        # 训练集只包含已知类数据
        X_train, y_train = X_known.iloc[train_index], y_known.iloc[train_index]
        X_test_known,  y_test_known= X_known.iloc[test_index], y_known.iloc[test_index]

        X_test=pd.concat([X_test_known, unknown_splits[i].iloc[:, :13]], ignore_index=True)
        y_test=pd.concat([y_test_known, unknown_splits[i].iloc[:, 13]], ignore_index=True)

        # 训练模型
        model.fit(X_train, y_train)
        # knn_threshold = 0.85
        #rf_threshold=0.55


        threshold=0.
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        max_prob = np.max(y_prob, axis=1)
        unknown_samples = np.max(y_prob, axis=1)
        y_pred[max_prob < threshold] = 51

        cm = confusion_matrix(y_test, y_pred)

        diam = np.trace(cm)
        sum = cm.sum()

        TN = cm[-1, -1]
        TP = diam - TN

        FP = cm[-1, :].sum() - TN
        FN = cm[:, -1].sum() - TN
        # FN=sum-TN-TP-FP

        accuracy = diam / sum
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        scores.append(accuracy)
        pre.append(precision)
        rec.append(recall)
        f1score.append(f1)

    acc = scores
    print("交叉验证得分:", scores)
    print("平均交叉验证得分:", np.mean(scores) * 100)

    print("精确率得分:", pre)
    print("平均精确率:", np.mean(pre) * 100)

    print("精确率得分:", rec)
    print("平均召回率:", np.mean(rec) * 100)

    print("F1值:", f1score)
    print("平均F1值:", np.mean(f1score))

    t1 = ti.time()
    time = t1 - t0
    train_time = time + train_time
    print(train_time)

    return acc






if __name__=='__main__':

    # original_score=original()

    rf_original_score=[0.94947209653092, 0.969076052796983, 0.9788812067881835,0.9634146341463415,0.9811415639929595]

    # knn_original_score=[0.9023261961406291, 0.912888301387971, 0.9294117647058823, 0.9079851930195663, 0.9205446853516658]

    original_score= rf_original_score

    # only_feature=feature_only()
    # t_stat, p_value = stats.ttest_ind(original_score,only_feature)
    # p_value = pg.ttest(original_score, only_feature)
    # print(p_value['p-val'])

    # only_sample=sample_only()
    #
    # t_stat, p_value = stats.ttest_ind(original_score, only_sample)
    # p_value = pg.ttest(original_score, only_sample)
    # print(p_value['p-val'])
    #
    both=both_data()
    t_stat, p_value = stats.ttest_ind( original_score,   both)
    p_value = pg.ttest(original_score, both)

    print(p_value['p-val'])