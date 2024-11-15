import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def draw_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    # 使用Seaborn绘制混淆矩阵
    plt.figure(figsize=(25, 25))
    sns.heatmap(cm, annot=True, fmt='d',cmap='Blues', xticklabels=range(50), yticklabels=range(50))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    # plt.show()

    report = classification_report(y_true, y_pred, zero_division=0,digits=4)

    print(report)


def rf_pre(val_data,val_label):

    threshold = 0.6
    rf = joblib.load("model/only_rf.pkl")

    pre = rf.predict(val_data)
    y_prob = rf.predict_proba(val_data)

    y_pred = np.argmax(y_prob, axis=1)
    max_prob = np.max(y_prob, axis=1)
    unknown_samples = np.max(y_prob, axis=1)

    pre[max_prob < threshold] = 51
    num = np.sum(unknown_samples)

    accuracy = accuracy_score(val_label, pre)
    print('Accuracy:', accuracy)
    return pre

def knn_pre(val_data, val_label):

    threshold = 0.9
    rf = joblib.load("model/only_knn.pkl")

    pre = rf.predict(val_data)
    y_prob = rf.predict_proba(val_data)

    y_pred = np.argmax(y_prob, axis=1)
    max_prob = np.max(y_prob, axis=1)
    unknown_samples = np.max(y_prob, axis=1)

    pre[max_prob < threshold] = 51
    num = np.sum(unknown_samples)

    accuracy = accuracy_score(val_label, pre)
    print('Accuracy:', accuracy)

    return pre


def svm_pre(val_data, val_label):
    threshold = 0.7
    rf = joblib.load("model/only_svm.pkl")

    pre = rf.predict(val_data)
    y_prob = rf.predict_proba(val_data)

    y_pred = np.argmax(y_prob, axis=1)
    max_prob = np.max(y_prob, axis=1)
    unknown_samples = np.max(y_prob, axis=1)

    pre[max_prob < threshold] = 51
    num = np.sum(unknown_samples)

    accuracy = accuracy_score(val_label, pre)
    print('Accuracy:', accuracy)

    return pre


def lstm_pre(val_data, val_label):
    val_data = np.array(val_data)
    val_label = np.array(val_label)

    val_data = torch.tensor(val_data, dtype=torch.float32)
    val_label=torch.tensor(val_label, dtype=torch.long)
    threshold = 0.7
    lstm = torch.jit.load('/content/model_scripted.pt')

    with torch.no_grad():
      pre = lstm(val_data)
      predicted_classes = torch.argmax(pre, dim=1)
      predicted_classes = pd.DataFrame(predicted_classes)
      print(predicted_classes)

      # pre = lstm(predicted_classes)

    max_prob = np.max(pre, axis=1)
    # unknown_samples = np.max(y_prob, axis=1)

    pre[max_prob < threshold] = 50
    # num = np.sum(unknown_samples)

    accuracy = accuracy_score(val_label, pre)
    print('Accuracy:', accuracy)

    return pre





if __name__=='__main__':
     data= pd.read_csv('new_gait_dataset/original_vaild_gait_dataset.csv')
     val_data=data.iloc[:,:17]
     val_label =data.iloc[:,17]
     # pre = rf_pre(val_data, val_label)
     # draw_matrix(val_label, pre)
     # pre=knn_pre(val_data,val_label )
     # draw_matrix(val_label, pre)

     Stand_X = StandardScaler()  # 特征进行标准化
     val_data= Stand_X.fit_transform(val_data)
     pre = svm_pre(val_data, val_label)
     draw_matrix(  val_label,pre)






    # pre=pd.read_csv('C:\\Users\\zhao\\Desktop\\thrs_pre.csv')



    # pre=lstm_pre(val_data,val_label)

    # pre=knn_pre(val_data,val_label )
    # draw_matrix(val_label, pre)
    #
    # pre = svm_pre(val_data, val_label)
    # draw_matrix(  val_label,pre)



