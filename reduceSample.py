import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def stand():
    data=pd.read_csv('C:\\Users\\zhao\\Desktop\\x_train_2.csv')
    train_data=data.iloc[:,:13]
    Stand_X = StandardScaler()
    train_data = Stand_X.fit_transform(train_data)
    print(train_data)






def calculate_sample():
    data = pd.read_csv('new_gait_dataset/only_feature_vaild.csv')
    column_means =data.mean()
    class_means = data.groupby('0').mean()
    class_var = data.var()
    each_classnumber=data.groupby('0').size()
    each_classnumber=np.array(each_classnumber)

    sample=np.array([])


    for i in range(8):

        number=0

        for j in range(13):

            x=class_var[j]
            k_mean=class_means.iloc[i, j]
            cl_mean=column_means[j]

            n= math.pow(1.96,2)*x/math.pow((k_mean-cl_mean),2)
            x=each_classnumber[1]

            if n>each_classnumber[i]:
                n=each_classnumber[i]
            number=n+number
        number=math.ceil(number/13)
        sample=np.append(sample,number)
    rows_to_delete = {i + 51: int(sample[i] )for i in range(len(sample))}

    print(rows_to_delete)

    for key, value in rows_to_delete.items():
        data = data.drop(data[data['0'] == key].sample(value).index)
        # filtered_df.to_csv('new_dataset/sample_train_gait_dataset.csv', index=False)
    each_classnumber = data.groupby('0').size()
    data.to_csv('new_gait_dataset/both_original_vaild_dataset.csv', index=False)
    print(each_classnumber)

def reduce_sample(data,rows_to_delete):
    # rows_to_delete = {
    #     1: 75,  # 类别1删除50行
    #     2: 99,  # 类别2删除30行
    #     3: 117,  # 类别3删除70行
    #     4: 141,  # 类别4删除40行
    #     5: 108,  # 类别5删除25行
    #     6: 121,  # 类别6删除55行
    #     7: 124,  # 类别7删除35行
    #     8: 220,  # 类别8删除45行
    #     9: 149,  # 类别9删除20行
    #     10: 135,  # 类别10删除60行
    #     11: 102,  # 类别12删除40行
    #     12: 235,  # 类别13删除30行
    #     13: 58,  # 类别14删除20行
    #     14: 106,  # 类别15删除15行
    #     15: 104,  # 类别16删除30行
    #     16: 154,  # 类别17删除40行
    #     17: 150,  # 类别18删除20行
    #     18: 146,  # 类别19删除30行
    #     19: 147,  # 类别20删除20行
    #     20: 148,  # 类别21删除15行
    #     21: 159,  # 类别22删除50行
    #     22: 131,  # 类别23删除30行
    #     23: 122,  # 类别24删除40行
    #     24: 87,  # 类别25删除20行
    #     25: 120,  # 类别26删除15行
    #     26: 124,  # 类别27删除30行
    #     27: 117,  # 类别28删除40行
    #     28: 179,  # 类别29删除50行
    #     29: 81,  # 类别30删除30行
    #     30: 162,  # 类别31删除40行
    #     31: 98,  # 类别32删除20行
    #     32: 178,  # 类别33删除15行
    #     33: 211,  # 类别34删除30行
    #     34: 185,  # 类别35删除40行
    #     35: 182,  # 类别36删除50行
    #     36: 180,  # 类别37删除30行
    #     37: 157,  # 类别37删除30行
    #     38: 325,  # 类别38删除40行
    #     39: 79,  # 类别39删除50行
    #     40: 214,  # 类别40删除30行
    #     41: 235,  # 类别41删除40行
    #     42: 239,  # 类别42删除50行
    #     43: 139,  # 类别43删除30行
    #     44: 245,  # 类别44删除40行
    #     45: 191,  # 类别45删除50行
    #     46: 290,  # 类别46删除30行
    #     47: 350,  # 类别47删除40行
    #     48: 206,  # 类别48删除50行
    #     49: 122,  # 类别49删除30行
    #     50: 237 # 类别50删除40行
    #
    # }

    # filtered_df = data.groupby('0').apply(lambda x: x.iloc[:-rows_to_delete.get(x.name, 0)]).reset_index(drop=True)

    for key, value in rows_to_delete.items():


        data = data.drop(data[data['0'] == key].sample(value).index)
    # filtered_df.to_csv('new_dataset/sample_train_gait_dataset.csv', index=False)
    each_classnumber = data.groupby('0').size()
    # data.to_csv('new_gait_dataset/only_sample_original_gait_dataset.csv', index=False)
    print(each_classnumber)
    # print(filtered_df)






if __name__ == '__main__':
    # stand()


    rows_to_delete=calculate_sample()
    # reduce_sample(data,rows_to_delete)


