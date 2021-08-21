# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt


#从predict_ret.txt中获取真实类别值和对应的概率得分值
filename = './predict_ret.txt'

file = open(filename)
lines = file.readlines()
#lines = np.array(lines)
# print lines
# ['0.94\t0.81\t...0.62\t\n', ... ,'0.92\t0.86\t...0.62\t\n']形式
rows = len(lines)  # 文件行数
datamat = np.zeros((rows, 4))  # 初始化矩阵
row = 0
for line in lines:
    line = line.strip().split(',')  # strip()默认移除字符串首尾空格或换行符
    datamat[row, :] = line[:]
    row += 1

real_cate = datamat


#for i in y_test:


##y_test相当于真实值，注意，roc曲线仅适用于二分类问题，多分类问题应先转化为二分类
y_test = real_cate[:,0]
# for i in range(len(y_test)):
#     if y_test[i] == 13:
#         y_test[i] = 1
#     else:
#         y_test[i] = 0
#y_score 根据x_test预测出的y_pre,根据出现的概率大小进行排列
y_score = real_cate[:,1]
##
fpr,tpr,thre = roc_curve(y_test,y_score)
##计算auc的值，就是roc曲线下的面积
auc = auc(fpr,tpr)

##画图
plt.plot(fpr,tpr,color = 'darkred',label = 'roc area:(%0.2f)'%auc)
plt.plot([0,1],[0,1],linestyle = '--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('roc_curve')
plt.legend(loc = 'lower right')
plt.show()

# from sklearn.preprocessing import LabelEncoder
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle
# import pandas as pd
# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
#
# data = pd.read_csv('/media/image/DevinWZM/C3D-tensorflow-master/predict_ret.csv', header=None)
# ##将第四列的无序非数值型数据转为数值型数据
# y = data[[4]]
# class_le = LabelEncoder()
# y = class_le.fit_transform(y.values.ravel())
#
# ##对数据进行改造，成为二分类问题
# X = data[[0, 1, 2, 3]][y != 2]
# y = y[y != 2]
#
# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
#
# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)
#
# ## Learn to predict each class against the other
# classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
#
# ##由decision_function函数得到y_score
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)
#
# fpr, tpr, thre = roc_curve(y_test, y_score)
# #
# roc_auc = auc(fpr, tpr)
# #
# plt.figure()
# lw = 2
# plt.figure(figsize=(9, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=lw,
#          label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw,
#          linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
