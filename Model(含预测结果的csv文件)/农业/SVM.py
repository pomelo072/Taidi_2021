import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # 忽略版本问题
from sklearn.model_selection import GridSearchCV
# 随机重复采样
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
print('------------svc------------')
#读取数据
data = pd.read_csv('../data/农、林、牧、渔业_处理.csv',engine = "python")
data2 = pd.read_csv('../data/农、林、牧、渔业_处理.csv',engine = "python")
# print("data：\n")
# print(data)
data2.drop(data2.columns[[0]],axis = 1,inplace = True)
# print("data2删除第一列：\n")
# print(data2)
data2.drop(columns = ['FLAG'],axis = 1,inplace = True)
# print("data2删除FLAG：\n")
#定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(sampling_strategy={1:21,0:196 },random_state=42)
# ros = RandomOverSampler(random_state=30,sampling_strategy=0.02)
X_smo,y_smo = smo.fit_resample(data2.iloc[:,:],data['FLAG'])
sc=StandardScaler()
sc.fit(X_smo)#计算样本的均值和标准差
X_smo=pd.DataFrame(sc.transform(X_smo))

#拆分专家样本集
data_tr, data_te, label_tr, label_te = train_test_split(X_smo,y_smo,test_size=0.3)
print(label_tr.groupby(label_tr).count())
#模型构建
tuned_parameters = {"C":[0.1, 1, 2], "gamma":[2,1, 0.1, 0.01]}
clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=8,scoring='roc_auc',n_jobs = 8, verbose=1)
clf.fit(data_tr,label_tr)
pre = clf.predict(data_te)
label_te_true = np.array(label_te)
print("the SVM model auc: %.4g" % metrics.roc_auc_score(label_te_true,pre))
print(classification_report(label_te_true,pre))
print("auc值为:", roc_auc_score(label_te_true,pre))


# 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
print(clf.best_params_)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

# 看一下具体的参数间不同数值的组合后得到的分数是多少
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

pre = clf.predict(data_te)



label_te_true = np.array(label_te)
print("the SVM model auc: %.4g" % metrics.roc_auc_score(label_te_true,pre))
print(classification_report(label_te_true,pre))
print("auc值为:", roc_auc_score(label_te_true,pre))

# 特征重要性排名
# lr_predictors = [i for i in data2.iloc[:,:].columns]
# coef_lr = pd.DataFrame({'var' : lr_predictors,
#                         'coef' :clf.best_estimator_.coef_.flatten()
#                         })
#
# index_sort =  np.abs(coef_lr['coef']).sort_values(ascending = False).index
# coef_lr_sort = coef_lr.loc[index_sort,:]
# print(coef_lr_sort[0:20])
# ROC曲线绘制
fpr1, tpr1, threshold1 = roc_curve(label_te_true,pre)
plt.plot(fpr1, tpr1, color='red')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.show()
plt.close()