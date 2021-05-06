#导入pandas库
import pandas as pd
#导入样本集拆分相关库
from sklearn.model_selection import train_test_split
#导入逻辑回归相关库
from sklearn.linear_model import LogisticRegression
# 导入模型保存库
import joblib
# 随机重复采样
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
#导入分类报告库
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # 忽略版本问题
# 具体的模型得分报告
from sklearn.metrics import classification_report, roc_auc_score
print('------------逻辑回归------------')
# 读取数据
data = pd.read_csv('../data/金融业_处理.csv',engine = "python")
data2 = pd.read_csv('../data/金融业_处理.csv',engine = "python")
# print("data：\n")
# print(data)
data2.drop(data2.columns[[0]],axis = 1,inplace = True)
# print("data2删除第一列：\n")
# print(data2)
data2.drop(columns = ['FLAG'],axis = 1,inplace = True)
# print("data2删除FLAG：\n")
# print(data2)
# print("data：\n")
# print(data)
#定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(sampling_strategy={1:30,0:525 },random_state=42)
#ros = RandomOverSampler(random_state=30,sampling_strategy=0.02)
X_smo,y_smo = smo.fit_resample(data2.iloc[:,:],data['FLAG'])
#标准化数据
sc=StandardScaler()
sc.fit(X_smo)#计算样本的均值和标准差
X_smo=pd.DataFrame(sc.transform(X_smo))

#拆分专家样本集
data_tr, data_te, label_tr, label_te = train_test_split(X_smo,y_smo,test_size=0.3)
print(label_tr.groupby(label_tr).count())

# 参数设置
params = {'C':[0.0001, 1, 100, 1000],
          'max_iter':[1, 10, 100, 500],
          'class_weight':['balanced', None],
          'solver':['liblinear','sag','lbfgs','newton-cg']
         }
lr = LogisticRegression()
clf = GridSearchCV(lr, param_grid=params, cv=10,scoring='roc_auc')
clf.fit(data_tr, label_tr)
print(clf.best_params_)
#模型预测
# pre = clf.predict_proba(data_te)[:,1]
pre = clf.predict(data_te)
label_te_true = np.array(label_te)
print("the LogisticRegression model auc: %.4g" % metrics.roc_auc_score(label_te_true,pre))
print(classification_report(label_te_true,pre))
print("auc值为:", roc_auc_score(label_te_true,pre))

# 特征重要性排名
lr_predictors = [i for i in data2.iloc[:,:].columns]
# print(clf.best_estimator_.coef_)
# lr_feat_imp = pd.Series(clf.best_estimator_.coef_.flatten, lr_predictors).sort_values(ascending=False)
# lr_feat_imp[0:20].plot(kind='bar', title='Feature Importance')
# plt.ylabel('Feature Importance Score')
# plt.show()
# plt.close()
# print(lr_feat_imp[0:20])
# 2、柱形图
#变量重要性排序
coef_lr = pd.DataFrame({'var' : lr_predictors,
                        'coef' :clf.best_estimator_.coef_.flatten()
                        })

index_sort =  np.abs(coef_lr['coef']).sort_values(ascending = False).index
coef_lr_sort = coef_lr.loc[index_sort,:]
print(coef_lr_sort[0:20])


#绘制ROC曲线
fpr1, tpr1, threshold1 = roc_curve(label_te_true,pre)
plt.plot(fpr1, tpr1, color='red')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LR ROC Curve')
plt.show()
plt.close()