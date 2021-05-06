import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# 随机重复采样
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
# 读取数据
data = pd.read_csv('../data/信息传输、软件和信息技术服务业_处理.csv' , engine = "python")
data2 = pd.read_csv('../data/信息传输、软件和信息技术服务业_处理.csv',engine = "python")
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
# smo = SMOTE(sampling_strategy={1:30,0:525 })
ros = RandomOverSampler(random_state=30,sampling_strategy=0.02)
X_smo,y_smo = ros.fit_resample(data2.iloc[:,:],data['FLAG'])
#标准化数据
sc=StandardScaler()
sc.fit(X_smo)#计算样本的均值和标准差
X_smo=pd.DataFrame(sc.transform(X_smo))

#拆分专家样本集
data_tr, data_te, label_tr, label_te = train_test_split(X_smo,y_smo,test_size=0.3)
print(label_tr.groupby(label_tr).count())

# 拆分专家样本集
# data_tr, data_te, label_tr, label_te = train_test_split(X_resampled, y_resampled,random_state=10,test_size=0.3)
# print("data_tr：\n")
# print(data_tr)

#定义优化参数
def rf_cv(n_estimators, min_samples_split, max_depth, max_features):
    val = cross_val_score(RandomForestClassifier(n_estimators=int(n_estimators),
                          min_samples_split=int(min_samples_split),
                          max_depth = int(max_depth),
                          max_features = min(max_features,0.999),
                          random_state = 2),
            data_tr,label_tr,scoring="roc_auc",cv=5).mean()
    return val
# 实例化一个bayes优化对象
# 贝叶斯优化
rf_bo = BayesianOptimization(rf_cv,
                             {
                                 "n_estimators":(10,250),
                                 "min_samples_split":(2,25),
                                 "max_features":(0.1,0.999),
                                 "max_depth":(10,18)
                             })
rf_bo.maximize(init_points=5,n_iter=25)
print(rf_bo.max)


#模型构建
rfc = RandomForestClassifier(n_estimators=10,min_samples_split=114,max_features=0.1,max_depth=11)
#模型训练
rfc = rfc.fit(data_tr,label_tr)

pre = rfc.predict(data_te)
score_r = rfc.score(data_te,label_te)

print("Random Forest:{}".format(score_r))
# 具体的模型得分报告
from sklearn.metrics import classification_report, roc_auc_score
label_te_true = np.array(label_te)




from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
# 特征重要性排名
rfc_predictors = [i for i in data2.iloc[:,:].columns]
rfc_feat_imp = pd.Series(rfc.feature_importances_, rfc_predictors).sort_values(ascending=False)
rfc_feat_imp[0:20].plot(kind='bar', title='Feature Importance')
plt.ylabel('Feature Importance Score')
plt.show()
plt.close()
print(rfc_feat_imp[0:20])

print(classification_report(label_te_true,pre))
print("auc值为:", roc_auc_score(label_te_true,pre))
# ROC曲线绘制
fpr1, tpr1, threshold1 = roc_curve(label_te_true,pre)
plt.plot(fpr1, tpr1, color='red')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RFC ROC Curve')
plt.show()
plt.close()