import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn import svm
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# 随机重复采样
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")  # 忽略版本问题
# 读取数据
data = pd.read_csv('../data/房地产业_处理.csv' , engine = "python")
data2 = pd.read_csv('../data/房地产业_处理.csv',engine = "python")
data3 = pd.read_csv('../预测data/房地产业_预测.csv',engine = "python")
data4 = pd.read_csv('../预测data/房地产业_预测.csv',engine = "python")
data3.drop(data3.columns[[0,1]],axis = 1,inplace = True)
pre_data = data3.iloc[:,:]
print(pre_data)
sc=StandardScaler()
sc.fit(np.array(pre_data))#计算样本的均值和标准差
zhiyaoye_pre_data=pd.DataFrame(sc.transform(np.array(pre_data)))
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
smo = SMOTE(sampling_strategy={1: 50,0:571},random_state=42)
# ros = RandomOverSampler(sampling_strategy=0.02)
X_smo,y_smo = smo.fit_resample(data2.iloc[:,:],data['FLAG'])
#标准化数据
sc=StandardScaler()
sc.fit(X_smo)#计算样本的均值和标准差
X_smo=pd.DataFrame(sc.transform(X_smo))

#拆分专家样本集
data_tr, data_te, label_tr, label_te = train_test_split(X_smo,y_smo,test_size=0.3,random_state=43)
print(label_tr.groupby(label_tr).count())
clf1 = xgb.XGBClassifier(n_estimators=475,learning_rate=0.5,
                        max_depth=2,min_child_weight=4,gamma=0.3,
                        subsample=0.79,colsample_bytree=0.73,
                        objective='binary:logistic',nthread=2,
                        scale_pos_weight=100,seed=10,Reg_lambda = 2.04,
                         Reg_alpha = 1.51)#scale_pos_weight很重要
clf2 = lgb.LGBMClassifier(boosting_type='gbdt',objective= 'binary',metric =  'binary_logloss,auc',
                          bagging_fraction = 0.63,bagging_freq = 3,
                          feature_fraction = 0.99,lambda_l1 =  0.3,lambda_l2 = 0.86,
                          learning_rate = 0.17,max_bin = 108,max_depth = 11,
                          min_data_in_leaf = 59,min_split_gain = 0.16,
                          n_estimators = 772,num_leaves = 27,
                          cat_smooth = 10,nthread = 4)
#cat_smooth：可以减少噪声对分类特征的影响，尤其是对于数据较少的分类
clf3 = CatBoostClassifier(loss_function="Logloss",eval_metric="AUC",
                          iterations=2500,learning_rate=0.01,depth=7,verbose=100,bagging_temperature = 0.42,
                          early_stopping_rounds=500)
clf4 = RandomForestClassifier(n_estimators=85,min_samples_split=9,max_features=0.158,max_depth=16)
clf5 = LogisticRegression(C= 1000, class_weight= 'balanced', max_iter= 500, solver='lbfgs')
clf6 = svm.SVC(C =2,gamma=0.1)

lr = LogisticRegression()  # clf1, clf2, clf3,clf4,clf5,clf6
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf4,clf5,clf6], meta_classifier=lr)

# for clf, label in zip(
#     [clf1, clf2, clf3,clf4,clf5,clf6, sclf],
#     ['xgb', 'lgb', 'catboost','RF','LR','svc', 'StackingClassifier']):
#
#     scores = model_selection.cross_val_score(clf, data_tr, label_tr, cv=3, scoring='accuracy')
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
sclf.fit(data_tr,label_tr)
label_te_true = np.array(label_te)
pre = sclf.predict(data_te)
zhizaoye = sclf.predict(zhiyaoye_pre_data)
print(zhizaoye)

number = data4['TICKER_SYMBOL'].values
dataframe = pd.DataFrame({'股票编号':number,'房地产业':zhizaoye})
dataframe.to_csv("房地产业-预测结果.csv", index=False, encoding='GBK')
print("the stacking model auc: %.4g" % metrics.roc_auc_score(label_te_true, pre))
print(classification_report(label_te_true, pre))
print("stacking auc值为:", roc_auc_score(label_te_true, pre))
# ROC曲线绘制
fpr1, tpr1, threshold1 = roc_curve(label_te_true,pre)
plt.plot(fpr1, tpr1, color='red')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stacking ROC Curve')
plt.show()
plt.close()