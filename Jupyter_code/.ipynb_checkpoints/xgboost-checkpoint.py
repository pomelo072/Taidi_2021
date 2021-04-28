import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer  #填补缺失值
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import cross_val_score
#导入数据
full_data = pd.read_csv('./Data/制造业.csv',encoding = 'utf8')
full_data.info()
#去除空值较多的特征因子
[m,n] = full_data.shape
nan_col = full_data.isnull().sum()
nan_col_list = nan_col[nan_col > m*0.7].index.to_list()
clean_data = full_data.drop(columns = nan_col_list,axis = 1)

data = clean_data.drop(columns = ['ACCOUTING_STANDARDS','REPORT_TYPE','CURRENCY_CD'])

dtrain = data.loc[:,'ACT_PUBTIME':'T_COGS']
dtarget = data.loc[:,'FLAG']
#去除方差小的特征因子
dtrain.drop(columns = ['ACT_PUBTIME','PUBLISH_DATE','END_DATE','FISCAL_PERIOD','MERGED_FLAG'],inplace = True)
#划分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(dtrain,dtarget,test_size=0.3,random_state=0)

#贝叶斯优化调参
def modelfit(learning_rate,n_estimators,max_depth,min_child_weight,gamma,subsample,colsample_bytree,
            Reg_lambda,Reg_alpha):
    xgb1 = xgb.XGBClassifier(objective='binary:logistic',nthread=2,
                             n_estimators=int(n_estimators),
                             learning_rate=float(learning_rate),
                             max_depth=int(max_depth),min_child_weight=int(min_child_weight),
                             gamma=float(gamma),subsample=float(subsample),
                             colsample_bytree=float(colsample_bytree),
                             Reg_lambda = float(Reg_lambda),Reg_alpha = float(Reg_alpha),
                             scale_pos_weight=1,seed=10,).fit(X_train,pd.DataFrame(y_train)).score(X_test,y_test)
    return xgb1

# 参数因子池
pool = {
    'learning_rate':(0.0001,0.9999),
    'n_estimators':(400,800),
    'max_depth':(1,20),              #树模型深度
    'min_child_weight':(1,5),
    'gamma':(0.1,0.7),
    'subsample':(0.6,0.9),
    'colsample_bytree':(0.6,0.9),
    'Reg_lambda':(0.001,3),
    'Reg_alpha':(0.001,2)

}
#定义贝叶斯调参模型
from bayes_opt import BayesianOptimization as bayes
optimizer = bayes(f=modelfit,
                 pbounds = pool,
                 verbose = 2,
                 random_state = 1,
                 )
#调参并输出最优因子
optimizer.maximize(
    init_points = 5,
    n_iter = 25,
)
print(optimizer.max)
#xgboost模型训练
xgb1 = xgb.XGBClassifier(n_estimators=545,learning_rate=0.0001,
                        max_depth=1,min_child_weight=5,gamma=0.1,
                        subsample=0.6,colsample_bytree=0.9,
                        objective='binary:logistic',nthread=2,
                        scale_pos_weight=1,seed=10,Reg_lambda = 3,
                         Reg_alpha = 0.001)
xgb1.fit(X_train,y_train)

y_test_pre = xgb1.predict(X_test)
y_test_true = np.array(y_test)
print ("the xgboost model Accuracy : %.4g" % metrics.accuracy_score(y_pred=y_test_pre, y_true=y_test_true))

#特征重要性排名
xgb_predictors = [i  for i in train_X.columns]
xgb_feat_imp = pd.Series(xgb1.feature_importances_,xgb_predictors).sort_values(ascending=False)
xgb_feat_imp.plot(kind = 'bar',title='Feature Importance')
plt.ylabel('Feature Importance Score')
plt.show()
print(xgb_feat_imp)

#具体的模型得分报告
from sklearn.metrics import classification_report,roc_auc_score
print(classification_report(y_test_true,y_test_pre))
print("auc值为:",roc_auc_score(y_test_true,y_test_pre))

#ROC曲线绘制
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr1,tpr1,threshold1 = roc_curve(y_test_true,y_test_pre)
plt.plot(fpr1, tpr1, color='r')
plt.plot([0, 1], [0, 1], color='blue',linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBClassifier ROC Curve')
