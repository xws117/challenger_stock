# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 17:24:22 2017

@author: xws
"""
from sklearn.model_selection import train_test_split
import pandas as pd
Train = pd.read_csv('C:/Users/xws/Desktop/challenger_socket/train.csv')
Test = pd.read_csv('C:/Users/xws/Desktop/challenger_socket/test.csv')

import xgboost as xgb


train ,test = train_test_split(Train[:300000],test_size=0.2)

Y_train = train['label']
X_train = train.drop(['id','label','era','weight'],axis=1)


X_test = test.drop(['id','label','era','weight'],axis=1)
Y_test = test['label']

Y_test = Y_test.reset_index().drop('index',axis=1)
Y_train = Y_train.reset_index().drop('index',axis=1)

dtrain = xgb.DMatrix(X_train,label=Y_train)
dvalid = xgb.DMatrix(X_test, label=Y_test)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
dtest = xgb.DMatrix(Test.drop(['id'],axis=1))


xgb_pars = {'min_child_weight': 2, 'eta': 0.2, 'colsample_bytree': 0.9, 
            'max_depth': 6,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 300, watchlist, early_stopping_rounds=10,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)


pred = model.predict(dtest)
submission = pd.concat([Test['id'], pd.DataFrame(pred)], axis=1)
submission.columns = ['id','proba']
submission.to_csv("submission.csv", index=False)
submission['proba'] = submission.apply(lambda x : 0 if (x['proba'] <= 0) else x['proba'], axis = 1)

submission['proba'] = submission.apply(lambda x : 1 if (x['proba'] >= 1) else x['proba'], axis = 1)

submission.to_csv("submission2.csv", index=False)