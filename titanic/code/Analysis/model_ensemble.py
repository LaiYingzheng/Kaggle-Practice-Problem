import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.ensemble import BaggingRegressor

from Feature_eng import *
from Modelling import *
from cross_vali import *
from Learning_curve import *

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

y = train_np[:, 0]
x = train_np[: , 1:]

clf = linear_model.LogisticRegression(C = 1.0, penalty = 'l1' , tol = 1e-6)

bagging_clf = BaggingRegressor (clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(x,y)

test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = bagging_clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv('/Users/lait/PycharmProjects/titanic/dataset/Bagging_runningRes.csv')

plot_learning_curve(bagging_clf, u"study curve",x, y)