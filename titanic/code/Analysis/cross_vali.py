
from Feature_eng import *
from Modelling import *
import pandas as pd
import sklearn

#cross_validation
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.values[:,1:]
Y  = all_data.values[:,0]

# print(sklearn.model_selection.cross_val_score(clf, X, Y, cv=5))

#bad case
    #Split data, train_data : Data =  7 : 3
split_train, split_cv = sklearn.model_selection.train_test_split(df, test_size= 0.3, random_state= 0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    #Modelling
clf = linear_model.LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-6)
clf.fit(train_df.values[:,1:], train_df.values[:,0])

    #prediction for cross validation
cv_df = split_cv.filter(regex = 'Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.values[:,1:])

origin_data_train = data_train

bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'].values)]