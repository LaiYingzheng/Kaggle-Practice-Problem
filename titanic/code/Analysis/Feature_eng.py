import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing


dataset ="/Users/lait/PycharmProjects/titanic/dataset/train.csv"
ResultPath = "/Users/lait/PycharmProjects/titanic/dataset/runningRes.csv"
data_train = pd.read_csv(dataset)


def set_missing_age(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    Known_age = age_df[age_df.Age.notnull()].values
    unKnown_age = age_df[age_df.Age.isnull()].values

    y = Known_age[:, 0]
    x= Known_age[: , 1:]

    rfr = RandomForestRegressor(random_state= 0, n_estimators= 2000, n_jobs=-1)
    rfr.fit(x,y)

    predicted_age = rfr.predict(unKnown_age[:, 1::])

    df.loc[ (df.Age.isnull()), "Age"] = predicted_age

    return df, rfr

def set_Cabin_type (df):
    df.loc [(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc [(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

data_train, rfr = set_missing_age(data_train)
data_train = set_Cabin_type(data_train)
rfr

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat ([data_train, dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass ] , axis = 1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
scaler = preprocessing.StandardScaler()
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1))

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

y = train_np[:, 0]

x = train_np[:, 1:]

clf = linear_model.LogisticRegression(C = 1.0, penalty = 'l1', tol = 1e-6)
clf.fit(x,y)
clf


