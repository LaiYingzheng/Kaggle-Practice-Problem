
from Feature_eng import *
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold


data_test_path = "/Users/lait/PycharmProjects/titanic/dataset/test.csv"

data_test = pd.read_csv(data_test_path)
data_test.loc[(data_test.Fare.isnull()), 'Fare' ] = 0

tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values

x = null_age[:, 1:]
predictAges = rfr.predict(x)
data_test.loc[(data_test.Age.isnull()),'Age'] = predictAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat ([data_test, dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass ] , axis = 1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
df_test['Age_scaled'] = scaler.transform(df_test['Age'].values.reshape(-1,1))
df_test['Fare_scaled'] = scaler.transform(df_test['Fare'].values.reshape(-1,1))
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

prediction = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values,'Survived':prediction.astype(np.int32)})



