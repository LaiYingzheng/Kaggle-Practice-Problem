import pandas as pd
import numpy as np

from pandas import Series, DataFrame


dataset ="/Users/lait/PycharmProjects/titanic/dataset/train.csv"
data_train = pd.read_csv(dataset)

g = data_train.groupby(['SibSp', 'Survived'])
df  = pd.DataFrame(g.count()['PassengerId'] )

print(df)

g = data_train.groupby(['Parch', 'Survived'])
df  = pd.DataFrame(g.count()['PassengerId'] )
print(df)