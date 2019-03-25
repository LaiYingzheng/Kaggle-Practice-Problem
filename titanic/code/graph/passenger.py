import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pandas import Series, DataFrame


# general status


dataset ="/Users/lait/PycharmProjects/titanic/dataset/train.csv"
data_train = pd.read_csv(dataset)

fig = plt.figure()
fig.set(alpha = 0.2)

plt.subplot2grid((2,5),(0,0))
data_train.Survived.value_counts().plot(kind= "bar")
plt.title(u"Survive status")
plt.ylabel(u"population")

plt.subplot2grid((2,5),(0,2))
data_train.Pclass.value_counts().plot(kind= "bar")
plt.ylabel(u"population")
plt.title(u"passenger class distribution")

plt.subplot2grid((2,5),(0,4))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"age")
plt.grid(b=True, which = 'major', axis = 'y')
plt.title(u"Survive distribution by age")

plt.subplot2grid((2,5),(1,0), colspan = 2)
data_train.Age[data_train.Pclass ==1].plot(kind='kde')
data_train.Age[data_train.Pclass ==2].plot(kind='kde')
data_train.Age[data_train.Pclass ==3].plot(kind='kde')
plt.xlabel(u"age")
plt.ylabel(u"population density")
plt.title(u"class age distribution")
plt.legend((u"first class", u"second class", u"third class"), loc ='best')

plt.subplot2grid((2,5),(1,3))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u'population embarked')
plt.ylabel(u'population')


plt.show()

