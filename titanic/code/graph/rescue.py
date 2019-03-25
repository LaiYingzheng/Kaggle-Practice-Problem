import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pandas import Series, DataFrame


dataset ="/Users/lait/PycharmProjects/titanic/dataset/train.csv"
data_train = pd.read_csv(dataset)

#Rescue Status

# plot 1 : Rescued Result in different classes
fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()

df = pd.DataFrame({u'rescued':Survived_1 , u'death':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'Rescued Result in different classes')
plt.xlabel(u'Class')
plt.ylabel(u'Population')

# plot 2 : gender and class, survive status
fig2 = plt.figure()
fig2.set(alpha = 0.65)
plt.title(u'gender and class, survive status')

ax1=fig2.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass !=3].value_counts().plot(kind = 'bar', label = "female highclass", color = '#FA2479')
ax1.set_xticklabels([u"Survived", u"Death"], rotation = 0)
ax1.legend([u"female/First Class"], loc = 'best')

ax1=fig2.add_subplot(142, sharey = ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass ==3].value_counts().plot(kind = 'bar', label = "female low lass", color = 'pink')
ax1.set_xticklabels([u"Survived", u"Death"], rotation = 0)
ax1.legend([u"female/Third Class"], loc = 'best')


ax1=fig2.add_subplot(143)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass !=3].value_counts().plot(kind = 'bar', label = "male highclass", color = 'lightblue')
ax1.set_xticklabels([u"Survived", u"Death"], rotation = 0)
ax1.legend([u"male/First Class"], loc = 'best')

ax1=fig2.add_subplot(144, sharey = ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass ==3].value_counts().plot(kind = 'bar', label = "male low class", color = 'steelblue')
ax1.set_xticklabels([u"Survived", u"Death"], rotation = 0)
ax1.legend([u"male/Third Class"], loc = 'best')

# plot 3 :survived status by embarked
fig3 = plt.figure()
fig3.set(alpha = 0.2)
Survived_00 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_01 = data_train.Embarked[data_train.Survived == 1].value_counts()

pl = pd.DataFrame({u'rescued':Survived_01 , u'death':Survived_00})
pl.plot(kind='bar', stacked=True)
plt.title(u'survived status by embarked')
plt.xlabel(u'embarked')
plt.ylabel(u"population")

plt.show()
