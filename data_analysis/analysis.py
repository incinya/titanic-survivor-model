import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

combined_train_test = pd.read_csv('../titanic/cleaned.csv')

Correlation = pd.DataFrame(combined_train_test[['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size',
                                                'Family_Size_Category', 'Fare', 'Fare_bin_id', 'Pclass',
                                                'Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])

"""我们挑选一些主要的特征，生成特征之间的热力图，查看特征与特征之间的相关性"""
colormap = getattr(plt.cm, 'viridis')
plt.figure(figsize=(14, 12))
plt.title('Pearson Correaltion of Feature', y=1.05, size=15)
# 默认线性相关
# 完全正比 : 1
# 完全反比 : -1
aa = Correlation.astype(float).corr()
sns.heatmap(Correlation.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
            annot=True)
plt.show()

g = sns.pairplot(combined_train_test[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',
                                      u'Family_Size', u'Title', u'Ticket_Letter']],
                 hue='Survived',
                 palette='seismic',
                 size=1.2,
                 diag_kind='kde',
                 diag_kws=
                 dict(shade=True),
                 plot_kws=dict(s=10))
g.set(xticklabels=[])
plt.show()
