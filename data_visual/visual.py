from unittest import TestCase

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_format.train_data_format import train_data

sns.set_style('whitegrid')


class RelationTest(TestCase):
    def test_survive(self):
        train_data['Survived'].value_counts().plot.pie(labeldistance=1.1, autopct='%1.2f%%',
                                                       shadow=False, startangle=90, pctdistance=0.6)
        plt.show()
        # labeldistance，文本的位置离远点有多远，1.1指1.1倍半径的位置
        # autopct，圆里面的文本格式，%3.1f%%表示小数有三位，整数有一位的浮点数
        # shadow，饼是否有阴影
        # startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
        # pctdistance，百分比的text离圆心的距离
        # patches, l_texts, p_texts，为了得到饼图的返回值，p_texts饼图内部文本的，l_texts饼图外label的文本

    def test_sex(self):
        print(train_data.groupby(['Sex', 'Survived'])['Survived'].count())
        print('-' * 60)
        print(train_data[['Sex', 'Survived']].groupby(['Sex']).mean())

        train_data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
        plt.show()

    # 船舱等级
    def test_p_class(self):
        print(train_data.groupby(['Pclass', 'Survived'])['Pclass'].count())
        print(train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean())
        train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar()
        plt.show()

        # 不同等级船舱的男女生存率：
        print(train_data.groupby(['Sex', 'Pclass', 'Survived'])['Survived'].count())
        train_data[['Sex', 'Pclass', 'Survived']].groupby(['Pclass', 'Sex']).mean().plot.bar()
        plt.show()

    def test_age(self):
        # 分别分析不同等级船舱和不同性别下的年龄分布和生存的关系
        fig, ax = plt.subplots(1, 2, figsize=(18, 5))
        ax[0].set_yticks(range(0, 110, 10))
        sns.violinplot("Pclass", "Age", hue="Survived", data=train_data, split=True, ax=ax[0])
        ax[0].set_title('Pclass and Age vs Survived')

        ax[1].set_yticks(range(0, 110, 10))
        sns.violinplot("Sex", "Age", hue="Survived", data=train_data, split=True, ax=ax[1])
        ax[1].set_title('Sex and Age vs Survived')

        plt.show()

        # 分析总体的年龄分布
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        train_data['Age'].hist(bins=100)
        plt.xlabel('Age')
        plt.ylabel('Num')
        plt.title('Age Distribution')

        plt.subplot(122)
        train_data.boxplot(column='Age', showfliers=True)
        plt.show()

        # 不同年龄下的生存和非生存的分布情况
        facet = sns.FacetGrid(train_data, hue="Survived", aspect=4)
        facet.map(sns.kdeplot, 'Age', shade=True)
        facet.set(xlim=(0, train_data['Age'].max()))
        facet.add_legend()
        plt.show()

        # 不同年龄下的平均生存率
        fig, axis1 = plt.subplots(1, 1, figsize=(18, 4))
        train_data['Age_int'] = train_data['Age'].astype(int)
        average_age = train_data[["Age_int", "Survived"]].groupby(['Age_int'], as_index=False).mean()
        sns.barplot(x='Age_int', y='Survived', data=average_age)
        plt.show()

        # 按照年龄，将乘客划分为儿童、少年、成年、老年，分析四个群体的生还情况
        bins = [0, 12, 18, 65, 100]
        train_data['Age_group'] = pd.cut(train_data['Age'], bins)
        by_age = train_data.groupby('Age_group')['Survived'].mean()
        print(by_age)
        by_age.plot(kind='bar')
        plt.show()

    # 称谓
    def test_title(self):
        train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        print(pd.crosstab(train_data['Title'], train_data['Sex']))

        # 观察不同称呼与生存率的关系
        train_data[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()
        plt.show()

    # 有无兄弟姐妹
    def test_SibSp(self):
        # 将数据分为有兄弟姐妹和没有兄弟姐妹的两组：
        sibsp_df = train_data[train_data['SibSp'] != 0]
        no_sibsp_df = train_data[train_data['SibSp'] == 0]

        plt.figure(figsize=(11, 5))
        plt.subplot(121)
        sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
        plt.xlabel('sibsp')

        plt.subplot(122)
        no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.1f%%')
        plt.xlabel('no_sibsp')

        plt.show()

    # 有无父母子女
    def test_parch(self):
        parch_df = train_data[train_data['Parch'] != 0]
        no_parch_df = train_data[train_data['Parch'] == 0]

        plt.figure(figsize=(11, 5))
        plt.subplot(121)
        parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.2f%%')
        plt.xlabel('parch')

        plt.subplot(122)
        no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct='%1.2f%%')
        plt.xlabel('no_parch')

        plt.show()

    # 亲友的人数
    def test_SibSp_and_parch(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        train_data[['Parch', 'Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
        ax[0].set_title('Parch and Survived')
        train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
        ax[1].set_title('SibSp and Survived')
        plt.show()

        train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1
        train_data[['Family_Size', 'Survived']].groupby(['Family_Size']).mean().plot.bar()
        plt.show()
        # 从图表中可以看出，若独自一人，那么其存活率比较低；但是如果亲友太多的话，存活率也会很低。

    # 票价
    def test_fare(self):
        plt.figure(figsize=(10, 5))
        train_data['Fare'].hist(bins=70)

        train_data.boxplot(column='Fare', by='Pclass', showfliers=True)
        plt.show()

        # 票价均值
        fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
        fare_survived = train_data['Fare'][train_data['Survived'] == 1]

        average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
        average_fare.plot(kind='bar', legend=False)

        plt.show()

    # 港口
    def test_port(self):
        """@author: fxh"""
        train_data[['Embarked', 'Survived']].groupby(['Embarked']).mean().plot.bar()
        plt.show()
