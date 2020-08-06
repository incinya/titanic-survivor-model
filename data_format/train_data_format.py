import re
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

train_data = pd.read_csv('../titanic/train.csv')
test_data = pd.read_csv('../titanic/test.csv')

sns.set_style('whitegrid')
train_data.head()

train_data.info()
print("-" * 40)
test_data.info()

# 处理空值
# embark 上船地点 填充众数
# cabin  船舱     填充U0
# age    年龄     随机森林预测
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
# replace missing value with U0
train_data['Cabin'] = train_data.Cabin.fillna('U0')
# train_data.Cabin[train_data.CAbin.isnull()]='U0'

# choose training data to predict age
age_df = train_data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]  # Age 非空的子集
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]  # Age 为空的子集
X = age_df_notnull.values[:, 1:]  # age        列的数据
Y = age_df_notnull.values[:, :1]  # 除age以外   列的数据

# use RandomForestRegression to train data
RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
RFR.fit(X, Y)
predictAges = RFR.predict(age_df_isnull.values[:, 1:])
train_data.loc[train_data['Age'].isnull(), ['Age']] = predictAges
print("-" * 40)
train_data.info()
# ####### 处理空值结束

# dummy
embark_dummies = pd.get_dummies(train_data['Embarked'])
train_data = train_data.join(embark_dummies)
# train_data.drop(['Embarked'], axis=1, inplace=True)

embark_dummies = train_data[['S', 'C', 'Q']]
print(embark_dummies.head())

# factoring
# 为舱位号码的字母部分创建分组
train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
_index = pd.factorize(train_data['CabinLetter'])
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]

print(train_data[['Cabin', 'CabinLetter']].head(20))

assert np.size(train_data['Age']) == 891
# 将年龄映射到范围(-1,1)
scaler = preprocessing.StandardScaler()
train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))

# 将数据分为5组(离散化)
train_data['Fare_bin'] = pd.qcut(train_data['Fare'], 5)
print(train_data['Fare_bin'].head())
train_data[['Fare']].mean().plot.bar()

# factorize
train_data['Fare_bin_id'] = pd.factorize(train_data['Fare_bin'])[0]
