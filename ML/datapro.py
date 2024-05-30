import warnings

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
# plot_confusion_matrix

df = pd.read_csv('transaction_dataset.csv', index_col=0)
# index_col=0 指定将 CSV 文件中的第一列作为索引（行标签）。 读取csv文件中的数据,存储到变量df中
print(df.shape)
# 获取读取到的数据的行列数, e.g.(9841, 50)  9841行  50列
print(df.head())
# 用于显示前几行,默认前五行

# 删除数据集中的前两列, csv文件中的index 以及address列
df = df.iloc[:,2:]
# 用于显示有关 DataFrame 的摘要信息，包括每列的非空值数量、数据类型等. 打印 DataFrame df 的摘要信息，以便快速查看数据的基本情况。
print(df.info())

# 将对象变量转换为 'category' 数据类型，以提高计算效率
categories = df.select_dtypes('O').columns.astype('category')
# 这边选择了所有字符串的列0 O 大写o
# 这行代码是利用上面步骤中得到的列索引，从 DataFrame df 中选择了包含 'category' 数据类型的列。
# 该操作返回一个新的 DataFrame，其中包含了被转换为 'category' 类型的列。
# 其他非字符串的数据都没有包含在里面


# 检查分类变量和数值变量的基本信息，包括唯一值的数量、描述性统计等。
# 这段代码通过循环遍历 DataFrame 中的分类变量列，并打印每个分类变量列的基本信息，包括唯一值的数量。
for i in df[categories].columns:
    print(f'The categorical column --{i}-- has --{len(df[i].value_counts())}-- unique values')


# Inspect numericals
numericals = df.select_dtypes(include=['float','int']).columns
# 选择所有float和int数据的列.储存到numericals里面
df[numericals].describe()
# 生成数值变量的描述性统计信息，包括计数、平均值、标准差、最小值、25th 百分位数、中位数、75th 百分位数和最大值。


# Inspect features variance
df[numericals].var()
# 计算所有数值的方差

# Inspect target distribution
print(df['FLAG'].value_counts())

pie, ax = plt.subplots(figsize=[15,10])
labels = ['Non-fraud', 'Fraud']
colors = ['#f9ae35', '#f64e38']
plt.pie(x = df['FLAG'].value_counts(), autopct='%.2f%%', explode=[0.02]*2, labels=labels, pctdistance=0.5, textprops={'fontsize': 14}, colors = colors)
plt.title('Target distribution')
plt.show()
# 通过饼状图查找存在欺诈的比例


# Correlation matrix
corr = df.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True
with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(18,18))
    sns.heatmap(corr,  mask=mask, annot=False, cmap='CMRmap', center=0, square=True)
# 计算数据集中数值型变量之间的相关性，并绘制相关性矩阵的热力图
# 若为正相关,说明数据一起增大

# Visualize s pattern of the dataframe
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False)
plt.show()
# 用于可视化数据框中缺失值的分布模式

# Drop the two categorical features
df.drop(df[categories], axis=1, inplace=True)
# 删除所有df中含有的文本数值

# Replace missings of numerical variables with median
df.fillna(df.median(), inplace=True)

# 缺失值都用中位值来填充

# Visualize s pattern of the dataframe
print(df.shape)
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False)
plt.show()
# 显示是否还有缺失值(无)

# Filtering the features with 0 variance
no_var = df.var() == 0
print(df.var()[no_var])
print('\n')
# 找出所有方差为0的特征值

# Drop features with 0 variance --- these features will not help in the performance of the model
df.drop(df.var()[no_var].index, axis = 1, inplace = True)
print("****************************")
print(df.var())
print(df.shape)
# 如果一整列的方差都为0,那么就是无意义数值,将整列数值删除

# Recheck the Correlation matrix
corr = df.corr()
# 重新计算了数据框 df 中各个数值型特征之间的相关系数矩阵

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(20, 20))  # 调整图形尺寸
    sns.heatmap(corr, mask=mask, annot=False, cmap='CMRmap', center=0, linewidths=0.1, square=True, ax=ax)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # 旋转特征名称并调整字体大小
    plt.yticks(rotation=0, fontsize=10)  # 调整字体大小
    plt.show()

drop = [ 'avg value sent to contract', ' ERC20 min val sent',' ERC20 max val sent',
        ' ERC20 avg val sent','max val sent to contract','total ether sent contracts'
         ,'Time Diff between first and last (Mins)', 'total ether balance', ' ERC20 max val rec'
         ,' ERC20 uniq rec token name',' ERC20 avg val rec','total transactions (including tnx to create contract'
         ,' ERC20 uniq sent token name',' Total ERC20 tnxs']

df.drop(drop, axis=1, inplace=True)
# 通过删除其中一个高相关性特征值的方法来减少多重共线性对模型的影响,提高模型的解释性和泛化能力
# 多重共线性的意思是,实例A和实例B高相关,那么只需要分析实例A的变化,就可以得到实例B的变化. 这个可以增家计算效率,减少过拟合.
# ????????????????????为什么是删一个?

# Recheck the Correlation matrix
corr = df.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(20, 20))  # 调整图形尺寸
    sns.heatmap(corr, mask=mask, annot=False, cmap='CMRmap', center=0, linewidths=0.1, square=True, ax=ax)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # 旋转特征名称并调整字体大小
    plt.yticks(rotation=0, fontsize=10)  # 调整字体大小
    plt.show()

# 重新绘制相关性矩阵的热力图，以便更好地可视化特征之间的相关性。观测现在的关联性还强吗?

columns = df.columns


# Investigate the distribution of our features using boxplots

# 获取数据框的列名
columns = df.columns

# 计算要创建的子图行数和列数
num_cols = len(columns)
num_rows = (num_cols - 1) // 5 + 1

# 创建子图网格
fig, axes = plt.subplots(num_rows, 5, figsize=(25, 25), constrained_layout=True)
plt.subplots_adjust(wspace=0.7, hspace=0.8)
plt.suptitle("Distribution of features", y=0.95, size=18, weight='bold')

# 遍历数据框的每一列，并在子图中绘制箱型图
for i, ax in enumerate(axes.flat):
    if i < num_cols:
        sns.boxplot(ax=ax, data=df, x=columns[i])
        ax.set_title(f'Distribution of {columns[i]}')
    else:
        # 如果索引超出了数据框的列数，则隐藏多余的子图
        ax.set_visible(False)

plt.show()


# Some features present a small distribution
for i in df.columns[1:]:
    if len(df[i].value_counts()) < 10:
        print(f'The column {i} has the following distribution: \n{df[i].value_counts()}')
        print('======================================')
# 检查数据集中每个特征的分布情况，并打印出分布中的唯一值及其计数，对于唯一值数量小于10的特征. 比如说The column min value sent to contract以及
# The column  ERC20 uniq sent addr.1方差都为0,说明是无意义数据.drop


drops = [' ERC20 uniq sent addr.1','min value sent to contract']
df.drop(drops, axis=1, inplace=True)
print(df.shape)
df.head()
# data preparation

df.to_csv('processed_data.csv')
print("saving successfully!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

y = df.iloc[:, 0]
X = df.iloc[:, 1:]
print(X.shape, y.shape)
# y表示flag列. x表示flag列之后的所有列

# Split into training (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# 划分训练集和数据集, 并通过随机种子来进行划分. 目的是使结果得到复现,便于调试代码,保持结果一致性.

# Normalize the training features
norm =QuantileTransformer()
norm_train_f = norm.fit_transform(X_train)
#将特征归一化,使用了powertransformer方法. 确保各个特征具有相似的尺度和范围，从而提高模型的训练效果和收敛速度

norm_df = pd.DataFrame(norm_train_f, columns=X_train.columns)

# 现在norm_df中包含之前df中所有数据归一化之后的数据.

# Distribution of features after log transformation

# 更新特征列名列表
columns = norm_df.columns

# 计算新的子图行数和列数
num_cols = len(columns)
num_rows = (num_cols - 1) // 5 + 1

# 创建新的子图网格
fig, axes = plt.subplots(num_rows, 5, figsize=(25, 25), constrained_layout=True)
plt.subplots_adjust(wspace=0.7, hspace=0.8)
plt.suptitle("Distribution of features", y=0.95, size=18, weight='bold')

# 遍历数据框的每一列，并在子图中绘制箱型图
for i, ax in enumerate(axes.flat):
    if i < num_cols:
        sns.boxplot(ax=ax, data=norm_df, x=columns[i])
        ax.set_title(f'Distribution of {columns[i]}')
    else:
        # 如果索引超出了数据框的列数，则隐藏多余的子图
        ax.set_visible(False)

plt.show()

# handling the imbalance
oversample = SMOTE()
print(f'Shape of the training before SMOTE: {norm_train_f.shape, y_train.shape}')

x_tr_resample, y_tr_resample = oversample.fit_resample(norm_train_f, y_train)
print(f'Shape of the training after SMOTE: {x_tr_resample.shape, y_tr_resample.shape}')
# 通过SMOTE方法对训练数据进行过采样, 处理类别不平衡,通过生成合成样本来增加少数类样本的数量.

# Target distribution before SMOTE
non_fraud = 0
fraud = 0

for i in y_train:
    if i == 0:
        non_fraud +=1
    else:
        fraud +=1

# Target distribution after SMOTE
no = 0
yes = 1

for j in y_tr_resample:
    if j == 0:
        no +=1
    else:
        yes +=1


print(f'BEFORE OVERSAMPLING \n \tNon-frauds: {non_fraud} \n \tFauds: {fraud}')
print(f'AFTER OVERSAMPLING \n \tNon-frauds: {no} \n \tFauds: {yes}')
# 打印使用SMOTE之前和之后分别的是否欺诈数量. 显示使用之后数据集的两者数据持平(数据平衡).
