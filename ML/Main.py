import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
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

# Visualize missings pattern of the dataframe
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

# Visualize missings pattern of the dataframe
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
print(df.var())
print(df.shape)
# 如果一整列的方差都为0,那么就是无意义数值,将整列数值删除

df.info()
# 显示处理到现在的数据的信息

# Recheck the Correlation matrix
corr = df.corr()
# 重新计算了数据框 df 中各个数值型特征之间的相关系数矩阵

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(corr,  mask=mask, annot=False, cmap='CMRmap', center=0, linewidths=0.1, square=True)

drop = ['total transactions (including tnx to create contract', 'total ether sent contracts', 'max val sent to contract', ' ERC20 avg val rec',
        ' ERC20 avg val rec',' ERC20 max val rec', ' ERC20 min val rec', ' ERC20 uniq rec contract addr', 'max val sent', ' ERC20 avg val sent',
        ' ERC20 min val sent', ' ERC20 max val sent', ' Total ERC20 tnxs', 'avg value sent to contract', 'Unique Sent To Addresses',
        'Unique Received From Addresses', 'total ether received', ' ERC20 uniq sent token name', 'min value received', 'min val sent', ' ERC20 uniq rec addr' ]
df.drop(drop, axis=1, inplace=True)
# 通过删除其中一个高相关性特征值的方法来减少多重共线性对模型的影响,提高模型的解释性和泛化能力
# 多重共线性的意思是,实例A和实例B高相关,那么只需要分析实例A的变化,就可以得到实例B的变化. 这个可以增家计算效率,减少过拟合.
# ????????????????????为什么是删一个?

# Recheck the Correlation matrix
corr = df.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]=True
with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(18,18))
    sns.heatmap(corr,  mask=mask, annot=False, cmap='CMRmap', center=0, linewidths=0.1, square=True)
# 重新绘制相关性矩阵的热力图，以便更好地可视化特征之间的相关性。观测现在的关联性还强吗?

columns = df.columns


# Investigate the distribution of our features using boxplots


fig, axes = plt.subplots(6, 3, figsize=(14, 14), constrained_layout=True)
plt.subplots_adjust(wspace=0.7, hspace=0.8)
plt.suptitle("Distribution of features", y=0.95, size=18, weight='bold')

ax = sns.boxplot(ax=axes[0, 0], data=df, x=columns[1])
ax.set_title(f'Distribution of {columns[1]}')

ax1 = sns.boxplot(ax=axes[0, 1], data=df, x=columns[2])
ax1.set_title(f'Distribution of {columns[2]}')

ax2 = sns.boxplot(ax=axes[0, 2], data=df, x=columns[3])
ax2.set_title(f'Distribution of {columns[3]}')

ax3 = sns.boxplot(ax=axes[1, 0], data=df, x=columns[4])
ax3.set_title(f'Distribution of {columns[4]}')

ax4 = sns.boxplot(ax=axes[1, 1], data=df, x=columns[5])
ax4.set_title(f'Distribution of {columns[5]}')

ax5 = sns.boxplot(ax=axes[1, 2], data=df, x=columns[6])
ax5.set_title(f'Distribution of {columns[6]}')

ax6 = sns.boxplot(ax=axes[2, 0], data=df, x=columns[7])
ax6.set_title(f'Distribution of {columns[7]}')

ax7 = sns.boxplot(ax=axes[2, 1], data=df, x=columns[8])
ax7.set_title(f'Distribution of {columns[8]}')

ax8 = sns.boxplot(ax=axes[2, 2], data=df, x=columns[9])
ax8.set_title(f'Distribution of {columns[9]}')

ax9 = sns.boxplot(ax=axes[3, 0], data=df, x=columns[10])
ax9.set_title(f'Distribution of {columns[10]}')

ax10 = sns.boxplot(ax=axes[3, 1], data=df, x=columns[11])
ax10.set_title(f'Distribution of {columns[11]}')

ax11 = sns.boxplot(ax=axes[3, 2], data=df, x=columns[12])
ax11.set_title(f'Distribution of {columns[12]}')

ax12 = sns.boxplot(ax=axes[4, 0], data=df, x=columns[13])
ax12.set_title(f'Distribution of {columns[13]}')

ax13 = sns.boxplot(ax=axes[4, 1], data=df, x=columns[14])
ax13.set_title(f'Distribution of {columns[14]}')

ax14 = sns.boxplot(ax=axes[4, 2], data=df, x=columns[15])
ax14.set_title(f'Distribution of {columns[15]}')

ax15 = sns.boxplot(ax=axes[5, 0], data=df, x=columns[16])
ax15.set_title(f'Distribution of {columns[16]}')

ax16 = sns.boxplot(ax=axes[5, 1], data=df, x=columns[17])
ax16.set_title(f'Distribution of {columns[17]}')

ax17 = sns.boxplot(ax=axes[5, 2], data=df, x=columns[18])
ax17.set_title(f'Distribution of {columns[18]}')

plt.show()
# 绘制每个特征图的箱型图,以分析数据

# Some features present a small distribution
for i in df.columns[1:]:
    if len(df[i].value_counts()) < 10:
        print(f'The column {i} has the following distribution: \n{df[i].value_counts()}')
        print('======================================')
# 检查数据集中每个特征的分布情况，并打印出分布中的唯一值及其计数，对于唯一值数量小于10的特征. 比如说The column min value sent to contract以及
# The column  ERC20 uniq sent addr.1方差都为0,说明是无意义数据.drop

drops = ['min value sent to contract', ' ERC20 uniq sent addr.1']
df.drop(drops, axis=1, inplace=True)
print(df.shape)
df.head()
# data preparation

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
norm = PowerTransformer()
norm_train_f = norm.fit_transform(X_train)
#将特征归一化,使用了powertransformer方法. 确保各个特征具有相似的尺度和范围，从而提高模型的训练效果和收敛速度

norm_df = pd.DataFrame(norm_train_f, columns=X_train.columns)
norm_df
# 现在norm_df中包含之前df中所有数据归一化之后的数据.

# Distribution of features after log transformation

fig, axes = plt.subplots(6, 3, figsize=(14, 14), constrained_layout=True)
plt.subplots_adjust(wspace=0.7, hspace=0.8)
axes[-1, -1].axis('off')  # hide axes
axes[-1, -2].axis('off')  # hide axes
plt.suptitle("Distribution of features after log", y=0.95, family='Sherif', size=18, weight='bold')

ax = sns.boxplot(ax=axes[0, 0], data=norm_df, x=norm_df.columns[0])
ax.set_title(f'Distribution of {norm_df.columns[0]}')

ax1 = sns.boxplot(ax=axes[0, 1], data=norm_df, x=norm_df.columns[1])
ax1.set_title(f'Distribution of {norm_df.columns[1]}')

ax2 = sns.boxplot(ax=axes[0, 2], data=norm_df, x=norm_df.columns[2])
ax2.set_title(f'Distribution of {norm_df.columns[2]}')

ax3 = sns.boxplot(ax=axes[1, 0], data=norm_df, x=norm_df.columns[3])
ax3.set_title(f'Distribution of {norm_df.columns[3]}')

ax4 = sns.boxplot(ax=axes[1, 1], data=norm_df, x=norm_df.columns[4])
ax4.set_title(f'Distribution of {norm_df.columns[4]}')

ax5 = sns.boxplot(ax=axes[1, 2], data=norm_df, x=norm_df.columns[5])
ax5.set_title(f'Distribution of {norm_df.columns[5]}')

ax6 = sns.boxplot(ax=axes[2, 0], data=norm_df, x=norm_df.columns[6])
ax6.set_title(f'Distribution of {norm_df.columns[6]}')

ax7 = sns.boxplot(ax=axes[2, 1], data=norm_df, x=norm_df.columns[7])
ax7.set_title(f'Distribution of {norm_df.columns[7]}')

ax8 = sns.boxplot(ax=axes[2, 2], data=norm_df, x=norm_df.columns[8])
ax8.set_title(f'Distribution of {norm_df.columns[8]}')

ax9 = sns.boxplot(ax=axes[3, 0], data=norm_df, x=norm_df.columns[9])
ax9.set_title(f'Distribution of {norm_df.columns[9]}')

ax10 = sns.boxplot(ax=axes[3, 1], data=norm_df, x=norm_df.columns[10])
ax10.set_title(f'Distribution of {norm_df.columns[10]}')

ax11 = sns.boxplot(ax=axes[3, 2], data=norm_df, x=norm_df.columns[11])
ax11.set_title(f'Distribution of {norm_df.columns[11]}')

ax12 = sns.boxplot(ax=axes[4, 0], data=norm_df, x=norm_df.columns[12])
ax12.set_title(f'Distribution of {norm_df.columns[12]}')

ax13 = sns.boxplot(ax=axes[4, 1], data=norm_df, x=norm_df.columns[13])
ax13.set_title(f'Distribution of {norm_df.columns[13]}')

ax14 = sns.boxplot(ax=axes[4, 2], data=norm_df, x=norm_df.columns[14])
ax14.set_title(f'Distribution of {norm_df.columns[14]}')

ax15 = sns.boxplot(ax=axes[5, 0], data=norm_df, x=norm_df.columns[15])
ax15.set_title(f'Distribution of {norm_df.columns[15]}')

plt.show()
# 得到每个特征值经过对数转换后的特征数据的分布情况

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

# *******************************************************************************************
LR = LogisticRegression(random_state=42)
LR.fit(x_tr_resample, y_tr_resample)

# Transform test features
norm_test_f = norm.transform(X_test)

preds_LR = LR.predict(norm_test_f)
# 实例化分类器, 用fit()方法将分类器应用到SMOTE过采样后的训练集上,进行训练模型. 使用transform()方法对x_test集进行归一化处理,这样可以和训练数据集拥有相同的数值范围
# 训练好的模型对归一化的测试数据norm_test_f进行预测, 将最终的结果存储在preds变量中

print(y_test.shape)
y_test.value_counts()

print(classification_report(y_test, preds_LR))
print(confusion_matrix(y_test, preds_LR))
# Calculate confusion matrix
cm_LR = confusion_matrix(y_test, preds_LR)

# 逻辑回归模型的混淆矩阵可视化
disp_LR = ConfusionMatrixDisplay(confusion_matrix=cm_LR, display_labels=LR.classes_)
disp_LR.plot(cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# *****************************************************************************************************************
RF = RandomForestClassifier(random_state=42)
RF.fit(x_tr_resample, y_tr_resample)
preds_RF = RF.predict(norm_test_f)

#plot_confusion_matrix(LR, norm_test_f, y_test)
cm_RF=confusion_matrix(y_test,preds_RF)

print(classification_report(y_test, preds_RF))
print(confusion_matrix(y_test, preds_RF))
# 随机森林模型的混淆矩阵可视化
disp_RF = ConfusionMatrixDisplay(confusion_matrix=cm_RF, display_labels=RF.classes_)
disp_RF.plot(cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# plot_confusion_matrix(RF, norm_test_f, y_test)

# ******************************************************************************************************************
# XGBoost的默认超参数:
# learning_rate（学习率）=3  学习率控制每次迭代中模型权重的调整幅度，较低的学习率通常需要更多的迭代次数，但可能会导致更好的泛化性能。
# n_estimators（迭代次数)=100    指定要构建的树的数量，通常是指定训练的迭代轮数。
# max_depth（树的最大深度）=6   指定每棵树的最大深度，用于控制树的复杂度。较大的值可以使模型更复杂，但也容易导致过拟合。
# min_child_weight（叶子节点的最小权重）=1 指定叶子节点上的所有实例权重之和的最小值。用于控制树的分支，避免过度分裂。
# subsample（样本采样比例）=1 指定用于训练每棵树的样本的比例。较小的值可以减少方差，但可能增加偏差。
# colsample_bytree（特征采样比例）=1    指定用于训练每棵树的特征的比例。较小的值可以减少方差，但可能增加偏差。
# gamma（分裂节点时的损失减少阈值）=0 指定节点分裂时必须损失减少的最小值。增加此值会导致更保守的树。
# reg_alpha（L1正则化项系数） 和 reg_lambda（L2正则化项系数）：默认为0   控制模型的复杂度

xgb_c = xgb.XGBClassifier(random_state=42)
xgb_c.fit(x_tr_resample, y_tr_resample)
preds_xgb = xgb_c.predict(norm_test_f)
# plot_confusion_matrix(xgb_c, norm_test_f, y_test)
cm_xgb=confusion_matrix(y_test,preds_xgb)

print(classification_report(y_test, preds_xgb))
print(confusion_matrix(y_test, preds_xgb))

# XGBoost模型的混淆矩阵可视化
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=xgb_c.classes_)
disp_xgb.plot(cmap='Blues')
plt.title('Confusion Matrix for XGBoost')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# **********************************************************************************************************************

params_grid = {'learning_rate':[0.01, 0.1, 0.5],
              'n_estimators':[100,200],
              'subsample':[0.3, 0.5, 0.9],
               'max_depth':[2,3,4],
               'colsample_bytree':[0.3,0.5,0.7]}

grid = GridSearchCV(estimator=xgb_c, param_grid=params_grid, scoring='recall', cv = 10, verbose = 0)

grid.fit(x_tr_resample, y_tr_resample)
print(f'Best params found for XGBoost are: {grid.best_params_}')
print(f'Best recall obtained by the best params: {grid.best_score_}')

preds_best_xgb = grid.best_estimator_.predict(norm_test_f)
print(classification_report(y_test, preds_best_xgb))
print(confusion_matrix(y_test, preds_best_xgb))
# 使用最佳模型的混淆矩阵可视化
cm_best_xgb = confusion_matrix(y_test, preds_best_xgb)
disp_best_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_best_xgb, display_labels=grid.best_estimator_.classes_)
disp_best_xgb.plot(cmap='Blues')
plt.title('Confusion Matrix for Best XGBoost Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ************************************************************************************

# Plotting AUC for untuned XGB Classifier
probs = xgb_c.predict_proba(norm_test_f)
pred = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12,8))
plt.title('ROC for tuned XGB Classifier')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Save the model for further use
pickle_out = open('XGB_FRAUD.pickle', 'wb')
pickle.dump(xgb_c, pickle_out)
pickle_out.close()


