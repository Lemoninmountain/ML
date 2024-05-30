import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import pickle
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble._hist_gradient_boosting import gradient_boosting
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler, MaxAbsScaler, \
    MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, BaggingClassifier, StackingClassifier, \
    AdaBoostClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import LSTM, GRU
from joblib import dump, load
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import optuna




warnings.filterwarnings('ignore')

# 加载处理后的数据
df = pd.read_csv('processed_data.csv', index_col=0)

y = df.iloc[:, 0]
X = df.iloc[:, 1:]
print(X.shape, y.shape)
# y表示flag列. x表示flag列之后的所有列

# Split into training (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
# 划分训练集和数据集, 并通过随机种子来进行划分. 目的是使结果得到复现,便于调试代码,保持结果一致性.

# Normalize the training features
norm =QuantileTransformer()
norm_train_f = norm.fit_transform(X_train)
#将特征归一化,使用了QuantileTransformer方法. 确保各个特征具有相似的尺度和范围，从而提高模型的训练效果和收敛速度

norm_df = pd.DataFrame(norm_train_f, columns=X_train.columns)

# handling the imbalance
oversample = SMOTE()

x_tr_resample, y_tr_resample = oversample.fit_resample(norm_train_f, y_train)

norm_test_f = norm.transform(X_test)


# ******************************************************************************************
print("RNN************************************************************************")
# 对特征进行归一化
# 对特征进行归一化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape)
print(X_test_scaled.shape)

X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
# 定义 RNN 模型的函数
sarte_time = time.time()
def create_rnn_model(optimizer='adam'):
    model = tf.keras.Sequential([
        LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 创建 KerasClassifier 包装器
rnn_model = KerasClassifier(build_fn=create_rnn_model, epochs=15, batch_size=32, verbose=0)

# 定义参数网格
param_grid = {'optimizer': ['adam', 'rmsprop']}

# 使用 GridSearchCV 进行搜索
grid_search = GridSearchCV(estimator=rnn_model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_result = grid_search.fit(X_train_reshaped, y_train)

# 输出最佳参数和最佳模型
print("Best Parameters:", grid_result.best_params_)
print("Best Score:", grid_result.best_score_)
best_rnn_model = grid_result.best_estimator_
training_time = time.time() - sarte_time
print(f"Training time for RNN: {training_time:.2f} seconds")

start_time = time.time()
# 使用最佳模型进行测试
y_pred = best_rnn_model.predict(X_test_reshaped)
print(classification_report(y_test, y_pred))
training_time = time.time() - start_time
print(f"Training time for RNN: {training_time:.2f} seconds")
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for Best RNN Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


'''
# ******Logistic regression******************************************************************************
print("Logistic regression********************************************")
# 定义要调优的参数网格
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 正则化参数
    'penalty': ['l1', 'l2']  # 正则化类型
}
train_start_time = time.time()
LR = LogisticRegression(random_state=42)
# 创建Grid Search对象
grid_search = GridSearchCV(LR, param_grid, cv=5, scoring='accuracy')

# 在训练集上拟合Grid Search对象
grid_search.fit(x_tr_resample, y_tr_resample)

# 获取最佳参数
best_LR = grid_search.best_estimator_

# 输出最佳参数组合和对应的准确率
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

train_end_time = time.time()

training_time = train_end_time - train_start_time
print(f"Training time for Logistic Regression: {training_time:.2f} seconds")

test_start_time = time.time()

# Transform test features


preds_LR = grid_search.predict(norm_test_f)
# 实例化分类器, 用fit()方法将分类器应用到SMOTE过采样后的训练集上,进行训练模型. 使用transform()方法对x_test集进行归一化处理,这样可以和训练数据集拥有相同的数值范围
# 训练好的模型对归一化的测试数据norm_test_f进行预测, 将最终的结果存储在preds变量中

# 记录测试结束时间
test_end_time = time.time()

# 计算测试时间
testing_time = test_end_time - test_start_time
print(f"Testing time for Logistic Regression: {testing_time:.2f} seconds")

print(y_test.shape)
y_test.value_counts()

print(classification_report(y_test, preds_LR))
print(confusion_matrix(y_test, preds_LR))
# Calculate confusion matrix
cm_LR = confusion_matrix(y_test, preds_LR)

# 逻辑回归模型的混淆矩阵可视化
disp_LR = ConfusionMatrixDisplay(confusion_matrix=cm_LR, display_labels=best_LR.classes_)
disp_LR.plot(cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# 保存模型
dump(best_LR, 'LR_model.joblib')
'''
'''
# *******LOF*********************************************************
print("LOF*********************************************************")

# Define the parameter grid
param_grid = {
    'n_neighbors': [10, 15, 20, 25],
    'contamination': [0.05, 0.1, 0.15, 0.2]
}
start_time = time.time()
# Create a LOF model
lof = LocalOutlierFactor()

# Create a GridSearchCV object
grid_search = GridSearchCV(lof, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(x_tr_resample, y_tr_resample)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)
train_time = time.time() - start_time

print(f"LOF model detection time: {train_time:.2f} seconds")
test_start_time = time.time()

# Use the best parameters to create the final LOF model
best_lof = LocalOutlierFactor(**best_params)

# Fit the final model to the training data
best_lof.fit(x_tr_resample)

# Predict outliers on the test data
outliers = best_lof.fit_predict(norm_test_f)

# Calculate detection time
test_time = time.time() - test_start_time
print(f"LOF model detection time: {test_time:.2f} seconds")

# Separate normal and outlier data
normal_data = norm_test_f[outliers == 1]
outlier_data = norm_test_f[outliers == -1]

# Visualize outliers and normal data
plt.scatter(normal_data[:, 0], normal_data[:, 1], color='blue', label='Normal')
plt.scatter(outlier_data[:, 0], outlier_data[:, 1], color='red', label='Outlier')
plt.title('LOF Outlier Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Create a virtual label, marking known outliers as 1 and non-outliers as 0
y_true = np.zeros(len(norm_test_f))
y_true[y_test] = 0

# Predict outliers using the final LOF model
predicted_labels = np.where(outliers == 1, 0, 1)

# Compute confusion matrix
cm = confusion_matrix(y_test, predicted_labels)

# Compute classification report
report = classification_report(y_test, predicted_labels)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Outlier'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for LOF')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\nClassification Report:")
print(report)
# 保存模型
dump(best_LOF, 'LOF_model.joblib')

# *******IsolationForest**************************************************************************

'''
'''
print("Isolation Forest**********************************")
# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': ['auto', 100, 200],
    'contamination': [0.05, 0.1, 0.15]
}
start_train=time.time()
# Create an Isolation Forest model
isolation_forest = IsolationForest(random_state=42, bootstrap=True)

# Create a GridSearchCV object
grid_search = GridSearchCV(isolation_forest, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(x_tr_resample, y_tr_resample)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Cross-Validation Accuracy:", best_score)
train_time = time.time() - start_train
print(f"Isolation Forest model detection time: {train_time:.2f} seconds")
# Use the best parameters to create the final Isolation Forest model
best_isolation_forest = IsolationForest(random_state=42, bootstrap=True, **best_params)

# Fit the final model to the training data
best_isolation_forest.fit(x_tr_resample)

# Predict outliers on the test data
test_start_time = time.time()
preds_isolation_forest = best_isolation_forest.predict(norm_test_f)


# Calculate detection time
detection_time = time.time() - test_start_time
print(f"Isolation Forest model detection time: {detection_time:.2f} seconds")

# Convert -1 to 0 and keep 1 unchanged to match the normal/anomaly labels
preds_isolation_forest[preds_isolation_forest == -1] = 0

# Print classification report and confusion matrix
print(classification_report(y_test, preds_isolation_forest))
print(confusion_matrix(y_test, preds_isolation_forest))

# Visualize confusion matrix
cm_isolation_forest = confusion_matrix(y_test, preds_isolation_forest)
disp_isolation_forest = ConfusionMatrixDisplay(confusion_matrix=cm_isolation_forest, display_labels=[0, 1])
disp_isolation_forest.plot(cmap='Blues')
plt.title('Confusion Matrix for Isolation Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 保存模型
dump(best_IF, 'IF_model.joblib')
'''
# **SVM ****************************************************************************************************

'''
print("SVM****************************************************************")

# 定义要搜索的参数网格
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

train_start_time = time.time()
# 创建SVM模型
SVM = SVC(random_state=42)

# 训练模型
SVM.fit(x_tr_resample, y_tr_resample)

# 创建Grid Search对象
grid_search = GridSearchCV(SVM, param_grid, cv=5, scoring='accuracy')

# 在训练集上拟合Grid Search对象
grid_search.fit(x_tr_resample, y_tr_resample)

# 输出最佳参数组合和对应的准确率
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

training_time = time.time() - train_start_time
print(f"SVM model training time: {training_time:.2f} seconds")

# 获取最佳参数的SVM模型
best_SVM = grid_search.best_estimator_

# 在测试集上进行预测
test_start_time = time.time()
preds_SVM = best_SVM.predict(norm_test_f)

# 计算检测时间
detection_time = time.time() - test_start_time
print(f"SVM model detection time: {detection_time:.2f} seconds")

# 计算混淆矩阵和分类报告
cm_SVM = confusion_matrix(y_test, preds_SVM)
print(classification_report(y_test, preds_SVM))
print(confusion_matrix(y_test, preds_SVM))

# 可视化混淆矩阵
disp_SVM = ConfusionMatrixDisplay(confusion_matrix=cm_SVM, display_labels=best_SVM.classes_)
disp_SVM.plot(cmap='Blues')
plt.title('Confusion Matrix for SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 保存模型
dump(best_SVM, 'SVM_model.joblib')
# *******Random Forest****************************************************************************************************
'''
'''
print("Random Forest*********************************")


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

train_start_time = time.time()
# 实例化随机森林模型
RF = RandomForestClassifier(random_state=42)

# 创建Grid Search对象
grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)

# 在训练集上拟合Grid Search对象
grid_search.fit(x_tr_resample, y_tr_resample)

# 获取最佳参数组合
best_params = grid_search.best_params_

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
# 使用最佳参数初始化随机森林模型
best_RF = RandomForestClassifier(random_state=42, **best_params)

# 训练模型

best_RF.fit(x_tr_resample, y_tr_resample)
train_end_time = time.time()
training_time = train_end_time - train_start_time
print(f"RandomForest model training time: {training_time:.2f} seconds")


# 在测试集上进行预测
test_start_time = time.time()
preds_RF = best_RF.predict(norm_test_f)
test_end_time = time.time()
detection_time = test_end_time - test_start_time
print(f"RandomForest model detection time: {detection_time:.2f} seconds")

# 计算混淆矩阵和分类报告
cm_RF = confusion_matrix(y_test, preds_RF)
print(classification_report(y_test, preds_RF))
print(confusion_matrix(y_test, preds_RF))

# 可视化混淆矩阵
disp_RF = ConfusionMatrixDisplay(confusion_matrix=cm_RF, display_labels=best_RF.classes_)
disp_RF.plot(cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 保存模型
dump(best_RF, 'RF_model.joblib')

'''
# ******XGBoost************************************************************************************************
# XGBoost的默认超参数:
# learning_rate（学习率）=3  学习率控制每次迭代中模型权重的调整幅度，较低的学习率通常需要更多的迭代次数，但可能会导致更好的泛化性能。
# n_estimators（迭代次数)=100    指定要构建的树的数量，通常是指定训练的迭代轮数。
# max_depth（树的最大深度）=6   指定每棵树的最大深度，用于控制树的复杂度。较大的值可以使模型更复杂，但也容易导致过拟合。
# min_child_weight（叶子节点的最小权重）=1 指定叶子节点上的所有实例权重之和的最小值。用于控制树的分支，避免过度分裂。
# subsample（样本采样比例）=1 指定用于训练每棵树的样本的比例。较小的值可以减少方差，但可能增加偏差。
# colsample_bytree（特征采样比例）=1    指定用于训练每棵树的特征的比例。较小的值可以减少方差，但可能增加偏差。
# gamma（分裂节点时的损失减少阈值）=0 指定节点分裂时必须损失减少的最小值。增加此值会导致更保守的树。
# reg_alpha（L1正则化项系数） 和 reg_lambda（L2正则化项系数）：默认为0   控制模型的复杂度
'''
print("XGBoost ***************************************************************")
xgb_c = xgb.XGBClassifier(random_state=42)
train_start_time = time.time()

xgb_c.fit(x_tr_resample, y_tr_resample)

train_end_time = time.time()
training_time = train_end_time - train_start_time
print(f"XGBoost model training time: {training_time:.2f} seconds")

test_start_time = time.time()

preds_xgb = xgb_c.predict(norm_test_f)

test_end_time = time.time()
detection_time = test_end_time - test_start_time
print(f"XGBoost model detection time: {detection_time:.2f} seconds")


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

# ******XGBoost with grid search*******************************************************************************************************

print("XGBoost with grid search********************************************************")
train_start_time = time.time()
params_grid = {'learning_rate':[0.01, 0.1, 0.5],
              'n_estimators':[100,200],
              'subsample':[0.3, 0.5, 0.9],
               'max_depth':[2,3,4],
               'colsample_bytree':[0.3,0.5,0.7]}

best_xgb = GridSearchCV(estimator=xgb_c, param_grid=params_grid, scoring='recall', cv = 10, verbose = 2)

best_xgb.fit(x_tr_resample, y_tr_resample)
print(f'Best params found for XGBoost are: {best_xgb.best_params_}')
print(f'Best recall obtained by the best params: {best_xgb.best_score_}')
train_end_time = time.time()
training_time = train_end_time - train_start_time
print(f"Grid search XGBoost model training time: {training_time:.2f} seconds")
test_start_time = time.time()
preds_best_xgb = best_xgb.best_estimator_.predict(norm_test_f)
test_end_time = time.time()
detection_time = test_end_time - test_start_time
print(f"Grid search XGBoost model detection time: {detection_time:.2f} seconds")

print(classification_report(y_test, preds_best_xgb))
print(confusion_matrix(y_test, preds_best_xgb))
# 使用最佳模型的混淆矩阵可视化
cm_best_xgb = confusion_matrix(y_test, preds_best_xgb)
disp_best_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_best_xgb, display_labels=best_xgb.best_estimator_.classes_)
disp_best_xgb.plot(cmap='Blues')
plt.title('Confusion Matrix for Best XGBoost Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 保存模型
dump(best_xgb, 'xgb_model.joblib')

# ************************************************************************************
'''
'''
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

'''

# ****************************************** SOFT VOTING CLASSIFIER
print("SOFT VOTING CLF**************************************************")


'''
xgboost_model = XGBClassifier(**best_params_xgboost)
rf_model = RandomForestClassifier(**best_params_rf)
svm_model = SVC(**best_params_svm,probability=True)
# 创建参数网格，指定每个基础模型的权重范围

param_grid = {
    'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2], [2, 2, 1], [1, 2, 2], [2, 1, 2], [2, 2, 2]]
}

# 创建投票分类器
voting_clf = VotingClassifier(
    estimators=[('xgboost', xgboost_model), ('random_forest', rf_model), ('svm', svm_model)],
    voting='soft'  # 使用软投票
)

# 创建GridSearchCV对象
grid_search = GridSearchCV(voting_clf, param_grid, cv=5, scoring='accuracy')

# 在训练集上拟合Grid Search对象
grid_search.fit(x_tr_resample, y_tr_resample)

# 输出最佳参数组合和对应的准确率
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# 获取最佳参数的投票分类器模型
best_voting_clf = grid_search.best_estimator_
training_time = time.time() - start_time
print(f"VotCLF model training time: {training_time:.2f} seconds")
start_time = time.time()

# 在测试集上进行预测
preds_best_voting_clf = best_voting_clf.predict(norm_test_f)

testing_time = time.time() - start_time
print(f"VotCLF model testing time: {testing_time:.2f} seconds")

# 计算混淆矩阵和分类报告
cm_best_voting_clf = confusion_matrix(y_test, preds_best_voting_clf)
print(classification_report(y_test, preds_best_voting_clf))
print(confusion_matrix(y_test, preds_best_voting_clf))

# 可视化混淆矩阵
disp_best_voting_clf = ConfusionMatrixDisplay(confusion_matrix=cm_best_voting_clf, display_labels=best_voting_clf.classes_)
disp_best_voting_clf.plot(cmap='Blues')
plt.title('Confusion Matrix for Best Voting Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
'''

'''
# stackingggggggggggggggggggggggggggg
print("STACKING")
start_time = time.time()
best_params_xgboost = {'colsample_bytree': 0.7, 'learning_rate': 0.5, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.9}
best_params_rf = {'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
best_params_svm = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
# 定义基础模型
xgboost_model = XGBClassifier(**best_params_xgboost)
rf_model = RandomForestClassifier(**best_params_rf)
svm_model = SVC(**best_params_svm)
lr = LogisticRegression(random_state=43)
rf = RandomForestClassifier(random_state=43)

# 定义final_estimator的参数搜索空间
def objective(trial):
    final_estimator = trial.suggest_categorical('final_estimator', ['xgboost', 'lr', 'rf'])
    if final_estimator == 'xgboost':
        params = {
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0)
        }
        model = XGBClassifier(**params)
    elif final_estimator == 'lr':
        lr_params = {
            'C': trial.suggest_float('lr_C', 0.1, 10.0),
            'penalty': trial.suggest_categorical('lr_penalty', ['l2'])
        }
        model = LogisticRegression(**lr_params)
    elif final_estimator == 'rf':
        rf_params = {
            'max_depth': trial.suggest_int('rf_max_depth', 1, 20),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300)
        }
        model = RandomForestClassifier()
    else:
        raise ValueError("Invalid final estimator.")

    stacking_clf = StackingClassifier(
        estimators=[
            ('xgboost', xgboost_model),
            ('random_forest', rf_model),
            ('svm', svm_model)
        ],
        final_estimator=model,
        cv=5
    )

    stacking_clf.fit(X_train, y_train)
    y_pred_stacking = stacking_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_stacking)
    return accuracy


# 使用Optuna搜索最佳参数
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 输出最佳参数
# 打印最佳模型和参数

print("Best trial:")
trial = study.best_trial
print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# 使用最佳参数创建最终的StackingClassifier
final_estimator = trial.params['final_estimator']
if final_estimator == 'xgboost':
    final_params = {
        'colsample_bytree': trial.params['colsample_bytree'],
        'learning_rate': trial.params['learning_rate'],
        'max_depth': trial.params['max_depth'],
        'n_estimators': trial.params['n_estimators'],
        'subsample': trial.params['subsample']
    }
    model = XGBClassifier(**final_params)
elif final_estimator == 'lr':
    lr_params = {
        'C': trial.suggest_float('lr_C', 0.1, 10.0),
        'penalty': trial.suggest_categorical('lr_penalty', ['l2'])
    }
    model = LogisticRegression(**lr_params)
elif final_estimator == 'rf':
    rf_params = {
        'max_depth': trial.suggest_int('rf_max_depth', 1, 20),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
        'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300)
    }
    model = RandomForestClassifier()

stacking_clf = StackingClassifier(
    estimators=[
        ('xgboost', xgboost_model),
        ('random_forest', rf_model),
        ('svm', svm_model)
    ],
    final_estimator=model,
    cv=5
)
print("Best Stacking Model:")
print(stacking_clf)

# 训练Stacking模型
stacking_clf.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"stacking model testing time: {training_time:.2f} seconds")
start_time = time.time()
# 预测并评估模型
y_pred_stacking = stacking_clf.predict(X_test)
print(f"stacking model testing time: {time.time() - start_time:}")

accuracy = accuracy_score(y_test, y_pred_stacking)
print("Accuracy of Stacking Model:", accuracy)

print(classification_report(y_test, y_pred_stacking))
print(confusion_matrix(y_test, y_pred_stacking))

# 计算混淆矩阵
cm_stacking = confusion_matrix(y_test, y_pred_stacking)
# 可视化混淆矩阵
disp_stacking = ConfusionMatrixDisplay(confusion_matrix=cm_stacking, display_labels=stacking_clf.classes_)
disp_stacking.plot(cmap='Blues')
plt.title('Confusion Matrix for Stacking Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
'''
'''
# 定义三个基学习器
best_params_xgboost = {'colsample_bytree': 0.7, 'learning_rate': 0.5, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.9}
best_params_rf = {'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
best_params_svm = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

xgboost_model = XGBClassifier(**best_params_xgboost)
rf_model = RandomForestClassifier(**best_params_rf)
svm_model = SVC(**best_params_svm)

start_time = time.time()

# 定义三个基学习器
best_params_xgboost = {'colsample_bytree': 0.7, 'learning_rate': 0.5, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.9}
best_params_rf = {'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
best_params_svm = {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}

xgboost_model = XGBClassifier(**best_params_xgboost)
rf_model = RandomForestClassifier(**best_params_rf)
svm_model = SVC(**best_params_svm)

# 训练 SVM 模型并增加错误实例的权重
svm_model.fit(X_train, y_train)
svm_errors = svm_model.predict(X_train) != y_train
svm_sample_weights = np.ones(len(y_train))
svm_sample_weights[svm_errors] *= 2  # 增加 SVM 模型预测错误实例的权重

# 训练 XGBoost 模型
xgboost_model.fit(X_train, y_train, sample_weight=svm_sample_weights)


# 获取 XGBoost 模型预测结果
xgboost_preds = xgboost_model.predict(X_test)

# 训练 Random Forest 模型并增加错误实例的权重
rf_model.fit(X_train, y_train)
rf_errors = rf_model.predict(X_train) != y_train
rf_sample_weights = np.ones(len(y_train))
rf_sample_weights[rf_errors] *= 2  # 增加 Random Forest 模型预测错误实例的权重

# 训练 Random Forest 模型

rf_model.fit(X_train, y_train, sample_weight=rf_sample_weights)
training_time = time.time() - start_time

start_time=time.time()
# 获取 Random Forest 模型预测结果
rf_preds = rf_model.predict(X_test)
testing_time = time.time() - start_time
# 输出 Random Forest 模型的混淆矩阵和分类报告
print("Confusion Matrix for Boosting Model:")
print(confusion_matrix(y_test, rf_preds))
print("\nClassification Report for Boosting Model:")
print(classification_report(y_test, rf_preds))

print("Training Time of boosting:", training_time, "seconds")
print("Testing Time of boosting", time.time() - start_time, "seconds")
'''

