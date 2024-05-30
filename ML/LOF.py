
# 创建 LOF 模型
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# 在训练集上拟合 LOF 模型
train_start_time = time.time()
# 在训练集上拟合 LOF 模型
lof.fit(x_tr_resample)
train_end_time = time.time()

training_time = train_end_time - train_start_time
print(f"LOF model training time: {training_time:.2f} seconds")

test_start_time = time.time()

# 使用 LOF 模型检测异常值
outliers = lof.fit_predict(norm_test_f)

test_end_time = time.time()

# 计算检测时间
detection_time = test_end_time - test_start_time
print(f"LOF model detection time: {detection_time:.2f} seconds")

# 将异常值和正常值分开
normal_data = norm_test_f[outliers == 1]
outlier_data = norm_test_f[outliers == -1]

# 可选：打印异常值和正常值的数量
print("Number of normal samples:", len(normal_data))
print("Number of outliers:", len(outlier_data))

# 可选：可视化异常值和正常值
plt.scatter(normal_data[:, 0], normal_data[:, 1], color='blue', label='Normal')
plt.scatter(outlier_data[:, 0], outlier_data[:, 1], color='red', label='Outlier')
plt.title('LOF Outlier Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 创建一个虚拟的标签，将已知的异常值标记为1，非异常值标记为0
y_true = np.zeros(len(norm_test_f))
y_true[y_test] = 0  # 将已知的异常值标记为1

# 使用 LOF 模型检测异常值
outliers = lof.fit_predict(norm_test_f)

# 将异常值的标签转换为二进制，1 表示正常值，-1 表示异常值
predicted_labels = np.where(outliers == 1, 0, 1)

# 计算混淆矩阵
cm = confusion_matrix(y_test, predicted_labels)

# 计算分类报告
report = classification_report(y_test, predicted_labels)

# 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Outlier'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for LOF')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\nClassification Report:")
print(report)


# ******************
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





#