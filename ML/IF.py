
train_start_time = time.time()
# 实例化 Isolation Forest 模型
isolation_forest = IsolationForest(random_state=42, bootstrap=True)

# 在训练集上拟合模型
isolation_forest.fit(x_tr_resample)
train_end_time = time.time()

training_time = train_end_time - train_start_time
print(f"Isolation Forest model training time: {training_time:.2f} seconds")

test_start_time = time.time()

# 使用模型进行异常检测，-1 表示异常，1 表示正常
preds_isolation_forest = isolation_forest.predict(norm_test_f)

test_end_time = time.time()
# 计算检测时间
detection_time = test_end_time - test_start_time
print(f"Isolation Forest model detection time: {detection_time:.2f} seconds")

# 将 -1 转换为 0，1 保持不变，以便与正常/异常标签匹配
preds_isolation_forest[preds_isolation_forest == -1] = 0

# 打印分类报告和混淆矩阵
print(classification_report(y_test, preds_isolation_forest))
print(confusion_matrix(y_test, preds_isolation_forest))

# 绘制混淆矩阵可视化
cm_isolation_forest = confusion_matrix(y_test, preds_isolation_forest)
disp_isolation_forest = ConfusionMatrixDisplay(confusion_matrix=cm_isolation_forest, display_labels=[0, 1])
disp_isolation_forest.plot(cmap='Blues')
plt.title('Confusion Matrix for Isolation Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# grid Search
