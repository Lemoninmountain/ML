
print("SVM****************************************************************")
train_start_time = time.time()
# 创建SVM模型
SVM = SVC(kernel='sigmoid', random_state=42)

# 训练模型
SVM.fit(x_tr_resample, y_tr_resample)
train_end_time = time.time()
training_time = train_end_time - train_start_time
print(f"SVM model training time: {training_time:.2f} seconds")
test_start_time = time.time()
# 预测
preds_SVM = SVM.predict(norm_test_f)
test_end_time = time.time()

# 计算检测时间
detection_time = test_end_time - test_start_time
print(f"SVM model detection time: {detection_time:.2f} seconds")
# 计算混淆矩阵和分类报告
cm_SVM = confusion_matrix(y_test, preds_SVM)
print(classification_report(y_test, preds_SVM))
print(confusion_matrix(y_test, preds_SVM))

# 可视化混淆矩阵
disp_SVM = ConfusionMatrixDisplay(confusion_matrix=cm_SVM, display_labels=SVM.classes_)
disp_SVM.plot(cmap='Blues')
plt.title('Confusion Matrix for SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# grid Search************************************************************