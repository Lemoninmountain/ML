# 对特征进行归一化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.shape)
print(X_test_scaled.shape)

X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# 定义 RNN 模型
model = tf.keras.Sequential([
    LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_train_time = time.time()

# 训练模型
model.fit(X_train_reshaped, y_train, epochs=15, batch_size=32, validation_split=0.2)
end_train_time = time.time()
train_time=end_train_time-start_train_time
print(f"Training time for RNN: {train_time:.2f} seconds")

# 评估模型
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

start_test_time=time.time()

# 预测测试集结果
y_pred_proba = model.predict(X_test_reshaped)
y_pred = (y_pred_proba > 0.5).astype(int)

end_test_time=time.time()

test_time=end_test_time-start_test_time
print(f"Testing time for RNN: {test_time:.2f} seconds")

print(classification_report(y_test, y_pred))
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 输出混淆矩阵
print("Confusion Matrix:")
print(cm)

# 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])  # 根据你的标签类别修改 display_labels
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for RNN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



# grid**************


