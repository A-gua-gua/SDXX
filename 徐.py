import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report

# 数据集路径
dataset_path = r'C:\Users\xmz\Desktop\flowers'
# 类别列表
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
# 目标图像尺寸
target_size = (64, 64)


def load_dataset(dataset_path, classes, target_size):
    images = []
    labels = []

    for i, cls in enumerate(classes):
        class_path = os.path.join(dataset_path, cls)
        print(f"加载 {cls} 类别的图像...")
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                # 加载图像并调整大小
                image = Image.open(image_path)
                image = image.resize(target_size, Image.LANCZOS)
                # 转换为数组
                image = np.array(image)
                # 归一化到 [0, 1]
                image = image / 255.0
                # 添加到列表
                images.append(image)
                labels.append(i)  # 类别索引
            except Exception as e:
                print(f"加载图像时出错 {image_path}: {e}")

    if len(images) == 0:
        raise ValueError("未加载到图像。请检查数据集路径和图像加载过程。")

    return np.array(images), np.array(labels)


try:
    # 加载数据集
    images, labels = load_dataset(dataset_path, classes, target_size)
except Exception as e:
    print(f"加载数据集时出错: {e}")
    exit(1)  # 如果数据集加载失败，则退出脚本

# 划分训练集、验证集和测试集
try:
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2,
                                                                            random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2,
                                                                          random_state=42)
except ValueError as ve:
    print(f"划分数据集时出错: {ve}")
    exit(1)  # 如果数据集划分失败，则退出脚本

# 数据增强
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# 显示示例图像
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(train_images[i])
    plt.title(classes[train_labels[i]])
    plt.axis('off')
plt.show()

# 输出数据集大小
print(f"训练集: {train_images.shape}, {train_labels.shape}")
print(f"验证集: {val_images.shape}, {val_labels.shape}")
print(f"测试集: {test_images.shape}, {test_labels.shape}")


# 构建原始卷积神经网络模型
def build_original_model(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(data_augmentation)  # 添加数据增强层
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(classes), activation='softmax'))  # 输出层，使用softmax激活函数

    return model


# 构建原始模型
input_shape = (*target_size, 3)
original_model = build_original_model(input_shape)

# 编译原始模型
original_model.compile(optimizer='rmsprop',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

# 打印原始模型架构
original_model.summary()

# 定义回调函数：学习率调度器和早期停止
lr_scheduler = callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
early_stopping = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# 训练原始模型
history_original = original_model.fit(train_images, train_labels, epochs=30, batch_size=32,
                                      validation_data=(val_images, val_labels),
                                      callbacks=[lr_scheduler, early_stopping])


# 构建优化后的卷积神经网络模型
def build_optimized_model(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(data_augmentation)  # 添加数据增强层
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))  # 调整 Dropout 率

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))  # 调整 Dropout 率

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))  # 调整 Dropout 率

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.4))  # 调整 Dropout 率
    model.add(layers.Dense(len(classes), activation='softmax'))  # 输出层，使用softmax激活函数

    return model


# 构建优化后的模型
optimized_model = build_optimized_model(input_shape)

# 编译优化后的模型
optimized_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# 打印优化后的模型架构
optimized_model.summary()

# 训练优化后的模型
history_optimized = optimized_model.fit(train_images, train_labels, epochs=30, batch_size=32,
                                        validation_data=(val_images, val_labels),
                                        callbacks=[lr_scheduler, early_stopping])

# 绘制训练过程中的准确率和损失曲线
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history_original.history['accuracy'], label='Original Training Accuracy')
plt.plot(history_original.history['val_accuracy'], label='Original Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Original Model Accuracy')

plt.subplot(2, 2, 2)
plt.plot(history_original.history['loss'], label='Original Training Loss')
plt.plot(history_original.history['val_loss'], label='Original Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Original Model Loss')

plt.subplot(2, 2, 3)
plt.plot(history_optimized.history['accuracy'], label='Optimized Training Accuracy')
plt.plot(history_optimized.history['val_accuracy'], label='Optimized Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Optimized Model Accuracy')

plt.subplot(2, 2, 4)
plt.plot(history_optimized.history['loss'], label='Optimized Training Loss')
plt.plot(history_optimized.history['val_loss'], label='Optimized Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Optimized Model Loss')

plt.tight_layout()
plt.show()




# 评估原始模型
test_loss_original, test_accuracy_original = original_model.evaluate(test_images, test_labels, verbose=2)
print(f"原始模型测试集损失：{test_loss_original:.4f}")
print(f"原始模型测试集准确率：{test_accuracy_original:.4f}")

# 评估优化后的模型
test_loss_optimized, test_accuracy_optimized = optimized_model.evaluate(test_images, test_labels, verbose=2)
print(f"优化模型测试集损失：{test_loss_optimized:.4f}")
print(f"优化模型测试集准确率：{test_accuracy_optimized:.4f}")

# 打印分类报告
predictions_original = original_model.predict(test_images)
predicted_labels_original = np.argmax(predictions_original, axis=1)
print("原始模型分类报告:")
print(classification_report(test_labels, predicted_labels_original, target_names=classes))

predictions_optimized = optimized_model.predict(test_images)
predicted_labels_optimized = np.argmax(predictions_optimized, axis=1)
print("优化模型分类报告:")
print(classification_report(test_labels, predicted_labels_optimized, target_names=classes))
