import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 设置数据集路径
data_dir = 'C:\\Users\\Agua\\Desktop\\Diana-256-v1_1'

# 图像大小
img_height, img_width = 256, 256
batch_size = 32

# 使用ImageDataGenerator进行数据预处理
datagen = ImageDataGenerator(validation_split=0.2)

# 训练集和验证集生成器
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# 将生成器中的数据转换为NumPy数组
def generator_to_numpy(generator):
    data = []
    labels = []
    for _ in range(len(generator)):
        batch_data, batch_labels = next(generator)
        data.append(batch_data)
        labels.append(batch_labels)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    return data, labels

# 获取训练集和验证集数据
train_data, train_labels = generator_to_numpy(train_generator)
validation_data, validation_labels = generator_to_numpy(validation_generator)

# 数据归一化
train_data = train_data / 255.0
validation_data = validation_data / 255.0

# 使用混合精度训练
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 构建模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 查看模型结构
model.summary()

# 训练模型
epochs = 50
history = model.fit(
    train_data, train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(validation_data, validation_labels))

from sklearn.metrics import classification_report, confusion_matrix

# 评估模型
validation_predictions = model.predict(validation_data)
y_pred = np.argmax(validation_predictions, axis=1)
y_true = np.argmax(validation_labels, axis=1)

print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))

print('Classification Report')
target_names = list(validation_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=target_names))

from tensorflow.keras.regularizers import l2

# 构建优化后的模型
optimized_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3), kernel_regularizer=l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(9, activation='softmax')
])

# 编译优化后的模型
optimized_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练优化后的模型
optimized_history = optimized_model.fit(
    train_data, train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(validation_data, validation_labels))
