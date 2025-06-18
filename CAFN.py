import os
import numpy as np
import pickle
import tensorflow as tf
from keras.models import Sequential, Model
from keras import regularizers
from keras.callbacks import Callback
from keras.layers import (Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Dense,Layer, Add,
                          Flatten, Input, LayerNormalization)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras import regularizers
import time, random
import seaborn as sns

# 设置 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 设置随机种子，保证可重复性
tf.random.set_seed(2)
np.random.seed(2)
random.seed(2)

class CustomEpochCallback(Callback):
    def __init__(self, start_epoch=1):
        super(CustomEpochCallback, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['epoch'] = epoch + self.start_epoch

class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_losses = []  # 初始化一个空列表来存储每个epoch的损失值

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.epoch_losses.append(logs.get('loss'))  # 在每个epoch结束时记录损失值

    def save_losses(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.epoch_losses, f)   # 使用pickle将损失值保存为.pkl文件


# 读取 pkl 文件
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# 构建模型
# 加性注意力（Additive Attention）实现
class AdditiveAttention(Layer):
    def __init__(self, attention_dim):
        super(AdditiveAttention, self).__init__()
        # 定义用于转换查询、键、值的权重层
        self.W_query = Dense(attention_dim, use_bias=False)
        self.W_key = Dense(attention_dim, use_bias=False)
        self.W_value = Dense(attention_dim, use_bias=False)
        # 定义用于计算注意力分数的权重层
        self.W_score = Dense(1, use_bias=False)

    def call(self, query, key, value):
        # 转换查询、键、值
        query_transformed = self.W_query(query)  # (batch_size, seq_len, attention_dim)
        key_transformed = self.W_key(key)  # (batch_size, seq_len, attention_dim)
        value_transformed = self.W_value(value)  # (batch_size, seq_len, attention_dim)

        # 计算注意力分数
        scores = self.W_score(tf.tanh(query_transformed + key_transformed))  # (batch_size, seq_len, 1)
        scores = tf.squeeze(scores, axis=-1)  # (batch_size, seq_len)
        attention_weights = tf.nn.softmax(scores, axis=1)  # (batch_size, seq_len)

        # 计算加权后的值
        attention_weights = tf.expand_dims(attention_weights, axis=-1)  # (batch_size, seq_len, 1)
        context_vector = tf.reduce_sum(attention_weights * value_transformed, axis=1)  # (batch_size, attention_dim)

        return context_vector, attention_weights


class SelfAttention(Layer):
    def __init__(self, attention_dim=64):
        super(SelfAttention, self).__init__()
        # 使用加性注意力
        self.attention = AdditiveAttention(attention_dim=attention_dim)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.add = Add()

    def call(self, x):
        # 计算自注意力
        attention_output, _ = self.attention(query=x, key=x, value=x)
        # 添加残差连接
        out = self.add([x, attention_output])
        # 层归一化
        out = self.layernorm(out)
        return out


def self_attention_block(x):
    # 创建自注意力块
    attention_output = SelfAttention(attention_dim=x.shape[-1])(x)
    # 可以进一步添加其他层，如前馈网络
    return attention_output


def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # 第一层卷积 + 自注意力模块
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = self_attention_block(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # 第二层卷积 + 自注意力模块
    x = Conv1D(filters=16, kernel_size=3, activation='relu')(x)
    x = self_attention_block(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # 第三层卷积 + 自注意力模块
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
    x = self_attention_block(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # 平展层
    x = Flatten()(x)

    # 全连接层
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(1))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 重新编译模型

    return model


# 读取初始训练和测试数据
def load_data(names, data_type='train'):
    data, labels = [], []
    for name in names:
        file_path = f'./data/{data_type}/{name}_label.pkl'
        loaded_data = load_pkl(file_path)
        data.append(loaded_data[0])
        labels.append(loaded_data[1])
    return np.vstack(data), np.concatenate(labels)


# 转换数据为 TensorFlow Tensors
def prepare_data(data, labels):
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
    return data_tensor, labels_tensor

# 绘制并保存混淆矩阵的函数
def save_confusion_matrix(cm, class_names, file_path, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(file_path)  # 保存图片
    plt.show()  # 展示当前图形


# 确保目录存在
os.makedirs('./loss_value/SFLSTM/', exist_ok=True)
os.makedirs('./confusion_matrix/SFLSTM/', exist_ok=True)


# 初始阶段
# 初始训练阶段
initial_classes = ['C0', 'C1', 'C2', 'C3']
train_data, train_labels = load_data(initial_classes, data_type='train')
test_data, test_labels = load_data(initial_classes, data_type='test')

# 重塑训练数据
input_shape = (train_data.shape[1], 1)
train_data = tf.reshape(train_data, (-1, 3072, 1))

# 构建并训练模型
model = build_model(input_shape=(input_shape), num_classes=4)
loss_value1 = LossHistory()  # 实例化自定义回调

# 记录模型训练的开始时间
print("Initial training phase...")
start_time_initial = time.time()  # 记录初始训练开始时间
history = model.fit(train_data, train_labels, epochs=100, batch_size=32, verbose=1, callbacks=[loss_value1])
# 打印摘要
model.summary()
# 记录初始训练结束时间
end_time_initial = time.time()
print(f"Initial training time: {end_time_initial - start_time_initial:.2f} seconds")

# 保存损失值
loss_value1.save_losses('./loss_value/SFLSTM/1.pkl')

# 模型评估
test_data0 = tf.reshape(test_data,(-1,3072,1))
test_loss0, test_accuracy0 = model.evaluate(test_data0, test_labels)
print(f'Test Loss: {test_loss0:.4f}, Test Accuracy: {test_accuracy0:.4f}')

# 生成分类报告
predicted_labels0 = np.argmax(model.predict(test_data0), axis=-1)
print(classification_report(np.argmax(test_labels, axis=-1), predicted_labels0, zero_division=0))

# 计算混淆矩阵
cm0 = confusion_matrix(np.argmax(test_labels, axis=-1), predicted_labels0)

# 保存混淆矩阵
save_confusion_matrix(cm0, initial_classes, './confusion_matrix/SFLSTM/initial_phase.png')



# 第二阶段
# 增量训练阶段
new_classes = ['C4', 'C5']
new_classes_confusion_matrix = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
new_train_data, new_train_labels = load_data(new_classes, data_type='train')
new_test_data, new_test_labels = load_data(new_classes, data_type='test')

# 合并训练和测试数据
print(train_data.shape, new_train_data.shape)
new_train_data = tf.reshape(new_train_data,(-1,3072,1))
combined_train_data = np.vstack((train_data, new_train_data))
combined_test_data = np.vstack((test_data, new_test_data))

# 更新类别数量并扩展 one-hot 标签
new_total_classes = 6
extended_train_labels = np.zeros((train_labels.shape[0], new_total_classes))
extended_test_labels = np.zeros((test_labels.shape[0], new_total_classes))

# 填充旧任务标签
extended_train_labels[:, :4] = train_labels
extended_test_labels[:, :4] = test_labels

# 填充新任务标签
combined_train_labels = np.vstack((extended_train_labels, new_train_labels))
combined_test_labels = np.vstack((extended_test_labels, new_test_labels))

# 重塑合并后的数据
combined_train_data = tf.reshape(combined_train_data, (-1, 3072, 1))

# 更新模型以适应新的类别
model = build_model(input_shape,new_total_classes)
out = model.layers[-2].output
model = Model(inputs=model.input,outputs=out)
out = Dense(new_total_classes, activation='softmax', kernel_regularizer=regularizers.l2(1))(out)
model = Model(inputs=model.input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 重新编译模型
loss_value2 = LossHistory()  # 实例化自定义回调

# 记录增量训练开始时间
print("Incremental training phase...")
start_time_incremental = time.time()
history2 = model.fit(combined_train_data, combined_train_labels, epochs=50, batch_size=32, verbose=1, callbacks=[loss_value2])
model.summary()
# 记录增量训练结束时间
end_time_incremental = time.time()
print(f"Incremental training time: {end_time_incremental - start_time_incremental:.2f} seconds")

# 保存损失值
loss_value2.save_losses('./loss_value/SFLSTM/2.pkl')

# 模型评估
combined_test_data2 = tf.reshape(combined_test_data,(-1,3072,1))
test_loss, test_accuracy = model.evaluate(combined_test_data2, combined_test_labels)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# 计算遗忘率
# 比较增量学习前后的性能
forgetting_rate_loss = (test_loss0 - test_loss)/test_loss0
print(f'遗忘率（损失）：{forgetting_rate_loss:.4f}')
forgetting_rate_accuracy = (test_accuracy0 - test_accuracy) / test_accuracy0
print(f'遗忘率（准确率）：{forgetting_rate_accuracy:.4f}')

# 生成分类报告
predicted_labels2 = np.argmax(model.predict(combined_test_data2), axis=-1)
print(classification_report(np.argmax(combined_test_labels, axis=-1), predicted_labels2, zero_division=0))

# 计算混淆矩阵
cm1 = confusion_matrix(np.argmax(combined_test_labels, axis=-1), predicted_labels2)

# 保存混淆矩阵
save_confusion_matrix(cm1, new_classes_confusion_matrix, './confusion_matrix/SFLSTM/Incremental_phase1.png')



# 第三阶段
# 增量训练阶段
new_classes3 = ['C6', 'C7']
new_classes3_confusion_matrix = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
new_train_data3, new_train_labels3 = load_data(new_classes3, data_type='train')
new_test_data3, new_test_labels3 = load_data(new_classes3, data_type='test')

# 合并训练和测试数据
print(combined_train_data.shape, new_train_data3.shape)
new_train_data3 = tf.reshape(new_train_data3,(-1,3072,1))
combined_train_data3 = np.vstack((combined_train_data, new_train_data3))
combined_test_data3 = np.vstack((combined_test_data, new_test_data3))

# 更新类别数量并扩展 one-hot 标签
new_total_classes3 = 8
extended_train_labels3 = np.zeros((combined_train_labels.shape[0], new_total_classes3))
extended_test_labels3 = np.zeros((combined_test_labels.shape[0], new_total_classes3))

extended_train_labels3[:, :6] = combined_train_labels
extended_test_labels3[:, :6] = combined_test_labels

combined_train_labels3 = np.vstack((extended_train_labels3, new_train_labels3))
combined_test_labels3 = np.vstack((extended_test_labels3, new_test_labels3))

# 重塑合并后的数据
combined_train_data3 = tf.reshape(combined_train_data3, (-1, 3072, 1))

# 创建自定义回调对象，设定起始 epoch
custom_epoch_callback3 = CustomEpochCallback(start_epoch=150)

# 更新模型以适应新的类别
model = build_model(input_shape,new_total_classes3)
out3 = model.layers[-2].output
model = Model(inputs=model.input,outputs=out3)
out3 = Dense(new_total_classes3, activation='softmax', kernel_regularizer=regularizers.l2(1))(out3)
model = Model(inputs=model.input, outputs=out3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 重新编译模型
loss_value3 = LossHistory()  # 实例化自定义回调

# 记录增量训练开始时间
print("Incremental training phase3...")
start_time_incremental3 = time.time()
history3 = model.fit(combined_train_data3, combined_train_labels3, epochs=50, batch_size=32, verbose=1, callbacks=[loss_value3])
model.summary()
# 记录增量训练结束时间
end_time_incremental3 = time.time()
print(f"Incremental training time: {end_time_incremental3 - start_time_incremental3:.2f} seconds")

# 保存损失值
loss_value3.save_losses('./loss_value/SFLSTM/3.pkl')

# 模型评估
combined_test_data33 = tf.reshape(combined_test_data3,(-1,3072,1))
test_loss3, test_accuracy3 = model.evaluate(combined_test_data33, combined_test_labels3)
print(f'Test Loss: {test_loss3:.4f}, Test Accuracy: {test_accuracy3:.4f}')

# 计算遗忘率
# 比较增量学习前后的性能
forgetting_rate_loss2 = (test_loss - test_loss3) / test_loss
print(f'遗忘率（损失）：{forgetting_rate_loss2:.4f}')
forgetting_rate_accuracy2 = (test_accuracy - test_accuracy3) / test_accuracy
print(f'遗忘率（准确率）：{forgetting_rate_accuracy2:.4f}')

# 生成分类报告
predicted_labels3 = np.argmax(model.predict(combined_test_data33), axis=-1)
print(classification_report(np.argmax(combined_test_labels3, axis=-1), predicted_labels3, zero_division=0))

# 计算混淆矩阵
cm2 = confusion_matrix(np.argmax(combined_test_labels3, axis=-1), predicted_labels3)

# 保存混淆矩阵
save_confusion_matrix(cm2, new_classes3_confusion_matrix, './confusion_matrix/SFLSTM/Incremental_phase2.png')



# 第四阶段
# 增量训练阶段
new_classes4 = ['C8', 'C9']
new_classes4_confusion_matrix = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
new_train_data4, new_train_labels4 = load_data(new_classes4, data_type='train')
new_test_data4, new_test_labels4 = load_data(new_classes4, data_type='test')

# 合并训练和测试数据
print(combined_train_data3.shape, new_train_data4.shape)
new_train_data4 = tf.reshape(new_train_data4,(-1,3072,1))
combined_train_data4 = np.vstack((combined_train_data3, new_train_data4))
combined_test_data4 = np.vstack((combined_test_data3, new_test_data4))

# 更新类别数量并扩展 one-hot 标签
new_total_classes4 = 10
extended_train_labels4 = np.zeros((combined_train_labels3.shape[0], new_total_classes4))
extended_test_labels4 = np.zeros((combined_test_labels3.shape[0], new_total_classes4))

extended_train_labels4[:, :8] = combined_train_labels3
extended_test_labels4[:, :8] = combined_test_labels3

combined_train_labels4 = np.vstack((extended_train_labels4, new_train_labels4))
combined_test_labels4 = np.vstack((extended_test_labels4, new_test_labels4))

# 重塑合并后的数据
combined_train_data4 = tf.reshape(combined_train_data4, (-1, 3072, 1))

# 更新模型以适应新的类别
model = build_model(input_shape,new_total_classes4)
out4 = model.layers[-2].output
model = Model(inputs=model.input,outputs=out4)
out4 = Dense(new_total_classes4, activation='softmax', kernel_regularizer=regularizers.l2(1))(out4)
model = Model(inputs=model.input, outputs=out4)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 重新编译模型
loss_value4 = LossHistory()  # 实例化自定义回调

# 记录增量训练开始时间
print("Incremental training phase4...")
start_time_incremental4 = time.time()
history4 = model.fit(combined_train_data4, combined_train_labels4, epochs=50, batch_size=32, verbose=1, callbacks=[loss_value4])
model.summary()
# 记录增量训练结束时间
end_time_incremental4 = time.time()
print(f"Incremental training time: {end_time_incremental4 - start_time_incremental4:.2f} seconds")

# 保存损失值
loss_value4.save_losses('./loss_value/SFLSTM/4.pkl')

# 模型评估
combined_test_data4 = tf.reshape(combined_test_data4,(-1,3072,1))
test_loss4, test_accuracy4 = model.evaluate(combined_test_data4, combined_test_labels4)
print(f'Test Loss: {test_loss4:.4f}, Test Accuracy: {test_accuracy4:.4f}')

# 计算遗忘率
# 比较增量学习前后的性能
forgetting_rate_loss3 = (test_loss3 - test_loss4) / test_loss3
print(f'遗忘率（损失）：{forgetting_rate_loss3:.4f}')
forgetting_rate_accuracy3 = (test_accuracy3 - test_accuracy4) / test_accuracy3
print(f'遗忘率（准确率）：{forgetting_rate_accuracy3:.4f}')

# 生成分类报告
predicted_labels4 = np.argmax(model.predict(combined_test_data4), axis=-1)
print(classification_report(np.argmax(combined_test_labels4, axis=-1), predicted_labels4, zero_division=0))

# 计算混淆矩阵
cm3 = confusion_matrix(np.argmax(combined_test_labels4, axis=-1), predicted_labels4)

# 保存混淆矩阵
save_confusion_matrix(cm3, new_classes4_confusion_matrix, './confusion_matrix/SFLSTM/Incremental_phase3.png')

