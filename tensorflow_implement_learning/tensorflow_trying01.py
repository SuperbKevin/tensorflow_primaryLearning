import tensorflow as tf
from tensorflow_implement_learning.UDF import load_data
from skimage.color import rgb2gray
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import random
# 忽略警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

images, labels = load_data('./Training')
images28 = [transform.resize(image, (28, 28)) for image in images]
# 将图片转化为数组
images28 = np.array(images28)
# 将图片处理成灰度图
images28 = rgb2gray(images28)

# 创建图
# graph = tf.Graph()

# # Initialize placeholder 为图像定义占位符
# with graph.as_default():
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# Flatten the input data将输入数据（图片）转换为集合
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
# t f.contrib.layers.fully_connection(F，num_output,activation_fn)
# 这个函数就是全连接层,F是输入，num_output是下一层单元的个数，activation_fn是激活函数
# tf.nn.relu 这个函数的作用是计算激活函数relu，即max(features, 0)，即将矩阵中每行的非最大值置0
# 激活函数就是在人工神经网络的神经元上运行的函数，负责将神经元的输入映射到输出端。
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
# spares_softmax_cross_entropy_wit_logits:根据稀疏表示的label（实际标签）和输出层数据（logits）计算交叉熵
# 这里测量的是离散分类任务中的概率误差，在这些任务中，类别是互斥的。
# 这意味着每一个条目（entry）都是一个单独的类别。在这种情况下，一个交通标志只会有一个标签
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# **回归（regression）被用于预测连续值，而分类（classification）则被用于预测离散值或数据点的类别**

# Define an optimizer 使用ADAM优化算法
# Adam 通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label
# 第一个参数是矩阵，第二个参数是0或者1。
# 0表示的是按列比较返回最大值的索引，1表示按行比较返回最大值的索引
# 返回最大可能性的标签（在62个标签之中）
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric 定义正确性评价指标
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# print("images_flat: ", images_flat)
# print("logits: ", logits)
# print("loss: ", loss)
# print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(501):
    print('EPOCH', i)
    accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
    if i % 10 == 0:
        print("Loss: ", loss)
    print('DONE WITH EPOCH')


# 评估结果
# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
# Print the real and predicted labels print(sample_labels)
print(predicted)

# Display the predictions and the ground truth
plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1+i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth: {0:9}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")

plt.show()


# 加载测试数据
# Load the test data
test_images, test_labels = load_data('./Testing')

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to gray scale
test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

sess.close()
