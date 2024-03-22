#
# (一)导入tensorflow模块（8分）
import tensorflow as tf
# (二)准备数据集（共12分，各6分）
x_data = [[0, 0, 1, 0],
          [1, 1, 1, 1],
          [1, 0, 1, 1],
          [0, 1, 1, 0]]
y_data = [[0],
          [1],
          [1],
          [0]]
# (三)定义变量X（8分）和Y（8分）
X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([4, 1]))
b = tf.Variable(tf.random_normal([1,]))
# (四)定义预测值，可用s神经网络igmoid函数（8分）
h = tf.sigmoid(tf.matmul(X, w) + b)
# (五)定义代价函数（8分）
cost = -tf.reduce_mean(Y*tf.log(h)+(1-Y)*tf.log(1-h))
predict = tf.cast(h > 0.5, dtype=tf.int32)
# (六)使用梯度下降优化器（8分）
op = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(cost)
# (七)创建会话（8分）
with tf.Session() as sess:
# (八)全局变量初始化（8分）
    sess.run(tf.global_variables_initializer())
# (九)迭代10001次，每500次输出一次代价值（8分）
    for i in range(10001):
        J,op_ = sess.run([cost,op],feed_dict={X: x_data,Y: y_data})
        if i % 500 == 0:
            print(J)
# (十)对数据[1,0,0,0]进行预测，输出结果。（8分）
    print(sess.run(predict, feed_dict={X: [[1, 0, 0, 0]]}))
# (十一)对数据[1,1,1,1]进行预测，输出结果。（8分）
    print(sess.run(predict, feed_dict={X: [[1, 1, 1, 1]]}))


