# 털과 날개가 있는지 없는지에 따라, 포유류인지 조류인지 분류하는 신경망 모델을 만들어봅니다.
import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# [기타, 포유류, 조류]
# 다음과 같은 형식을 one-hot 형식의 데이터라고 합니다.
y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# 신경층 모델 구성

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([2,3], -1., 1.))

b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X,W), b)
# relu 함수를 activation function 으로
L = tf.nn.relu(L)

model = tf.nn.softmax(L)

# axit = 0 col을 더함. # axis = 1 row를 더함.
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
optimizer = tf.train.AdamOptimizer (learning_rate = 0.01)
train_op = optimizer.minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(1000):
    val_list = sess.run(train_op, feed_dict={X : x_data, Y: y_data})
    print(val_list)
    if (step + 1) % 10 == 0:
        print(step+1 , sess.run(cost, feed_dict={ X : x_data, Y: y_data}))


# 결과 확인
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)

print("예측값 : ", sess.run(prediction, feed_dict={X: x_data}))
print("실제값 : ", sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도 : %.2f" % sess.run(accuracy * 100, feed_dict={X : x_data, Y: y_data}))

