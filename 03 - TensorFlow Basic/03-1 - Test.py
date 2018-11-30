import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

hypothesis = W*X +b

# cost 함수를 정의하고
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
# optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)

train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 횟수가 증가하면 오차가 줄어듬
    for step in range(100):
    # for step in range(1000):
        train_op_val, cost_val = sess.run([train_op, cost], feed_dict={ X: x_data , Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))
        print("train_op_val ; ",train_op_val)

    print("\n=====test======")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
