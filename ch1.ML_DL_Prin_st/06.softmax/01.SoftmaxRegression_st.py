import tensorflow as tf
import numpy as np

def softmax_regression_1():
    # y = x
    x = [[1., 2.],  # C
         [2., 1.],
         [4., 5.],  # B
         [5., 4.],
         [8., 9.],  # A
         [9., 8.]]
    y = [[0., 0., 1.],
         [0., 0., 1.],
         [0., 1., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [1., 0., 0.]]

    w = tf.Variable(tf.random_normal([?, ?]))
    b = tf.Variable(tf.random_normal([?]))

    # (6, 3) = (6, 2) @ (?, ?)
	# z = tf.matmul(x, w) + b
    z = x @ w + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizeer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizeer.minimize(loss)

    sess = tf.Session()
    sess.run((tf.global_variables_initializer()))

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    print(preds)
    print(np.argmax(preds, axis=1))
    sess.close()


softmax_regression_1()
    
#%%
# x --> ph_x,  평균 예측율 출력, 예측 학점 출력
# 3시간 공부하고 7번 출석한 학생과, 6시간 공부하고 2번 출석한 학생의 학점??
preds = sess.run(z, {ph_x: [[3., 7.], [6., 2.]]})

