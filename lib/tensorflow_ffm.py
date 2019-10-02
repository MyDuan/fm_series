import matplotlib.pyplot as plt
import tensorflow as tf

loss_list = []

# Treat each feature as a dummy field
# feature example: 1:1:5.1, 2:2:3.5, 3:3:1.4, 4:4:0.2


def model_1(x, n, fields, k):
    w_0 = tf.Variable([0.0], trainable=True, name='w_0')
    w = tf.Variable(tf.truncated_normal([n, 1], stddev=0.1), name='w')
    v = tf.Variable(tf.truncated_normal([n, fields, k], stddev=0.1), name='v')
    interaction = tf.constant(0, dtype='float32')
    for i in range(n):
        for j in range(i + 1, n):
            vifj = v[i, j]
            vjfi = v[j, i]
            vivj = tf.reduce_sum(tf.multiply(vifj, vjfi))
            xixj = tf.multiply(x[:, i], x[:, j])
            interaction += tf.multiply(vivj, xixj)
    output = w_0 + tf.matmul(x, w) + interaction
    return output


# Discretize each numerical feature to a categorical one.
# feature example: 1:5.1:1, 2:3.5:1, 3:1.4:1, 4:0.2:1


def model_2(x, n, fields, k):
    w_0 = tf.Variable([0.0], trainable=True, name='w_0')
    w = tf.Variable(tf.truncated_normal([n, 1], stddev=0.1), name='w')
    v = tf.Variable(tf.truncated_normal([n, fields, k], stddev=0.1), name='v')
    interaction = tf.constant(0, dtype='float32')
    for i in range(n):
        fi = 0
        if x[:, i] != 0:
            for j in range(i + 1, n):
                fj = fi + 1
                if x[:, j] != 0:
                    vifj = v[i, fj]
                    vjfi = v[j, fi]
                    vivj = tf.reduce_sum(tf.multiply(vifj, vjfi))
                    xixj = tf.multiply(x[:, i], x[:, j])
                    interaction += tf.multiply(vivj, xixj)
                    fj += 1
            fi += 1
    output = w_0 + tf.matmul(x, w) + interaction
    return output


def loss_func(output, y, train_type):
    if train_type == 'regr':
        loss = tf.reduce_mean(tf.square(output - y))
    elif train_type == '2_class':
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(tf.multiply(output, y))))
    return loss


def training(loss, learning_rate):
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train


def run(model, x, y, learning_rate, epoch, n, f, k, train_type,
        x_train_data, y_train_data):
    feed_dict = {x: x_train_data, y: y_train_data}
    with tf.Session() as sess:
        if model == 'model_1':
            output = model_1(x, n, f, k)
        elif model == 'model_2':
            output = model_2(x, n, f, k)
        loss = loss_func(output, y, train_type)
        train = training(loss, learning_rate)

        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(epoch):
            sess.run(train, feed_dict=feed_dict)
            loss_results = sess.run(loss, feed_dict=feed_dict)
            loss_list.append(loss_results)
            if i % (epoch / 10) == 0:
                print('step[{}]  loss : {}'.format(i, loss_results))
        y_results = sess.run(output, feed_dict=feed_dict)
        if train_type == 'regr':
            plot_graph(x_train_data, y_train_data, y_results)
        elif train_type == '2_class':
            acc = 0
            for i in range(len(y_results)):
                if y_results[i][0] > 0:
                    y_test = 1
                else:
                    y_test = -1
                if y_train_data[i] == y_test:
                    acc += 1
            print("accuracy:", acc / 100.0)
            plt.plot(loss_list)
            plt.show()


def plot_graph(x_train_data, y_train_data, y_results):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 4))
    ax1.plot(x_train_data, y_results, color='red', )
    ax1.plot(x_train_data, y_train_data, color='blue')
    ax2.set_ylabel('loss')
    ax2.plot(loss_list)
    plt.show()