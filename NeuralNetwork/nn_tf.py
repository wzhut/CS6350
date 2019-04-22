# encoder and decoder for logistic function gradient
import tensorflow as tf
import numpy as np
import math

def main():
    # encoder and decoder configuration
    dim = 5  # gradient dimension
    npt = 10 # number of points in each worker node
    nw = 5 # number of worker
    nr = 1 # redundancy, number of encoder outputs
    nu = 1 # number of unreliable sources
    hn = 500 
   # enc_layer_size = [dim * nw * npt, hn, hn, hn, dim * npt] # encoder layer sizes
    # 4 hidden layer
    enc_layer_size = [dim * nw * npt, hn, hn, hn, hn, hn,hn, hn, hn, hn, dim * npt] # encoder layer sizes
#    enc_layer_act = [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None]
    enc_layer_act = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu,tf.nn.relu, tf.nn.relu,tf.nn.relu, None]
   # dec_layer_size = [dim * (nw + nr), hn, hn, hn, nu * dim] # decoder layer sizes
    dec_layer_size = [dim * (nw + nr), hn, hn, hn, hn, hn, hn, hn, hn, hn, nu * dim]
 #   dec_layer_act = [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None]
    dec_layer_act = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu,tf.nn.relu, tf.nn.relu, None]
    ## encoder
    # encoder input
    x = tf.placeholder(tf.float32, [None, enc_layer_size[0]])
    
    n_hidden_layer = len(enc_layer_size) - 1
    enc_param = [dict() for i in range(n_hidden_layer)]
    # encoder network
    for i in range(n_hidden_layer):
        # weight
        if enc_layer_act[i] == tf.nn.tanh :
            enc_param[i]['w'] = tf.Variable(tf.random_normal([enc_layer_size[i], enc_layer_size[i+1]], stddev=math.sqrt(2/(enc_layer_size[i] + enc_layer_size[i+1]))))
        else:
            enc_param[i]['w'] = tf.Variable(tf.random_normal([enc_layer_size[i], enc_layer_size[i+1]], stddev=math.sqrt(2/enc_layer_size[i])))
        # bias
        enc_param[i]['b'] = tf.Variable(tf.zeros(enc_layer_size[i+1]))

        if i == 0:
            flow_in = x
        else:
            flow_in = enc_param[i-1]['y']
        if enc_layer_act[i] == None:
            enc_param[i]['y'] = tf.matmul(flow_in, enc_param[i]['w']) + enc_param[i]['b']
        else:
            enc_param[i]['y'] = enc_layer_act[i](tf.matmul(flow_in, enc_param[i]['w']) + enc_param[i]['b'])
    ## decoder
    # decoder input
    # graidient
    g = tf.placeholder(tf.float32, [None, dim * nw])
    # corresponding weight
    w = tf.placeholder(tf.float32, [None, dim])
    # mising weight
    m = tf.placeholder(tf.float32, [None, dim * nu])
    # calculate gradient of info summary
    x_coded = tf.reshape(enc_param[n_hidden_layer-1]['y'],[-1, dim])
    w_reshaped = tf.reshape(tf.tile(w, [1, npt]), [-1, dim])
    wx = tf.reduce_sum(tf.multiply(x_coded, -w_reshaped), 1)
    f = tf.reshape(tf.divide(1 , 1 +  tf.exp(wx)),[-1,1])
    new_g = tf.multiply(tf.tile(tf.multiply(f, 1 - f), [1,dim]), x_coded)
    new_g = tf.reduce_sum(tf.reshape(new_g, [npt, -1]),0)
    new_g = tf.reshape(new_g,[-1, dim])

    # missing mask
    mask = tf.placeholder(tf.float32, [None, dim * nw]) # unreliable pattern
    # incomplete gradients + summary gradient
    g_dec = tf.concat([tf.multiply(g, mask), new_g], 1)
   
    #
    n_hidden_layer = len(dec_layer_size) - 1
    dec_param = [dict() for i in range(n_hidden_layer)]
    # encoder network
    for i in range(n_hidden_layer):
        # weight
        if dec_layer_act[i] == tf.nn.tanh :
            dec_param[i]['w'] = tf.Variable(tf.random_normal([dec_layer_size[i], dec_layer_size[i+1]], stddev=math.sqrt(2/(dec_layer_size[i] + dec_layer_size[i+1]))))
        else:
            dec_param[i]['w'] = tf.Variable(tf.random_normal([dec_layer_size[i], dec_layer_size[i+1]], stddev=math.sqrt(2/dec_layer_size[i])))
        # bias
        dec_param[i]['b'] = tf.Variable(tf.zeros(dec_layer_size[i+1]))

        if i == 0:
            flow_in = g_dec
        else:
            flow_in = dec_param[i-1]['y']
        if dec_layer_act[i] == None:
            dec_param[i]['y'] = tf.matmul(flow_in, dec_param[i]['w']) + dec_param[i]['b']
        else:
            dec_param[i]['y'] = dec_layer_act[i](tf.matmul(flow_in, dec_param[i]['w']) + dec_param[i]['b'])

    # loss
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(((m - dec_param[n_hidden_layer - 1]['y']))**2, axis = 1)))
    # optimizer
    lr = 0.0001   # learning rate
    opt = tf.train.AdamOptimizer(lr)
    train = opt.minimize(loss)

    # init
    init_op = tf.global_variables_initializer()

    total_step =  100000
    with tf.Session() as sess:
        sess.run(init_op)
        ## load data
        # train
        train_w = np.genfromtxt('./train_w.csv', delimiter=',')
        train_x = np.genfromtxt('./train_x.csv', delimiter=',')
        train_g = np.genfromtxt('./train_g.csv', delimiter=',')
        train_m = np.genfromtxt('./train_m.csv', delimiter=',')
        train_mask = np.genfromtxt('./train_mask.csv', delimiter=',')
        # test
        test_w = np.genfromtxt('./test_w.csv', delimiter=',')
        test_x = np.genfromtxt('./test_x.csv', delimiter=',')
        test_g = np.genfromtxt('./test_g.csv', delimiter=',')
        test_m = np.genfromtxt('./test_m.csv', delimiter=',')
        test_mask = np.genfromtxt('./test_mask.csv', delimiter=',')

        feed_dict = {x: train_x, mask: train_mask, w: train_w, g: train_g, m: train_m}
        for i in range(total_step):
            _,train_err =  sess.run([train, loss], feed_dict=feed_dict)
            if i % 100 == 0:
                # test
                feed_dict = {x: test_x, mask: test_mask, w: test_w, g: test_g, m:test_m}
                test_err = sess.run(loss, feed_dict = feed_dict)
                print('train_error:', train_err, ' test_error:', test_err)


if __name__ == '__main__':
    main()
