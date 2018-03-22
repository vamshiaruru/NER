from __future__ import division
from sklearn.metrics import precision_recall_fscore_support as score
import tensorflow as tf
import numpy as np
import time

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
window_size = 3
tag_num_dict = {"O": 0, "PERSON": 1, "MISC": 2, "ORG": 3, "LOC": 4}


def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def forward_prop(X, w_1, w_2):
    h = tf.nn.relu(tf.matmul(X, w_1))
    y_hat = tf.matmul(h, w_2)
    return y_hat


def generate_lines(fp):
    docs = []
    cur_line = []
    for line in fp:
        if line == "\n":
            docs.append(cur_line)
            cur_line = []
        else:
            words = line.strip().split(" ")
            cur_line.append(words)
    if not len(cur_line) == 0:
        docs.append(cur_line)
    return docs


def generate_window(words, index):
    window = []
    for i in range(index-int(window_size/2), index+int(window_size/2)+1):
        window.append(words[i])
    return window


def full_o(tags):
    for tag in tags:
        if not tag == "O":
            return False
    return True


def generate_sequences(lines, vectors):
    x = []
    y = []
    j = 0
    k = len(lines)
    for line in lines:
        print "{}/{}\r".format(j,k),
        j += 1
        for i in xrange(int(window_size/2)):
            line.insert(0, ["<s>", "O"])
            line.append(["</s>", "O"])
        words, tags = zip(*line)
        # if full_o(tags):
        #     continue
        for i in xrange(len(words)):
            if words[i] == "<s>" or words[i] == "</s>":
                continue
            window = generate_window(words, i)
            window = [vectors[word] for word in window]
            window = [vector for sublist in window for vector in sublist]
            x.append(np.array(window))
            output = [0, 0, 0, 0, 0]
            output[tag_num_dict[tags[i]]] = 1
            y.append(np.array(output))
    x = np.array(x)
    y = np.array(y)
    print "reached here"
    return x, y


def load_data(trainfile="train.txt", testfile="test.txt"):
    vocab = open("fasttext_words.txt")
    vocab_vectors = open("fasttext_vectors.txt")
    train = open(trainfile)
    test = open(testfile)
    temp_word_dict = dict()
    temp_vec_dict = dict()
    i = 0
    print "building dictionary for word vectors...."
    for line in vocab:
        temp_word_dict[i] = line.strip()
        i += 1
    i = 0
    for line in vocab_vectors:
        vector = [float(word) for word in line.strip().split(" ")]
        vector.insert(0, float(1))
        temp_vec_dict[i] = vector
        i += 1
    word_vec_dict = dict()
    vec_word_dict = dict()
    for key in temp_word_dict.keys():
        word_vec_dict[temp_word_dict[key]] = temp_vec_dict[key]
        vec_word_dict[tuple(temp_vec_dict[key])] = temp_word_dict[key]
    print "generating sentences....."
    train_lines = generate_lines(train)
    test_lines = generate_lines(test)
    print "generating sequences of window size {}......".format(window_size)
    train_x, train_y = generate_sequences(train_lines, word_vec_dict)
    print "generating sequences out of test data....."
    test_x, test_y = generate_sequences(test_lines, word_vec_dict)
    train.close()
    test.close()
    vocab.close()
    vocab_vectors.close()
    return train_x, train_y, test_x, test_y, word_vec_dict, vec_word_dict


def main():
    print "loading data...."
    report = open("fasttext_results.txt", "w")
    train_x, train_y, test_x, test_y, word_vec_dict, vec_word_dict = load_data()
    true_y = []
    for i in range(len(test_y)):
        true_y.append(np.argmax(test_y[i:i + 1], axis=1))
    true_y = [y[0] for y in true_y]
    print "data loaded, initializing tensorflow session..."
    x_size = train_x.shape[1]
    h_size = 100
    # check with different hidden layer sizes
    y_size = train_y.shape[1]
    X = tf.placeholder("float", shape=[None, x_size])
    Y = tf.placeholder("float", shape=[None, y_size])
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))
    y_hat = forward_prop(X, w_1, w_2)
    predict = tf.argmax(y_hat, axis=1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                                  logits=y_hat))
    cost += 0.001 * (tf.nn.l2_loss(w_1) + tf.nn.l2_loss(w_2))
    updates = tf.train.AdamOptimizer(0.001).minimize(cost)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print "session initialized, commencing optimization...."
    for epoch in range(30):
        cost_val = 9999
        start_time = time.time()
        print "epoch:\t {}".format(epoch)
        for i in range(len(train_x)):
            print "\r{0}".format(i),
            _, cost_val = sess.run([updates, cost],
                                   feed_dict={X: train_x[i:i+1],
                                              Y: train_y[i:i+1]})
        end_time = time.time()
        r_str = "finished the current epoch, total time taken:{}, " \
                "cost:{}".format(end_time - start_time, str(cost_val))
        print r_str
        report.write(r_str + "\n")
        predicted = sess.run(predict, feed_dict={X: test_x, Y: test_y})
        precision, recall, fscore, support = score(true_y, predicted)
        print precision, recall, fscore, support
        report.write(str(precision) + "\n")
        report.write(str(recall) + "\n")
        report.write(str(fscore) + "\n")
        report.write("***************************" + "\n")

    sess.close()

if __name__ == "__main__":
    main()
