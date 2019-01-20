import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

#funkcja pobierajaca sciezke do zdjecia ktora jest zapisana w pliku tekstowym.
def get_img():
    file=open("zdjecie.txt",'r')
    img_path=file.readlines();
    file.close();
    return img_path


def reset_graph(seed=42):    #funkcja sluzaca do resetowania struktury grafu modulu tensorflow
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#funckja ktora odtwarza model sieci i wczytuje wytrenowane parametry
#nastepnie podaje na wejscie zdjecie wybrane w interfejsie matlaba, do ktorego sciezke
#odczytala funkcja get_img
def predict(img_path=get_img()[0]):
    img_height=150
    img_width=150
    pre_img=cv2.imread(img_path,cv2.IMREAD_COLOR)
    img=cv2.resize(pre_img,( img_height,img_width))
    #pool_dropout_rate = 0.25
    fc_dropout_rate = 0.5

    reset_graph()  # zresetowanie grafu

#architektura sieci została juz opisana w pliku z google collab
    with tf.name_scope("wejscia"):
        X = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 3], name="X")
        y = tf.placeholder(tf.float32, shape=[None, 2], name="y")
        training = tf.placeholder_with_default(False, shape=[], name='uczenie')
    with tf.name_scope("splotowe"):
        conv1 = tf.layers.conv2d(X, filters=64, kernel_size=3,
                                 strides=1, padding="SAME",
                                 activation=tf.nn.relu, name="splot1")
        #print(conv1.shape)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2], padding="VALID")
        #print(pool1.shape)
        conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=3,
                                 strides=1, padding="SAME",
                                 activation=tf.nn.relu, name="splot2")
        #print(conv2.shape)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2], padding="VALID")
        #print(pool2.shape)
        conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=3,
                                 strides=1, padding="SAME",
                                 activation=tf.nn.relu, name="splot3")
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=[2, 2], padding="VALID")
        #print(conv3.shape)
        #print(pool3.shape)
        conv4 = tf.layers.conv2d(pool3, filters=64, kernel_size=3,
                                 strides=1, padding="SAME",
                                 activation=tf.nn.relu, name="splot4")
        #print(conv4.shape)
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=[2, 2], padding="VALID")
        #print(pool4.shape)

    with tf.name_scope("pp"):
        flatten1 = tf.layers.flatten(pool4)
        #print(flatten1.shape)
        fc1 = tf.layers.dense(flatten1, 64, activation=tf.nn.relu, name="pp1")
        #print(fc1.shape)
        fc1_drop = tf.layers.dropout(fc1, fc_dropout_rate, training=training)
        #print(fc1_drop.shape)
    with tf.name_scope("wyjscie"):
        logits = tf.layers.dense(fc1_drop, 2, name="wyjscie")
        #print(logits.shape)
        Y_proba = tf.nn.softmax(logits, name="Y_prawd")

    with tf.name_scope("uczenie"):
        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y) #w przypadku zadania klasyfikacji cross_entropy jest duzo lepsza metoda optymalizacj niz RMS
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)

    with tf.name_scope("ocena"):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #ilosc poprawnie zakwalifikowanych przez wszystkie

    saver=tf.train.Saver()
    prep_img=img.reshape(-1,img_height,img_width,3) #przygotowanie zdjecia do podania na wejscie sieci

    # b, g, r = cv2.split(pre_img)
    # img2 = cv2.merge([r, g, b])
    # plt.imshow(img2)
    # plt.title("Oryginalne zdjecie")
    # plt.show()
    with tf.Session() as sess:
        saver.restore(sess, "./mojModelProjekt")
        prediction = sess.run([Y_proba], feed_dict={X: prep_img})
        #print(prediction[0][0]) #prawdopodobienstwa przynaleznosci do danej klasy
        #print((prediction[0][0][0]+prediction[0][0][1])) #suma wynosi 1 (softmax )
        if prediction[0][0][0] > prediction[0][0][1]:
            return "CAT"
        else:
            return "DOG"



#otwarcie pliku tekstowego i zapisanie zwroconego wyniku.
f= open("wyniki.txt","w+")
f.write(predict())
f.close()