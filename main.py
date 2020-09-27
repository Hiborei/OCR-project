import numpy as np, cv2
# import tensorflow
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from PIL import Image, ImageChops
from sklearn import tree

letters_big = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
letters_small = ['a_small', 'b_small', 'c_small', 'd_small', 'e_small', 'f_small', 'g_small', 'h_small', 'i_small', 'j_small', 'k_small', 'l_small', 'm_small', 'n_small', 'o_small', 'p_small', 'q_small', 'r_small', 's_small', 't_small', 'u_small', 'v_small', 'w_small', 'x_small', 'y_small', 'z_small']
numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
symbols = letters_big+letters_small+numbers
for_labels = letters_big+letters_big+numbers

def trim(im):
    im = Image.fromarray(im)
    bg = Image.new(im.mode, im.size, (255,255,255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        img = np.array(im.crop(bbox))
        shape = img.shape
        if abs(shape[0]-shape[1]) > 2:
            if shape[0] > shape[1]:
                diff = shape[0]-shape[1]
                border = np.full((shape[0], int(diff/2), 3), 255, dtype='uint8')
                img = np.hstack((img, border))
                img = np.hstack((border, img))
            else:
                diff = shape[1]-shape[0]
                border = np.full((int(diff/2), shape[1], 3), 255, dtype='uint8')
                img = np.vstack((img, border))
                img = np.vstack((border, img))
        return img


def labeling(how_many):
    labels = []

    for a in for_labels:
        for x in range(0, how_many):
            labels.append(a)
    return labels


def labeling_num(how_many):
    labels = []

    for a in range(0, len(for_labels)):
        for x in range(0, how_many):
            labels.append(a)
    return labels

def image_loader_vec(start, how_many):
    database = []

    for s in symbols:
        for x in range(start, start+how_many):
            num = str(x)
            num = num.zfill(4)
            img = cv2.imread('E:/class/'+s + '/hsf_0/hsf_0_0'+num+'.png')
            img = trim(img)
            ret, processed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(processed, (16, 16), interpolation=cv2.INTER_AREA)
            final = np.resize(resized, (16*16))
            database.append(final)

    return database

    # print(len(database))
    # for i in database:
    #     k = '0'
    #     while not k == ord('n'):
    #         cv2.imshow('img', i)
    #         k = cv2.waitKey()


def image_loader_one_hot(start, how_many):
    database = []

    for s in symbols:
        for x in range(start, start + how_many):
            num = str(x)
            num = num.zfill(4)
            img = cv2.imread('E:/class/' + s + '/hsf_0/hsf_0_0' + num + '.png')
            img = trim(img)
            img = np.array(img)
            ret, processed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(processed, (16, 16), interpolation=cv2.INTER_AREA)
            database.append(resized)
    database = to_categorical(database)
    return database


def MNB(data_train_X, data_train_Y):
    print('Naive Bayes:')
    NB = MultinomialNB()
    NB.fit(data_train_X, data_train_Y)

    test_x = image_loader_vec(200, 10)
    test_y = labeling(10)
    results = NB.predict(test_x)
    score = accuracy_score(results, test_y)
    print(score)

    for_show = image_loader_vec(250, 1)
    results = NB.predict(for_show)


    #
    # for r in range(0,len(symbols)):
    #     k = '0'
    #     while not k == ord('n'):
    #          cv2.imshow(str(results[r]), cv2.resize(for_show[r].reshape((16, 16)),(200,200)))
    #          k = cv2.waitKey()

def ComNB(data_train_X, data_train_Y):
    print('Complement Naive Bayes:')
    CNB = ComplementNB()
    CNB.fit(data_train_X, data_train_Y)

    test_x = image_loader_vec(200, 10)
    test_y = labeling(10)
    results = CNB.predict(test_x)
    score = accuracy_score(results, test_y)
    print(score)

    for_show = image_loader_vec(250, 1)
    results = CNB.predict(for_show)



    for r in range(0,len(symbols)):
        k = '0'
        while not k == ord('n'):
             cv2.imshow(str(results[r]), cv2.resize(for_show[r].reshape((16, 16)),(200,200)))
             k = cv2.waitKey()



def CNN(data_train_X, data_train_Y):
    test_x = image_loader_one_hot(200, 10)
    test_y = labeling(10)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape = (16, 16, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(data_train_X, data_train_Y, validation_data=(test_x, test_y), epochs=3)

    model.predict(test_x[:4])


def SupportVector(data_train_X, data_train_Y):

    print('Support Vector:')
    svc = SVC(kernel='poly', C=1, gamma=0.001, random_state=1)
    svc.fit(data_train_X, data_train_Y)
    test_x = image_loader_vec(200, 10)
    test_y = labeling(10)

    results = svc.predict(test_x)
    score = accuracy_score(test_y, results)
    print(score)

    test_x = image_loader_vec(200, 1)

    results = svc.predict(test_x)

    for r in range(0, len(symbols)):
        k = '0'
        while not k == ord('n'):
            cv2.imshow(str(results[r]), cv2.resize(test_x[r].reshape((16, 16)), (200, 200)))
            k = cv2.waitKey()


def Trees(data_x, data_y):
    print('Decision Trees:')
    tr = tree.DecisionTreeClassifier()
    tr = tr.fit(data_x, data_y)

    test_x = image_loader_vec(200, 10)
    test_y = labeling(10)
    results = tr.predict(test_x)
    score = accuracy_score(test_y, results)
    print(score)

    # for r in range(0, len(symbols)):
    #     k = '0'
    #     while not k == ord('n'):
    #         cv2.imshow(str(results[r]), cv2.resize(test_x[r].reshape((16, 16)), (200, 200)))
    #         k = cv2.waitKey()

def neural(data_x, data_y):
    print('Neural Regressor:')
    regr = MLPClassifier(random_state=1, max_iter=500)
    regr = regr.fit(data_x, data_y)

    test_x = image_loader_vec(200, 10)
    test_y = labeling_num(10)
    results = regr.predict(test_x)
    score = accuracy_score(test_y, results)
    print(score)

data_x = image_loader_vec(0,200)
data_y = labeling_num(200)
# ComNB(data_x, data_y)
# MNB(data_x, data_y)
# SupportVector(data_x, data_y)
# Trees(data_x, data_y)
neural(data_x, data_y)

