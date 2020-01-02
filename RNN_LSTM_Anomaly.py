from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from malfunction.CANPacket import CANPacket_t, CANPacket_q
from malfunction.Parser import *
import csv

__all__ = ['Parser', 'CANPacket', 'rnn.py']

def lstm_learning(train_payload, train_flag, test_payload, test_flag):

    # fix random seed for reproducibility
    np.random.seed(7)
    max_payload = 20
    max_size = len(train_flag)

    X_train = []
    Y_train = []
    for i in range(0,len(train_payload)):
        X_train.append(train_payload[i])
        Y_train.append(train_flag[i])

    X_test = []
    Y_test = []
    for i in range(0,len(test_payload)):
        X_test.append(test_payload[i])
        Y_test.append(test_flag[i])

    tok = Tokenizer(num_words=max_size)
    tok.fit_on_texts(X_train)
    X_train = tok.texts_to_sequences(X_train)
    X_train = sequence.pad_sequences(X_train, maxlen=max_payload)

    tok.fit_on_texts(X_test)
    X_test = tok.texts_to_sequences(X_test)
    X_test = sequence.pad_sequences(X_test, maxlen=max_payload)

    # create the model
    embedding_vecor_length = 3
    model = Sequential()
    model.add(Embedding(input_dim=max_size, output_dim=embedding_vecor_length, input_length=max_payload))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Y_train, epochs=2, batch_size=32, validation_data=(X_test, Y_test))

    # Final evaluation of the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict_classes(X_test, verbose = 0)
    return predictions

if __name__ == '__main__':

    result = open('result.csv', 'r+',newline='')
    csvwriter = csv.writer(result)

    trainset = parse('trainfile')

    train_cid = []
    for p in trainset:
        train_cid.append(str(p.cid))
    train_data = []
    for p in trainset:
        pdata = ''.join(p.data_field)
        train_data.append(pdata)
    train_flag = []
    for p in trainset:
        if p.ccls == 'R':
            train_flag.append(1)
        if p.ccls == 'T':
            train_flag.append(0)
    train_payload = []
    for i in range (0,len(train_cid)):
        train_payload.append(train_cid[i]+train_data[i])

    testset = parse('testfile')

    test_cid = []
    for q in testset:
        test_cid.append(str(q.cid))
    test_data = []
    for q in testset:
        pdata = ''.join(q.data_field)
        test_data.append(pdata)
    test_flag = []
    for i in range(0, len(test_data)):
        test_flag.append(1)
    test_payload = []
    for i in range(0, len(test_cid)):
        test_payload.append(test_cid[i] + test_data[i])
    prior_len = len(test_flag)

    #if train set and test set have different size, repeat the last data of the smaller one until the bigger one
    if (len(train_flag) < len(test_flag)):
        for i in range(0, len(test_flag) - len(train_flag)):
            train_payload.append(train_payload[len(train_payload) - 1])
            train_flag.append(train_flag[len(train_flag) - 1])
    elif (len(train_flag) > len(test_flag)):
        for i in range(0, len(train_flag) - len(test_flag)):
            test_payload.append(test_payload[len(test_payload) - 1])
            test_flag.append(test_flag[len(test_flag) - 1])

    temp = []
    ydata = lstm_learning(train_payload,train_flag,test_payload,test_flag)

    for d in ydata:
        temp.append(d)
        print(d)
    temp = temp[0:prior_len]

    print(temp)

    for i in range(0,len(temp)):
        if temp[i] == 1:
            csvwriter.writerow([i+1, 'R'])
        elif temp[i] == 0:
            csvwriter.writerow([i+1, 'T'])
