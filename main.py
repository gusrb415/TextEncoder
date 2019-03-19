import argparse
import json
import os
import time

from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, Dropout, TimeDistributed, LSTM
from keras.optimizers import Adam
import numpy as np
from data_helper import load_data, build_input_data
from scorer import scoring
from utils import TestCallback, make_submission


def build_model(embedding_dim, hidden_size, drop1, drop2, sequence_length, vocabulary_size):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    drop_embed = Dropout(drop1)(embedding)

    lstm_out = LSTM(units=hidden_size, dropout=drop1, recurrent_dropout=drop2, return_sequences=True,
                    activation='selu')(drop_embed)

    outputs = TimeDistributed(Dense(units=vocabulary_size, activation='softmax'))(lstm_out)

    model = Model(inputs=inputs, outputs=outputs)
    adam = Adam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

    print(model.summary())
    return model


def predict_final_word(model_list, vocabulary, filename):
    list_prob = []
    id_list = []
    for model in model_list:
        id_list = []
        prev_tokens_list = []
        prev_tokens_lens = []
        with open(filename, "r") as fin:
            fin.readline()
            for line in fin:
                id_, prev_sent, grt_last_token = line.strip().split(",")
                id_list.append(id_)
                prev_tokens = prev_sent.split()
                prev_tokens_list.append(prev_tokens)
                prev_tokens_lens.append(len(prev_tokens))
        X = np.array([build_input_data(t, vocabulary)[0][0].tolist() for t in prev_tokens_list])
        y_prob = (model.predict(X, batch_size=32))
        last_token_probs = np.array([y_prob[b, prev_tokens_lens[b] - 1, :] for b in range(y_prob.shape[0])])
        list_prob.append(last_token_probs)

    list_sum = 0
    for a in list_prob:
        list_sum = list_sum + a
    last_token_probs = list_sum / (len(list_prob))
    return dict(zip(id_list, last_token_probs))


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    if opt.mode == "train":
        st = time.time()
        print('Loading data')
        x_train, y_train, x_valid, y_valid, vocabulary_size = load_data("data", opt.debug)

        num_training_data = x_train.shape[0]
        sequence_length = x_train.shape[1]
        print(num_training_data)

        print('Vocab Size', vocabulary_size)

        model = build_model(opt.embedding_dim, opt.hidden_size, opt.drop1, opt.drop2, sequence_length, vocabulary_size)
        print("Training Model...")
        model.fit(x_train, y_train, batch_size=opt.batch_size, epochs=opt.epochs, verbose=2,
                  callbacks=[TestCallback((x_valid, y_valid), model=model)])
        model.save(opt.saved_model)
        print("Training cost time: ", time.time() - st)
    else:
        if opt.mode == "score_valid":
            model = load_model(opt.saved_model)
            vocabulary = json.load(open(os.path.join("data", "vocab.json")))
            predict_dict = predict_final_word([model], vocabulary, opt.input)
            sub_file = make_submission(predict_dict, opt.student_id, opt.input)
            scoring(sub_file, os.path.join("data"), type="valid")
        else:
            model0 = load_model('models/model0.h5')
            model1 = load_model('models/model1.h5')
            model2 = load_model('models/model2.h5')
            model3 = load_model('models/model3.h5')
            model4 = load_model('models/model4.h5')
            model5 = load_model('models/model5.h5')
            model6 = load_model('models/model6.h5')
            model7 = load_model('models/model7.h5')
            model8 = load_model('models/model8.h5')
            model9 = load_model('models/model9.h5')
            model_list = [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9]
            vocabulary = json.load(open(os.path.join("data", "vocab.json")))
            predict_dict = predict_final_word(model_list, vocabulary, opt.input)
            sub_file = make_submission(predict_dict, opt.student_id, opt.input)
            scoring(sub_file, os.path.join("data"), type="valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default="train", choices=["train", "test", "score_valid"],
                        help="Train or test mode")
    parser.add_argument("-saved_model", type=str, default="model.h5",
                        help="saved model path")
    parser.add_argument("-input", type=str, default=os.path.join("data", "valid.csv"),
                        help="Input path for generating submission")
    parser.add_argument("-debug", action="store_true",
                        help="Use validation data as training data if it is true")
    parser.add_argument("-score", action="store_true",
                        help="Report score if it is")
    parser.add_argument("-student_id", default=None, required=True,
                        help="Student id number is compulsory!")
    parser.add_argument("-epochs", type=int, default=1,
                        help="training epoch num")
    parser.add_argument("-batch_size", type=int, default=32,
                        help="training batch size")
    parser.add_argument("-embedding_dim", type=int, default=100,
                        help="word embedding dimension")
    parser.add_argument("-hidden_size", type=int, default=500,
                        help="rnn hidden size")
    parser.add_argument("-drop1", type=float, default=0.5,
                        help="dropout1")
    parser.add_argument("-drop2", type=float, default=0.5,
                        help="dropout2")
    parser.add_argument("-gpu", type=str, default="",
                        help="dropout")
    opt = parser.parse_args()
    main(opt)
