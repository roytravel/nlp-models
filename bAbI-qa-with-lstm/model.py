# -*- coding:utf-8 -*-
"""Reference: https://wikidocs.net/82475 (딥러닝을 이용한 자연어 처리 입문) """
import os
import re
import tarfile
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nltk import FreqDist
from functools import reduce
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Permute, dot, add, concatenate
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Activation
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)


def check_execution(func):
    def wrapper(*args, **kwargs):
        #print (f'[*] TRACE: calling {func.__name__}')
        result = func(*args, **kwargs)
        print (f'[*] TRACE: {func.__name__}() is called.')
        return result
    return wrapper


@check_execution
def download_dataset():
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/' 'babi_tasks_1-20_v1-2.tar.gz')
    with tarfile.open(path) as tar:
        tar.extractall()


@check_execution
def read_data(fpath):
    stories, questions, answers = [], [], [] # story, question, answer 저장
    story_temp = [] # 현 시점의 story 임시 저장
    with open(fpath, mode ="rb") as lines:
        for line in lines:
            line = line.decode("utf-8") # b' 제거
            line = line.strip() # '\n' 제거
            idx, text = line.split(" ", 1) # 맨 앞에 있는 숫자 분리

            if int(idx) == 1:
                story_temp = []

            if "\t" in text: # 현재 읽는 줄이 Q/A 인 경우
                question, answer, _ = text.split("\t") # Q/A를 각각 저장
                stories.append([x for x in story_temp if x]) # 지금까지의 누적 story를 story에 저장
                questions.append(question)
                answers.append(answer)

            else: # 현재 읽는 줄이 story인 경우
                story_temp.append(text) # 임시 저장

    return stories, questions, answers


def tokenize(sent):
    return [ x.strip() for x in re.split('(\W+)', sent) if x and x.strip()]


@check_execution
def preprocess(train_data, test_data):
    counter = FreqDist() # 토큰의 사용빈도를 담는 클래스 객체 생성

    # 두 문장의 story를 하나의 문장으로 통합하는 함수
    flatten = lambda data: reduce(lambda x, y: x + y, data)

    # 각 샘플의 길이를 저장하는 리스트
    story_len = []
    question_len = []

    for stories, questions, answers in [train_data, test_data]:
        for story in stories:
            stories = tokenize(flatten(story)) # story의 문장들을 펼친 후 토큰화
            story_len.append(len(stories)) # 각 story의 길이 저장
            for word in stories: # 단어 집합에 단어 추가
                counter[word] += 1

        for question in questions:
            question = tokenize(question)
            question_len.append(len(question))
            for word in question:
                counter[word] += 1

        for answer in answers:
            answer = tokenize(answer)
            for word in answer:
                counter[word] += 1

    # 단어 집합 생성
    word2idx = {word : (idx + 1) for idx, (word, _) in enumerate(counter.most_common())}
    idx2word = {idx : word for word, idx in word2idx.items()}

    story_max_len = np.max(story_len) # 가장 긴 story의 길이
    question_max_len = np.max(question_len) # 가장 긴 question의 길이
    print(f'   [+] Max length of story: {story_max_len}')
    print(f'   [+] Max length of question: {question_max_len}')
    return word2idx, idx2word, story_max_len, question_max_len


@check_execution
def vectorize(data, word2idx, story_maxlen, question_maxlen):
    Xs, Xq, Y = [], [], []
    flatten = lambda data: reduce(lambda x, y: x + y, data)

    stories, questions, answers = data
    for story, question, answer in zip(stories, questions, answers):
        xs = [word2idx[w] for w in tokenize(flatten(story))]
        xq = [word2idx[w] for w in tokenize(question)]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2idx[answer])

        # story와 question은 각각의 최대 길이로 패딩. 정답은 원-핫 인코딩
    return pad_sequences(Xs, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen), to_categorical(Y, num_classes=len(word2idx) + 1)


@check_execution
def build_model(story_max_len, question_max_len):
    input_sequence = Input((story_max_len,)) # 입력을 담는 변수인 플레이스 홀더 객체 생성
    question = Input((question_max_len,)) # 입력을 담는 변수인 플레이스 홀더 객체 생성
    # print('Stories :', input_sequence)
    # print('Question:', question)

    # story를 위한 첫번째 임베딩. 그림에서의 Embedding A
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=args.embed_size))
    input_encoder_m.add(Dropout(args.dropout_rate))
    # 결과 : (samples, story_max_len, embedding_dim) / 샘플의 수, 문장의 최대 길이, 임베딩 벡터의 차원

    # story를 위한 두번째 임베딩. 그림에서의 Embedding C
    # 임베딩 벡터의 차원을 question_max_len(질문의 최대 길이)로 한다.
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=question_max_len))
    input_encoder_c.add(Dropout(args.dropout_rate))
    # 결과 : (samples, story_max_len, question_max_len) / 샘플의 수, 문장의 최대 길이, 질문의 최대 길이(임베딩 벡터의 차원)

    # 질문을 위한 임베딩. 그림에서의 Embedding B
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size, output_dim=args.embed_size, input_length=question_max_len))
    question_encoder.add(Dropout(args.dropout_rate))
    # 결과 : (samples, question_max_len, embedding_dim) / 샘플의 수, 질문의 최대 길이, 임베딩 벡터의 차원

    # 실질적인 임베딩 과정
    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

    print(f'   [+] Input encoded m : {input_encoded_m}')
    print(f'   [+] Input encoded c : {input_encoded_c}')
    print(f'   [+] Question encoded : {question_encoded}')

    # story 단어들과 질문 단어들 간의 유사도를 구하는 과정. 유사도는 내적을 사용.
    match = dot([input_encoded_m, question_encoded], axes=-1, normalize=False)
    match = Activation('softmax')(match)
    print(f'   [+] Match shape : {match}')
    # 결과 : (samples, story_maxlen, question_max_len) / 샘플의 수, 문장의 최대 길이, 질문의 최대 길이

    response = add([match, input_encoded_c])  # (samples, story_max_len, question_max_len)
    response = Permute((2, 1))(response)  # (samples, question_max_len, story_max_len)
    print(f'   [+] Response shape : {response}')

    # 질문 벡터와 답변 벡터를 연결
    answer = concatenate([response, question_encoded])
    print(f'  [*] Answer shape : {answer}')

    answer = LSTM(args.lstm_size)(answer)
    answer = Dropout(args.dropout_rate)(answer)
    answer = Dense(vocab_size)(answer)
    answer = Activation('softmax')(answer)

    model = Model([input_sequence, question], answer)
    return model


@check_execution
def plot(history):
    plt.subplot(2, 1, 1)
    plt.title("Accuracy")
    plt.plot(history.history["acc"], color="g", label="train")
    plt.plot(history.history["val_acc"], color="b", label="validation")
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="g", label="train")
    plt.plot(history.history["val_loss"], color="b", label="validation")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()


@check_execution
def inference(Xstest, Xqtest, Ytest):
    ytest = np.argmax(Ytest, axis=1)
    Ytest_ = model.predict([Xstest, Xqtest])
    ytest_ = np.argmax(Ytest_, axis=1)

    real = []
    for i in test_answers:
        real.append(word2idx[i])

    result = f1_score(real, ytest_, average='micro')
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--train_epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='number of batch size')
    parser.add_argument('--embed_size', type=int, default=96, help='size of embedding')
    parser.add_argument('--lstm_size', type=int, default=96, help='size of LSTM')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='rate of dropout')
    parser.add_argument('--plot', type=bool, default=True, help='"True" if you plot else "False"')
    args = parser.parse_args()

    download_dataset() # Babi 데이터셋 다운로드

    DATA_DIR = 'tasks_1-20_v1-2/en-10k' # qa 데이터셋 경로
    TRAIN_FILE = os.path.join(DATA_DIR, "qa1_single-supporting-fact_train.txt") # train 파일 경로
    TEST_FILE = os.path.join(DATA_DIR, "qa1_single-supporting-fact_test.txt") # test 파일 경로
    
    train_data = read_data(TRAIN_FILE) # story, question, answer 데이터 read
    test_data = read_data(TEST_FILE) # story, question, answer 데이터 read
    train_stories, train_questions, train_answers = read_data(TRAIN_FILE) # story, question, answer 데이터 read
    test_stories, test_questions, test_answers = read_data(TEST_FILE) # story, question, answer 데이터 read

    print (f"   [+] Train: stroy({len(train_stories)}), question({len(train_questions)}), answers({len(train_answers)})") # story, question, answer 길이 출력
    print (f"   [+] Test: stroy({len(test_stories)}), question({len(test_questions)}), answers({len(test_answers)})") # story, question, answer 길이 출력

    word2idx, idx2word, story_max_len, question_max_len = preprocess(train_data, test_data) # 전처리를 통해 word2idx, idx2word와 story, question에 대한 max length 값 get.
    vocab_size = len(word2idx) + 1 # 왜 + 1?
    
    Xstrain, Xqtrain, Ytrain = vectorize(train_data, word2idx, story_max_len, question_max_len) # 학습을 위한 벡터화
    Xstest, Xqtest, Ytest = vectorize(test_data, word2idx, story_max_len, question_max_len) # 학습을 위한 벡터화
    #print(Xstrain.shape, Xqtrain.shape, Ytrain.shape, Xstest.shape, Xqtest.shape, Ytest.shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=100)

    model = build_model(story_max_len, question_max_len)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    #print(model.summary())

    history = model.fit([Xstrain, Xqtrain], 
                        Ytrain, 
                        args.batch_size, 
                        args.train_epochs, 
                        validation_data=([Xstest, Xqtest], Ytest),
                        callbacks=[early_stop])

    model.save('model.h5')
    print("\n   [*] Eval Accuracy: %.4f" % (model.evaluate([Xstest, Xqtest], Ytest)[1]))

    if args.plot:
        plot(history)

    if 'model' not in globals():
        model = load_model('model.h5')
    
    result = inference(Xstest, Xqtest, Ytest)
    print (f'[*] Inference : {round(result, 2)}')