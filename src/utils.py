import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def split_data(data, target_col, test_size=0.2, random_state=42):
    num_test = int(len(data) * test_size)
    num_train = len(data) - num_test
    train, test = train_test_split(data, train_size=num_train, test_size=num_test, random_state=random_state)

    train_X = train.drop(columns=target_col)
    train_y = train[target_col]

    test_X = test.drop(columns=target_col)

    train_X.to_csv('artefacts/train_x.csv', index=False)
    train_y.to_csv('artefacts/train_y.csv', index=False)
    test_X.to_csv('artefacts/test_x.csv', index=False)


def train_log_reg(filename='model.pkl', penalty='l2', max_iter=1000, random_state=42):
    assert penalty in ['none', 'l2'], 'Wrong penalty'
    assert max_iter > 0, 'Maximum quantity of iterations is wrong'

    train_X = pd.read_csv('artefacts/train_x.csv')
    train_y = pd.read_csv('artefacts/train_y.csv')

    model = LogisticRegression(penalty=penalty, random_state=random_state, max_iter=max_iter)
    model.fit(train_X, train_y)

    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def inference(filename='model.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    test_X = pd.read_csv('artefacts/test_x.csv')

    preds = model.predict(test_X)
    pd.DataFrame(preds).to_csv('artefacts/preds.csv', index=False)
