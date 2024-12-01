# helpful command to prevent automatic transformation of datasets when pushing:
# git config --global core.autocrlf false

# Requirements
# Code and demo will be marked individually based on
# 1. Executability (2)
# 2. Functionality (2)
# 3. Accuracy (1)
# 4. Clarity and comments (1)
# 5. Explainability and demo (2)
# 6. Extensions made to the original code (2)

import numpy as np
from sklearn.preprocessing import StandardScaler

# DATA INITIALIZATION ###############################################
# movie dataset
mov_data = np.genfromtxt('data/movie.csv', delimiter=',')
mov_data.shape
mov_X = mov_data[1:, 0] # all rows except the first one, text column
mov_y = mov_data[1:, 1] # all rows except the first one, label column

# chat gpt dataset
cgpt_data = np.genfromtxt('data/chatgpt_sentiment_analysis.csv', delimiter=',')
cgpt_data.shape
cgpt_X = cgpt_data[1:, 1] # all rows except the first one, text column
cgpt_y = cgpt_data[1:, 2] # all rows except the first one, label column

# social media dataset
sm_data = np.genfromtxt('data/soc_med_sentiment_analysis.csv', delimiter=',')
sm_data.shape
sm_X = sm_data[1:, 4] # all rows except the first one, text column
sm_y = sm_data[1:, 5] # all rows except the first one, label column

keys = ["mov", "cgpt", "sm"]

X = {
    "mov": mov_X,
    "cgpt": cgpt_X,
    "sm": sm_X
}

y = {
    "mov": mov_y,
    "cgpt": cgpt_y,
    "sm": sm_y
}

X_train = { }

X_test = { }

y_train = { }

y_test = { }

# PRE-PROCESSING ###################################################
# convert the labels of the cgpt and sm data to numeric labels
# should i do a check to see make sure there's nothing outside of these labels?????????????
labels = np.where(y["cgpt"] == "good", 1, np.where(y["cgpt"] == "neutral", 0, np.where(y["cgpt"] == "bad", -1, y)))
y.update({"cgpt": labels})

labels = np.where(y["sm"] == "positive", 1, np.where(y["sm"] == "neutral", 0, np.where(y["sm"] == "negative", -1, y)))
y.update({"sm": labels})

def preprocess(X, y):
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Shuffle the data
    indices = []

    for i in range(len(X)):
        indices.append(i)

    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split data: training 80%, testing 20%
    split = int(0.8 * X.shape[0])
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # return X, X_train, X_test, y, y_train, y_test
    return X_train, X_test, y_train, y_test

for i in range(3):
    key = keys[i]

    # a, a_tr, a_ts, b, b_tr, b_ts = preprocess(X[key], y[key])
    A_tr, A_ts, b_tr, b_ts = preprocess(X[key], y[key])

    # X.update({key: A}) # does the X and y even matter anymore??
    X_train.update({key: A_tr})
    X_test.update({key: A_ts})
    # y.update({key: b})
    y_train.update({key: b_tr})
    y_test.update({key: b_ts})

# cnn
def model():
    pass

def predict():
    pass

def accuracy():
    pass

