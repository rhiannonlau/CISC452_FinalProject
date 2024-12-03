# if needed: helpful command to prevent automatic transformation of datasets when pushing:
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
import pandas as pd
from sklearn.preprocessing import StandardScaler

# DATA INITIALIZATION ###############################################
# movie dataset
mov_data = np.genfromtxt('data/movie.csv', delimiter=',')
# mov_data = pd.read_csv('data/movie.csv', quotechar='"')
mov_data.shape
mov_X = mov_data[1:, 0] # all rows except the first one, text column
mov_y = mov_data[1:, 1] # all rows except the first one, label column

# chat gpt dataset
cgpt_data = np.genfromtxt('data/chatgpt_sentiment_analysis.csv', delimiter=',')
# cgpt_data = pd.read_csv('data/chatgpt_sentiment_analysis.csv', quotechar='"')
cgpt_data.shape
cgpt_X = cgpt_data[1:, 1] # all rows except the first one, text column
cgpt_y = cgpt_data[1:, 2] # all rows except the first one, label column

# social media dataset
sm_data = np.genfromtxt('data/soc_med_sentiment_analysis.csv', delimiter=',')
# sm_data = pd.read_csv('data/soc_med_sentiment_analysis.csv', quotechar='"')
sm_data.shape
sm_X = sm_data[1:, 4] # all rows except the first one, text column
sm_y = sm_data[1:, 5] # all rows except the first one, label column

# Dictionaries and peripherals for data storage
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
# convert the 0 labels from the mov data to -1
labels = np.where(y["mov"] == "0", -1, y)
y.update({"mov": labels})

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

# CNN MODEL ########################################################
class cnn:
    def __init__(self, n, f, K, s):
        self.n = n
        self.f = f
        self.K = K
        self.s = s
        self.p = 0

        self.weights = np.random.randn(self.K, self.f) / np.sqrt(self.f)
        self.b = np.zeros(self.K) # self.b = np.zeros((1, self.K))?

        # Fully connected layer weights
        flattened_size = self.n[0] - self.f + 1
        self.fc_weights = np.random.randn(3, flattened_size * self.K) / np.sqrt(flattened_size * self.K)
        self.fc_bias = np.zeros(3)

    def convolve(self, input):
        output = []

        output_size = (self.n + 2 * self.p - self.f) / self.s + 1

        # check if it's a whole number
        while not isinstance(output_size, int):
            # check there has been an attempt at a padding calculation
            if self.p == 0:
                self.p = (self.f - 1) / 2 # if not, add padding using the formula

            else:
                self.p += 1
            
            output_size = (self.n + 2 * self.p - self.f) / self.s + 1
            
        # if we need padding, re-make the layer with the padding
        if self.p != 0:
            padded = np.zeros(self.n + 2 * self.p, self.n + 2 * self.p)

            for i in range(self.n):
                for j in range(self.n):
                    padded[i + self.p][j + self.p] = input[i][j]
            
        # make the output layer
        for k in range(self.K):
            for i in range(output_size):
                output[k, i] = np.sum(input[i:i + self.f] * self.weights[self.f]) + self.b[self.f]

    def relu(x):
        return np.maximum(0, x)

    def pool(output):
        return np.max(output, axis=1)

    def forward():
        pass

    def backprop():
        pass

    def predict():
        pass

    def accuracy():
        pass

    def train(self, X, y, epochs, learning_rate):
        input = X

        for epoch in range(epochs):
            for i in range(len(X)):
                output = self.convolve(input)
                output = self.relu(output)
                output = self.convolve(output)
                output = self.relu(output)
                pooled = self.pool(output)

                # Flatten
                flattened = pooled.reshape(-1)
                
                # Fully connected layer
                fc_output = np.dot(self.fc_weights, flattened) + self.fc_bias
                
                # Softmax for classification
                probabilities = self.softmax(fc_output)

                # Compute loss (here using simple cross-entropy)
                true_label = y[i]
                loss = -np.log(probabilities[true_label])
                total_loss += loss

                # backprop

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X)}")



model = cnn(n=1, f=1, K=32, s=1)
# try combining the data and try keeping them separate to see what works better
