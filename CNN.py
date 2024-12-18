# CNN - Rhiannon Lau

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# DATA INITIALIZATION ###############################################
# movie dataset
mov_data = pd.read_csv('data/movie.csv', quotechar='"')
mov_data.shape
mov_X = mov_data.iloc[:, 0] # text column
mov_y = mov_data.iloc[:, 1] # all rows except the first one, label column

# chat gpt dataset
cgpt_data = pd.read_csv('data/chatgpt_sentiment_analysis.csv', quotechar='"')
cgpt_data.shape
cgpt_X = cgpt_data.iloc[1:, 1] # all rows except the first one (because of the header), text column
cgpt_y = cgpt_data.iloc[1:, 2] # all rows except the first one, label column

# social media dataset
sm_data = pd.read_csv('data/soc_med_sentiment_analysis.csv', quotechar='"')
sm_data.shape
sm_X = sm_data.iloc[:, 4] # text column
sm_y = sm_data.iloc[:, 5] # label column

# PRE-PROCESSING ###################################################
# convert the 0 labels from the mov data to -1
mov_y = np.where(mov_y == "0", -1, 1).astype(int)

# # convert the labels of the cgpt and sm data to numeric labels
cgpt_y = np.where(cgpt_y == "good", 1, np.where(cgpt_y == "neutral", 0, -1)).astype(int)
sm_y = np.where(sm_y == "positive", 1, np.where(sm_y == "neutral", 0, -1)).astype(int)

# vectorize features for text analysis
vectorizer = TfidfVectorizer(max_features=750)
mov_X = vectorizer.fit_transform(mov_X).toarray()
cgpt_X = vectorizer.fit_transform(cgpt_X).toarray()
sm_X = vectorizer.fit_transform(sm_X).toarray()

X = np.concatenate((mov_X, cgpt_X, sm_X), axis=0)
y = np.concatenate((mov_y, cgpt_y, sm_y), axis=0)

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

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess(X, y)

# CNN MODEL ########################################################
class cnn:
    def __init__(self, N, F, K, S, P, pool_F, pool_S):
        self.N = N # input size
        self.F = F # filter size
        self.K = K # number of filters
        self.S = S # stride
        self.P = P # padding
        self.pool_F = pool_F # pooling filter size
        self.pool_S = pool_S # pooling stride

        # convolution layer weights and biases
        self.conv_W = np.random.randn(self.K, self.F) / np.sqrt(self.N)
        self.conv_b = np.zeros(self.K)

        # fully connected layer weights and biases
        self.fc_W = np.random.randn(3, self.K) * np.sqrt(2.0 / (self.N + 3))
        self.fc_b = np.zeros(3)

        # store values for backpropagation
        self.cache = {}

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

    def convolve(self, input):
        # output = []
        output = np.zeros((self.K, input.shape[0] - self.F + 1))

        output_size = (self.N + 2 * self.P - self.F) / self.S + 1

        # check if it's a whole number
        while output_size % 1 != 0:
            # check there has been an attempt at a padding calculation
            if self.P == 0:
                self.P = (self.F - 1) / 2 # if not, add padding using the formula

            else:
                self.P += 1
            
            output_size = (self.N + 2 * self.P - self.F) / self.S + 1
        
        output_size = int(output_size)
        # if we need padding, re-make the layer with the padding
        if self.P != 0:
            padded = np.zeros(self.N + 2 * self.P, self.N + 2 * self.P)

            for i in range(self.N):
                for j in range(self.N):
                    padded[i + self.P][j + self.P] = input[i][j]
            
        # print("making output layer...")
        # make the output layer
        for k in range(self.K):
            for i in range(output_size):
                output[k, i] = np.sum(input[i:i + self.F] * self.conv_W[self.F]) + self.conv_b[self.F]

        # Store input and pre-activation for backprop
        self.cache['conv_input'] = input
        self.cache['conv_pre_activation'] = output
        
        return self.relu(output)

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def pool(self, output):
        # max pooling
        pooled_output = np.max(output, axis=1)
        
        # store indices of max values for backprop
        self.cache['max_indices'] = np.argmax(output, axis=1)
        self.cache['conv_output'] = output
        
        return pooled_output

    def forward(self, input):
        conv_output = self.convolve(input)
        # conv_output = self.convolve(conv_output)
        
        pooled_output = self.pool(conv_output)
        
        # store pooled layer
        # pooled = pooled_output.reshape(-1)
        pooled = pooled_output
        self.cache['pooled'] = pooled
        
        # fully connected layer
        fc_output = np.dot(self.fc_W, pooled) + self.fc_b
        self.cache['fc_pre_activation'] = fc_output
        
        # softmax for classification
        probabilities = self.softmax(fc_output)
        self.cache['probabilities'] = probabilities
        
        return probabilities

    def backprop(self, X, y, learning_rate):
        # one-hot encode the labels
        y_enc = np.zeros_like(self.cache['probabilities'])
        y_enc[y] = 1
        
        # compute gradients
        dL_dz = self.cache['probabilities'] - y_enc
        
        # fully connected layer gradients
        dL_dfc_W = np.outer(dL_dz, self.cache['pooled'])
        dL_dfc_b = dL_dz
        
        # pooled layer gradient
        dL_dpooled = np.dot(self.fc_W.T, dL_dz)
        
        # reshape gradient back to pooled output shape
        dL_dpooled = dL_dpooled.reshape(self.K)
        
        # max pooling backward pass
        dL_dconv = np.zeros_like(self.cache['conv_output'])
        for f in range(self.K):
            max_idx = self.cache['max_indices'][f]
            dL_dconv[f, max_idx] = dL_dpooled[f]
        
        # relu derivative
        dL_dconv_pre = dL_dconv * self.relu_derivative(self.cache['conv_pre_activation'])
        
        # convolutional layer gradients
        dL_dconv_W = np.zeros_like(self.conv_W)
        dL_dconv_b = np.zeros_like(self.conv_b)
        
        for f in range(self.K):
            for i in range(dL_dconv_pre.shape[1]):
                # gradient for convolution weights
                dL_dconv_W[f] += dL_dconv_pre[f, i] * self.cache['conv_input'][i:i+self.F]
                
            # bias gradient
            dL_dconv_b[f] = np.sum(dL_dconv_pre[f])
        
        # update weights and biases
        self.fc_W -= learning_rate * dL_dfc_W
        self.fc_b -= learning_rate * dL_dfc_b
        self.conv_W -= learning_rate * dL_dconv_W
        self.conv_b -= learning_rate * dL_dconv_b
        
        return dL_dz

    def cross_entropy_loss(self, y_pred, y_true):
        # one-hot encode the true label
        y_enc = np.zeros_like(y_pred)
        y_enc[y_true] = 1
        
        # clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # compute loss
        return -np.sum(y_enc * np.log(y_pred_clipped))
    
    def predict(self, X):
        return self.forward(X)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):

            total_loss = 0
            
            for i in range(len(X)):
                pred = self.forward(X[i])
                
                # Compute loss
                loss = self.cross_entropy_loss(pred, y[i])
                total_loss += loss
                
                self.backprop(X[i], y[i], learning_rate)
            
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X)}")

model = cnn(N=(X_train.shape[1]), F=5, K=64, S=1, P=0, pool_F=2, pool_S=2)
model.train(X_train[:len(X_train)//100, :], y_train[:len(X_train)//100], 5, 0.001)