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
from sklearn.feature_extraction.text import TfidfVectorizer

# DATA INITIALIZATION ###############################################
# movie dataset
# mov_data = np.genfromtxt('data/movie.csv', delimiter=',')
mov_data = pd.read_csv('data/movie.csv', quotechar='"')
mov_data.shape
mov_X = mov_data.iloc[:, 0] # text column
mov_y = mov_data.iloc[:, 1] # all rows except the first one, label column

# chat gpt dataset
# cgpt_data = np.genfromtxt('data/chatgpt_sentiment_analysis.csv', delimiter=',')
cgpt_data = pd.read_csv('data/chatgpt_sentiment_analysis.csv', quotechar='"')
cgpt_data.shape
cgpt_X = cgpt_data.iloc[1:, 1] # all rows except the first one, text column
cgpt_y = cgpt_data.iloc[1:, 2] # all rows except the first one, label column

# social media dataset
# sm_data = np.genfromtxt('data/soc_med_sentiment_analysis.csv', delimiter=',')
sm_data = pd.read_csv('data/soc_med_sentiment_analysis.csv', quotechar='"')
sm_data.shape
sm_X = sm_data.iloc[:, 4] # text column
sm_y = sm_data.iloc[:, 5] # label column

# Dictionaries and peripherals for data storage
# keys = ["mov", "cgpt", "sm"]

# X = {
#     "mov": mov_X,
#     "cgpt": cgpt_X,
#     "sm": sm_X
# }

# y = {
#     "mov": mov_y,
#     "cgpt": cgpt_y,
#     "sm": sm_y
# }

# X_train = { }

# X_test = { }

# y_train = { }

# y_test = { }


# PRE-PROCESSING ###################################################
# check for unique values?????????????
# convert the 0 labels from the mov data to -1
# y["mov"] = np.where(y["mov"] == "0", -1, 1).astype(int)
mov_y = np.where(mov_y == "0", -1, 1).astype(int)

# # convert the labels of the cgpt and sm data to numeric labels
# y["cgpt"] = np.where( y["cgpt"] == "good", 1, np.where(y["cgpt"] == "neutral", 0, -1)).astype(int)
# y["sm"] = np.where( y["sm"] == "positive", 1, np.where(y["sm"] == "neutral", 0, -1)).astype(int)
cgpt_y = np.where(cgpt_y == "good", 1, np.where(cgpt_y == "neutral", 0, -1)).astype(int)
sm_y = np.where(sm_y == "positive", 1, np.where(sm_y == "neutral", 0, -1)).astype(int)

# vectorize features for text analysis
vectorizer = TfidfVectorizer(max_features=500)
# X["mov"] = vectorizer.fit_transform(X["mov"]).toarray()
# X["cgpt"] = vectorizer.fit_transform(X["cgpt"]).toarray()
# X["sm"] = vectorizer.fit_transform(X["sm"]).toarray()
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

# for i in range(3):
    # key = keys[i]

    # A_tr, A_ts, b_tr, b_ts = preprocess(X[key], y[key])
    
    # X_train.update({key: A_tr})
    # X_test.update({key: A_ts})
    # y_train.update({key: b_tr})
    # y_test.update({key: b_ts})

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
        # self.conv_W = np.random.randn(self.K, self.F) / np.sqrt(self.F)
        self.conv_W = np.random.randn(self.K, self.F) / np.sqrt(self.N)
        self.conv_b = np.zeros(self.K) # self.b = np.zeros((1, self.K))?
        # self.conv1_W = np.random.randn(self.K, self.F) / np.sqrt(self.N)
        # self.conv1_b = np.zeros(self.K) # self.b = np.zeros((1, self.K))?
        # self.conv2_W = np.random.randn(self.K, self.F) / np.sqrt(self.N)
        # self.conv2_b = np.zeros(self.K) # self.b = np.zeros((1, self.K))?

        # fully connected layer weights and biases
        flattened_size = self.N - self.F + 1
        # self.fc_W = np.random.randn(3, flattened_size * self.K) / np.sqrt(flattened_size * self.K)
        self.fc_W = np.random.randn(3, self.K) / np.sqrt(self.N)
        self.fc_b = np.zeros(3)

        # Store intermediate values for backpropagation
        self.cache = {}

        print(f"initial conv w {self.conv_W}")
        print(f"conv b {self.conv_b}")
        print(f"fc w {self.fc_W}")
        print(f"fc b {self.fc_b}")

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)

    ################## VER 1 ###############################
    def convolve(self, input):
        # output = []
        output = np.zeros((self.K, input.shape[0] - self.F + 1))

        output_size = (self.N + 2 * self.P - self.F) / self.S + 1
        # print(f"output size: {output_size}")

        # check if it's a whole number
        while output_size % 1 != 0:
            # print("needs padding... calculating...")
            # check there has been an attempt at a padding calculation
            if self.P == 0:
                self.P = (self.F - 1) / 2 # if not, add padding using the formula

            else:
                self.P += 1
            
            output_size = (self.N + 2 * self.P - self.F) / self.S + 1
            # print(f"padding {self.P}, output size: {output_size}")
        
        # print(f"output size: {output_size}")
        output_size = int(output_size)

        # print("remaking layer with padding...")
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


    ############# VER 2 ###################################
    # def convolve(self, input, conv_W, conv_b):
    #     # Determine output size
    #     output_size = (self.N + 2 * self.P - self.F) / self.S + 1

    #     # Ensure output size is a whole number by adjusting padding if necessary
    #     while output_size % 1 != 0:
    #         if self.P == 0:
    #             # Add padding based on the formula
    #             self.P = (self.F - 1) // 2
    #         else:
    #             # Increment padding until the output size becomes valid
    #             self.P += 1

    #         output_size = (self.N + 2 * self.P - self.F) / self.S + 1

    #     output_size = int(output_size)

    #     # Apply padding to the input if required
    #     if self.P > 0:
    #         padded_input = np.zeros((input.shape[0] + 2 * self.P, input.shape[1] + 2 * self.P))
    #         padded_input[self.P:-self.P, self.P:-self.P] = input
    #     else:
    #         padded_input = input

    #     # Initialize the output array
    #     output = np.zeros((conv_W.shape[0], output_size, output_size))

    #     # Perform the convolution operation for each filter
    #     for k in range(conv_W.shape[0]):  # Loop over filters
    #         for i in range(output_size):
    #             for j in range(output_size):
    #                 # Define the region of the input corresponding to the current kernel position
    #                 region = padded_input[
    #                     i * self.S : i * self.S + self.F,
    #                     j * self.S : j * self.S + self.F,
    #                 ]
    #                 # Convolve the filter with the input region and add the bias
    #                 output[k, i, j] = np.sum(region * conv_W[k]) + conv_b[k]

    #     return output

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def pool(self, output):
        # # check to make sure that the pooling filter and stride will fit
        # for i in range(0, len(output), self.pool_S):
        #     pass
        # return np.max(output, axis=1)

        # Simple max pooling with tracking for backprop
        pooled_output = np.max(output, axis=1)
        
        # Store indices of max values for backprop
        self.cache['max_indices'] = np.argmax(output, axis=1)
        self.cache['conv_output'] = output
        
        return pooled_output

    def forward(self, input):
        # output = self.convolve(input)
        # output = self.relu(output)
        # output = self.convolve(output)
        # output = self.relu(output)
        # pooled = self.pool(output)

        # # Flatten
        # flattened = pooled.reshape(-1)
        
        # # Fully connected layer
        # fc_output = np.dot(self.fc_weights, flattened) + self.fc_b
        
        # # Softmax for classification
        # probabilities = self.softmax(fc_output)

        # return probabilities

        ########## VER 2 #####################
        # print("Convolutional layer...")
        # Convolution layer
        conv_output = self.convolve(input)
        # conv_output = self.convolve(conv_output)
        
        # print("Pooling...")
        # Max pooling
        pooled_output = self.pool(conv_output)
        
        # Flatten
        # flattened = pooled_output.reshape(-1)
        flattened = pooled_output
        self.cache['flattened'] = flattened
        
        # Fully connected layer
        fc_output = np.dot(self.fc_W, flattened) + self.fc_b
        self.cache['fc_pre_activation'] = fc_output
        
        # Softmax for classification
        probabilities = self.softmax(fc_output)
        self.cache['probabilities'] = probabilities
        
        return probabilities
    
        ######## VER 3 ###############################
        # Cache input for backprop
        # self.cache['conv_input'] = input

        # # First Convolution + ReLU
        # conv1_output = self.convolve(input, self.conv1_W, self.conv1_W)
        # conv1_output = self.relu(conv1_output)
        # self.cache['conv1_output'] = conv1_output
        # self.cache['conv1_pre_activation'] = input  # Input to first conv layer (before ReLU)

        # # Second Convolution + ReLU
        # conv2_output = self.convolve(conv1_output, self.conv2_W, self.conv2_W)
        # conv2_output = self.relu(conv2_output)
        # self.cache['conv2_output'] = conv2_output
        # self.cache['conv2_pre_activation'] = conv1_output  # Input to second conv layer (before ReLU)

        # # Max Pooling
        # pooled_output = self.pool(conv2_output)
        # self.cache['pooled_output'] = pooled_output

        # # Flatten
        # flattened = pooled_output.reshape(-1)
        # self.cache['flattened'] = flattened

        # # Fully Connected Layer
        # fc_output = np.dot(self.fc_W, flattened) + self.fc_b
        # self.cache['fc_pre_activation'] = fc_output

        # # Softmax for Classification
        # probabilities = self.softmax(fc_output)
        # self.cache['probabilities'] = probabilities

        # return probabilities

    def backprop(self, X, y, learning_rate):
    # def backprop(self, y, output):
        # # calculate the error for the output layer
        # output_error = y - output

        # # calculate the gradient
        # output_delta = output_error  # * 1 since the output layer uses a linear activation function with derivative of 1
        
        # # calculate the error for the hidden layer
        # hidden_error = np.dot(output_delta, np.transpose(self.W2))

        # # calculate the gradient
        # # hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)

        # # return output_delta, hidden_delta

        ################ VER 2 ###############################
        # One-hot encode the labels
        y_one_hot = np.zeros_like(self.cache['probabilities'])
        y_one_hot[y] = 1
        
        # Compute gradients
        # Softmax and Cross-Entropy Loss Gradient
        dL_dz = self.cache['probabilities'] - y_one_hot
        
        # Fully Connected Layer Gradients
        dL_dfc_weights = np.outer(dL_dz, self.cache['flattened'])
        dL_dfc_bias = dL_dz
        
        # Gradient w.r.t flattened layer
        dL_dflattened = np.dot(self.fc_W.T, dL_dz)
        
        # Reshape flattened gradient back to pooled output shape
        dL_dpooled = dL_dflattened.reshape(self.K)
        
        # Max Pooling Backward Pass
        dL_dconv = np.zeros_like(self.cache['conv_output'])
        for f in range(self.K):
            max_idx = self.cache['max_indices'][f]
            dL_dconv[f, max_idx] = dL_dpooled[f]
        
        # ReLU Derivative
        dL_dconv_pre = dL_dconv * self.relu_derivative(self.cache['conv_pre_activation'])
        
        # Convolutional Layer Gradients
        dL_dconv_weights = np.zeros_like(self.conv_W)
        dL_dconv_bias = np.zeros_like(self.conv_b)
        
        for f in range(self.K):
            for i in range(dL_dconv_pre.shape[1]):
                # Gradient for convolution weights
                dL_dconv_weights[f] += dL_dconv_pre[f, i] * self.cache['conv_input'][i:i+self.F]
                
            # Bias gradient
            dL_dconv_bias[f] = np.sum(dL_dconv_pre[f])
        
        # print("Updating weights and biases...")
        # Update weights and biases
        self.fc_W -= learning_rate * dL_dfc_weights
        self.fc_b -= learning_rate * dL_dfc_bias
        self.conv_W -= learning_rate * dL_dconv_weights
        self.conv_b -= learning_rate * dL_dconv_bias
        
        return dL_dz

        ##################### VER 3 ######################
        # # One-hot encode the labels
        # y_one_hot = np.zeros_like(self.cache['probabilities'])
        # y_one_hot[y] = 1

        # # Compute gradients
        # # Softmax and Cross-Entropy Loss Gradient
        # dL_dz = self.cache['probabilities'] - y_one_hot

        # # Fully Connected Layer Gradients
        # dL_dfc_weights = np.outer(dL_dz, self.cache['flattened'])
        # dL_dfc_bias = dL_dz

        # # Gradient w.r.t flattened layer
        # dL_dflattened = np.dot(self.fc_W.T, dL_dz)

        # # Reshape flattened gradient back to pooled output shape
        # dL_dpooled = dL_dflattened.reshape(self.cache['pooled_output'].shape)

        # # Max Pooling Backward Pass
        # dL_dconv2_output = np.zeros_like(self.cache['conv2_output'])
        # for f in range(self.K):
        #     max_idx = self.cache['max_indices'][f]
        #     dL_dconv2_output[f, max_idx] = dL_dpooled[f]

        # # Backpropagate through ReLU2
        # dL_dconv2_pre = dL_dconv2_output * self.relu_derivative(self.cache['conv2_pre_activation'])

        # # Backpropagate through Convolution Layer 2
        # dL_dconv2_weights = np.zeros_like(self.conv2_W)
        # dL_dconv2_bias = np.zeros_like(self.conv2_b)
        # dL_dconv1_output = np.zeros_like(self.cache['conv1_output'])

        # for f in range(self.K):
        #     for i in range(dL_dconv2_pre.shape[1]):
        #         # Gradient for convolution weights (Conv2)
        #         dL_dconv2_weights[f] += dL_dconv2_pre[f, i] * self.cache['conv1_output'][i:i+self.F]

        #     # Bias gradient (Conv2)
        #     dL_dconv2_bias[f] = np.sum(dL_dconv2_pre[f])

        #     # Gradient for input to Conv2 (output of Conv1)
        #     dL_dconv1_output[f, i:i+self.F] += dL_dconv2_pre[f, i] * self.conv2_W[f]

        # # Backpropagate through ReLU1
        # dL_dconv1_pre = dL_dconv1_output * self.relu_derivative(self.cache['conv1_pre_activation'])

        # # Backpropagate through Convolution Layer 1
        # dL_dconv1_weights = np.zeros_like(self.conv1_W)
        # dL_dconv1_bias = np.zeros_like(self.conv1_b)

        # for f in range(self.K):
        #     for i in range(dL_dconv1_pre.shape[1]):
        #         # Gradient for convolution weights (Conv1)
        #         dL_dconv1_weights[f] += dL_dconv1_pre[f, i] * self.cache['conv_input'][i:i+self.F]

        #     # Bias gradient (Conv1)
        #     dL_dconv1_bias[f] = np.sum(dL_dconv1_pre[f])

        # # Update weights and biases
        # self.fc_W -= learning_rate * dL_dfc_weights
        # self.fc_b -= learning_rate * dL_dfc_bias
        # self.conv2_W -= learning_rate * dL_dconv2_weights
        # self.conv2_b -= learning_rate * dL_dconv2_bias
        # self.conv1_W -= learning_rate * dL_dconv1_weights
        # self.conv1_b -= learning_rate * dL_dconv1_bias

        # return dL_dz

    def cross_entropy_loss(self, y_pred, y_true):
        # One-hot encode the true label
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[y_true] = 1
        
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Compute cross-entropy loss
        return -np.sum(y_true_one_hot * np.log(y_pred_clipped))

    def train(self, X, y, epochs, learning_rate):
        # input = X

        # for epoch in range(epochs):
        #     for i in range(len(X)):
                
        #         prediction = self.forward(input)

        #         # Compute loss (here using simple cross-entropy)
        #         true_label = y[i]
        #         loss = -np.log(prediction[true_label])
        #         total_loss += loss

        #         # backprop ?

        #     if epoch % 10 == 0:
        #         print(f"Epoch {epoch}, Loss: {total_loss/len(X)}")

        for epoch in range(epochs):
            # print(f"Epoch {epoch}")

            total_loss = 0
            
            # print(f"len(X) {len(X)}")
            for i in range(len(X)):
                # if i % 100 == 0:
                #     print(f"i {i}")

                # Forward propagation
                # print("Forward propagating now...")
                pred = self.forward(X[i])
                
                # Compute loss (cross-entropy)
                loss = self.cross_entropy_loss(pred, y[i])
                total_loss += loss
                
                # print("Backpropagating now...")
                # Backpropagation
                self.backprop(X[i], y[i], learning_rate)
            
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X)}")
            # print(f"Loss: {total_loss/len(X)}")

        print(f"final conv w {self.conv_W}")
        print(f"conv b {self.conv_b}")
        print(f"fc w {self.fc_W}")
        print(f"fc b {self.fc_b}")



# model = cnn(N=1, F=1, K=32, S=1, P=0, pool_F=2, pool_S=2)
# try combining the data and try keeping them separate to see what works better

# print(f"X train: {X_train}")
# print(f"X test: {X_test}")
# print(f"y train: {y_train}")
# print(f"y test: {y_test}")

# 40000 + 499 + 219,293 = 259,792

# print(f"X expected length: {259792}, len(X): {len(X)}")
# print(f"y expected length: {259792}, actual: {len(y)}")
# print(f"X train expected length: {259792 * .8}, actual: {len(X_train)}")
# print(f"X test expected length: {259792 * .2}, actual: {len(X_test)}")
# print(f"y train expected length: {259792 * .8}, actual: {len(y_train)}")
# print(f"y test expected length: {259792 * .2}, actual: {len(y_test)}")

model = cnn(N=(X_train.shape[1]), F=3, K=64, S=1, P=0, pool_F=2, pool_S=2)
model.train(X_train[:len(X_train)//100, :], y_train[:len(X_train)//100], 10, 0.001)
# [:len(X_train)//100, :]