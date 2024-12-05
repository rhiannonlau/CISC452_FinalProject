
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # standard between all 3 of our models
import pandas as pd

# Preprocessing function
def preprocess(X, y): 
    indices = []

    for i in range(X.shape[0]):
        indices.append(i)

    np.random.shuffle(indices) # shuffle the data
    X = X[indices]
    y = y[indices]

    # Use about half
    mid = X.shape[0]//2
    Xnew = X[:mid]
    ynew = y[:mid]

    # Split data: training 80%, testing 20%
    split = int(0.8 * Xnew.shape[0])
    X_train, X_test = Xnew[:split], Xnew[split:]
    y_train, y_test = ynew[:split], ynew[split:]

    return X_train, X_test, y_train, y_test


# Perceptron-related functions
def perceptron_activation(x): # if the value is positive, the perceptron will activate
    if x>=0:
        ans = 1
    else:
        ans = 0
    return ans

def perceptron_predict(x, w):
    if hasattr(x, "toarray"):
        x = x.toarray().flatten() # x was originally a sparse matrix, which is the output of TfidfVectorizer
    predict = np.dot(x, w[1:]) + w[0] # dot product and w0 represents the sum of all weighted inputs
    final = perceptron_activation(predict)
    return final

def train_perceptron(X, y, learning_rate, epochs):
    num_samples, num_features = X.shape
    w = np.zeros(num_features +1) # initialize bias and weights to zero
    for epoch in range(epochs):
        for i in range(num_samples):
            prediction = perceptron_predict(X[i], w)
            w[1:] += learning_rate*(y[i]-prediction)*X[i]
            w[0] += learning_rate*(y[i]-prediction)    
    return w

def multiplePerceptrons(X, y, classes, learning_rate, epochs): # work-around for using a perceptron for binary classification
    classifier = {} # initialize dict
    for c in classes: 
        y_hot = np.where(y == c, 1, 0) # hot encode for current class

        # train perceptron
        classifier[c] = train_perceptron(X, y_hot, learning_rate, epochs)
    return classifier

def multPredict(X, classifier):
    pred = []

    for x in X:
        if hasattr(x, "toarray"):
            x = x.toarray().flatten() # x was originally a sparse matrix
    
        value = {c: np.dot(x, w[1:]) + w[0] for c, w in classifier.items()} #use dot prod
        answer = max(value, key=value.get) # find highest value
        pred.append(answer)
    
    pred = np.array(pred)
    return pred

def calculate_accuracy(predicted, y_test):
    accuracy = np.mean(predicted == y_test)
    return accuracy






# Initialize and Preprocess

# Initialize Data
# Movie dataset
mov_data = pd.read_csv('data/movie.csv')
mov_X = mov_data.iloc[:, 0].values  # First column
mov_y = mov_data.iloc[:, 1].values   # second column

# ChatGPT dataset
cgpt_data = pd.read_csv('data/chatgpt_sentiment_analysis.csv')
cgpt_X = cgpt_data.iloc[:, 1].values  # 2nd column 
cgpt_y = cgpt_data.iloc[:, 2].values   # 3rd column 
cgpt_y = np.where(cgpt_y == "good", 1, np.where(cgpt_y == "neutral", 0, -1))

# Social Media dataset
sm_data = pd.read_csv('data/soc_med_sentiment_analysis.csv')
sm_X = sm_data.iloc[:, 4].values  # 5th column 
sm_y = sm_data.iloc[:, 5].values   # 6th column 
sm_y = np.where(sm_y == "positive", 1, np.where(sm_y == "neutral", 0, -1))

# Combine all datasets
Xtemp = np.concatenate((mov_X, cgpt_X), axis=0)
X = np.concatenate((Xtemp, sm_X), axis=0)
ytemp = np.concatenate((mov_y, cgpt_y), axis=0)
y = np.concatenate((ytemp, sm_y), axis=0)


# Vectorize the data
vectorizer = TfidfVectorizer(max_features=800, norm="l2")
X = vectorizer.fit_transform(X)

# separate train and test
X_train, X_test, y_train, y_test = preprocess(X, y)



# Training, Testing, Accuracy
classes = [-1.0,1] # possible classes it can fall into
classed = multiplePerceptrons(X_train, y_train, classes, learning_rate=0.001, epochs=8)
# FINAL RESULTS
for c, w in classed.items():
    print(f"Class {c}: Weights = {w[1:]}, Bias = {w[0]}")
# Call training
predicted = multPredict(X_test, classed)


# calculate and print accuracy
accuracy = calculate_accuracy(predicted, y_test)
print("Accuracy(%):", accuracy*100)
