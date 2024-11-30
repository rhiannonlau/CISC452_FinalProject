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
# this one has some hindi in it??
# twtData = np.genfromtxt('data/Twitter_Data.csv', delimiter=',')
# twtData.shape

mov_data = np.genfromtxt('data/movie.csv', delimiter=',')
mov_data.shape
# skip the first row
mov_X = mov_data[1:, 0] # all rows except the first one, text column
mov_y = mov_data[1:, 1] # all rows except the first one, label column

# this one's data seems incomplete - lots of 0's
# amzData = np.genfromtxt('data/amazon_reviews.csv', delimiter=',')
# amzData.shape


# PRE-PROCESSING ###################################################
# Normalize the data
scaler = StandardScaler()
mov_X = scaler.fit_transform(mov_X)

# Shuffle the data - necessary?
# ind = []
# for i in range(len(mov_X)):
#     ind.append(i)
# np.random.shuffle(ind)
# mov_X = mov_X[ind]
# mov_y = mov_y[ind]

# Split data: training 80%, testing 20%
split = int(0.8 * mov_X.shape[0])
X_train, X_test = mov_X[:split], mov_X[split:]
y_train, y_test = mov_y[:split], mov_y[split:]

