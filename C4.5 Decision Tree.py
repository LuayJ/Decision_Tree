import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
from random import randint

# Getting the dataset and partitioning it into training, validation, and test sets
df = pd.read_csv('adult.csv')
# train_df = df.iloc[0:22793, :]
train_df = df.iloc[0:2000, :]  # Smaller dataset used to test the algorithm during development
val_df = df.iloc[22793:26049, :]
test_df = df.iloc[26049:, :]

# Splitting the validation set into data and labels
val_data = val_df.drop([val_df.columns[-1]], axis=1)
val_labels = val_df[val_df.columns[-1]]

# Splitting the test set into data and labels
test_data = test_df.drop([test_df.columns[-1]], axis=1)
test_labels = test_df[test_df.columns[-1]]

theta = [0.2, 0.4, 0.6, 0.8]  # Theta (cutoff) values


# Returns entropy of the parent node, takes the training set as the input argument
def parent_get_entropy(df):
    entropy = 0

    for label in np.unique(df['income']):
        frac = df['income'].value_counts()[label] / len(df['income'])
        entropy += -frac * np.log2(frac)

    return entropy


# Returns entropy of the current feature, takes the training set and the feature in question as input arguments
# This is part of the algorithm used to decide what feature to split at
def feature_get_entropy(df, feature):
    entropy = 0
    threshold = 0

    if df[feature].dtypes == object:
        for feature_val in np.unique(df[feature]):
            feature_entropy = 0

            for label in np.unique(df['income']):
                numer = len(df[feature][df[feature] == feature_val][df['income'] == label])
                denom = len(df[feature][df[feature] == feature_val])

                frac = numer / (denom + np.finfo(float).eps)  # EPS is epsilon, used to ensure no dividing by 0 errors

                if frac > 0:
                    feature_entropy += -frac * np.log2(frac)

            weight = len(df[feature][df[feature] == feature_val]) / len(df)
            entropy += weight * feature_entropy

    else:
        entropy = 1  # Max entropy in a binary tree
        prev = 0

        for feature_val in np.unique(df[feature]):
            current_entropy = 0
            cutoff = (feature_val + prev) / 2

            for op in [operator.le, operator.gt]:
                feature_entropy = 0

                for label in np.unique(df['income']):
                    numer = len(df[feature][op(df[feature], cutoff)][df['income'] == label])
                    denom = len(df[feature][op(df[feature], cutoff)])

                    frac = numer / (denom + np.finfo(float).eps)  # EPS is epsilon, to ensure no dividing by 0 errors

                    if frac > 0:
                        feature_entropy += -frac * np.log2(frac)

                weight = denom / len(df)
                current_entropy += weight * feature_entropy

            if current_entropy < entropy:
                entropy = current_entropy
                threshold = cutoff

    return entropy, threshold


# Decides which feature is best to use for splitting, takes the training set as input
# Decides based on the maximum information gain
def best_split(df):
    info_gains = []
    thresholds = []

    for feature in list(df.columns[:-1]):
        parent_entropy = parent_get_entropy(df)
        feature_entropy, threshold = feature_get_entropy(df, feature)

        info_gain = parent_entropy - feature_entropy

        info_gains.append(info_gain)
        thresholds.append(threshold)

    return df.columns[:-1][np.argmax(info_gains)], thresholds[np.argmax(info_gains)], feature_entropy


# Splits the training set rows given the best feature and the criteria to base the split on
# Inputs: the training set, chosen feature, sub-feature, and left or right side depending on the type of feature it is
def split_rows(df, feature, feature_val, op):
    return df[op(df[feature], feature_val)].reset_index(drop=True)


# Grows the tree, takes the training set and theta value as input
# Returns the grown tree
def grow_tree(df, theta, tree=None):
    global depth
    feature, cutoff, entropy = best_split(df)  # Finds the best feature to split at

    if tree is None:
        tree = {}
        tree[feature] = {}

    # And thus begins the recursion
    if df[feature].dtypes == object:
        # To handle categorical data
        for feature_val in np.unique(df[feature]):
            new_df = split_rows(df, feature, feature_val, operator.eq)
            labels, count = np.unique(new_df['income'], return_counts=True)

            if len(count) == 1:
                # Pure leaf
                tree[feature][feature_val] = labels[0]
            else:
                if depth >= max_depth or entropy < theta:
                    # Makes a leaf if either of the above conditions are true (not pure)
                    tree[feature][feature_val] = labels[np.argmax(count)]
                else:
                    # Continue the recursion
                    depth += 1
                    tree[feature][feature_val] = grow_tree(new_df, theta)
    else:
        # To handle numeric data
        new_df = split_rows(df, feature, cutoff, operator.le)
        labels, count = np.unique(new_df['income'], return_counts=True)

        if len(count) == 1:
            # Pure leaf
            tree[feature]['<=' + str(cutoff)] = labels[0]
        else:
            if depth >= max_depth or entropy < theta:
                # Makes a leaf if either of the conditions are true (not a pure leaf)
                tree[feature]['<=' + str(cutoff)] = labels[np.argmax(count)]
            else:
                # Continue the recursion for values <= numerical cutoff (cutoff decided in the best split function)
                depth += 1
                tree[feature]['<=' + str(cutoff)] = grow_tree(new_df, theta)

        new_df = split_rows(df, feature, cutoff, operator.gt)
        labels, count = np.unique(new_df['income'], return_counts=True)

        if len(count) == 1:
            # Pure leaf
            tree[feature]['>' + str(cutoff)] = labels[0]
        else:
            if depth >= max_depth or entropy < theta:
                # Makes a leaf if either of the conditions are true (not a pure leaf)
                tree[feature]['>' + str(cutoff)] = labels[np.argmax(count)]
            else:
                # Continue the recursion for values > numerical cutoff (cutoff decided in the best split function)
                tree[feature]['>' + str(cutoff)] = grow_tree(new_df, theta)

    return tree


# Predicts given row (x) with features, traverses the existing tree to find and return the prediction
# x is a SINGLE ROW, NOT a dataframe
def predict_target(features, x, tree):
    for node in tree.keys():
        val = x[node]

        # To avoid a potential problem where a certain value may not be found in the tree, reason unknown
        # Makes a prediction randomly
        if val not in tree[node].keys():
            rand = randint(0, 1)

            if rand == 0:
                prediction = '<=50K'
            else:
                prediction = '>50K'

            return prediction

        if type(val) == str:
            tree = tree[node][val]
        else:
            cutoff = str(list(tree[node].keys())[0]).split('<=')[1]

            if val <= float(cutoff):
                tree = tree[node]['<=' + cutoff]
            else:
                tree = tree[node]['>' + cutoff]

        if type(tree) is dict:
            prediction = predict_target(features, x, tree)
        else:
            prediction = tree
            return prediction

    return prediction


# Predictor function, test dataset as input, returns array of predictions
def predict(x):
    results = []
    features = {key: i for i, key in enumerate(list(x.columns))}

    for i in range(len(x)):
        results.append((predict_target(features, x.iloc[i], tree)))

    return np.array(results)


# Accuracy of the model
# Inputs are the true labels and the predicted labels
def accuracy(y_true, y_pred):
    return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100, 2)


train_acc = []
test_acc = []
max_depth = int(input('Enter a maximum depth for the trees (the root node is 1): '))
print('Growing Trees...')

for idx in range(len(theta)):
    depth = 1
    tree = grow_tree(train_df, theta[idx])
    print('\nCutoff:', theta[idx])
    # print('\nDecision Tree(depth = {}) : \n {}'.format(depth, tree))
    train_acc.append(accuracy(val_labels, predict(val_data)))
    print('Train Accuracy:', train_acc[idx])
    test_acc.append(accuracy(test_labels, predict(test_data)))
    print('Test Accuracy:', test_acc[idx])

plt.xlabel('Cutoff Values')
plt.ylabel('Accuracy')
plt.plot(theta, train_acc, label='Training Accuracy')
plt.plot(theta, test_acc, label='Test Accuracy')
plt.legend()
plt.show()
