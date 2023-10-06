import math

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Assign1_DecisionTree import DecisionTree
import numpy as np

class RandomForest():
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.
    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    max_features: int
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    """

    def __init__(self, n_estimators=100, max_features=None, min_samples_split=50,
                 min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators  # Number of trees
        self.max_features = max_features  # Maxmimum number of features per tree
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain  # Minimum information gain req. to continue
        self.max_depth = max_depth  # Maximum depth for tree


        # Initialize decision trees
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                DecisionTree())

    def get_random_subsets(self, X, y, n_subsets, replacements=True):
        """ Return random subsets (with replacements) of the data """
        n_samples = np.shape(X)[0]
        # Concatenate x and y and do a random shuffle
        X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
        np.random.shuffle(X_y)
        subsets = []

        # Uses 50% of training samples without replacements
        subsample_size = int(n_samples // 2)
        if replacements:
            subsample_size = n_samples  # 100% with replacements

        for _ in range(n_subsets):
            idx = np.random.choice(
                range(n_samples),
                size=np.shape(range(subsample_size)),
                replace=replacements)
            X = X_y[idx][:, :-1]
            y = X_y[idx][:, -1]
            subsets.append([X, y])
        return subsets

    def fit(self, X, y):
        n_features = np.shape(X)[1]
        # If max_features have not been defined => select it as
        # sqrt(n_features)
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))

        # Choose one random subset of the data for each tree
        subsets = self.get_random_subsets(X, y, self.n_estimators)

        for i in range(self.n_estimators):
            X_subset, y_subset = subsets[i]
            # Feature bagging (select random subsets of the features)
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            # Save the indices of the features for prediction
            self.trees[i].feature_indices = idx
            # Choose the features corresponding to the indices
            X_subset = X_subset[:, idx]
            # Fit the tree to the data
            self.trees[i].learn(np.column_stack((X_subset,y_subset)))

    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        # Let each tree make a prediction on the data
        for i, tree in enumerate(self.trees):
            # Indices of the features that the tree has trained on
            idx = tree.feature_indices
            # Make a prediction based on those features
            results = []
            for instance in X:
                result = tree.classify(instance[idx])
                results.append(result)
            y_preds[:, i] = results

        y_pred = []
        # For each sample
        for sample_predictions in y_preds:
            # Select the most common class prediction
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def run_random_forest():
    # Load data set
    input_file = "./dataset/spam.data.txt"
    data = np.loadtxt(input_file)
    print("Number of records: %d" % len(data))
    print("Train dataset size = %s\n" % data.shape.__str__())

    X, y = data[:, :-1], data[:, -1]

    # Split training/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    max_features_array = np.array([4,7,10,20])
    accuracies = []
    oob_errors = []
    for instance in max_features_array:
        clf = RandomForest(n_estimators=10,max_features=instance)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        TP, FP, TN, FN = perf_measure(y_test, y_pred)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        print("accuracy = %.3f when max_features = %.1f" % (accuracy,instance))
        accuracies.append(accuracy)

        sensitivity = TP/(TP+FN)
        print("Sensitivity = %.3f when max_features = %.1f\n" % (sensitivity,instance))

if __name__ == "__main__":
    run_random_forest()