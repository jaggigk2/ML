# FoML Assign 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import numpy as np

myname = "Jagadeesh Krishnan"


# Implement your decision tree below
class DecisionTree:

    def __init__(self):
        # initialize the tree
        self.tree = {"attribute": None, "threshold": None, "left_data": None, "right_data": None,
                     "information_gain": None, "leaf_value": None, "y_data": None, "prob": None}

    def build_tree(self, dataset):

        x_data, y_data = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(x_data)

        tree_split = {"attribute": None, "threshold": None, "left_data": None, "right_data": None,
                      "information_gain": None, "leaf_value": None, "y_data": None, "prob": None}

        # split the decision node when the data is more than 10
        if num_samples > 10:
            # find the best split
            tree_split = self.make_tree_split(dataset, num_samples, num_features)
            # decision node
            if tree_split["information_gain"] > 0:
                left_tree = self.build_tree(tree_split["left_data"])
                right_tree = self.build_tree(tree_split["right_data"])

                self.tree["attribute"] = tree_split["attribute"]
                self.tree["threshold"] = tree_split["threshold"]
                self.tree["left_data"] = left_tree
                self.tree["right_data"] = right_tree
                self.tree["information_gain"] = tree_split["information_gain"]
                self.tree["leaf_value"] = None
                self.tree["y_data"] = tree_split["y_data"]
                self.tree["prob"] = tree_split["prob"]

                tree_split["left_data"] = left_tree
                tree_split["right_data"] = right_tree
                tree_split["leaf_value"] = None
                return tree_split

        # make it leaf node when the data is less than 10
        leaf_value = self.get_leaf_value(y_data)
        self.tree["leaf_value"] = leaf_value
        tree_split["leaf_value"] = leaf_value

        self.tree["left_data"] = None
        self.tree["right_data"] = None
        tree_split["left_data"] = None
        tree_split["right_data"] = None

        self.tree["y_data"] = y_data
        tree_split["y_data"] = y_data

        prob = self.get_leaf_value_Prob(y_data)
        # leaf_info_gain = 1-self.entropy(y_data)
        self.tree["prob"] = prob
        tree_split["prob"] = prob
        return tree_split

    def make_tree_split(self, dataset, num_samples, num_features):

        tree_split = {}
        maximum_gain = -float("inf")

        # loop over all the features
        for attribute in range(num_features):
            feature_values = dataset[:, attribute]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                left_data, right_data = self.split_left_right(dataset, attribute, threshold)

                if len(left_data) > 0 and len(right_data) > 0:
                    y, left_y, right_y = dataset[:, -1], left_data[:, -1], right_data[:, -1]
                    # compute information gain
                    calculated_info_gain = self.information_gain(y, left_y, right_y)

                    # update the best split if needed
                    if calculated_info_gain > maximum_gain:
                        tree_split["attribute"] = attribute
                        tree_split["threshold"] = threshold
                        tree_split["left_data"] = left_data
                        tree_split["right_data"] = right_data
                        tree_split["information_gain"] = calculated_info_gain
                        tree_split["y_data"] = y
                        maximum_gain = calculated_info_gain
                        tree_split["prob"] = self.get_leaf_value_Prob(y)
        return tree_split

    def split_left_right(self, dataset, attribute, threshold):

        left_data = np.array([row for row in dataset if row[attribute] <= threshold])
        right_data = np.array([row for row in dataset if row[attribute] > threshold])
        return left_data, right_data

    def information_gain(self, parent, left_child, right_child):

        weight_l = len(left_child) / len(parent)
        weight_r = len(right_child) / len(parent)
        gain = self.entropy(parent) - (weight_l * self.entropy(left_child) + weight_r * self.entropy(right_child))
        return gain

    def entropy(self, y_value):

        class_labels = np.unique(y_value)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y_value[y_value == cls]) / len(y_value)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def get_leaf_value(self, y_value):

        y_value = list(y_value)
        return max(y_value, key=y_value.count)

    def get_leaf_value_Prob(self, y_value):

        y_value_list = list(y_value)
        y_value_max = max(y_value_list, key=y_value_list.count)
        if y_value_max == 0:
            countMax = np.count_nonzero(y_value == 0)
            countMin = np.count_nonzero(y_value == 1)
        else:
            countMax = np.count_nonzero(y_value == 1)
            countMin = np.count_nonzero(y_value == 0)
        prob_max = countMax / (countMax + countMin)

        return prob_max

    def prediction(self, x_value, tree):
        if tree is not None:
            if tree["leaf_value"] is not None: return tree["leaf_value"]
            feature_val = x_value[tree["attribute"]]
            if feature_val <= tree["threshold"]:
                return self.prediction(x_value, tree["left_data"])
            else:
                return self.prediction(x_value, tree["right_data"])

    def post_prune(self, tree, threshold):
        # simplify tree bottom-up
        if tree is None: return tree
        left_data = None
        right_data = None
        for key, value in tree.items():
            if key == "left_data" and value is not None:
                left_data = value
            if key == "right_data" and value is not None:
                right_data = value
            if key == "prob":
                # stop the tree if it is greater than treshold and make it a leaf
                if value > threshold:
                    tree["left_data"] = None
                    tree["right_data"] = None
                    tree["leaf_value"] = self.get_leaf_value(tree["y_data"])
                    tree["prob"] = self.get_leaf_value_Prob(tree["y_data"])
                    return tree
                else:
                    left_child = self.post_prune(left_data, threshold)
                    tree["left_data"] = left_child

                    right_child = self.post_prune(right_data, threshold)
                    tree["right_data"] = right_child
                    return tree

    def learn(self, training_set):
        training_set_arr = np.array(training_set)
        self.tree = self.build_tree(training_set_arr)

    def classify(self, test_instance):
        result = self.prediction(test_instance, self.tree)
        return result


def run_decision_tree():
    # Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [np.array(line).astype(float) for line in csv.reader(f, delimiter=",")]
    print("Number of records: %d" % len(data))

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    training_set = [x for i, x in enumerate(data) if i % K != 5]  #5
    test_set = [x for i, x in enumerate(data) if i % K == 5]

    tree = DecisionTree()
    # Construct a tree using training set
    tree.learn(training_set)

    # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set:
        result = tree.classify(instance[:-1])
        results.append(result == instance[-1])

    accuracy = float(results.count(True)) / float(len(results))
    print("accuracy: %.4f" % accuracy)

    # Post Pruning. do not break down the tree further when the probability of that node value is more than 85%
    tree.tree = tree.post_prune(tree.tree, 0.85)  #.85
    results1 = []
    for instance in test_set:
        result1 = tree.classify(instance[:-1])
        results1.append(result1 == instance[-1])

    accuracy1 = float(results1.count(True)) / float(len(results1))
    print("after post pruning accuracy: %.4f" % accuracy1)


if __name__ == "__main__":
    run_decision_tree()
