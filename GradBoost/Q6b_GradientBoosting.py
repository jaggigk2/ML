
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def run_grad_boost():
    #load data
    loan = pd.read_csv('./dataset/loan_train.csv', encoding="ISO-8859-1", low_memory=False)
    print("Train dataset size = %s" % loan.shape.__str__())

    loan2 = pd.read_csv('./dataset/loan_test.csv', encoding="ISO-8859-1", low_memory=False)
    print("Test dataset size = %s\n" % loan2.shape.__str__())

    missing = round(100 * (loan.isnull().sum() / len(loan.id)), 2)

    columns_with_missing_values = list(missing[missing >= 50].index)
    #print(len(columns_with_missing_values))

    loan = loan.drop(columns_with_missing_values, axis=1)
    loan2 = loan2.drop(columns_with_missing_values, axis=1)

    loan = loan.drop('desc', axis=1) # removing description column
    loan2 = loan2.drop('desc', axis=1)  # removing description column

    #below three columns are categorical with only 0 or nan. Hence removed
    drop_columnlist = ['collections_12_mths_ex_med', 'chargeoff_within_12_mths', 'tax_liens']
    loan = loan.drop(drop_columnlist, axis=1)
    loan2 = loan2.drop(drop_columnlist, axis=1)

    #removing null records for important attributes
    loan = loan[~loan.pub_rec_bankruptcies.isnull()]
    loan2 = loan2[~loan2.pub_rec_bankruptcies.isnull()]

    missing = round(100 * (loan.isnull().sum() / len(loan.id)), 2)

    # removing the missing values record from the dataset
    loan = loan[~loan.emp_title.isnull()]
    loan = loan[~loan.emp_length.isnull()]
    loan = loan[~loan.title.isnull()]
    loan = loan[~loan.revol_util.isnull()]
    loan = loan[~loan.last_pymnt_d.isnull()]

    loan2 = loan2[~loan2.emp_title.isnull()]
    loan2 = loan2[~loan2.emp_length.isnull()]
    loan2 = loan2[~loan2.title.isnull()]
    loan2 = loan2[~loan2.revol_util.isnull()]
    loan2 = loan2[~loan2.last_pymnt_d.isnull()]

    # unique IDs and other features that are not required for analysis are dropped
    columns_tobe_dropped = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'pymnt_plan', 'url', 'zip_code',
                            'initial_list_status', 'policy_code', 'application_type', 'acc_now_delinq', 'delinq_amnt',
                            'emp_title','issue_d','title','addr_state','earliest_cr_line','last_pymnt_d','last_credit_pull_d']
    clean_loan = loan.drop(columns_tobe_dropped, axis=1)
    clean_loan2 = loan2.drop(columns_tobe_dropped, axis=1)


    #cleaning the records for int_rate and revol_rate. Sometimes the record had 10.9% ... so we need to strip %
    clean_loan['int_rate'] = clean_loan['int_rate'].str.strip('%').astype('float')
    clean_loan['revol_util'] = clean_loan['revol_util'].str.strip('%').astype('float')

    clean_loan2['int_rate'] = clean_loan2['int_rate'].str.strip('%').astype('float')
    clean_loan2['revol_util'] = clean_loan2['revol_util'].str.strip('%').astype('float')

    #cleaning the emp_length feature because some records have data likes '<1' , '10 year'. Make it numeric
    emp_length_dict = {
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10
    }
    clean_loan = clean_loan.replace({"emp_length": emp_length_dict})
    clean_loan2 = clean_loan2.replace({"emp_length": emp_length_dict})

    #
    clean_loan['term'] = clean_loan.term.apply(lambda x: x.split()[0])
    clean_loan2['term'] = clean_loan2.term.apply(lambda x: x.split()[0])

    #keep only the 'Fully Paid' and 'Charged Off' loan_status
    clean_loan = clean_loan[clean_loan['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    clean_loan2 = clean_loan2[clean_loan2['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    mapping_dict = {'loan_status': {'Fully Paid': 1, 'Charged Off': 0}}
    clean_loan = clean_loan.replace(mapping_dict)
    clean_loan2 = clean_loan2.replace(mapping_dict)

    # trasformation of categorical data to binary
    headers = ["grade", "sub_grade", "home_ownership",
               "verification_status", "purpose"]
    #
    for col in headers:
        enc = LabelEncoder()
        enc.fit(clean_loan[col])
        enc_name_mapping = dict(zip(enc.classes_, enc.transform(enc.classes_)))

        clean_loan[col] = enc.transform(clean_loan[col])
        clean_loan2[col] = clean_loan2[col].map(enc_name_mapping).fillna(-1).astype('str')

    #drop the loan status from X_data
    final_loan_X_train = clean_loan.drop('loan_status', axis=1).to_numpy()
    final_loan_y_train = clean_loan.loc[:, ["loan_status"]].to_numpy()

    final_loan_X_test = clean_loan2.drop('loan_status', axis=1).to_numpy()
    final_loan_y_test = clean_loan2.loc[:, ["loan_status"]].to_numpy()


    # prediction and error
    estimators = np.array([50,200, 500])
    accuracies = []
    for instance in estimators:

        clf = GradientBoostingClassifier(n_estimators=instance, learning_rate=0.1, max_depth=1, random_state=0)
        clf.fit(final_loan_X_train, np.ravel(final_loan_y_train))
        y_pred = clf.predict(final_loan_X_test)

        #accracy
        accuracy = accuracy_score(final_loan_y_test, y_pred)
        print("accuracy = %.3f when Estimators = %.3f" % (accuracy,instance))
        accuracies.append(accuracy)

        # Precision
        precision = precision_score(final_loan_y_test, y_pred)
        print("precision = %.3f when Estimators = %.3f" % (precision,instance))

        # recall
        recall = recall_score(final_loan_y_test, y_pred)
        print("recall = %.3f when Estimators = %.3f\n" % (recall,instance))

    best_accuracy = np.max(accuracies)
    best_accuracy_index = [i for i, x in enumerate(accuracies) if x == best_accuracy]
    print("The best accuracy of Gradient Boosting clasifier = %.3f was for Estimators = %.3f, learning_rate=0.1, max_depth=1 " % (best_accuracy, estimators[best_accuracy_index]))

    clf_DT = DecisionTreeClassifier(max_depth=None, min_samples_split=500,random_state = 0)
    clf_DT.fit(final_loan_X_train, final_loan_y_train)
    y_pred_DT = clf_DT.predict(final_loan_X_test)

    # accracy of simple Decision tree
    accuracy_DT = accuracy_score(final_loan_y_test, y_pred_DT)
    print("accuracy = %.3f of simple Decision Tree classifier was when min_samples_split = 500" % (accuracy_DT))

if __name__ == "__main__":
    run_grad_boost()
