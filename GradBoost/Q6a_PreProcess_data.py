import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def run_process_data():
    # load data
    loan = pd.read_csv('./dataset/loan_train.csv', encoding="ISO-8859-1", low_memory=False)
    print("before feature engineering & before cleaning = " + loan.shape.__str__())

    # to check which columns have more missing data to be removed
    missing = round(100 * (loan.isnull().sum() / len(loan.id)), 2)

    # Cloumns with more than 50% missing data to be removed
    columns_with_missing_values = list(missing[missing >= 50].index)
    # print(len(columns_with_missing_values))
    # There were 57 columns with more than 50% values with missing data

    loan = loan.drop(columns_with_missing_values, axis=1)
    # print(loan.shape)   #After removing such columns. 54 columns remain.

    loan = loan.drop('desc', axis=1)  # removing description column

    # below three columns are categorical with only 0 or nan. Hence removed
    drop_columnlist = ['collections_12_mths_ex_med', 'chargeoff_within_12_mths', 'tax_liens']
    loan = loan.drop(drop_columnlist, axis=1)

    # removing null records for important attributes
    loan = loan[~loan.pub_rec_bankruptcies.isnull()]

    # check for more missing data
    missing = round(100 * (loan.isnull().sum() / len(loan.id)), 2)
    # print(missing[missing != 0])

    # removing the missing values record from the dataset
    loan = loan[~loan.emp_title.isnull()]
    loan = loan[~loan.emp_length.isnull()]
    loan = loan[~loan.title.isnull()]
    loan = loan[~loan.revol_util.isnull()]
    loan = loan[~loan.last_pymnt_d.isnull()]

    # unique IDs and other features that are not required for analysis are dropped
    columns_tobe_dropped = ['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 'pymnt_plan', 'url', 'zip_code',
                            'initial_list_status', 'policy_code', 'application_type', 'acc_now_delinq', 'delinq_amnt', ]
    clean_loan = loan.drop(columns_tobe_dropped, axis=1)

    # cleaning the records for int_rate and revol_rate. Sometimes the record had 10.9% ... so we need to strip %
    clean_loan['int_rate'] = clean_loan['int_rate'].str.strip('%').astype('float')
    clean_loan['revol_util'] = clean_loan['revol_util'].str.strip('%').astype('float')

    # cleaning the emp_length feature because some records have data likes '<1' , '10 year'. Make it numeric
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

    #
    clean_loan['term'] = clean_loan.term.apply(lambda x: x.split()[0])

    # keep only the 'Fully Paid' and 'Charged Off' loan_status
    clean_loan = clean_loan[clean_loan['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    mapping_dict = {'loan_status': {'Fully Paid': 1, 'Charged Off': 0}}
    clean_loan = clean_loan.replace(mapping_dict)

    print("after feature engineering & after cleaning = " + clean_loan.shape.__str__())

    # trasformation of categorical data to binary
    headers = ["grade", "sub_grade", "home_ownership",
               "verification_status", "purpose"]
    #
    for col in headers:
        enc = LabelEncoder()
        enc.fit(clean_loan[col])
        enc_name_mapping = dict(zip(enc.classes_, enc.transform(enc.classes_)))

    # drop the loan status from X_data
    final_loan_X_train = clean_loan.drop('loan_status', axis=1).to_numpy()
    final_loan_y_train = clean_loan.loc[:, ["loan_status"]].to_numpy()


if __name__ == "__main__":
    run_process_data()
