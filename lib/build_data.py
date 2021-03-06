import numpy as np
import pandas as pd


def build_sin():
    x_train_data = np.arange(0, 2 * np.pi, 0.01).reshape([629, 1])
    y_train_data = np.sin(x_train_data).reshape([629, 1])
    return x_train_data, y_train_data


def build_line():
    x_train_data = np.arange(0, 2 * np.pi, 0.01).reshape([629, 1])
    y_train_data = (5 * x_train_data + 6).reshape([629, 1])
    return x_train_data, y_train_data


def build_iris(path):
    data = open(path)
    label_list = []
    features_list = []
    for line in data.readlines():
        feature_label = line.strip().split(',')
        label_list.append(int(feature_label[-1]))
        features = list(map(lambda x: float(x), feature_label[0:-1]))
        features_list.append(features)

    features_train = np.array(features_list).reshape([100, 4])
    label_train = np.array(label_list).reshape([100, 1])
    return label_train, features_train


def build_iris_with_field(path):
    data = open(path)
    label_list = []
    features_list = []

    for line in data.readlines():
        features = [0] * 8
        feature_label = line.strip().split(',')
        label_list.append(int(feature_label[-1]))
        for i in range(len(feature_label[0:-1])):
            if features[int(float(feature_label[i]))] and int(float(feature_label[i])) == 7:
                features[int(float(feature_label[i])) + 1] = 1
            else:
                features[int(float(feature_label[i]))] = 1

        features_list.append(features)
    features_train = np.array(features_list).reshape([100, 8])
    label_train = np.array(label_list).reshape([100, 1])
    return label_train, features_train


def build_iris_deep_fm(path):
    data = open(path)
    label_list = []
    features_list = []
    feild_list = []
    for line in data.readlines():
        feature_label = line.strip().split(',')
        label_list.append(int(feature_label[-1]))
        features = list(map(lambda x: float(x), feature_label[0:-1]))
        features_list.append(features)
        feild_list.append([0, 1, 2, 3])

    features_train = np.array(features_list).reshape([100, 4])
    feild_of_train = np.array(feild_list).reshape([100, 4])
    label_train = np.array(label_list).reshape([100, 1])
    return label_train, features_train, feild_of_train


def build_census_income(train_file, test_file):

    feature_colums = ["age", "workclass", "fnlwgt", "education", "education_num",
                      "marital_status", "occupation", "relationship", "race", "gender",
                      "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
    df_train = pd.read_csv(train_file, names=feature_colums, skipinitialspace=True)
    df_test = pd.read_csv(test_file, names=feature_colums, skipinitialspace=True, skiprows=1)
    df_train['label'] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    df_test['label'] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
    return df_train, df_test
