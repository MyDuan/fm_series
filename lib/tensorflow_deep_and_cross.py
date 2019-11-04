import tensorflow as tf
from lib import build_data

train_file = './data/adult.data'
test_file = './data/adult.test'
df_train, df_test = build_data.build_census_income(train_file, test_file)

_HASH_BUCKET_SIZE = 1000


def input_fn(df):
    categorical_colums = ["workclass", "education", "marital_status", "occupation",
                          "relationship", "race", "gender", "native_country"]
    continuous_colums = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in continuous_colums}
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
                        for k in categorical_colums}
    feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))
    label = tf.constant(df['label'].values)
    return feature_cols, label


def train_input_fn():
    return input_fn(df_train)


def eval_input_fn():
    return input_fn(df_test)


def model():
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])
    workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=_HASH_BUCKET_SIZE)
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=_HASH_BUCKET_SIZE)

    columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.indicator_column(occupation)]


    model_dir = "./model"

    input_layer = tf.feature_column.input_layer(features=features, feature_columns=columns)
    '''
    estimator = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50]
    )
    '''
    return estimator
