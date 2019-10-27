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
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
    race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other",
                                                       "White"])
    education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=_HASH_BUCKET_SIZE)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=_HASH_BUCKET_SIZE)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=_HASH_BUCKET_SIZE)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=_HASH_BUCKET_SIZE)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=_HASH_BUCKET_SIZE)

    age = tf.contrib.layers.real_valued_column("age")
    age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    education_num = tf.contrib.layers.real_valued_column("education_num")
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

    base_columns = [
        gender, race, native_country, education, occupation, workclass, relationship, age_buckets
    ]

    crossed_columns = [
        tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=_HASH_BUCKET_SIZE),
        tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=_HASH_BUCKET_SIZE),
        tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=_HASH_BUCKET_SIZE)
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        tf.contrib.layers.embedding_column(workclass, dimension=8),
        tf.contrib.layers.embedding_column(education, dimension=8),
        tf.contrib.layers.embedding_column(gender, dimension=8),
        tf.contrib.layers.embedding_column(relationship, dimension=8),
        tf.contrib.layers.embedding_column(native_country, dimension=8),
        tf.contrib.layers.embedding_column(occupation, dimension=8),
        age, education_num, capital_gain, capital_loss, hours_per_week]

    model_dir = "./model"
    model = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
    return model
