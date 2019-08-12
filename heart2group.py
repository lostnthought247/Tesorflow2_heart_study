# import all the required python libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# create pandas dataframe from csv file
df = pd.read_csv("heart.csv")

# split the data into train, test, and validate groups
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size =0.2)


# Convert python dataframe into tensor dataset and identify target variable
def df_to_dataset(df, shuffle=True, batch_size=32):
    df = df.copy()
    labels = df.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size = len(df))
    ds = ds.batch(batch_size)
    return ds

# create a blank feature list to be populated with converted dataset columns
feature_columns = []

# numeric cols - converts simple numerical columns in dataframe to tf datasets
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))

# bucketized cols - categorizes age ranged and add output dataset to features
age = feature_column.numeric_column("age")
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols - categorized string column and add output dataset to features
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols -
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

# combines feature columns into a "Feature layer"
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# converts split dataframe into split tf datasets
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Builds the neural network model with 2 hidden layers and one output layer
model = tf.keras.Sequential([
feature_layer,
layers.Dense(128, activation="relu"),
layers.Dense(128, activation="relu"),
layers.Dense(1, activation="sigmoid"),
])

# compiles model using standard calibrations
model.compile(optimizer="adam",
            loss = "binary_crossentropy",
            metrics = ['accuracy'],
            run_eagerly=True)

# fits model to train and validation datasets, runs 5 passes through model (epochs)
model.fit(train_ds,
validation_data=val_ds,
epochs=5)

# Shows model's estimated accuracy
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
