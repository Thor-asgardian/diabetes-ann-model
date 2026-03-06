# Diabetes Prediction using ANN

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load dataset
df = pd.read_csv("NBD.csv")

# Features and label
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Model
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(500, activation="relu", input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(50, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Prediction
sample = [[45, 63]]

sample = scaler.transform(sample)

prediction = model.predict(sample)

print("Prediction:", prediction)

if prediction[0][0] > 0.5:
    print("Diabetes detected")
else:
    print("No diabetes detected")