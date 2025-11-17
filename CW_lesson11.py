import pandas as pd      #робота з csv
import numpy as np
import tensorflow as tf      #нейронка
from tensorflow import keras      #розширення до ts
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder      #перетворює текст в числа
import matplotlib.pyplot as plt      #графіки

df = pd.read_csv("data/figures.csv")
print(df.head())

encoder = LabelEncoder()
df["label_enc"] = encoder.fit_transform(df["label"])
X = df[["area", "perimeter", "corners"]]
y = df["label_enc"]

model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(3,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(8, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X, y, epochs=300, verbose=0)

plt.plot(history.history['loss'], label = 'Втрати')
plt.plot(history.history['accuracy'], label = 'Точність')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title('Процес навчання моделі')
plt.legend()
plt.show()


test = np.array([25, 20, 0])
pred = model.predict(test)
print(pred)
print(encoder.inverse_transform(np.argmax(pred)))