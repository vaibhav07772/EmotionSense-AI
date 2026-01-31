import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("fer2013.csv")

# Prepare data
pixels = data['pixels'].tolist()
faces = []

for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(48, 48)
    faces.append(face)

faces = np.asarray(faces)
faces = faces / 255.0
faces = faces.reshape(-1, 48, 48, 1)

emotions = to_categorical(data['emotion'], num_classes=7)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    faces, emotions, test_size=0.2, random_state=42
)

# Build CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Save model
model.save("emotion_model.h5")

print("✅ Emotion model trained and saved as emotion_model.h5")