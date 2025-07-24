import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob

# Define action labels
actions = ["kick", "punch", "idle"]

# Constants
sequence_length = 10
expected_features = 99  # 33 landmarks × (x, y, z)

# Data containers
data = []
labels = []

# Process each action
for action in actions:
    file_pattern = f"action_data/{action}_*.csv"
    csv_files = glob.glob(file_pattern)

    if len(csv_files) == 0:
        print(f"❌ No files found for action: {action}")
        continue

    for file_path in csv_files:
        df = pd.read_csv(file_path, header=None)

        if df.shape[1] != expected_features:
            print(f"⚠ Skipping {file_path}: expected 99 columns, got {df.shape[1]}")
            continue

        try:
            sequence_data = df.values.astype(np.float32)
            num_frames = sequence_data.shape[0]

            if num_frames < sequence_length:
                print(f"⚠ Skipping {file_path}: not enough frames ({num_frames})")
                continue

            # Create sliding windows
            for start in range(0, num_frames - sequence_length + 1, 5):  # step=5 for overlap
                window = sequence_data[start:start + sequence_length]  # shape (10, 99)
                data.append(window)
                labels.append(action)

            print(f"✅ Processed {file_path} with {len(data)} sequences")

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

# Final checks
if len(data) == 0:
    print("❌ No valid data found. Check your CSV files.")
    exit()

# Convert to NumPy arrays
X = np.array(data, dtype=np.float32)  # shape: (samples, 10, 99)
y = np.array(labels)

print(f"✅ Final X shape: {X.shape}, Number of labels: {len(y)}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(sequence_length, expected_features)))
model.add(Dense(32, activation="relu"))
model.add(Dense(len(actions), activation="softmax"))

# Compile and train
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=16)

# Save model
model.save("action_model.h5")
print("✅ Model trained and saved as action_model.h5!")