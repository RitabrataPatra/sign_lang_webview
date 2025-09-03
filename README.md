# Real-Time Sign Language Recognition using a Transformer Model

This project is a complete system for real-time sign language recognition using a webcam. It leverages a state-of-the-art Transformer architecture to translate common sign language gestures into text, demonstrating a modern approach to sequence-based machine learning tasks.

### Google-Colab Live Code Link

[Open in Google Colab](https://colab.research.google.com/drive/1-gAgzVn9pHJw68kWEJmmUeG_U8Azfvs9?usp=sharing)

---

## 🚀 Project Overview

The primary goal of this project is to build a proof-of-concept application that can accurately identify a predefined set of sign language gestures (`hello`, `thanks`, `iloveyou`) from a live video feed. The system is built from the ground up, covering the full machine learning pipeline: data collection, model training, and real-time inference.

A key focus of this project is not just building a functional model, but also demonstrating and understanding a critical challenge in machine learning: **overfitting**. The project is designed to show the dramatic difference in performance when a model is trained on insufficient versus adequate data.

---

## Core Technologies

The system is powered by two main components: the "eyes" that see the gesture and the "brain" that understands it.

| Component   | Technology             | Role                                                                                   |
|-------------|-----------------------|----------------------------------------------------------------------------------------|
| The "Eyes"  | Google's MediaPipe    | Analyzes the webcam feed to extract a detailed 3D skeleton of 1662 landmark points from the body, face, and hands. It transforms raw video into structured, numerical data. |
| The "Brain" | Transformer Model (TensorFlow/Keras) | A state-of-the-art neural network that processes the sequence of landmark data. Its Self-Attention Mechanism allows it to analyze the entire gesture at once, focusing on the most important movements to understand the full context. |

---

## 🔬 Key Challenge Demonstrated: Overfitting

This project serves as a practical experiment in **overfitting**, one of the most fundamental challenges in AI.

A powerful model trained on a tiny dataset doesn't learn the general patterns of a sign; it simply memorizes the few examples it has seen. We call this the "Lazy Student" problem.

- **Small Dataset (3 video clips per sign):** The model overfits severely. It might learn a lazy shortcut (e.g., a specific shadow in the background of the "hello" videos) and incorrectly predict "hello" for every sign.
- **Larger Dataset (15 video clips per sign):** The model is forced to ignore random coincidences and learn the actual pattern of the gesture itself, leading to accurate predictions.

> **Visualization:**  
> The "Large Dataset" shows training and validation accuracies rising together, indicating good learning, while the "Small Dataset" shows a large gap, indicating memorization.

---

## 📂 Project Structure

The project is organized into several key Python scripts:

- `mediapipe_utils.py`: Helper functions to process webcam frames with MediaPipe and extract landmark data.
- `model.py`: Contains the custom `TransformerBlock` and `PositionalEncoding` layers, defining the architecture of the AI model.
- `train.py`: Loads the collected data, trains the Transformer model, and saves the final weights to an `.h5` file.
- `detect_live.py`: Loads the trained model and runs real-time detection using the webcam.

---

## ⚙️ How to Run This Project

Follow these steps to set up and run the project on your local machine.

### Step 1: Clone the Repository

```sh
git clone <your-repository-url>
cd <your-repository-name>
```

### Step 2: Set Up the Environment

It is highly recommended to use a virtual environment. The project is confirmed to work with Python 3.9 and the following library versions.

```sh
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# .\venv\Scripts\activate  # On Windows

# Install the required packages
pip install tensorflow==2.10.0 numpy==1.26.4 mediapipe==0.10.9 opencv-python==4.8.0.76 scikit-learn
```

### Step 3: Collect Data

Run the data collection script. This will use your webcam to record 15 video sequences for each of the three signs and save the landmark data into a newly created `MP_Data` folder.

```sh
python data_collection.py
```

This will guide you through recording the data for `hello`, `thanks`, and `iloveyou`.

### Step 4: Train the Model

Once your data is collected, run the training script. This will load the data, build the Transformer model, and train it for 200 epochs. The final model will be saved as `action_transformer.h5`.

```sh
python train.py
```

### Step 5: Run the Live Demo

With the model trained, you can now run the live detection application.

```sh
python detect_live.py
```

This will open your webcam and display the real-time predictions at the top of the screen.

---

## 📜 The Code

Here is a look at the core components of the project's code.

<details>
<summary><strong>model.py (The Transformer Architecture)</strong></summary>

```python
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "position": self.pos_encoding.shape[1],
            "d_model": self.pos_encoding.shape[2],
        })
        return config

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model,
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.att.key_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate,
        })
        return config

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_model(num_actions, sequence_length=30, input_dim=1662, embed_dim=128, num_heads=8, ff_dim=128):
    inputs = Input(shape=(sequence_length, input_dim))
    x = Dense(embed_dim)(inputs)
    x = PositionalEncoding(sequence_length, embed_dim)(x)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_actions, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
    )
    return model
```
</details>

<details>
<summary><strong>train.py (The Training Script)</strong></summary>

```python
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from model import create_model

# --- Load Data ---
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 15
sequence_length = 30
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

print("Loading data...")
for action in actions:
    for sequence in range(no_sequences):
        sequence_path = os.path.join(DATA_PATH, action, str(sequence))
        if os.path.exists(sequence_path) and len(os.listdir(sequence_path)) == sequence_length:
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(sequence_path, f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"Warning: Skipping incomplete sequence {sequence} for action '{action}'")

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# --- Build and Train the Model ---
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = create_model(len(actions))
model.summary()

print("\nStarting model training...")
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))
print("Training complete!")

# --- Save the Model ---
model.save('action_transformer.h5')
print("Model saved