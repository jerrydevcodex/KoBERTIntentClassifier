# modules/train.py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

def prepare_datasets(data):
    label_encoder = LabelEncoder()
    data['encoded_intent'] = label_encoder.fit_transform(data['intent'])

    X_train, X_val, y_train, y_val = train_test_split(data['text'], data['encoded_intent'], test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val, label_encoder

def encode_texts(tokenizer, texts, max_len=128):
    return tokenizer(
        list(texts),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

def train_model(model, train_dataset, val_dataset, epochs=3):
    model.compile(
        optimizer=Adam(learning_rate=2e-5),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()]
    )

    model.fit(
        train_dataset.shuffle(100).batch(16),
        validation_data=val_dataset.batch(16),
        epochs=epochs
    )
