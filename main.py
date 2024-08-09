# main.py
import numpy as np
import tensorflow as tf
from modules.data_processing import load_and_preprocess_data
from modules.model import load_model_and_tokenizer, save_model_and_tokenizer
from modules.train import prepare_datasets, encode_texts, train_model
from modules.predict import load_test_data, predict_intent, save_results

# 데이터 로드 및 전처리
train_data = load_and_preprocess_data('data/train_data.csv')

# 모델 및 토크나이저 로드
num_labels = train_data['intent'].nunique()
model, tokenizer = load_model_and_tokenizer(num_labels)

# 데이터셋 준비
X_train, X_val, y_train, y_val, label_encoder = prepare_datasets(train_data)
train_encodings = encode_texts(tokenizer, X_train)
val_encodings = encode_texts(tokenizer, X_val)
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_val))

# 모델 학습
train_model(model, train_dataset, val_dataset)

# 모델 및 토크나이저 저장
save_model_and_tokenizer(model, tokenizer, label_encoder)

# 테스트 데이터 로드 및 예측
test_data = load_test_data('data/test_data.csv')
predicted_intents = predict_intent(model, tokenizer, test_data['text'].tolist(), label_encoder.classes_)

# 결과 저장 및 확인
result_df = save_results(test_data, predicted_intents)
print(result_df.head())
