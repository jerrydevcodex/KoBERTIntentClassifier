# modules/predict.py
import pandas as pd
import tensorflow as tf

def load_test_data(filepath):
    return pd.read_csv(filepath)

def predict_intent(model, tokenizer, texts, label_classes):
    inputs = tokenizer(texts, return_tensors='tf', padding=True, truncation=True, max_length=128)
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    predictions = tf.argmax(outputs.logits, axis=-1)
    predicted_labels = [label_classes[prediction] for prediction in predictions]
    return predicted_labels

def save_results(test_data, predicted_intents, output_filepath='result.csv'):
    result_df = test_data.copy()
    result_df['predicted_intent'] = predicted_intents
    result_df['match'] = (result_df['intent'] == result_df['predicted_intent']).astype(int)
    result_df.to_csv(output_filepath, index=False)
    return result_df
