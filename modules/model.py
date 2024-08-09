# modules/model.py
from transformers import BertTokenizer, TFBertForSequenceClassification

def load_model_and_tokenizer(num_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)
    return model, tokenizer

def save_model_and_tokenizer(model, tokenizer, label_encoder, directory='intent_classifier_model'):
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    np.save(f'{directory}/label_classes.npy', label_encoder.classes_)
