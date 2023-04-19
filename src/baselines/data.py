import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class FakeNewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        
        self.df = df
        self.label_encoder = LabelEncoder()
        
        self.texts = self.df['Full text'].values
        self.labels = self.encode_label(self.df['Label'].values)
    
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, index):
        input_ids, attention_mask = self.tokenize_text(self.texts[index])
        label = self.labels[index]
        return input_ids, attention_mask, label
    
    def __len__(self):
        return len(self.df)
    
    def encode_label(self, labels):
        return self.label_encoder.fit_transform(labels)
    
    def decode_label(self, encoded_labels):
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def tokenize_text(self, text):
        encoded_dict = self.tokenizer.encode_plus(
                text,                      # Sentence to encode.
                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                max_length = self.max_length,         # Pad & truncate all sentences.
                truncation = True, 
                padding = 'max_length',
                return_attention_mask = True,   # Construct attn. masks.
                return_tensors = 'pt',     # Return pytorch tensors.
            )
        return encoded_dict['input_ids'][0], encoded_dict['attention_mask'][0]
        
    
    