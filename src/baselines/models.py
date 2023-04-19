import torch
import torch.nn as nn
from transformers import AutoModel

class BertClassifier(nn.Module):

    def __init__(self, name_model, num_class=2,  dropout=0.2):

        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(name_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_class)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)

        return linear_output
    

class fakeBERT(nn.Module):

    def __init__(self, name_model, num_class = 2,  dropout=0.2):

        super(fakeBERT, self).__init__()

        # BERT embed
        self.bert = AutoModel.from_pretrained(name_model)
        
        # parallel Conv1D
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(768, 128, 3), # in_chanel = 768 (get_all), out_chanel = 128 (num_filter)
            nn.ReLU(),
            nn.MaxPool1d(5)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(768, 128, 4), # in_chanel = 768 (get_all), out_chanel = 128 (num_filter)
            nn.ReLU(),
            nn.MaxPool1d(5)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(768, 128, 5), # in_chanel = 768 (get_all), out_chanel = 128 (num_filter)
            nn.ReLU(),
            nn.MaxPool1d(5)
        )
        
        # After concatenate
        self.conv_block_4_5 = nn.Sequential(
            nn.Conv1d(128, 128, 5), # in_chanel = 128 (get_all), out_chanel = 128 (num_filter)
            nn.ReLU(),
            nn.MaxPool1d(5),
            
            nn.Conv1d(128, 128, 5), # in_chanel = 128 (get_all), out_chanel = 128 (num_filter)
            nn.ReLU(),
            nn.MaxPool1d(30)
        )
        
        self.dense_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.Dropout(dropout),
            nn.Linear(128, num_class)
        )
        
        

    def forward(self, input_id, mask):

        #embed size: [bacth_size, token_length, 768(num_feature)]
        embed, _ = self.bert(input_ids = input_id, attention_mask = mask, return_dict=False)
        
        # Convert embedsize to [bacth_size, 768, token_length] <=> (N, Cin, L) of conv1D
        embed = torch.transpose(embed,1,2).contiguous()
        
        # Parallel conv1D
        out_1 = self.conv_block_1(embed)
        out_2 = self.conv_block_2(embed)
        out_3 = self.conv_block_3(embed)
        
        out_cat = torch.cat((out_1, out_2, out_3), 2)
        
        # After concatenate
        out_4_5 = self.conv_block_4_5(out_cat)
        
        # # Convert to [batch_size, token_length, num_feature] to flatten
        out_4_5 = torch.transpose(out_4_5,1,2).contiguous()
        
        out = self.dense_block(out_4_5)

        return out
    

class BertLSTM(nn.Module):
    def __init__(self , name_model,  num_class = 2, embedding_dim= 768, hidden_dim= 128, n_layers = 1, dropout=0.2):
        super(BertLSTM, self).__init__()
        
        self.bert = AutoModel.from_pretrained(name_model)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, 128),
            nn.Dropout(dropout),
            nn.Linear(128, num_class)
        )
        
    def forward(self, input_id, mask):
        embeds, _ = self.bert(input_ids = input_id, attention_mask = mask, return_dict=False)
        
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out[:,-1]

        out = self.classifier(lstm_out)

        return out