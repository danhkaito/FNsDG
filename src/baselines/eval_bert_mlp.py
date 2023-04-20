import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from  model.model import BertClassifier
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# import matplotlib.pyplot as plt
from tqdm import tqdm
# import seaborn as sns
import utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from early_stoping import EarlyStopping

FOLDER = '../BERT_fine_tune'

parser = utils.get_parser()

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

if args.cuda:    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model = BertClassifier(args.name_model, args.num_class, args.dropout)
model.load_state_dict(torch.load(f"{FOLDER}/{args.dataset}.pt"))
model = model.to(device)

tokenizer = BertTokenizer.from_pretrained(args.name_model)

def evaluate_prediction():
    df = pd.read_csv(f"../clean data/{args.dataset}/test.csv")
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    texts=df['Full text']
    label=df['Label']

    labelEncoder=LabelEncoder()
    encoded_label=labelEncoder.fit_transform(label)
    y=np.reshape(encoded_label,(-1,1))
    labels = y[:,0]
    sentences = texts.values


    # Load the BERT tokenizer.
    input_ids = []
    attention_masks = []
    tokens = dict()
    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = args.token_length,         # Pad & truncate all sentences.
                        truncation = True, 
                        padding = 'max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'][0])
        # print(encoded_dict['input_ids'].shape)
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'][0])

    # Convert the lists into tensors.
    tokens['input_ids'] = torch.stack(input_ids)
    tokens['attention_mask'] = torch.stack(attention_masks)

    labels = torch.from_numpy(labels)


    # Combine the training inputs into a TensorDataset.
    test_dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'],  labels)

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    test_dataloader = DataLoader(
                test_dataset,  # The training samples.
                sampler = SequentialSampler(test_dataset), # Select batches randomly
                batch_size = args.batch_size # Trains with this batch size.
            )
    predictions , true_labels = [], []
    model.eval()
    for batch in test_dataloader:
    # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            output = model(b_input_ids, b_input_mask)

        

        # Move logits and labels to CPU
        output = output.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(np.argmax(output, axis=1).flatten())
        true_labels.append(label_ids.flatten())

    print('DONE.')
    pred_labels = (np.concatenate(predictions, axis=0))
    true_labels = (np.concatenate(true_labels, axis=0))
    
    print(f"Accuracy: {accuracy_score(true_labels, pred_labels)}\n \
          Precsion, Recall, F1-Score Label 1: {precision_recall_fscore_support(true_labels, pred_labels, average='binary', pos_label = 1)}\n\
          Precsion, Recall, F1-Score Label 0: {precision_recall_fscore_support(true_labels, pred_labels, average='binary', pos_label = 0)}")

evaluate_prediction()