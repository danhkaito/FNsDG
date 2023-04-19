import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
# import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import AutoTokenizer

from models import *
from data import FakeNewsDataset
import utils


parser = utils.get_parser()
args = parser.parse_args()

#-----------------------------------#
BATCH_SIZE = args.batch_size
EPOCH = args.epoch
LR = args.lr
EPS = args.eps

TRAIN_PATH = f'../../data/{args.dataset}/train.csv'
SAVE_PATH = f'../../saved/baselines/{args.model_name}_{args.model}_{args.dataset}.pt'
#-----------------------------------#

utils.set_seed(args.seed)

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


df = pd.read_csv(TRAIN_PATH)
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

print('Loading BERT tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Dataset
dataset = FakeNewsDataset(df, tokenizer, args.max_length)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = BATCH_SIZE # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = BATCH_SIZE # Evaluate with this batch size.
        )


if args.model == 'mlp':
    model = BertClassifier(args.model_name)
elif args.model == 'lstm':
    model = BertLSTM(args.model_name)
elif args.model == 'cnn':
    model = fakeBERT(args.model_name)
    
if args.freeze_pretrain is True:
    print("Freeze BERT")
    for param in model.bert.parameters():
        param.requires_grad = False


criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr= LR, eps=EPS)
total_steps = len(train_dataloader) * EPOCH

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                    num_warmup_steps = 0, # Default value in run_glue.py
                                    num_training_steps = total_steps)

model = model.to(device)
criterion = criterion.to(device)
early_stopping = utils.EarlyStopping(patience=10, mode='loss')


def train(epochs):
    training_stats = []
    print("=====================================")
    print(f"Train model: {args.model_name} + {args.model}")
    print(f'Use batch size: {BATCH_SIZE}')
    print(f'Use epoch: {EPOCH}') 
    
    for epoch_num in range(epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_num + 1, epochs))
        print('Training...')


        total_acc_train = 0
        total_loss_train = 0
        model.train()
        for train_input_ids, train_input_mask, train_label in tqdm(train_dataloader):
            train_input_ids = train_input_ids.to(device)
            train_input_mask = train_input_mask.to(device)
            train_label = train_label.to(device)

            output = model(train_input_ids, train_input_mask)
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()
            for val_input_ids, val_input_mask, val_label in validation_dataloader:

                val_input_ids = val_input_ids.to(device)
                val_input_mask = val_input_mask.to(device)
                val_label = val_label.to(device)

                output = model(val_input_ids, val_input_mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / train_size: .4f} \
            | Train Accuracy: {total_acc_train / train_size: .4f} \
            | Val Loss: {total_loss_val / val_size: .4f} \
            | Val Accuracy: {total_acc_val / val_size: .4f}')
        
        # Saving best model
        if early_stopping.early_stop(total_loss_val, model, SAVE_PATH):
                print(f"Early stopping at epoch: {epoch_num + 1}")
                break
        
        training_stats.append(
        {
        'epoch': epoch_num + 1,
        'Training Loss': total_loss_train / train_size,
        'Training Acc': total_acc_train / train_size,
        'Valid Loss': total_loss_val / val_size,
        'Valid Accur.': total_acc_val / val_size,
        # 'Training Time': training_time,
        # 'Validation Time': validation_time
        })
        
    return training_stats

training_stats = train(EPOCH)