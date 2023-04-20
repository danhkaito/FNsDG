import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
# import matplotlib.pyplot as plt
from tqdm import tqdm
# import seaborn as sns
import utils
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

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

TEST_PATH = f'../../data/{args.test}/test.csv'
SAVE_PATH = f'../../saved/baselines/{args.model_name}_{args.model}_{args.train}.pt'
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


df = pd.read_csv(TEST_PATH)
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

print('Loading BERT tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Dataset
test_dataset = FakeNewsDataset(df, tokenizer, args.max_length)
test_size = int(len(test_dataset))

print('{:>5,} testing samples'.format(test_size))

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
test_dataloader = DataLoader(
            test_dataset,  # The training samples.
            sampler = SequentialSampler(test_dataset), # Select batches randomly
            batch_size = BATCH_SIZE # Trains with this batch size.
        )

if args.model == 'mlp':
    model = BertClassifier(args.model_name)
elif args.model == 'lstm':
    model = BertLSTM(args.model_name)
elif args.model == 'cnn':
    model = fakeBERT(args.model_name)

model.load_state_dict(torch.load(SAVE_PATH))

model = model.to(device)


def test():
    pred_prob, true_labels = [], []
    model.eval()
    for test_input_ids, test_input_mask, test_label in tqdm(test_dataloader):
        
        test_input_ids = test_input_ids.to(device)
        test_input_mask = test_input_mask.to(device)
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            output = model(test_input_ids, test_input_mask)

        # Move logits and labels to CPU
        output = output.detach().cpu()
        label_ids = test_label
        
        # Store predictions and true labels
        pred_prob.append(output)
        # predictions.append(np.argmax(output, axis=1).flatten())
        true_labels.append(label_ids)

    print('DONE.')
    pred_prob = torch.cat(pred_prob, axis=0)
    true_labels = torch.cat(true_labels, axis=0)
    
    utils.print_metrics(true_labels, pred_prob)

    # print(f"Accuracy: {accuracy_score(true_labels, pred_labels)}\n \
    #     Precsion, Recall, F1-Score Label 1: {precision_recall_fscore_support(true_labels, pred_labels, average='binary', pos_label = 1)}\n\
    #     Precsion, Recall, F1-Score Label 0: {precision_recall_fscore_support(true_labels, pred_labels, average='binary', pos_label = 0)}")
    
test()