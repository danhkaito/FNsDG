import torch
import random
import numpy as np
import argparse
import os

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, mode = 'acc'): # mode is 'acc' or 'loss'
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        if mode == 'acc':
            self.value = -np.inf
        else:
            self.value = np.inf

    def early_stop(self, new_value, model, path):
        if (self.mode == 'acc' and self.value < new_value) or (self.mode == 'loss' and self.value > new_value):
            self.value = new_value
            self.counter = 0
            
            print("Saving best model...")
            torch.save(model.state_dict(), path)
        
        elif (self.mode == 'acc' and self.value - new_value > self.min_delta) or (self.mode == 'loss' and new_value - self.value> self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
        

def set_seed(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True,
                    help='Enable CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='Liar')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument('--model', type=str, default='mlp', help='Use: mlp, lstm, cnn')
    parser.add_argument('--batch_size', type=int, default=32, help='number of size per batch')
    parser.add_argument('--preload', type=bool, default=False, help='Preload data')
    
    # Use early stopping?
    parser.add_argument('--early_stopping', type=bool, default=True, help='Use early stopping?')
    parser.add_argument('--patience', type=int, default=5, help='Patience for suing early stoping')
    parser.add_argument('--freeze_pretrain', type=bool, default=False, help='Use freeze pretrain')

    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs to train.')
    
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eps', type=float, default=1e-8)

    # parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--max_length', type=int, default=512)