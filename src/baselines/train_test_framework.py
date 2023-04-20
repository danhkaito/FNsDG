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

class TrainTestFramework:
    def __init__(self, model_type, model_name, train_data, max_length, folder_data, folder_model, freeze_pretrain = False):
        
        if freeze_pretrain:
            t = 'freezed'
        else: t = 'unfreezed'
        
        self.model_name = model_name
        self.train_data = train_data
        self.model_type = model_type
        self.max_length = max_length
        self.folder_data = folder_data
        self.freeze_pretrain = freeze_pretrain
        
        self.train_data_path = f'{folder_data}/{train_data}/train.csv'
        self.model_path = f'{folder_model}/{model_name}_{model_type}_{train_data}_{t}.pt'
        
        if model_type == 'mlp':
            self.model = BertClassifier(model_name)
        elif model_type == 'lstm':
            self.model = BertLSTM(model_name)
        elif model_type == 'cnn':
            self.model = fakeBERT(model_name)
            
        if freeze_pretrain is True:
            print("Freeze pretrained model")
            for param in self.model.bert.parameters():
                param.requires_grad = False
    
    
    def prepare_data(self, path, batch_size ,train_split_rate = 1): # Return dataloader(s)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        df = pd.read_csv(path)
        dataset = FakeNewsDataset(df, tokenizer, self.max_length)
        
        train_size = int(train_split_rate * len(dataset))
        val_size = len(dataset) - train_size
        
        # Testing
        if train_split_rate == 1:
            print('{:>5,} Testing samples'.format(train_size))
            dataloader = DataLoader(
                        dataset,  # The training samples.
                        sampler = SequentialSampler(dataset), # Select batches sequencely
                        batch_size = batch_size # Trains with this batch size.
                    )
            return dataloader, train_size
        
        # Training
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
        
        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))
        
        return train_dataloader, validation_dataloader, train_size, val_size
    
    def train(self, batch_size, epoch, learning_rate, eps, use_early_stopping = True, patience = 5, cuda = True):
        
        train_dataloader, validation_dataloader, train_size, val_size = self.prepare_data(self.train_data_path, batch_size, 0.8)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.model.parameters(), lr= learning_rate, eps=eps)
        total_steps = len(train_dataloader) * epoch

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

        device = utils.get_device(cuda)
        model = self.model.to(device)
        criterion = criterion.to(device)
        
        if use_early_stopping:
            early_stopping = utils.EarlyStopping(patience=patience, mode='loss')
    
        # Begin Training
        training_stats = []
        print("")
        print(f"Train model: {self.model_name} + {self.model_type}")
        print(f"Freezed pretrain: {self.freeze_pretrain}")
        print(f"Train dataset: {self.train_data}")
        print(f'Use batch size: {batch_size}')
        print(f'Use epoch: {epoch}') 
        print(f'Use learning_rate: {learning_rate}') 
        print(f'Use eps: {eps}') 
        print(f'Use early stopping: {use_early_stopping}') 
        print(f'Use patience: {patience}') 
        print(f'Use device: {device}') 
        
        for epoch_num in range(epoch):

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_num + 1, epoch))
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
            if use_early_stopping:
                if early_stopping.early_stop(total_loss_val, model, self.model_path):
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
        
        if not use_early_stopping:
            print("Saving model...")
            torch.save(model.state_dict(), self.model_path)
            
        return training_stats
    
    def test(self, test_data, batch_size=32, load_from_disk = True, cuda = True):
        
        test_data_path = f'{self.folder_data}/{test_data}/test.csv'
        test_dataloader, test_size = self.prepare_data(test_data_path, batch_size)
        
        if load_from_disk:
            self.model.load_state_dict(torch.load(self.model_path))
        
        device = utils.get_device(cuda)
        model = self.model.to(device)
        
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
        print('')
        print(f"Train model: {self.model_name} + {self.model_type}")
        print(f"Freezed pretrain: {self.freeze_pretrain}")
        print(f"Train dataset: {self.train_data}")
        print(f"Test dataset: {test_data}")
        print('')
        
        pred_prob = torch.cat(pred_prob, axis=0)
        true_labels = torch.cat(true_labels, axis=0)
        
        utils.print_metrics(true_labels, pred_prob)
        
        
        
            
            
            


            
            
            