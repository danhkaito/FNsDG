from train_test_framework import TrainTestFramework
from utils import get_parser
import importlib

import sys
import os

OUTPUT_FOLDER = './Log'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

parser = parser = get_parser()
args = parser.parse_args()

conf = importlib.import_module(args.config)

model_conf = conf.MODEL_CONF

model_args = {
    'model_type': 'cnn', 
    'model_name': model_conf['model_name'], 
    'train_data': 'Liar',
    'max_length': model_conf['max_length'],
    'folder_data': model_conf['folder_data'], 
    'folder_model': model_conf['folder_model'],
    'freeze_pretrain': model_conf['freeze_pretrain']
}

train_args = {
    'batch_size': model_conf['batch_size'], 
    'epoch': model_conf['epoch'], 
    'learning_rate': model_conf['learning_rate'], 
    'eps': model_conf['eps'], 
    'use_early_stopping': model_conf['use_early_stopping'], 
    'patience': model_conf['patience'],
    'cuda': model_conf['cuda']
}

# Erase file
open(f'{OUTPUT_FOLDER}/train.txt', 'w').close()
open(f'{OUTPUT_FOLDER}/test.txt', 'w').close()

for model_type in conf.MODEL_TYPES[1:2]:
    for data in conf.DATA[1:2]:
        
        model_args['train_data'] = data
        model_args['model_type'] = model_type
        train_args['epoch'] = 1
        
        train_test = TrainTestFramework(**model_args)
        
        with open(f'{OUTPUT_FOLDER}/train.txt', 'a') as f:
            sys.stdout = f
            
            print(f"\n\n#====================[{model_type}+{data}]======================#")
            train_test.train(**train_args)
        
        
        with open(f'{OUTPUT_FOLDER}/test.txt', 'a') as f:
            sys.stdout = f
            
            for test_data in conf.DATA:
                print(f"\n\n#==============[Train:{model_type}+{data}/Test:{test_data}]=============#")
                train_test.test(test_data, batch_size=32, load_from_disk=False)
        

# train_test = TrainTestFramework(**model_args)
# train_test.train(**train_args)
# train_test.test(32)