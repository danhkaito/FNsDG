from train_test_framework import TrainTestFramework
from utils import get_parser, set_seed
import importlib

import sys
import os

#==============[SETING ENV PARAM]===============#
sys.path.append("./configs")

parser = parser = get_parser()
args = parser.parse_args()

print(f"Using config file: {args.config}.py")

conf = importlib.import_module(args.config)

model_conf = conf.MODEL_CONF
#===============================================#


#==============[SETING MODEL ARGS]===============#
model_args = {
    'model_type': 'cnn', 
    'model_name': model_conf['model_name'], 
    'train_data': 'Liar',
    'max_length': model_conf['max_length'],
    'folder_data': model_conf['folder_data'], 
    'folder_model': model_conf['folder_model'] + f'/{args.config}',
    'freeze_pretrain': model_conf['freeze_pretrain']
}

train_args = {
    'batch_size': model_conf['batch_size'], 
    'epoch': model_conf['epoch'], 
    'learning_rate': model_conf['learning_rate'], 
    'eps': model_conf['eps'], 
    'use_early_stopping': model_conf['use_early_stopping'], 
    'patience': model_conf['patience'],
    'cuda': model_conf['cuda'],
    'save_model': model_conf['save_model']
}
#=================================================#


# ============== Set seed ===============
set_seed(model_conf['seed'])

LOG_FOLDER = model_args['folder_model'] + '/Log'

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
    
TRAIN_LOG = f'{LOG_FOLDER}/train.txt'
TEST_LOG = f'{LOG_FOLDER}/test.txt'

# Erase file
open(TRAIN_LOG, 'w').close()
open(TEST_LOG, 'w').close()

for model_type in conf.MODEL_TYPES:
    for data in conf.DATA:
        
        model_args['train_data'] = data
        model_args['model_type'] = model_type
        # train_args['epoch'] = 1
        
        train_test = TrainTestFramework(**model_args)
        
        with open(TRAIN_LOG, 'a') as f:
            sys.stdout = f
            
            print(f"\n\n#====================[{model_type}+{data}]======================#")
            train_test.train(**train_args)
        
        
        with open(TEST_LOG, 'a') as f:
            sys.stdout = f
            
            for test_data in conf.DATA:
                print(f"\n\n#==============[Train:{model_type}+{data}/Test:{test_data}]=============#")
                train_test.test(test_data, batch_size=32, load_from_disk=False)
        

# train_test = TrainTestFramework(**model_args)
# train_test.train(**train_args)
# train_test.test(32)