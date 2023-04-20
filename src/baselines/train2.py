from train_test_framework import TrainTestFramework
from utils import get_parser
import importlib

parser = parser = get_parser()
args = parser.parse_args()

conf = importlib.import_module(args.config)

model_conf = conf.MODEL_CONF

model_args = {
    'model_type': 'cnn', 
    'model_name': model_conf['model_name'], 
    'train_data': 'Liar', 
    'test_data': 'Liar',
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

train_test = TrainTestFramework(**model_args)
# train_test.train(**train_args)
train_test.test(32)