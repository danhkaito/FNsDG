# Train test one
MODEL_TYPES = ['mlp', 'lstm']

DATA = ['medical']

MODEL_CONF = {
   'seed': 40,
   'batch_size': 16,
   'use_early_stopping': False,
   'patience': 5,
   'freeze_pretrain': False,
   'model_name': 'vinai/phobert-base',
   'epoch': 10,
   'learning_rate': 2e-5,
   'eps': 1e-8,
   'max_length': 256,
   'cuda': True,
   'folder_data': "../../data",
   'folder_model':"../../saved/baselines",
   'save_model': False
}