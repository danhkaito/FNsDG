# Train test one
MODEL_TYPES = ['mlp', 'cnn', 'lstm']

DATA = ['covid3', 'liar_fnd_for', 'liar_fnd_for_covid3']

MODEL_CONF = {
   'seed': 40,
   'batch_size': 8,
   'use_early_stopping': False,
   'patience': 5,
   'freeze_pretrain': False,
   'model_name': 'distilroberta-base',
   'epoch': 10,
   'learning_rate': 2e-5,
   'eps': 1e-8,
   'max_length': 512,
   'cuda': True,
   'folder_data': "../../data",
   'folder_model':"../../saved/baselines",
   'save_model': False
}