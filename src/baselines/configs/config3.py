# Train test all
MODEL_TYPES = ['mlp', 'cnn', 'lstm']

DATA = ['Liar', 'FakeorReal', 'FND']

MODEL_CONF = {
   'seed': 40,
   'batch_size': 32,
   'use_early_stopping': False,
   'patience': 5,
   'freeze_pretrain': False,
   'model_name': 'bert-base-cased',
   'epoch': 10,
   'learning_rate': 1e-5,
   'eps': 1e-8,
   'max_length': 512,
   'cuda': True,
   'folder_data': "../../data",
   'folder_model':"../../saved/baselines",
   'save_model': False
}