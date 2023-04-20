MODEL_TYPES = ['mlp', 'cnn', 'lstm']

DATA = ['Liar', 'FakeorReal', 'FND']

MODEL_CONF = {
   'seed': 40,
   'batch_size': 1,
   'use_early_stopping': True,
   'patience': 5,
   'freeze_pretrain': False,
   'model_name': 'bert-base-cased',
   'epoch': 10,
   'learning_rate': 2e-5,
   'eps': 1e-8,
   'max_length': 512,
   'cuda': True,
   'folder_data': "../../data",
   'folder_model':"../../saved/baselines"
}