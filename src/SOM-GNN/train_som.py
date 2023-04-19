import math
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import numpy as np
import torch
from tqdm import tqdm
from utils import *
from conf.parse_config import get_parser, parse_args
from model.SOM.utils import *
from model.SOM.som import Som
from model.SOM.vizual_som import SOMVisualize
import os

parser = get_parser()
arg = parser.parse_args()

args = parse_args(parser, arg.conf)
args = args[0]

set_seed(args.seed)

print(args)

train_X = np.load(f"../../saved/embeddings/{args.train}/{args.embedding_model}/train_embedding.npy")
train_Y = np.load(f"../../saved/embeddings/{args.train}/{args.embedding_model}/train_label.npy")
bs = args.batch_size_som

dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y))
sampler = RandomSampler(dataset)
data = DataLoader(dataset, sampler=sampler, batch_size=bs)
rad = math.floor(math.sqrt(5*math.sqrt(train_X.shape[0])))

som = Som((rad,rad,train_X.shape[1]), lr=args.lr_som)

device = torch.device(args.device)

epochs = args.epoch_som
som = som.to(device)
losses = []
for ep in range(epochs):
    decay = math.exp(-ep/(epochs/math.log(rad)))
    loss = 0
    print(f"Epoch {ep+1}/{epochs}\n")
    for batch in tqdm(data):
        batch = batch[0].to(device)
        bmus = som(batch)
        som.backward()
        som.adjust_hyperparam(decay, decay)
        loss += mean_quantization_err(bmus, som).cpu().item()
    
    loss/= len(data)
    losses.append({"Loss": loss, "Learning_rate": som.lr, "dRadius": som.dRadius,
                    'epoch': ep})

som_viz = SOMVisualize(som)
som_viz.viz_loss(losses)

FOLDER = f"../saved/SOM-GNN/model/SOM/{args.train}/{args.embedding_model}/"
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

if args.save_som:
    torch.save(som, f"{FOLDER}som_{epochs}_{bs}.pt")