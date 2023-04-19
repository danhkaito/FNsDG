import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, RGCNConv, GatedGraphConv, GATv2Conv, GINConv
from transformers import BertModel


class FNsGCN(torch.nn.Module):
    def __init__(self, hidden_channels_1, hidden_channels_2, num_feature_concat, num_content_feature, num_style_feature, num_classes):
        super(FNsGCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        self.conv1 = SAGEConv(num_content_feature, hidden_channels_1)
        self.conv2 = SAGEConv(hidden_channels_1, hidden_channels_2)
        
        self.out = nn.Linear(hidden_channels_2, num_classes)

    def forward(self, x_content, edge_index):
        
        # x_content_enc = self.post_enc(x_content)
        # x_style_content = self.style_enc(x_style)
        # x = torch.cat((x_content_enc, x_style_content),1)
        # # First Message Passing Layer (Transformation)
        x = self.conv1(x_content, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv4(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv 3(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = self.conv3(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        x = self.out(x)
        return x

