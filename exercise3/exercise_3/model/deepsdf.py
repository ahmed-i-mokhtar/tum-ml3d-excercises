import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.w1 = nn.utils.weight_norm(nn.Linear(latent_size+3, 512))
        self.w2 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.w3 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.w4 = nn.utils.weight_norm(nn.Linear(512, 512-latent_size-3))
        self.w5 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.w6 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.w7 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.w8 = nn.utils.weight_norm(nn.Linear(512, 512))

        self.fc = nn.Linear(512,1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = self.dropout(self.relu(self.w1(x_in)))
        x = self.dropout(self.relu(self.w2(x)))
        x = self.dropout(self.relu(self.w3(x)))
        x = self.dropout(self.relu(self.w4(x)))
        
        x = torch.cat((x,x_in),dim=1)
        
        x = self.dropout(self.relu(self.w5(x)))
        x = self.dropout(self.relu(self.w6(x)))
        x = self.dropout(self.relu(self.w7(x)))
        x = self.dropout(self.relu(self.w8(x)))
        
        x = self.fc(x)
        return x
