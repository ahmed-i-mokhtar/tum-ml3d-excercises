import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels=2, out_channels=self.num_features, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(in_channels=self.num_features, out_channels=self.num_features*2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(self.num_features * 2)
        self.conv3 = nn.Conv3d(in_channels=self.num_features*2, out_channels=self.num_features*4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(self.num_features * 4)
        self.conv4 = nn.Conv3d(in_channels=self.num_features*4, out_channels=self.num_features*8, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(self.num_features * 8)
        # TODO: 2 Bottleneck layers
        self.bottleneck = nn.Sequential(
            nn.Linear(self.num_features*8,self.num_features*8),
            nn.ReLU(),
            nn.Linear(self.num_features*8,self.num_features*8),
            nn.ReLU()
        )
        # TODO: 4 Decoder layers
        self.decoder1 = nn.ConvTranspose3d(in_channels=self.num_features * 2 * 8, out_channels=self.num_features * 4, kernel_size=4, stride=1, padding=0)
        self.dbn1 = nn.BatchNorm3d(self.num_features * 4)
        self.decoder2 = nn.ConvTranspose3d(in_channels=self.num_features * 4 * 2, out_channels=self.num_features * 2, kernel_size=4, stride=2, padding=1)
        self.dbn2 = nn.BatchNorm3d(self.num_features * 2)
        self.decoder3 = nn.ConvTranspose3d(in_channels=self.num_features * 2 * 2, out_channels=self.num_features, kernel_size=4, stride=2, padding=1)
        self.dbn3 = nn.BatchNorm3d(self.num_features)
        self.decoder4 = nn.ConvTranspose3d(in_channels=self.num_features * 2, out_channels=1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        x1 = self.leaky_relu(self.conv1(x))
        x2 = self.leaky_relu(self.bn1(self.conv2(x1)))
        x3 = self.leaky_relu(self.bn2(self.conv3(x2)))
        x4 = self.leaky_relu(self.bn3(self.conv4(x3)))
        # Reshape and apply bottleneck layers
        x = x4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        x = self.relu(self.dbn1(self.decoder1(torch.cat((x,x4),dim=1)))) # Might need to change the dim later
        x = self.relu(self.dbn2(self.decoder2(torch.cat((x,x3),dim=1))))
        x = self.relu(self.dbn3(self.decoder3(torch.cat((x,x2),dim=1))))
        x = self.decoder4(torch.cat((x,x1),dim=1))
        
        x = torch.squeeze(x, dim=1)
        # TODO: Log scaling
        x = torch.log(torch.abs(x) + 1)
        return x
