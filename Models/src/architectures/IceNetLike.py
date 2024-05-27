import torch
from torch import nn

class IceNetLike(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super().__init__()
        self.architecture_name = 'IceNetLike'

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024)
        )

        self.upconv_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        self.upconv_2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        self.upconv_3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.upconv_4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_block_9 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv_block_10 = nn.Conv2d(128, 1, kernel_size=1)

        
    def forward(self, x):
        x1 = self.conv_block_1(x)
        x = self.pool_1(x1)

        x2 = self.conv_block_2(x)
        x = self.pool_2(x2)

        x3 = self.conv_block_3(x)
        x = self.pool_3(x3)

        x4 = self.conv_block_4(x)
        x = self.pool_4(x4)

        x = self.conv_block_5(x)

        x = self.upconv_1(x)
        x = torch.cat([x4, x], dim=1)
        x = self.conv_block_6(x)

        x = self.upconv_2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_block_7(x)

        x = self.upconv_3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_block_8(x)

        x = self.upconv_4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_block_9(x)

        x = self.conv_block_10(x)

        return x