import torch
import torch.nn as nn
# https://arxiv.org/pdf/1409.1556.pdf
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        #block1
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, padding = 1, kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, padding = 1, kernel_size = 3, stride = 1)
        #block2
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, padding = 1, kernel_size = 3, stride = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, padding = 1, kernel_size = 3, stride = 1)
        #block3
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, padding = 1, kernel_size = 3, stride = 1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, padding = 1, kernel_size = 3, stride = 1)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 256, padding = 1, kernel_size = 1, stride = 1)
        #block4
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 512, padding = 1, kernel_size = 3, stride = 1)
        self.conv9 = nn.Conv2d(in_channels = 512, out_channels = 512, padding = 1, kernel_size = 3, stride = 1)
        self.conv10 = nn.Conv2d(in_channels = 512, out_channels = 512, padding = 1, kernel_size = 1, stride = 1)
        #block5
        self.conv11 = nn.Conv2d(in_channels = 512, out_channels = 512, padding = 1, kernel_size = 3, stride = 1)
        self.conv12 = nn.Conv2d(in_channels = 512, out_channels = 512, padding = 1, kernel_size = 3, stride = 1)
        self.conv13 = nn.Conv2d(in_channels = 512, out_channels = 512, padding = 1, kernel_size = 1, stride = 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        #block1

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        #block2
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)

        #block3
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool(x)

        #block4
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.pool(x)

        #block5
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1) #flatten
        x = self.classifier(x)
        return x

#if __name__ == "__main__":
