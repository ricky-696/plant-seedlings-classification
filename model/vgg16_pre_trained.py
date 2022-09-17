import torch.nn as nn
import torch
from torchvision import models
class VGG16(nn.Module):
    def __init__(self, num_classes = 12):
        super(VGG16, self).__init__()
        net = models.vgg16(pretrained=True)   #從預訓練模型加載VGG16網絡參數
        net.classifier = nn.Sequential()	#將分類層置空，下面將改變我們的分類層
        self.features = net		#保留VGG16的特徵層
        self.classifier = nn.Sequential(    #定義自己的分類層
                nn.Linear(512 * 7 * 7, 512),  #512 * 7 * 7不能改變 ，由VGG16網絡決定的，第二個參數爲神經元個數可以微調
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = VGG16(12)
    print(model)