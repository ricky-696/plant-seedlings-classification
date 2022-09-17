import load_data
from .model.vgg16_pre_trained import VGG16

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torchvision

ans = []
device_num = 0
label = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
        'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 
        'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

data_img, img_name = load_data.read_test_img((224, 224), "./test")
dataset = load_data.TestDataset(data_img)
loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
with torch.cuda.device(device_num):
    model = VGG16(num_classes = 12).cuda()
    model.load_state_dict(torch.load('./model/VGG16.pt'))
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device_num)
            output = model(data)
            res = output.argmax(dim = 1, keepdim = True)
            ans.append(label[res])

dict = {'file' : img_name, 'species' : ans}
df = pd.DataFrame(dict)
df.to_csv('predict.csv', index = False)