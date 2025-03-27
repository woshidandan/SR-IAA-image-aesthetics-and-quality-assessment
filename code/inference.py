import os
import torch
import torchvision

import numpy as np
import cv2
from tqdm import tqdm

from option.config import Config
from model.backbone import resnet50_backbone
from model.model_main import IAARegression


# configuration
config = Config({
    # device
    'gpu_id': "0",  
    'num_workers': 0,

    # data
    'db_name': 'AADB-M',                                     
    'db_path': '/fast_dataset/AADB/datasetImages_originalSize/',    
    'txt_file_name': './IAA_list/aadb_mini_train.txt',            
    'train_size': 0.8,                                        
    'scenes': 'all',
    'batch_size': 2,
    'patch_size': 32,

    # ViT structure
    'n_enc_seq': 32*24 + 12*9 + 7*5,        
    'n_layer': 14,                         
    'd_hidn': 384,                          
    'i_pad': 0,
    'd_ff': 384,                            
    'd_MLP_head': 1152,                     
    'n_head': 6,                           
    'd_head': 384,                         
    'dropout': 0.1,                        
    'emb_dropout': 0.1,                     
    'layer_norm_epsilon': 1e-12,
    'n_output': 1,                          
    'Grid': 10,                            

    # load & save checkpoint
    'snap_path': './IAA_weights/Tmp',        
    'checkpoint': './IAA_weights/AADB-20230419-16_13_00-epoch1-100/epoch80.pth',
})

# device setting
config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU %s' % config.gpu_id)
else:
    print('Using CPU')

# input normalize
class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
    def __call__(self, sample):
        sample[:, :, 0] = (sample[:, :, 0] - self.mean[0]) / self.var[0]
        sample[:, :, 1] = (sample[:, :, 1] - self.mean[1]) / self.var[1]
        sample[:, :, 2] = (sample[:, :, 2] - self.mean[2]) / self.var[2]
        return sample

# data selection
if config.db_name == 'AVA':
    from data.koniq import IQADataset
if config.db_name == 'AADB':
    from data.koniq import IQADataset
if config.db_name == 'PARA':
    from data.koniq import IQADataset
if config.db_name == 'SPAQ':
    from data.koniq import IQADataset

# numpy array -> torch tensor
class ToTensor(object):
    def __call__(self, sample):
        sample = np.transpose(sample, (2, 0, 1))
        sample = torch.from_numpy(sample)
        return sample

# create model
model_backbone = resnet50_backbone().to(config.device)
model_transformer = IAARegression(config).to(config.device)

# load weights
checkpoint = torch.load(config.checkpoint)
model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
model_backbone.eval()
model_transformer.eval()

# input transform
transforms = torchvision.transforms.Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensor()])

filenames = os.listdir(config.dirname)
filenames.sort()
f = open(config.result_score_txt, 'w')

# input mask (batch_size x len_sqe+1)
mask_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)

# inference
for filename in tqdm(filenames):
    d_img_name = os.path.join(config.dirname, filename)
    ext = os.path.splitext(d_img_name)[-1]
    if ext == '.jpg':
        # multi-scale feature extraction
        d_img_org = cv2.imread(d_img_name)
        d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2RGB)
        d_img_org = np.array(d_img_org).astype('float32') / 255
        d_img_org = cv2.resize(d_img_org, dsize=(1024, 768), interpolation=cv2.INTER_CUBIC)

        h, w, c = d_img_org.shape
        d_img_scale_1 = cv2.resize(d_img_org, dsize=(config.scale_1, int(h*(config.scale_1/w))), interpolation=cv2.INTER_CUBIC)
        d_img_scale_2 = cv2.resize(d_img_org, dsize=(config.scale_2, int(h*(config.scale_2/w))), interpolation=cv2.INTER_CUBIC)
        d_img_scale_2 = d_img_scale_2[:160, :, :]

        d_img_org = transforms(d_img_org)
        d_img_org = torch.tensor(d_img_org.to(config.device)).unsqueeze(0)
        d_img_scale_1 = transforms(d_img_scale_1)
        d_img_scale_1 = torch.tensor(d_img_scale_1.to(config.device)).unsqueeze(0)
        d_img_scale_2 = transforms(d_img_scale_2)
        d_img_scale_2 = torch.tensor(d_img_scale_2.to(config.device)).unsqueeze(0)

        feat_dis_org = model_backbone(d_img_org)
        feat_dis_scale_1 = model_backbone(d_img_scale_1)
        feat_dis_scale_2 = model_backbone(d_img_scale_2)

        # quality prediction
        pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2)

        # result save
        line = '%s\t%f\n' % (filename, float(pred.item()))
        f.write(line)
f.close()
        






