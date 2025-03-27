import torch
import numpy as np
    

class RandHorizontalFlip(object):
    def __call__(self, sample):
        # r_img: H x W x C (numpy)
        d_img_org = sample['d_img_org']
        d_img_ref_1 = sample['d_img_ref_1']
        d_img_ref_2 = sample['d_img_ref_2']
        score = sample['score']

        prob_lr = np.random.random()
        # np.fliplr needs HxWxC
        if prob_lr > 0.5:
            d_img_org = np.fliplr(d_img_org).copy()
            d_img_ref_1 = np.fliplr(d_img_ref_1).copy()
            d_img_ref_2 = np.fliplr(d_img_ref_2).copy()


        sample = {'d_img_org': d_img_org, 'd_img_ref_1': d_img_ref_1, 'd_img_ref_2': d_img_ref_2, 'score': score}
        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        d_img_org = sample['d_img_org']
        d_img_ref_1 = sample['d_img_ref_1']
        d_img_ref_2 = sample['d_img_ref_2']
        score = sample['score']

        d_img_org[:, :, 0] = (d_img_org[:, :, 0] - self.mean[0]) / self.var[0]
        d_img_org[:, :, 1] = (d_img_org[:, :, 1] - self.mean[1]) / self.var[1]
        d_img_org[:, :, 2] = (d_img_org[:, :, 2] - self.mean[2]) / self.var[2]

        d_img_ref_1[:, :, 0] = (d_img_ref_1[:, :, 0] - self.mean[0]) / self.var[0]
        d_img_ref_1[:, :, 1] = (d_img_ref_1[:, :, 1] - self.mean[1]) / self.var[1]
        d_img_ref_1[:, :, 2] = (d_img_ref_1[:, :, 2] - self.mean[2]) / self.var[2]
        
        d_img_ref_2[:, :, 0] = (d_img_ref_2[:, :, 0] - self.mean[0]) / self.var[0]
        d_img_ref_2[:, :, 1] = (d_img_ref_2[:, :, 1] - self.mean[1]) / self.var[1]
        d_img_ref_2[:, :, 2] = (d_img_ref_2[:, :, 2] - self.mean[2]) / self.var[2]

        sample = {'d_img_org': d_img_org, 'd_img_ref_1': d_img_ref_1, 'd_img_ref_2': d_img_ref_2, 'score': score}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        d_img_org = sample['d_img_org']
        d_img_ref_1 = sample['d_img_ref_1']
        d_img_ref_2 = sample['d_img_ref_2']
        score = sample['score']

        d_img_org = np.transpose(d_img_org, (2, 0, 1))
        d_img_org = torch.from_numpy(d_img_org)

        d_img_ref_1 = np.transpose(d_img_ref_1, (2, 0, 1))
        d_img_ref_1 = torch.from_numpy(d_img_ref_1)

        d_img_ref_2 = np.transpose(d_img_ref_2, (2, 0, 1))
        d_img_ref_2 = torch.from_numpy(d_img_ref_2)

        score = torch.from_numpy(score)

        sample = {'d_img_org': d_img_org, 'd_img_ref_1': d_img_ref_1, 'd_img_ref_2': d_img_ref_2, 'score': score}
        return sample


def RandShuffle(config):
    train_size = config.train_size

    if config.scenes == 'all':
        if config.db_name == 'AVA':
            scenes = list(range(229951))
        elif config.db_name == 'AADB':
            scenes = list(range(8959))
        elif config.db_name == 'PARA':
            scenes = list(range(27219))
        elif config.db_name == 'SPAQ':
            scenes = list(range(11126))
        elif config.db_name == 'AADB-M':
            scenes = list(range(101))
    else:
        scenes = config.scenes
        
    n_scenes = len(scenes)
    n_train_scenes = int(np.floor(n_scenes * train_size))
    n_test_scenes = n_scenes - n_train_scenes

    seed = np.random.random()
    random_seed = int(seed*10)
    np.random.seed(random_seed)
    np.random.shuffle(scenes)
    train_scene_list = scenes[:n_train_scenes]
    test_scene_list = scenes[n_test_scenes:]
    
    return train_scene_list, test_scene_list