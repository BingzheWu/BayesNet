import sys
import os
import torch
import torchvision
import torch.nn.functional as F
from mmcv import Config
sys.path.append('.')
from attack.utils import load_model
from dataset.dataset_utils import pil_loader


def whbox_attack(attack_cfg):
    image_file_list = attack_cfg['image_file_list']
    save_dir = os.path.join(attack_cfg['experimental_dir'], 'attack_results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_file = os.path.join(save_dir, 'critic_scores')
    d_net = load_model(attack_cfg)
    trans = torchvision.transforms
    image_size = attack_cfg['imageSize']
    transform_val = trans.Compose([trans.Resize((image_size, image_size)), trans.ToTensor()])
    save_results = open(save_file, 'w')
    img_size = attack_cfg['imageSize']
    with open(image_file_list, 'r') as f:
        for data in f.readlines():
            image_file, label, membership = data.split(',')
            image = pil_loader(image_file)
            image = transform_val(image)
            image = image.view(1,3,img_size,img_size)
            label = int(label.strip())
            tensor=torch.ones(1)
            label = tensor.new_tensor([label], dtype=torch.int64)
            pred = d_net(image).detach()
            loss = F.cross_entropy(pred, label, reduction='sum')
            loss = loss.numpy()
            save_results.writelines(image_file+','+str(label)+','+str(loss)+','+str(membership)+'\n')


if __name__=='__main__':
    cfg_file = sys.argv[1]
    attack_cfg = Config.fromfile(cfg_file)
    whbox_attack(attack_cfg)