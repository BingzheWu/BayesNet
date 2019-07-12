import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import KFold


def patient_ids_to_item_files(dataroot, patient_ids, mode='train'):
    items = []
    if mode=='train':
        membership = 1
    else:
        membership = 0
    for id in patient_ids:
        data_root = os.path.join(dataroot, 'brust', id)
        for label in [0,1]:
            image_path = os.path.join(data_root, str(label), "*.png")
            print(image_path)
            image_file = glob.glob(image_path)
            image_lable_pair = [(path, label, membership) for path in image_file]
            items += image_lable_pair
    return items


def _test_id2item():
    dataroot = '/home/bingzhe/dataset/dp/'
    patient_id_file = os.path.join(dataroot, 'cases_train.txt')
    patient_ids = open(patient_id_file, 'r').readlines()
    patient_ids = [id.strip() for id in patient_ids]
    items = patient_ids_to_item_files(dataroot, patient_ids)
    print(items)
    save_test_file = '__pycache__/items.txt'
    with open(save_test_file, 'w') as f:
        for image_file, label, membership in items:
            f.writelines(image_file+','+str(label)+','+str(membership)+'\n')


def write_items_into_file(items, save_file_path):
    with open(save_file_path, 'w') as f:
        for image_file, label, membership in items:
            f.writelines(image_file+','+str(label)+','+str(membership)+'\n')


def kfold_patient_ids(dataroot, patient_ids, n_splits=2):
    kf = KFold(n_splits=n_splits)
    save_dir = os.path.join(dataroot, 'kfold'+str(n_splits))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for idx, (train_id_idxs, test_id_idxs) in enumerate(kf.split(patient_ids)):
        train_ids = [patient_ids[id_idx] for id_idx in train_id_idxs]
        test_ids = [patient_ids[id_idx] for id_idx in test_id_idxs]
        train_items = patient_ids_to_item_files(dataroot, train_ids, mode='train')
        test_items = patient_ids_to_item_files(dataroot, test_ids, mode='test')
        save_train = os.path.join(save_dir, 'fold_'+str(idx)+'_train.txt')
        save_test = os.path.join(save_dir, 'fold_' + str(idx) + '_test.txt')
        write_items_into_file(train_items, save_train)
        write_items_into_file(test_items, save_test)


def _test_kfold():
    dataroot = '/home/bingzhe/dataset/dp/'
    patient_id_file = os.path.join(dataroot, 'cases_train.txt')
    patient_ids = open(patient_id_file, 'r').readlines()
    patient_ids = [id.strip() for id in patient_ids]
    kfold_patient_ids(dataroot, patient_ids)


if __name__ == '__main__':
    _test_kfold()