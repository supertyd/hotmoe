import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.hotmoe import HotMoETrack
from  lib.test.tracker.ostrack import OSTrack
import lib.test.parameter.hotmoe_hsi as rgbt_params
import multiprocessing
import torch
import time
from glob import glob

def X2Cube(img, B=[4, 4], skip = [4, 4], bandNumber=16):
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//B[0], N//B[1], bandNumber)

    hsi_min = np.min(DataCube, axis=(0,1), keepdims=True)
    hsi_max = np.max(DataCube, axis=(0,1), keepdims=True)
    hsi_normed = (DataCube - hsi_min) / ((hsi_max - hsi_min) + 1e-6) * 255
    hsi_normed = hsi_normed.astype(np.uint8)
    return hsi_normed

def genConfig(seq_path, set_type):

    if 'HOT23VAL' in set_type:
        RGB_img_list = sorted(glob(join(seq_path, '*.jpg')))
        HSI_img_list = sorted(glob(join(seq_path.replace('-FalseColor',''), '*.png')))
        try:
            RGB_gt = np.loadtxt(join(seq_path, 'groundtruth_rect.txt'), delimiter='\t')
        except:
            RGB_gt = np.loadtxt(join(seq_path, 'groundtruth_rect.txt'))
    elif 'HOT23TEST' in set_type:
        RGB_img_list = sorted(glob(join(seq_path, '*.jpg')))
        HSI_img_list = sorted(glob(join(seq_path.replace('-FalseColor',''), '*.png')))
        try:
            RGB_gt = np.loadtxt(join(seq_path, 'init_rect.txt'), delimiter='\t')
        except:
            RGB_gt = np.loadtxt(join(seq_path, 'init_rect.txt'))
    else:
        raise ValueError

    if RGB_gt.ndim == 1:
        RGB_gt  = RGB_gt[np.newaxis, :]

    return RGB_img_list, RGB_gt, HSI_img_list


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=2, debug=0, epoch=None):

    if seq_name.startswith("HSI-RedNIR"):
        B=[4, 4]
        skip = [4, 4]
        bandNumber=16
        train_data_type='rednir'  # RedNIR(15)
    elif seq_name.startswith("HSI-VIS"):
        B=[4, 4]
        skip = [4, 4]
        bandNumber=16
        train_data_type='vis'  # VIS(16)
    else:
        B=[5, 5]
        skip = [5, 5]
        bandNumber=25
        train_data_type='nir'  # NIR(25)

    seq_txt = seq_name.replace('/', '-')
    save_name = '{}'.format(yaml_name)
    save_path = f"./results/{dataset_name}/" + save_name + '/' + seq_txt + '.txt'
    save_folder = f"./results/{dataset_name}/" + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return

    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass


    params = rgbt_params.parameters(yaml_name, epoch)  # 'vitb_256_mae_ce_32x4_ep300'

    hotmoe = HotMoETrack(params)
    tracker = OSTrack_HOT(tracker=hotmoe)

    # ostrack = OSTrack(params)
    # tracker = OSTrack_HOT(tracker=ostrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+ seq_name +'——————————————')
    RGB_img_list, RGB_gt, HSI_img_list = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    for frame_idx, rgb_img in enumerate(RGB_img_list):
        tic = cv2.getTickCount()
        image = cv2.cvtColor(cv2.imread(rgb_img), cv2.COLOR_BGR2RGB)
        hsi_img = cv2.imread(HSI_img_list[frame_idx], -1)
        hsi_img_norm = X2Cube(hsi_img, B=B, skip=skip, bandNumber=bandNumber)
        if seq_name.startswith("HSI-RedNIR"):
            hsi_img_norm = hsi_img_norm[:, :, :-1]
        frame = np.concatenate((image, hsi_img_norm), axis=2)
        if frame_idx == 0:
            # initialization
            tracker.initialize(frame, RGB_gt[0].tolist())  # xywh
            tic = cv2.getTickCount()
        elif frame_idx > 0:
            # track
            region = tracker.track(frame, train_data_type)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()

    np.savetxt(save_path, result, fmt='%.2f', delimiter=',')
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))
    # except Exception as e:
    #     print(e)
    #     print(f"{seq_name} has probelm!")

class OSTrack_HOT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB, train_data_type):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB, train_data_type=train_data_type)
        pred_bbox = outputs['target_bbox']
        # pred_score = outputs['best_score']
        return pred_bbox


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on hsi dataset.')
    parser.add_argument('--yaml_name', type=str, default='deep_all', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='HOT23VAL', help='Name of dataset (HOT23VAL, HOT23TEST).')
    parser.add_argument('--threads', default=2, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--mode', default='parallel', type=str, help='running mode: [sequential , parallel]')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--epoch', default=60, type=int, help='to vis tracking results')
    parser.add_argument('--video', type=str, default='', help='Sequence name for debug.')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    cur_dir = abspath(dirname(__file__))
    ## path initialization
    seq_list = None
    list = []
    if dataset_name == 'HOT23VAL':
        seq_home = '/home/ubuntu/Downloads/challenge2023/datasets/validation'   # replace it with your path


        #seq1 = sorted([join('HSI-NIR-FalseColor', i) for i in os.listdir(join(seq_home, 'HSI-NIR-FalseColor')) if isdir(join(seq_home, 'HSI-NIR-FalseColor', i))])
        seq2 = sorted([join('HSI-VIS-FalseColor', i) for i in os.listdir(join(seq_home, 'HSI-VIS-FalseColor')) if isdir(join(seq_home, 'HSI-VIS-FalseColor', i))])
        #seq3 = sorted([join('HSI-RedNIR-FalseColor', i) for i in os.listdir(join(seq_home, 'HSI-RedNIR-FalseColor')) if isdir(join(seq_home, 'HSI-RedNIR-FalseColor', i))])
        #seq_list = seq1 + seq2 + seq3
        seq_list = seq2
        seq_list.sort()
    elif dataset_name == 'HOT23TEST':
        seq_home = '/your_path/ranking'
        seq1 = sorted([join('HSI-NIR-FalseColor', i) for i in os.listdir(join(seq_home, 'HSI-NIR-FalseColor')) if isdir(join(seq_home, 'HSI-NIR-FalseColor', i))])
        seq2 = sorted([join('HSI-VIS-FalseColor', i) for i in os.listdir(join(seq_home, 'HSI-VIS-FalseColor')) if isdir(join(seq_home, 'HSI-VIS-FalseColor', i))])
        seq3 = sorted([join('HSI-RedNIR-FalseColor', i) for i in os.listdir(join(seq_home, 'HSI-RedNIR-FalseColor')) if isdir(join(seq_home, 'HSI-RedNIR-FalseColor', i))])
        seq_list = seq1 + seq2 + seq3
        seq_list.sort()
    else:
        raise ValueError("Error dataset!")

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.debug, args.epoch) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.debug, args.epoch) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")

