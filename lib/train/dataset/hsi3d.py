import os
import os.path
import torch
import numpy as np
import pandas
from glob import glob
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import cv2

class HSI3D(BaseVideoDataset):
    """ HSI dataset.

    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().hsi_dir if root is None else root
        if 'train' in  split:
            root = os.path.join(root, 'training')
        elif 'val' in  split:
            root = os.path.join(root, 'validation')
        else:
            raise ValueError('Unknown split name.')
        super().__init__('HSI3D', root, image_loader)

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train_vis':
                file_path = os.path.join(ltr_path, 'data_specs', 'hsi_vis_train_split.txt')
            elif split == "train_vis_2023":
                file_path = os.path.join(ltr_path, 'data_specs', 'hsi_vis_train_split.txt')
            elif split == 'val_vis':
                file_path = os.path.join(ltr_path, 'data_specs', 'hsi_vis_val_split.txt')
            elif split == 'train_nir':
                file_path = os.path.join(ltr_path, 'data_specs', 'hsi_nir_train_split.txt')
            elif split == 'val_nir':
                file_path = os.path.join(ltr_path, 'data_specs', 'hsi_nir_val_split.txt')
            elif split == 'train_rednir':
                file_path = os.path.join(ltr_path, 'data_specs', 'hsi_rednir_train_split.txt')
            elif split == 'val_rednir':
                file_path = os.path.join(ltr_path, 'data_specs', 'hsi_rednir_val_split.txt')
            elif split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'hsi_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'hsi_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list


    def get_name(self):
        return 'hsi3d'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        try:
            gt = np.loadtxt(bb_anno_file, delimiter='\t', dtype=np.float32)
        except:
            gt = np.loadtxt(bb_anno_file, dtype=np.float32)

        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name_rgb = self.sequence_list[seq_id]
        seq_path_rgb = os.path.join(self.root, seq_name_rgb)
        return seq_path_rgb

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_rgb_hsi_frame_path(self, seq_path, frame_id):
        seq_paths = sorted(glob(os.path.join(seq_path, '*.jpg')))
        seq_paths_rgb = seq_paths[frame_id]
        seq_paths_hsi = seq_paths_rgb.replace("-FalseColor", "").replace("jpg", "png")
        return (seq_paths_rgb, seq_paths_hsi)

    def X2Cube(self,img, B=[4, 4], skip=[4, 4], bandNumber=16):
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
        DataCube = out.reshape(M // B[0], N // B[1], bandNumber)
        return DataCube




    def _get_frame(self, seq_path, frame_id):
        rgb_hsi_frame_path = self._get_rgb_hsi_frame_path(seq_path, frame_id)
        rgb = self.image_loader(rgb_hsi_frame_path[0])
        hsi = cv2.imread(rgb_hsi_frame_path[1],cv2.IMREAD_UNCHANGED)

        #hsi = np.load(rgb_hsi_frame_path[1])  # np.uint16 h,w,16
        hsi_min = np.min(hsi, axis=(0,1), keepdims=True)
        hsi_max = np.max(hsi, axis=(0,1), keepdims=True)
        hsi_normed = (hsi - hsi_min) / ((hsi_max - hsi_min) + 1e-6) * 255
        hsi_normed = hsi_normed.astype(np.uint8)
        

        if 'HSI-RedNIR-FalseColor' in rgb_hsi_frame_path[0]:  #15
            hsi_normed = self.X2Cube(hsi_normed,B=[4, 4], skip=[4, 4], bandNumber=16)
            frame = np.concatenate((rgb, hsi_normed[:,:,:-1]), axis=2)
        elif 'HSI-VIS-FalseColor' in rgb_hsi_frame_path[0]:  #16
            hsi_normed = self.X2Cube(hsi_normed, B=[4, 4], skip=[4, 4], bandNumber=16)
            frame = np.concatenate((rgb, hsi_normed), axis=2)
        elif 'HSI-NIR-FalseColor' in rgb_hsi_frame_path[0]:  #25
            hsi_normed = self.X2Cube(hsi_normed, B=[5, 5], skip=[5, 5], bandNumber=25)
            frame = np.concatenate((rgb, hsi_normed), axis=2)
        else:
            raise ValueError
        return frame

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'seq_belong_mask':
                continue
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

