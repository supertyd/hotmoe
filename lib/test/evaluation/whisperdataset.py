import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
from PIL import Image
from pathlib import Path


class WhisperDataset(BaseDataset):

    def __init__(self, vos_mode=False):
        super().__init__()


        self.base_path = self.env_settings.hot2022_path

        self.sequence_list = self._get_sequence_list()

        self.vos_mode = vos_mode

        self.mask_path = None
        if self.vos_mode:
            self.mask_path = self.env_settings.got10k_mask_path

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = np.loadtxt(anno_path, dtype=np.float32)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frames_path_hsi = frames_path.replace("HSI-VIS-FalseColor", "HSI-VIS")
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frames_list_hsi = [frame for frame in os.listdir(frames_path_hsi) if frame.endswith(".png")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        frames_list_hsi.sort(key=lambda f: int(f[:-4]))
        frames_list_hsi = [os.path.join(frames_path_hsi, frame) for frame in frames_list_hsi]

        masks = None
        if self.vos_mode:
            seq_mask_path = '{}/{}'.format(self.mask_path, sequence_name)
            masks = [self._load_mask(Path(self._get_anno_frame_path(seq_mask_path, f[:-3] + 'png'))) for f in
                     frame_list[0:1]]

        return Sequence(sequence_name, frames_list, frames_list_hsi, 'whisper', ground_truth_rect.reshape(-1, 4))

    @staticmethod
    def _load_mask(path):
        if not path.exists():
            print('Error: Could not read: ', path, flush=True)
            return None
        im = np.array(Image.open(path))
        im = np.atleast_3d(im)[..., 0]
        return im

    def _get_anno_frame_path(self, seq_path, frame_name):
        return os.path.join(seq_path, frame_name)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open('{}/readme.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        return sequence_list
