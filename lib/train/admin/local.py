class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = ''    # Directory for tensorboard files.
        self.pretrained_networks = ''
        self.got10k_val_dir = ''
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_lmdb_dir = ''
        self.coco_dir = ''
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.depthtrack_dir = ''
        self.lasher_dir = ''
        self.visevent_dir = ''
        self.hsi_dir = "/home/ubuntu/Downloads/challenge2023/datasets"
