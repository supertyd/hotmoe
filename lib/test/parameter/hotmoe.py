from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.hotmoe.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/hotmoe/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(prj_dir,"/media/ubuntu/b47114be-f454-4377-bf84-0e81da2a42bc/hotmoe/lib/train/output/checkpoints/train/hotmoe/deep_all/HotMoE_ep0003.pth.tar")
    # params.checkpoint = os.path.join(prj_dir, "./models/ViPT_%s.pth"%yaml_name)
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
