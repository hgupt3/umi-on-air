import pathlib
import numpy as np

### Task parameters
DATA_DIR = '/home/harsh/flyingumi/data/bc/'
TYPE = 'ee'
STATE_DIM = 14
ACTION_DIM = 8

# Base task configs (shared properties across robot types)
TASK_CONFIGS = {
    'peg': {
        'episode_len': 2500,
        'camera_names': ['ee'],
    },
    'pick': {
        'episode_len': 3000,
        'camera_names': ['ee'],
    },
    'cabinet': {
        'episode_len': 3000,
        'camera_names': ['ee'],
    },
    'valve': {
        'episode_len': 2500,
        'camera_names': ['ee'],
    },
}

# Robot-specific task configs
SIM_TASK_CONFIGS = {
    'uam_peg': {
        **TASK_CONFIGS['peg'],
        'dataset_dir': DATA_DIR + '/uam_peg/demonstration/',
        'base_x_ub': 1.2,
    },
    'umi_peg': {
        **TASK_CONFIGS['peg'],
        'dataset_dir': DATA_DIR + '/umi_peg/demonstration/',
    },
    'ur10e_peg': {
        **TASK_CONFIGS['peg'],
        'dataset_dir': DATA_DIR + '/ur10e_peg/demonstration/',
    },
    
    'uam_pick': {
        **TASK_CONFIGS['pick'],
        'dataset_dir': DATA_DIR + '/uam_pick/demonstration/',
        'base_x_ub': 0.4,
    },
    'umi_pick': {
        **TASK_CONFIGS['pick'],
        'dataset_dir': DATA_DIR + '/umi_pick/demonstration/',
    },
    'ur10e_pick': {
        **TASK_CONFIGS['pick'],
        'dataset_dir': DATA_DIR + '/ur10e_pick/demonstration/',
    },
    
    'uam_cabinet': {
        **TASK_CONFIGS['cabinet'],
        'dataset_dir': DATA_DIR + '/uam_cabinet/demonstration/',
        'base_x_ub': -0.02,
    },
    'umi_cabinet': {
        **TASK_CONFIGS['cabinet'],
        'dataset_dir': DATA_DIR + '/umi_cabinet/demonstration/',
    },
    'ur10e_cabinet': {
        **TASK_CONFIGS['cabinet'],
        'dataset_dir': DATA_DIR + '/ur10e_cabinet/demonstration/',
    },
    
    'uam_valve': {
        **TASK_CONFIGS['valve'],
        'dataset_dir': DATA_DIR + '/uam_valve/demonstration/',
        'base_x_ub': 0.4,
    },
    'umi_valve': {
        **TASK_CONFIGS['valve'],
        'dataset_dir': DATA_DIR + '/umi_valve/demonstration/',
    },
    'ur10e_valve': {
        **TASK_CONFIGS['valve'],
        'dataset_dir': DATA_DIR + '/ur10e_valve/demonstration/',
    },
}

### Simulation envs fixed constants
DT = 0.02
START_UAM_POSE = [-0.5, 0.0, 1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
XML_DIR = str(pathlib.Path(__file__).parent.parent.resolve()) + '/envs/assets/'

# UAM Gripper helper functions
UAM_GRIPPER_POSITION_OPEN = 0.06
UAM_GRIPPER_POSITION_CLOSE = 0.001
UAM_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - UAM_GRIPPER_POSITION_CLOSE) / (UAM_GRIPPER_POSITION_OPEN - UAM_GRIPPER_POSITION_CLOSE)
UAM_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (UAM_GRIPPER_POSITION_OPEN - UAM_GRIPPER_POSITION_CLOSE) + UAM_GRIPPER_POSITION_CLOSE

# UMI Gripper constants (used by UMI Oracle, UR10e, and UAM robots)
UMI_GRIPPER_OPEN = 0.06
UMI_GRIPPER_CLOSE = 0.001