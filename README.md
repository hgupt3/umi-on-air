
# UMI-on-Air

<p align="center">
  <img src="assets/bench.gif" alt="UMI-on-Air Demo" width="800"/>
</p>

This repository contains the simulation benchmark, environments, and evaluation tooling for **UMI-on-Air: Embodiment-Aware Guidance for Embodiment-Agnostic Visuomotor Policies**. 

For full details on the method and experimental results, see the [UMI-on-Air project website](https://umi-on-air.github.io/) and the [UMI-on-Air paper](https://umi-on-air.github.io/static/umi-on-air.pdf).

## Table of Contents

- [Installation](#installation)
- [Code Structure](#code-structure)
- [Available Tasks and Embodiments](#available-tasks-and-embodiments)
- [Collect Your Own Dataset](#collect-your-own-dataset)
- [Training Policies](#training-policies)
- [Policy Evaluation](#policy-evaluation)
- [Ablation Studies](#ablation-studies)
- [Contact](#contact)

## Quick Start

**To quickly evaluate pre-trained policies:**
1. Complete the [Installation](#installation) steps (including downloading pre-trained models)
2. Jump directly to [Policy Evaluation](#policy-evaluation) for usage examples

## Installation

### Conda Environment Setup
Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate flyingumi
```

### ACADOS Installation
ACADOS is required for the MPC trajectory controller. Follow these steps (one-time setup):

#### 1. Clone ACADOS Repository
```bash
cd am_mujoco_ws/am_trajectory_controller
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
```

#### 2. Build ACADOS C Library
```bash
mkdir -p build
cd build
cmake -DACADOS_INSTALL_DIR=.. ..
make -j$(nproc)
make install
cd ../../..
```

#### 3. Install ACADOS Python Interface
```bash
cd acados/interfaces/acados_template
pip install -e .
cd ../../../../../..
```

**Note:** On first run, ACADOS will prompt to automatically download the `tera_renderer` binary. Press `y` to agree.

### Setup ACADOS Environment

**Important:** You must source the ACADOS environment in each terminal session:
```bash
source am_mujoco_ws/am_trajectory_controller/setup_ee_mpc.sh
```

### Pre-trained Models

We provide pre-trained diffusion policy checkpoints trained on UMI demonstration data collected via motion capture for all four tasks (cabinet, peg, pick, valve). These policies can be directly evaluated on any embodiment using the `imitate_episodes.py` script with EADP guidance.

#### Download Pre-trained Checkpoints

```bash
# Download all pre-trained models
wget <DOWNLOAD_LINK_HERE>
tar -xzf pretrained_models.tar.gz -C checkpoints/
```

This will extract the checkpoints to:
```
checkpoints/
├── umi_cabinet/
├── umi_peg/
├── umi_pick/
└── umi_valve/
```

### Navigate to Working Directory

All evaluation and data collection scripts are in `am_mujoco_ws/policy_learning/`:
```bash
cd am_mujoco_ws/policy_learning
```

## Code Structure

```
.
├── am_mujoco_ws/
│   ├── am_trajectory_controller/     # MPC trajectory controller and configuration
│   ├── universal_manipulation_interface/  # Diffusion policy with EADP
│   ├── policy_learning/             # Simulation environments and evaluation scripts
│   │   ├── constants.py            # Task configs
│   │   ├── ee_sim_env.py          # Base environment classes
│   │   ├── imitate_episodes.py    # Policy evaluation script
│   │   └── run_ablation.py        # Ablation study runner
│   └── envs/assets/                # MuJoCo XML scene definitions
│       ├── hexa_scorpion_4dofarm_*.xml  # UAM scenes
│       ├── umi_*.xml              # UMI robot scenes  
│       ├── ur10e_umi_*.xml        # UR10e robot scenes
│       ├── meshes/                # 3D model files
│       └── textures/              # Texture files
└── data/                           # Datasets and results
```

## Available Tasks and Embodiments

### Task Naming Format: `EMBODIMENT_TASK`

### Embodiments

| Embodiment | Description | Constraints |
|------------|-------------|-------------|
| `umi` | Universal Manipulation Interface - handheld gripper (oracle) | Unconstrained, perfectly tracks desired trajectories |
| `ur10e` | UR10e robotic arm with UMI gripper | Fixed-base arm, highly capable tracking |
| `uam` | Unmanned Aerial Manipulator - hexarotor with 4-DoF scorpion arm | Constrained dynamics, cannot follow desired trajectories closely |

### Tasks

| Task | Description | Episode Length |
|------|-------------|----------------|
| `cabinet` | Open cabinet drawer, retrieve can, place on cabinet top | 50 s |
| `peg` | High-precision peg-in-hole insertion | 50 s |
| `pick` | Pick and place can from table on to the bowl | 50 s |
| `valve` | Rotate valve handle 180 degrees | 50 s |

## Collect Your Own Dataset

You can collect demonstration data using keyboard teleoperation with `record_episodes_keyboard.py`:

```bash
python record_episodes_keyboard.py \
    --task_name EMBODIMENT_TASK \
    [--onscreen_render | --use_3d_viewer] \
    [--disturb]
```

**Example:**
```bash
python record_episodes_keyboard.py \
    --task_name uam_cabinet \
    --onscreen_render
```

Episodes are saved as HDF5 files in `data/bc/<EMBODIMENT_TASK>/demonstration/` (e.g., `data/bc/uam_cabinet/demonstration/episode_0.hdf5`). The dataset directory can be configured in `constants.py`.

### Visualization Options

- **`--onscreen_render`**: Ego-centric camera view with keyboard controls for teleoperation
- **`--use_3d_viewer`**: Third-person 3D MuJoCo viewer for scene inspection
- **`--disturb`**: Enable wind disturbances for UAM embodiment

### Keyboard Controls (Teleoperation)
- `W/A/S/D`: Move horizontally
- `Space/Shift`: Move up/down
- `Q/E`: Close/open gripper
- `Arrow keys`: Rotate pitch/yaw
- `Z/C`: Roll left/right
- `P`: Start recording
- `R`: Reset scene
- `ESC`: Exit

## Training Policies

### Convert Demonstrations to Training Format

Convert recorded HDF5 episodes to UMI zarr format:

```bash
python ../universal_manipulation_interface/convert_hdf5_to_umi_zarr.py \
    --input_dir /path/to/episode/directory \
    --output_path dataset.zarr.zip \
    --camera_name ee \
    --image_size 224
```

### Train Diffusion Policy

The simulation runs at **50 Hz**. Configure your policy's query frequency:

```
query_frequency = action_horizon × obs_down_sample_steps
```

The policy will be called every `query_frequency` timesteps (each timestep = 1/50 second).

**Example:** If `action_horizon = 8` and `obs_down_sample_steps = 4`, then `query_frequency = 32` (policy runs every 0.64 seconds).

These parameters can be configured in the training config file at `am_mujoco_ws/universal_manipulation_interface/diffusion_policy/config/train_diffusion_unet_timm_umi_workspace.yaml`. Review and edit the config as needed before training.

#### Single-GPU Training

```bash
python ../universal_manipulation_interface/train.py \
    --config-name=train_diffusion_unet_timm_umi_workspace \
    task.dataset_path=dataset.zarr.zip
```
## Policy Evaluation

The `imitate_episodes.py` script evaluates trained UMI policies in simulation. It supports standard diffusion policies and Embodiment-Aware Diffusion Policy (EADP) with MPC guidance.

### Basic Usage

```bash
python imitate_episodes.py \
    --task_name EMBODIMENT_TASK \
    [--ckpt_dir <path_to_checkpoint_folder>] \
    --output_dir <results_folder> \
    --num_rollouts <number_of_trials> \
    [--guidance <guidance_strength>] \
    [--use_3d_viewer | --onscreen_render] \
    [--disturb]
```

**Note:** The `--ckpt_dir` argument is optional. If omitted, it will auto-detect based on task name (e.g., `uam_cabinet` → `checkpoints/umi_cabinet/`). This works seamlessly with the pre-trained models downloaded during installation.

### Guided Diffusion (EADP)

- **`--guidance`**: Scalar strength for MPC-based guidance (default: `0.0` for no guidance, `1.5` for our tested guidance)

### Keyboard Controls During Evaluation (on window)

- **`1`**: Discard current episode and retry
- **`2`**: Save current episode as failed and move to next
- **`SPACE`**: Pause/Resume simulation
- **`ESC`**: Exit program
- **`F`**: Toggle fullscreen (with `--onscreen_render`)

### Output Structure

Results are saved in the output directory:
```
output_dir/
├── episode_000/
│   ├── video.mp4              # Episode video
│   ├── metrics.json           # Episode metrics
│   └── tracking_errors.png    # Tracking error plots
├── episode_001/
│   └── ...
├── experiment_summary.json     # Aggregate statistics
├── experiment_summary.txt      # Human-readable summary
├── episode_metrics.csv         # All episodes in CSV format
└── summary_plots.png          # Visualization of results
```

## Ablation Studies

The `run_ablation.py` script runs parallel ablation sweeps over guidance parameters, reproducing the simulation studies from the UMI-on-Air paper.

### Usage

```bash
python run_ablation.py \
    --task_name EMBODIMENT_TASK \
    [--ckpt_dir /path/to/checkpoint] \
    --guidances 0.0,0.5,1.0,1.5 \
    --num_rollouts 30 \
    --max_workers 4 \
    [--output_dir /custom/output/path]
```

**Note:** The `--ckpt_dir` argument is optional and will auto-detect based on task name if not provided.

By default, this creates `<ckpt_dir>/ablation_results/` with one subfolder per guidance value. Use `--output_dir` to specify a custom output location. The script:
- Runs experiments in parallel with automatic retry on failure
- Provides live progress monitoring with ETAs
- Generates summary heatmaps showing success rate and episode duration

**Example output:**
```
ablation_results/
├── guidance0.0_s1/
├── guidance0.5_s1/
├── guidance1.0_s1/
├── ...
└── summary_heatmaps.png
```

## Contact

If you have any questions, feel free to reach out:

- 📧 Email: hgupt3@illinois.edu
