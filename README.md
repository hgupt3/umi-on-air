
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
wget https://huggingface.co/LeCAR-Lab/umi-on-air_checkpoints/resolve/main/checkpoints.tar.gz
tar -xzf checkpoints.tar.gz
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

**Note:** These scripts require a display/GUI environment to run. They will not work on headless servers (SSH without X11 forwarding, cloud instances without display) even in "headless" mode, as MuJoCo needs to render camera images for the vision-based policies. There are likely workarounds for CLI-only usage - if you implement a solution, please consider submitting a pull request!

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
| `cabinet` | Open cabinet drawer, retrieve can, place on cabinet top | 60 s |
| `peg` | High-precision peg-in-hole insertion | 50 s |
| `pick` | Pick and place can from table on to the bowl | 60 s |
| `valve` | Rotate valve handle 180 degrees | 70 s |

## Collect Your Own Dataset

You can collect demonstration data using keyboard teleoperation.

### Usage

```bash
python record_episodes_keyboard.py \
    --task_name EMBODIMENT_TASK \
    [--onscreen_render | --use_3d_viewer] \
    [--disturb]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task_name` | *required* | Task in format `EMBODIMENT_TASK` (e.g., `uam_cabinet`) |
| `--onscreen_render` | disabled | Ego-centric camera view for teleoperation |
| `--use_3d_viewer` | disabled | Third-person 3D MuJoCo viewer for scene inspection |
| `--disturb` | disabled | Enable wind disturbances for UAM embodiment |

### Example

```bash
python record_episodes_keyboard.py \
    --task_name uam_cabinet \
    --onscreen_render
```

Episodes are saved as HDF5 files in `data/bc/<EMBODIMENT_TASK>/demonstration/` (e.g., `data/bc/uam_cabinet/demonstration/episode_0.hdf5`).

### Keyboard Controls

| Key | Action |
|-----|--------|
| `W/A/S/D` | Move horizontally |
| `Space/Shift` | Move up/down |
| `Q/E` | Close/open gripper |
| `Arrow keys` | Rotate pitch/yaw |
| `Z/C` | Roll left/right |
| `P` | Start recording |
| `R` | Reset scene |
| `ESC` | Exit |

## Training Policies

### Convert Demonstrations to Training Format

Convert recorded HDF5 episodes to UMI zarr format:

```bash
python ../universal_manipulation_interface/convert_hdf5_to_umi_zarr.py \
    --input_dir <path_to_episode_directory> \
    --output_path dataset.zarr.zip \
    --camera_name ee \
    --image_size 224
```

### Train Diffusion Policy

The simulation runs at **50 Hz**. Configure your policy's query frequency in the training config:

```
query_frequency = action_horizon × obs_down_sample_steps
```

The policy will be called every `query_frequency` timesteps (each timestep = 1/50 second).

**Example:** If `action_horizon = 8` and `obs_down_sample_steps = 4`, then `query_frequency = 32` (policy runs every 0.64 seconds).

Edit the config file at `am_mujoco_ws/universal_manipulation_interface/diffusion_policy/config/train_diffusion_unet_timm_umi_workspace.yaml` before training.

### Single-GPU Training

```bash
python ../universal_manipulation_interface/train.py \
    --config-name=train_diffusion_unet_timm_umi_workspace \
    task.dataset_path=dataset.zarr.zip
```
## Policy Evaluation

The `imitate_episodes.py` script evaluates trained policies in simulation with support for Embodiment-Aware Diffusion Policy (EADP) guidance.

### Usage

```bash
python imitate_episodes.py \
    --task_name EMBODIMENT_TASK \
    [--ckpt_dir <checkpoint_directory>] \
    [--output_dir <results_directory>] \
    [--num_rollouts <number>] \
    [--guidance <strength>] \
    [--use_3d_viewer | --onscreen_render] \
    [--disturb]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task_name` | *required* | Task in format `EMBODIMENT_TASK` (e.g., `uam_cabinet`, `ur10e_peg`) |
| `--ckpt_dir` | *auto-detect* | Checkpoint directory. Auto-detects to `checkpoints/umi_{TASK}/` if not provided |
| `--output_dir` | `results/eval/<task>/<timestamp>/` | Directory to save results (videos, metrics, plots) |
| `--num_rollouts` | 10 | Number of evaluation episodes to run |
| `--guidance` | 0.0 | MPC guidance strength for EADP (0.0=disabled, 1.5=tested value) |
| `--use_3d_viewer` | disabled | Interactive 3D MuJoCo viewer with mouse controls |
| `--onscreen_render` | disabled | Ego-centric camera view in fullscreen window |
| `--disturb` | disabled | Enable wind disturbances for UAM embodiment |
| `--resume` | disabled | Continue most recent run (auto-finds latest timestamp) |

**Note:** If neither `--use_3d_viewer` nor `--onscreen_render` is specified, evaluation runs headless (no visualization, fastest).

### Example

Evaluate UAM on cabinet task with EADP guidance, disturbances, and 3D visualization:

```bash
python imitate_episodes.py \
    --task_name uam_cabinet \
    --num_rollouts 30 \
    --guidance 1.5 \
    --use_3d_viewer \
    --disturb
```

**Note:** Each run creates a timestamped directory. Use `--resume` to automatically continue the most recent run for that task in case there is some failure.

### Keyboard Controls

Available when using `--use_3d_viewer` or `--onscreen_render`:

| Key | Action |
|-----|--------|
| `1` | Discard current episode and retry |
| `2` | Save current episode as failed and move to next |
| `SPACE` | Pause/Resume simulation |
| `ESC` | Exit program |
| `F` | Toggle fullscreen (only with `--onscreen_render`) |

### Output Structure

Results are saved in timestamped directories:

```
results/eval/uam_cabinet/2025-11-25_14-30-00/
├── episode_000/
│   ├── video.mp4              # Episode recording
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

The `run_ablation.py` script runs parallel ablation sweeps over guidance parameters.

### Usage

```bash
python run_ablation.py \
    --task_name EMBODIMENT_TASK \
    [--ckpt_dir <checkpoint_directory>] \
    [--guidances <comma_separated_values>] \
    [--guided_steps <comma_separated_values>] \
    [--num_rollouts <number>] \
    [--max_workers <number>] \
    [--output_dir <custom_output_path>] \
    [--disturb]
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task_name` | *required* | Task in format `EMBODIMENT_TASK` (e.g., `uam_cabinet`) |
| `--ckpt_dir` | *auto-detect* | Checkpoint directory (auto-detects from task name if not provided) |
| `--guidances` | `0.0,0.5,1.0,1.5` | Comma-separated guidance values to test |
| `--guided_steps` | `1` | Comma-separated guided step thresholds |
| `--num_rollouts` | 30 | Episodes per configuration |
| `--max_workers` | 4 | Number of parallel experiments |
| `--output_dir` | `results/ablations/<task>/<timestamp>/` | Output directory for results |
| `--resume` | disabled | Resume most recent sweep (auto-finds latest) |
| `--disturb` | disabled | Enable wind disturbances for UAM embodiment |

### Example

Run ablation sweep on UAM peg task with 4 guidance values in parallel:

```bash
python run_ablation.py \
    --task_name uam_cabinet \
    --guidances 0.0,1.5 \
    --num_rollouts 30 \
    --disturb
```

The script:
- Runs experiments in parallel with automatic retry on failure
- Provides live progress monitoring with ETAs
- Generates summary heatmaps showing success rate and episode duration

### Output Structure

Results are saved in timestamped directories:

```
results/ablations/uam_peg/2025-11-25_18-00-00/
├── guidance0.0_s1/
│   ├── episode_000/
│   ├── episode_001/
│   └── experiment_summary.json
├── guidance0.5_s1/
│   └── ...
├── guidance1.0_s1/
│   └── ...
├── guidance1.5_s1/
│   └── ...
└── summary_heatmaps.png        # Combined results visualization
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{gupta2025umionairembodimentawareguidanceembodimentagnostic,
      title={UMI-on-Air: Embodiment-Aware Guidance for Embodiment-Agnostic Visuomotor Policies}, 
      author={Harsh Gupta and Xiaofeng Guo and Huy Ha and Chuer Pan and Muqing Cao and Dongjae Lee and Sebastian Scherer and Shuran Song and Guanya Shi},
      year={2025},
      eprint={2510.02614},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.02614}, 
}
```

## Contact

If you have any questions, feel free to reach out:

- 📧 Email: hgupt3@illinois.edu
