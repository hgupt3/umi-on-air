#!/usr/bin/env python3
"""
run_ablation.py – parallel 2-D sweep over (guidance, guided_steps) or (scale, guided_steps)

Example (guidance mode):
    python run_ablation.py --ckpt_dir /path/to/ckpt \
        --guidances 0.0,0.2,0.5,1.0 --guided_steps 5,10,15,20 \
        --max_workers 4

Example (scale mode):
    python run_ablation.py --ckpt_dir /path/to/ckpt \
        --scales 0.8,1.0,1.2,1.5 --guided_steps 5,10,15,20 \
        --max_workers 4

The script calls imitate_episodes.py in parallel for each combination and stores each
run in a dedicated output directory:
  - Guidance mode: <ckpt_dir>/ablation_results/g{guidance}_s{steps}  
  - Scale mode: <ckpt_dir>/ablation_results/scale{scale}_s{steps}
"""
import argparse
import itertools
import os
import signal
import subprocess
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time
import shutil
import threading
import math

# Global variable to hold the executor for signal handling
current_executor = None

def get_timestamp():
    """Get current timestamp in YYYY-MM-DD_HH-MM-SS format"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def find_latest_timestamped_dir(base_path):
    """Find the most recent timestamped directory"""
    if not os.path.exists(base_path):
        return None
    dirs = [d for d in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, d))]
    if not dirs:
        return None
    dirs.sort(reverse=True)
    return os.path.join(base_path, dirs[0])


def signal_handler(signum, frame):
    """Handle Ctrl+C by immediately terminating all workers and exiting"""
    print("\n🛑 Ctrl+C detected! Terminating all workers immediately...", file=sys.stderr)
    
    if current_executor is not None:
        # Cancel all pending futures and shutdown without waiting
        current_executor.shutdown(wait=False, cancel_futures=True)
        
        # Kill any remaining child processes
        try:
            import psutil
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Give processes a moment to terminate gracefully
            time.sleep(0.5)
            
            # Force kill any that didn't terminate
            for child in children:
                try:
                    if child.is_running():
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
        except ImportError:
            # Fallback if psutil not available - use os.killpg
            try:
                os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
                time.sleep(0.5)
                os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
            except:
                pass
    
    print("🛑 Forced shutdown complete.", file=sys.stderr)
    os._exit(130)  # Standard exit code for Ctrl+C


class ProgressMonitor:
    """Monitor and display progress of running experiments"""
    
    def __init__(self, experiments_root, experiment_params_list, max_workers, scale_mode, param_name):
        self.experiments_root = experiments_root
        self.experiment_params_list = experiment_params_list
        self.max_workers = max_workers
        self.scale_mode = scale_mode
        self.param_name = param_name
        self.total_experiments = len(experiment_params_list)
        self.completed_experiments = {}  # {experiment_id: (success_rate, actual_duration_minutes)}
        self.running_experiments = {}   # {experiment_id: progress_data}
        self.experiment_start_times = {}  # {experiment_id: start_timestamp}
        self.lock = threading.Lock()
        
    def format_eta(self, minutes):
        """Format ETA in human-readable format"""
        if minutes < 60:
            return f"{int(minutes)}min"
        else:
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            return f"{hours}h {mins}min"
    
    def get_experiment_progress(self, experiment_id):
        """Read progress status file for an experiment"""
        exp_dir = os.path.join(self.experiments_root, experiment_id)
        progress_file = os.path.join(exp_dir, 'progress_status.json')
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return None
    
    def mark_experiment_started(self, experiment_id):
        """Mark when an experiment starts"""
        with self.lock:
            self.experiment_start_times[experiment_id] = time.time()
    
    def update_progress(self):
        """Update progress for all experiments"""
        with self.lock:
            # Check for completed experiments
            for g, s, *_ in self.experiment_params_list:
                experiment_id = f"{self.param_name}{g}_s{s}"
                exp_dir = os.path.join(self.experiments_root, experiment_id)
                summary_file = os.path.join(exp_dir, "experiment_summary.json")
                
                # If experiment completed, move from running to completed
                if experiment_id in self.running_experiments and os.path.exists(summary_file):
                    # Read final results
                    try:
                        with open(summary_file, 'r') as f:
                            summary_data = json.load(f)
                        success_rate = summary_data['experiment_summary']['success_rate']
                        
                        # Calculate actual duration from start time
                        start_time = self.experiment_start_times.get(experiment_id, time.time())
                        actual_duration_minutes = (time.time() - start_time) / 60
                        
                        self.completed_experiments[experiment_id] = (success_rate, actual_duration_minutes)
                        del self.running_experiments[experiment_id]
                        if experiment_id in self.experiment_start_times:
                            del self.experiment_start_times[experiment_id]
                    except:
                        pass
                
                # Update running experiment progress
                elif experiment_id not in self.completed_experiments:
                    progress_data = self.get_experiment_progress(experiment_id)
                    if progress_data and progress_data.get('status') in ['running', 'starting']:
                        self.running_experiments[experiment_id] = progress_data
                        # Track start time if not already tracked
                        if experiment_id not in self.experiment_start_times:
                            self.experiment_start_times[experiment_id] = time.time()
    
    def display_progress(self):
        """Display current progress"""
        with self.lock:
            # Clear previous lines and show updated progress
            print("\n" + "="*80)
            
            # Overall progress bar
            completed_count = len(self.completed_experiments)
            total_count = self.total_experiments
            progress_percent = (completed_count / total_count) * 100 if total_count > 0 else 0
            
            # ASCII progress bar
            bar_width = 20
            filled_width = int((progress_percent / 100) * bar_width)
            bar = "█" * filled_width + "░" * (bar_width - filled_width)
            
            print(f"📊 ABLATION PROGRESS {bar} {completed_count}/{total_count} ({progress_percent:.0f}%)")
            
            # Completed experiments
            if self.completed_experiments:
                completed_list = []
                for exp_id, (success_rate, duration) in self.completed_experiments.items():
                    completed_list.append(f"{exp_id}({success_rate:.0%})")
                print(f"   ✅ Completed: {', '.join(completed_list)}")
            
            # Running experiments with individual progress
            if self.running_experiments:
                running_count = len(self.running_experiments)
                print(f"   🔄 Running: {running_count} experiments")
                
                for exp_id, progress in self.running_experiments.items():
                    current_ep = progress.get('current_episode', 0)
                    total_ep = progress.get('total_episodes', 1)
                    success_count = progress.get('successful_episodes', 0)
                    success_rate = progress.get('success_rate', 0.0)
                    eta_min = progress.get('estimated_remaining_time', 0) / 60  # Convert to minutes
                    
                    ep_percent = (current_ep / total_ep) * 100
                    eta_str = self.format_eta(eta_min) if eta_min > 0 else "calculating..."
                    
                    # Find experiment index
                    exp_index = next((i+1 for i, (g, s, *_) in enumerate(self.experiment_params_list) 
                                    if f"{self.param_name}{g}_s{s}" == exp_id), "?")
                    
                    print(f"   🚀 [{exp_index}/{total_count}] {exp_id} | Episode {current_ep}/{total_ep} ({ep_percent:.0f}%) | Success: {success_count}/{current_ep} ({success_rate:.0%}) | ETA: {eta_str}")
            
            # Pending experiments
            pending_experiments = []
            for g, s, *_ in self.experiment_params_list:
                exp_id = f"{self.param_name}{g}_s{s}"
                if exp_id not in self.completed_experiments and exp_id not in self.running_experiments:
                    pending_experiments.append(exp_id)
            
            if pending_experiments:
                if len(pending_experiments) <= 8:  # Show all if few remaining
                    print(f"   ⏳ Pending: {', '.join(pending_experiments)}")
                else:  # Show count if many remaining
                    print(f"   ⏳ Pending: {len(pending_experiments)} experiments")
            
            # Overall ETA - Parallelism-aware calculation
            if self.running_experiments or pending_experiments:
                pending_count = len(pending_experiments)
                running_count = len(self.running_experiments)
                
                # Get average experiment duration estimate
                if self.completed_experiments:
                    # Use actual durations from completed experiments (stored in minutes)
                    avg_experiment_duration_seconds = sum(
                        duration * 60 for _, duration in self.completed_experiments.values()
                    ) / len(self.completed_experiments)
                elif self.running_experiments:
                    # Estimate from running experiments: use current progress to extrapolate
                    durations = []
                    for progress in self.running_experiments.values():
                        current_ep = progress.get('current_episode', 0)
                        total_ep = progress.get('total_episodes', 1)
                        avg_ep_duration = progress.get('avg_episode_duration', 0)
                        if current_ep > 0 and avg_ep_duration > 0:
                            # Estimated total duration = avg_ep_duration * total_ep
                            durations.append(avg_ep_duration * total_ep)
                    if durations:
                        avg_experiment_duration_seconds = sum(durations) / len(durations)
                    else:
                        avg_experiment_duration_seconds = 30 * 60  # Fallback: 30 min
                else:
                    avg_experiment_duration_seconds = 30 * 60  # Fallback: 30 min
                
                # Calculate total remaining work in experiment-seconds
                # Running experiments: sum of their remaining times
                running_remaining_seconds = sum(
                    p.get('estimated_remaining_time', 0) 
                    for p in self.running_experiments.values()
                )
                
                # Pending experiments: each takes avg_experiment_duration
                pending_total_seconds = pending_count * avg_experiment_duration_seconds
                
                # Total remaining work (in experiment-time units)
                total_remaining_work_seconds = running_remaining_seconds + pending_total_seconds
                
                # With max_workers running in parallel, divide by parallelism
                # But we can't have more workers than remaining experiments
                effective_parallelism = min(self.max_workers, running_count + pending_count)
                if effective_parallelism > 0:
                    total_eta_seconds = total_remaining_work_seconds / effective_parallelism
                else:
                    total_eta_seconds = 0
                
                total_eta_minutes = total_eta_seconds / 60
                print(f"   🕐 Overall ETA: {self.format_eta(total_eta_minutes)}")
            
            print("="*80)


def run_single_experiment(params):
    """
    Worker function that runs one experiment with automatic retry logic.
    If the experiment fails, its output directory is deleted and the run is
    repeated up to `max_retries` additional times.
    """
    g, s, ckpt_dir, task_name, num_rollouts, extra_args, script_dir, max_retries, scale_mode, param_name, experiments_root, disturb = params

    out_dir_base = os.path.join(experiments_root, f'{param_name}{g}_s{s}')

    attempt = 0
    last_duration = 0.0
    while attempt <= max_retries:
        attempt += 1

        # Start from a clean directory
        if os.path.isdir(out_dir_base):
            shutil.rmtree(out_dir_base, ignore_errors=True)
        os.makedirs(out_dir_base, exist_ok=True)

        # Paths
        log_file = os.path.join(out_dir_base, "experiment.log")
        error_file = os.path.join(out_dir_base, "experiment_error.log")
        imitate_ep_path = os.path.join(script_dir, 'imitate_episodes.py')

        # Choose python interpreter
        conda_python = os.path.expanduser("~/miniconda3/envs/flyingumi/bin/python")
        if not os.path.exists(conda_python):
            conda_python = os.path.expanduser("~/anaconda3/envs/flyingumi/bin/python")
        if not os.path.exists(conda_python):
            conda_python = sys.executable

        cmd = [
            conda_python, imitate_ep_path,
            '--ckpt_dir', ckpt_dir,
            '--task_name', task_name,
            '--num_rollouts', str(num_rollouts),
            '--output_dir', out_dir_base,
        ]
        
        if disturb:
            cmd.append('--disturb')
        
        # Add scale or guidance parameter based on mode
        if scale_mode:
            cmd.extend(['--scale', str(g)])
        else:
            cmd.extend(['--guidance', str(g)])
        
        cmd.extend(['--guided_steps', str(s)])

        if extra_args:
            cmd.extend(extra_args.split())

        acados_build_dir = os.path.join(out_dir_base, "acados_build")
        cmd.extend(['--acados_build_dir', acados_build_dir])

        start_time = time.time()

        with open(log_file, 'w') as f:
            f.write(f"=== Experiment {param_name}{g}_s{s} attempt {attempt} started at {datetime.now()} ===\n")
            f.write(f"Using Python: {conda_python}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 60 + "\n\n")

        env = os.environ.copy()
        env['ACADOS_SOURCE_DIR'] = '/home/harsh/flyingumi/am_mujoco_ws/am_trajectory_controller/acados'
        env['LD_LIBRARY_PATH'] = f"/home/harsh/flyingumi/am_mujoco_ws/am_trajectory_controller/acados/lib:{env.get('LD_LIBRARY_PATH', '')}"

        try:
            with open(log_file, 'a') as stdout_f, open(error_file, 'w') as stderr_f:
                result = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f, text=True, env=env)
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"Subprocess failed to launch: {e}\n")

        last_duration = time.time() - start_time

        # Success check based on expected output files
        experiment_summary_json = os.path.join(out_dir_base, "experiment_summary.json")
        experiment_summary_txt  = os.path.join(out_dir_base, "experiment_summary.txt")
        episode_metrics_csv     = os.path.join(out_dir_base, "episode_metrics.csv")

        episode_dirs = [
            d for d in os.listdir(out_dir_base)
            if d.startswith('episode_') and os.path.isdir(os.path.join(out_dir_base, d))
        ]
        valid_episodes = sum(
            os.path.isfile(os.path.join(out_dir_base, d, 'metrics.json'))
            for d in episode_dirs
        )

        files_success = (
            os.path.isfile(experiment_summary_json) and
            os.path.isfile(experiment_summary_txt) and
            os.path.isfile(episode_metrics_csv) and
            valid_episodes >= num_rollouts
        )

        if files_success:
            return (g, s, 0, last_duration)

        # If failed and no retries left, exit loop
        if attempt > max_retries:
            break

        with open(log_file, 'a') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Attempt {attempt} failed – retrying ({attempt}/{max_retries})\n")

    # All retries exhausted – report failure
    return (g, s, 1, last_duration)


def auto_detect_checkpoint_dir(task_name):
    """
    Auto-detect checkpoint directory based on task name.
    
    Task naming format: EMBODIMENT_TASK (e.g., 'uam_cabinet', 'ur10e_peg')
    Looks for checkpoints in: checkpoints/umi_{TASK}/
    
    Args:
        task_name: Task name in format 'embodiment_task'
    
    Returns:
        Path to checkpoint directory if found, None otherwise
    """
    # Extract task name (part after embodiment prefix)
    parts = task_name.split('_', 1)
    if len(parts) < 2:
        return None
    
    task = parts[1]  # Get task name (cabinet, peg, pick, valve)
    
    # Get workspace root (2 levels up from policy_learning)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Look for checkpoints/umi_{task}/ directory
    checkpoint_dir = os.path.join(workspace_root, 'checkpoints', f'umi_{task}')
    
    if os.path.exists(checkpoint_dir):
        print(f"✨ Auto-detected checkpoint directory: {checkpoint_dir}")
        return checkpoint_dir
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=False, help='Checkpoint directory (optional - will auto-detect from task_name if not provided)')
    parser.add_argument('--task_name', default='uam_peg')
    parser.add_argument('--guidances', default='0.0,0.5,1.0,1.5',
                        help='Comma-separated list of guidance values')
    parser.add_argument('--guided_steps', default='1',
                        help='Comma-separated list of guided_step thresholds')
    parser.add_argument('--scales', default='',
                        help='Comma-separated list of scale values (mutually exclusive with guidances)')
    parser.add_argument('--num_rollouts', type=int, default=30)
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of parallel workers')
    parser.add_argument('--extra_args', default='',
                        help='Additional flags forwarded verbatim to imitate_episodes.py')
    parser.add_argument('--max_retries', type=int, default=4,
                        help='Number of times to retry a failed experiment (each retry starts from a clean directory)')
    parser.add_argument('--output_dir', default='',
                        help='Output directory for ablation results (default: <ckpt_dir>/ablation_results)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume most recent sweep (auto-finds latest, skips completed)')
    parser.add_argument('--disturb', action='store_true',
                        help='Enable wind disturbances for UAM embodiment')
    args = parser.parse_args()
    
    # Auto-detect checkpoint directory if not provided
    if args.ckpt_dir is None:
        args.ckpt_dir = auto_detect_checkpoint_dir(args.task_name)
        if args.ckpt_dir is None:
            print(f"❌ Error: Could not auto-detect checkpoint directory for task '{args.task_name}'")
            print(f"   Please provide --ckpt_dir explicitly or ensure checkpoints/umi_{{task}}/ exists")
            sys.exit(1)

    # Determine sweep mode
    scale_mode = bool(args.scales.strip())
    
    if scale_mode:
        scale_values = [float(x) for x in args.scales.split(',') if x.strip()]
        param_values = scale_values
        param_name = "scale"
    else:
        guidance_values = [float(x) for x in args.guidances.split(',') if x]
        param_values = guidance_values  
        param_name = "guidance"
    
    step_values = [int(x) for x in args.guided_steps.split(',') if x]

    # Use custom output directory if provided, otherwise default to results/ablations/<task>/<timestamp>
    if args.output_dir:
        experiments_root = os.path.abspath(args.output_dir)
    else:
        # Get workspace root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(script_dir))
        
        # For UAM tasks with disturbance, use uam-disturb_* folder naming
        folder_task_name = args.task_name
        if args.disturb and args.task_name.startswith('uam_'):
            folder_task_name = args.task_name.replace('uam_', 'uam-disturb_', 1)
        
        ablation_base = os.path.join(workspace_root, 'results', 'ablations', folder_task_name)
        
        if args.resume:
            # Auto-find most recent sweep
            experiments_root = find_latest_timestamped_dir(ablation_base)
            if experiments_root is None:
                print(f"❌ Error: --resume specified but no previous sweeps found in {ablation_base}")
                sys.exit(1)
            print(f"📂 Resuming from: {experiments_root}")
        else:
            # Create new timestamped directory
            timestamp = get_timestamp()
            experiments_root = os.path.join(ablation_base, timestamp)
    
    os.makedirs(experiments_root, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Prepare experiment parameters
    experiments_to_run = []
    skipped_experiments = []
    
    for g, s in itertools.product(param_values, step_values):
        out_dir = os.path.join(experiments_root, f'{param_name}{g}_s{s}')
        
        # Skip combo if we already have results from a previous run (only when resuming)
        if args.resume:
            summary_file = os.path.join(out_dir, "experiment_summary.json")
            done_flag = os.path.join(out_dir, "DONE.txt")
            if os.path.isfile(summary_file) or os.path.isfile(done_flag):
                skipped_experiments.append(f"{param_name}{g}_s{s}")
                continue
        
        # Add to experiments to run
        experiment_params = (g, s, args.ckpt_dir, args.task_name, 
                           args.num_rollouts, args.extra_args, script_dir, args.max_retries, 
                           scale_mode, param_name, experiments_root, args.disturb)
        experiments_to_run.append(experiment_params)
    
    if skipped_experiments:
        print(f"⚠️  Skipping {len(skipped_experiments)} experiments (results already present):")
        for exp in skipped_experiments:
            print(f"    {exp}")
        print()
    
    if not experiments_to_run:
        print("✅ All experiments already completed!")
        # Still generate summary plots
    else:
        print(f"🚀 Starting {len(experiments_to_run)} experiments with {args.max_workers} parallel workers...")
        print(f"📁 Results will be saved to: {experiments_root}")
        print()
        
        # Initialize progress monitor with max_workers
        progress_monitor = ProgressMonitor(experiments_root, experiments_to_run, args.max_workers, scale_mode, param_name)
        
        # Progress monitoring function
        def monitor_progress():
            while True:
                progress_monitor.update_progress()
                progress_monitor.display_progress()
                time.sleep(10)  # Update every 10 seconds
        
        # Start progress monitoring thread
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        # Set signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)
        global current_executor
        current_executor = ProcessPoolExecutor(max_workers=args.max_workers)
        
        # Run experiments in parallel
        completed_experiments = []
        failed_experiments = []
        
        start_time = time.time()
        
        with current_executor as executor:
            # Submit all experiments
            future_to_params = {
                executor.submit(run_single_experiment, params): params 
                for params in experiments_to_run
            }
            
            # Process completed experiments
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                g, s = params[0], params[1]
                
                try:
                    result_g, result_s, return_code, duration = future.result()
                    
                    out_dir = os.path.join(experiments_root, f'{param_name}{result_g}_s{result_s}')
                    
                    if return_code == 0:
                        # Mark success
                        Path(out_dir, "DONE.txt").touch()
                        completed_experiments.append(f"{param_name}{result_g}_s{result_s}")
                        print(f"✅ {param_name}{result_g}_s{result_s} completed successfully ({duration:.1f}s)")
                    else:
                        # Mark failure
                        Path(out_dir, "FAIL.txt").write_text(str(return_code))
                        failed_experiments.append(f"{param_name}{result_g}_s{result_s}")
                        print(f"❌ {param_name}{result_g}_s{result_s} failed (return code: {return_code})")
                        
                except Exception as e:
                    failed_experiments.append(f"{param_name}{g}_s{s}")
                    print(f"💥 {param_name}{g}_s{s} crashed with exception: {e}")
        
        # Reset global executor reference
        current_executor = None
        
        total_time = time.time() - start_time
        
        # Final progress update
        progress_monitor.update_progress()
        progress_monitor.display_progress()
        
        print(f"\n🏁 Parallel ablation sweep finished in {total_time:.1f}s")
        print(f"✅ Completed: {len(completed_experiments)}")
        print(f"❌ Failed: {len(failed_experiments)}")
        
        if failed_experiments:
            print(f"Failed experiments: {', '.join(failed_experiments)}")

    # --------------------------------------------------
    # Cleanup ACADOS build directories
    # --------------------------------------------------
    print("\n🧹 Cleaning up ACADOS build directories...")
    cleaned_count = 0
    for g, s in itertools.product(param_values, step_values):
        out_dir = os.path.join(experiments_root, f'{param_name}{g}_s{s}')
        acados_build_dir = os.path.join(out_dir, "acados_build")
        if os.path.isdir(acados_build_dir):
            try:
                shutil.rmtree(acados_build_dir)
                cleaned_count += 1
            except OSError as e:
                print(f"   Error removing {acados_build_dir}: {e}")
    if cleaned_count > 0:
        print(f"   Removed {cleaned_count} build directories.")
    else:
        print("   No build directories to clean.")


    # --------------------------------------------------
    # Build summary DataFrame across all completed runs
    # --------------------------------------------------
    print(f"\n📊 Generating summary plots...")
    
    records = []
    for g, s in itertools.product(param_values, step_values):
        out_dir = os.path.join(experiments_root, f'{param_name}{g}_s{s}')
        summary_path = os.path.join(out_dir, "experiment_summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path, "r") as f:
            data = json.load(f)
        episodes = data.get("per_episode_metrics", [])
        total_eps = len(episodes)
        succ_eps  = [ep for ep in episodes if ep.get("success", False)]

        success_rate    = len(succ_eps) / total_eps if total_eps else np.nan
        avg_time_succ   = np.mean([ep.get("episode_duration", np.nan) for ep in succ_eps]) if succ_eps else np.nan

        records.append({
            param_name: g,
            "steps": s,
            "success_rate": success_rate,
            "avg_time_succ": avg_time_succ,
        })

    if not records:
        print("No completed runs found – skipping summary figure.")
        return

    df = pd.DataFrame(records)

    def pivot(metric):
        return df.pivot(index=param_name, columns="steps", values=metric).sort_index().sort_index(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(len(step_values)*1.6, 7))
    heatmap_cfg = [
        ("success_rate",   "Success rate",        "viridis", ".0%"),
        ("avg_time_succ",  "Avg episode time (s)", "Blues_r",  ".1f"),
    ]

    for ax, (metric, title, cmap, fmt) in zip(axes, heatmap_cfg):
        data_piv = pivot(metric)
        sns.heatmap(data_piv, cmap=cmap, annot=True, fmt=fmt, linewidths=0.5, linecolor='grey', ax=ax,
                    cbar_kws={"shrink": 0.8}, vmin=0 if metric=="success_rate" else None,
                    vmax=1 if metric=="success_rate" else None)
        ax.set_title(title)
        ax.set_xlabel("guided_steps")
        ax.set_ylabel(param_name)

    fig.tight_layout()
    fig_path = os.path.join(experiments_root, "summary_heatmaps.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"🖼  Saved summary figure → {fig_path}")


if __name__ == '__main__':
    main() 