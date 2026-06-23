# SO100 + LeRobot + RDK End-to-End Guide

[Chinese](./README_cn.md)

## Open-Source Resources

The datasets and trained model related to this project are available on Hugging Face. They can be used to reproduce experiments, compare training results, or inspect the expected dataset format.

Open-source datasets from this project:

- [`taoz-member/tidy_desktop`](https://huggingface.co/datasets/taoz-member/tidy_desktop): desktop tidying task dataset.
- [`taoz-member/grab_blocks`](https://huggingface.co/datasets/taoz-member/grab_blocks): block grasping task dataset.

Open-source trained model from this project:

- [`taoz-member/eval_tidy_desktop`](https://huggingface.co/taoz-member/eval_tidy_desktop): trained/evaluation model for the `tidy_desktop` task.

## Directory Layout

The project directory is roughly organized as follows:

```text
lerobot_grasp/
  lerobot/                  Main LeRobot source code
  examples/                 Official/local examples; SO100 guide is in examples/10_use_so100.md
  docs/                     LeRobot documentation
  media/                    Example images and tutorial assets
  rdk_LeRobot_tools/        RDK S100/S100P BPU export and on-board inference tools
  bpu_output/               Example BPU inference artifacts
  pyproject.toml            Python package configuration
```

The `stable` branch of `rdk_LeRobot_tools` mainly targets older LeRobot workflows and v2.1 datasets. It has primarily been verified for deploying ACT models on RDK S100. For SO100, data collection and training should use `--robot.type=so100` with the ACT policy. If the BPU inference script still defaults to `so101`, change it to `so100` before deployment.

## Overall Workflow

1. Prepare the development/RDK environments.
2. Configure SO100 motor IDs, arm ports, and cameras.
3. Calibrate the leader and follower arms.
4. Test teleoperation and camera synchronization.
5. Record the `taoz/so100_total` dataset.
6. Train an ACT policy on a machine with an NVIDIA GPU.
7. Export the ACT model to ONNX and compile it into a BPU model with the OpenExplorer toolchain.
8. Copy `bpu_output` to the RDK board and run BPU inference.

## Environment Setup

Python 3.10 is recommended. The development machine is used for training and model export. The RDK board is used for calibration, data collection, teleoperation, and final inference.

### Development Machine or Training Server

```bash
cd /root/Desktop/lerobot

conda create -y -n lerobot python=3.10
conda activate lerobot

pip install -e ".[feetech]"
pip install onnx onnxsim termcolor tqdm
```

If you hit LeRobot version or dataset-format compatibility issues, prefer the D-Robotics LeRobot fork, or make sure `datasets==2.19.0` is installed:

```bash
pip install "datasets==2.19.0"
```

ACT training requires an NVIDIA GPU and a CUDA-enabled PyTorch installation. Check it with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

### RDK Board

```bash
cd /root/Desktop/lerobot

pip install -e ".[feetech]"
pip install hbm-runtime
```

`hbm-runtime` is not required if the board is only used for collection and teleoperation. It is required for BPU inference.

## SO100 Hardware Setup

### Find Arm Serial Ports

Connect the leader/follower bus servo adapters, then run:

```bash
python lerobot/scripts/find_motors_bus_port.py
```

Follow the prompt to unplug/replug USB devices and record the leader/follower ports, for example:

```text
leader:   /dev/ttyUSB0
follower: /dev/ttyUSB1
```

On Linux/RDK, if the serial ports are not accessible, temporarily grant permissions:

```bash
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1
```

### Configure Motor IDs

The SO100 leader and follower arms each use six Feetech `sts3215` motors. Their IDs are usually `1` to `6`. Connect only one motor at a time and set IDs according to the actual port:

```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/ttyUSB0 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```

Repeat with `--ID` set to `2`, `3`, `4`, `5`, and `6` for both leader and follower arms. Label the motors, for example `L1-L6` and `F1-F6`, to make troubleshooting much easier.

### Update SO100 Port Configuration

Open:

```text
lerobot/common/robot_devices/robots/configs.py
```

Find `So100RobotConfig`, then update the `port` values in `leader_arms` and `follower_arms`:

```python
@RobotConfig.register_subclass("so100")
@dataclass
class So100RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so100"

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyUSB0",
                motors={...},
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyUSB1",
                motors={...},
            ),
        }
    )
```

## Camera Setup

Connect the cameras and inspect their indices:

```bash
python lerobot/common/robot_devices/cameras/opencv.py \
  --images-dir outputs/images_from_opencv_cameras
```

Check the generated images and confirm which view corresponds to `camera_index=0` and `camera_index=1`. Then update the `cameras` section in `So100RobotConfig`:

```python
cameras: dict[str, CameraConfig] = field(
    default_factory=lambda: {
        "laptop": OpenCVCameraConfig(
            camera_index=0,
            fps=30,
            width=640,
            height=480,
        ),
        "phone": OpenCVCameraConfig(
            camera_index=1,
            fps=30,
            width=640,
            height=480,
        ),
    }
)
```

Camera names are written into the dataset and also affect the normalization file names generated during BPU export. Use the same camera names for training, export, and board-side inference.

## Calibration

Before calibration, make sure the arm is not blocked by cables, emergency-stop conditions, or mechanical limits. It is best to first test the arm without task objects.

### Calibrate the Follower Arm

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'
```

### Calibrate the Leader Arm

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'
```

Calibration files are written to `.cache/calibration/so100`. Recalibrate after changing arms, rebuilding the mechanism, changing port mappings, or seeing obvious motion offsets.

## Teleoperation Check

First test leader/follower tracking without cameras:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

Then enable cameras and data streaming to verify image, joint state, and action synchronization:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=teleoperate \
  --control.fps=30
```

If the camera windows do not open, go back to the camera setup step and verify the indices. If the terminal reports a headless environment, keyboard shortcuts and on-screen display may not be available.

## Record a Dataset

The following command is the dataset collection command used by this project. It records `taoz/so100_total`, with 60 episodes, 40 seconds per episode, and no upload to the Hub:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a pencil and put it in the box." \
  --control.repo_id=taoz/so100_total \
  --control.tags='["so100","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=40 \
  --control.reset_time_s=10 \
  --control.num_episodes=60 \
  --control.push_to_hub=false
```

Useful keyboard shortcuts during recording:

- Right arrow: end the current recording stage early and enter reset; during reset, skip to the next episode.
- Left arrow: discard the current episode and record it again.
- Esc: stop the full recording session and save completed episodes.

Note: the command above does not explicitly set `--control.root`. Depending on the LeRobot defaults, the data may be saved in the Hugging Face/LeRobot cache instead of `/root/Desktop/lerobot/so100_total`. The training command below reads from `/root/Desktop/lerobot/so100_total`, so after collection, confirm that the dataset is located there. If not, copy the dataset to that path, or add this argument in the next recording run:

```bash
--control.root=/root/Desktop/lerobot/so100_total
```

It is not included in the official command above in order to keep the command exactly aligned with the requested collection command.

## Dataset Validation

After collection, check three things:

1. The number of episodes is close to 60.
2. Sample videos are clear, camera views are fixed, and the object is visible.
3. Replay one episode and confirm action/image synchronization.

Local visualization:

```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id taoz/so100_total \
  --local-files-only 1
```

Replay episode 0:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=taoz/so100_total \
  --control.episode=0
```

If the dataset is stored at a specific local path, add the corresponding root argument when needed.

## Use Open-Source Datasets

If you do not want to collect a new dataset, you can use the datasets already open-sourced on Hugging Face. For online training, replace `--dataset.repo_id` with the target dataset:

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=taoz-member/tidy_desktop \
  --policy.type=act \
  --output_dir=/root/gpufree-data/outputs/train/act_tidy_desktop \
  --job_name=act_tidy_desktop \
  --policy.device=cuda \
  --wandb.enable=false
```

Another dataset example:

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=taoz-member/grab_blocks \
  --policy.type=act \
  --output_dir=/root/gpufree-data/outputs/train/act_grab_blocks \
  --job_name=act_grab_blocks \
  --policy.device=cuda \
  --wandb.enable=false
```

If the network is unstable, download the dataset first and point `--dataset.root` to the local directory:

```bash
huggingface-cli download \
  --repo-type dataset \
  taoz-member/tidy_desktop \
  --local-dir /root/Desktop/lerobot/tidy_desktop
```

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=taoz-member/tidy_desktop \
  --dataset.root=/root/Desktop/lerobot/tidy_desktop \
  --policy.type=act \
  --output_dir=/root/gpufree-data/outputs/train/act_tidy_desktop \
  --job_name=act_tidy_desktop \
  --policy.device=cuda \
  --wandb.enable=false
```

If an open-source dataset uses the v3.0 structure but the current RDK export path expects v2.1, convert it first using the next section.

## Convert Dataset v3.0 to v2.1

The `stable` version of `rdk_LeRobot_tools` is mainly verified with LeRobot v2.1 datasets. If a dataset was collected with a newer LeRobot version and is in v3.0 format, convert it to v2.1 before training, ACT export, and BPU compilation.

This project already includes a downgrade script:

```text
lerobot/common/datasets/v30/convert_dataset_v30_to_v21.py
```

Common conversion command:

```bash
python lerobot/common/datasets/v30/convert_dataset_v30_to_v21.py \
  --input-dir /root/Desktop/lerobot/so100_total_v30 \
  --output-dir /root/Desktop/lerobot/so100_total \
  --overwrite
```

For a quick parquet/metadata test without video conversion:

```bash
python lerobot/common/datasets/v30/convert_dataset_v30_to_v21.py \
  --input-dir /root/Desktop/lerobot/so100_total_v30 \
  --output-dir /root/Desktop/lerobot/so100_total_v21_test \
  --skip-videos \
  --overwrite
```

The script mainly does the following:

- Splits v3.0 `data/chunk-xxx/file-xxx.parquet` shards back into v2.1 per-episode files: `data/chunk-xxx/episode_000000.parquet`.
- Converts v3.0 `meta/tasks.parquet` to v2.1 `meta/tasks.jsonl`.
- Converts v3.0 `meta/episodes/chunk-xxx/file-xxx.parquet` to `meta/episodes.jsonl`.
- Rebuilds `meta/episodes_stats.jsonl` from `stats/...` fields in v3.0 episode metadata.
- Splits v3.0 merged videos `videos/{video_key}/chunk-xxx/file-xxx.mp4` into v2.1 files: `videos/chunk-xxx/{video_key}/episode_000000.mp4`.
- Rewrites `meta/info.json`, sets `codebase_version` to `v2.1`, and updates `total_episodes`, `total_frames`, `total_tasks`, `total_videos`, `data_path`, and `video_path`.
- Copies dataset card files such as `README.md` and `.gitattributes`.

Recommended post-conversion check:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/root/Desktop/lerobot/so100_total")
info = json.loads((root / "meta/info.json").read_text())
print("codebase_version:", info.get("codebase_version"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
print("data files:", len(list((root / "data").glob("chunk-*/*.parquet"))))
print("episode metadata:", (root / "meta/episodes.jsonl").exists())
print("episode stats:", (root / "meta/episodes_stats.jsonl").exists())
PY
```

Confirm that `codebase_version` is `v2.1` and that the number of parquet data files equals the number of episodes. The later training command can then keep using:

```bash
--dataset.root=/root/Desktop/lerobot/so100_total
```

## Train an ACT Policy

Training should run on a development machine or server with an NVIDIA GPU. The following command is the training command used by this project:

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=taoz/so100_total \
  --policy.type=act \
  --output_dir=/root/gpufree-data/outputs/train/act_so100_total \
  --job_name=act_so100_total \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.root=/root/Desktop/lerobot/so100_total
```

Argument notes:

- `--dataset.repo_id=taoz/so100_total`: dataset ID. Even when using a local path, keep it consistent with the recording command.
- `--dataset.root=/root/Desktop/lerobot/so100_total`: local dataset path for training.
- `--policy.type=act`: use the ACT policy.
- `--output_dir=/root/gpufree-data/outputs/train/act_so100_total`: training output directory.
- `--job_name=act_so100_total`: job name.
- `--policy.device=cuda`: use an NVIDIA GPU.
- `--wandb.enable=false`: disable Weights & Biases.

During training, watch for:

- Whether the loss generally decreases.
- Whether GPU memory usage is stable.
- Missing frames, missing cameras, or missing fields during data loading.
- Whether `checkpoints` are generated under the output directory.

Resume training:

```bash
python lerobot/scripts/train.py \
  --config_path=/root/gpufree-data/outputs/train/act_so100_total/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

After training, the common model path is:

```text
/root/gpufree-data/outputs/train/act_so100_total/checkpoints/last/pretrained_model
```

To use a fixed-step checkpoint, replace `last` with the actual checkpoint directory.

## Trajectory Smoothing Tuning

This project already includes three layers of trajectory smoothing: dataset-time smoothing, training-time smooth loss, and inference-time action queue smoothing. They address different stages, so do not increase all parameters aggressively at once.

### 1. Dataset-Time Action Smoothing

Location:

```text
lerobot/common/datasets/lerobot_dataset.py:836
```

Core logic:

```python
def _smooth_action_sequence(actions: list, window_radius: int = 5) -> list:
    ...

if "action" in episode_buffer:
    episode_buffer["action"] = self._smooth_action_sequence(episode_buffer["action"])
```

Purpose: before each episode is saved, the `action` sequence is smoothed with a moving average. The default `window_radius=5` means the actual window size is `2 * 5 + 1 = 11` frames. At 30 FPS, this covers about 0.37 seconds. The first and last 5 frames are kept unchanged to avoid biasing episode boundaries.

Tuning suggestions:

- `window_radius=3`: light smoothing, suitable for gripper actions, fast grasping, or already clean trajectories.
- `window_radius=5`: current default, suitable for medium-speed SO100 tasks such as grasping a pencil and placing it in a box.
- `window_radius=8`: stronger smoothing, useful when teleoperation jitter is obvious, but may slow down grasp/release timing.
- `window_radius=0` or removing the call inside `save_episode`: keep raw actions with no dataset-level smoothing.

If gripper opening/closing is being blurred or grasp timing becomes late, reduce `window_radius` from 5 to 3 first. If large arm joints are jittery but the gripper must stay fast, consider smoothing only the first five joint dimensions and skipping the gripper dimension.

### 2. Training-Time Smooth Loss

Location:

```text
lerobot/common/policies/act/modeling_act.py:79
```

Current parameters:

```python
self.smooth_loss_kernel_size = 11
self.smooth_loss_weight = 1.0
```

Purpose: during training, encourage the predicted action chunks to stay close to a moving-average trajectory. It does not directly modify the dataset. Instead, it adds a smoothing regularizer to predicted actions:

```text
total loss = original ACT loss + smooth_loss * smooth_loss_weight
```

Tuning suggestions:

- `smooth_loss_weight=0.2-0.5`: more flexible actions, less over-smoothing.
- `smooth_loss_weight=1.0`: current default and a good baseline.
- `smooth_loss_weight=1.5-2.0`: increase only when model rollout is clearly jittery.
- `smooth_loss_kernel_size=7`: faster response, weaker smoothing.
- `smooth_loss_kernel_size=11`: current default.
- `smooth_loss_kernel_size=15`: stronger smoothing, but may slow action changes.

Use an odd `smooth_loss_kernel_size` when possible. Even values are automatically reduced by 1 in the code. If training loss decreases normally but real-robot actions feel sluggish, do not keep increasing `smooth_loss_weight`; reduce it instead.

### 3. Inference-Time Action Queue Smoothing

Location:

```text
lerobot/common/policies/act/modeling_act.py:79
```

Current parameters:

```python
self.inference_smoothing_window = 8
self.max_action_increment = 0.06
```

Purpose:

- `inference_smoothing_window`: applies a moving average to the predicted action queue. The default value 8 means a window size of `17`.
- `max_action_increment`: limits action jumps. If the first action of a new chunk differs too much from the last executed action, transitional actions are inserted to avoid sudden robot jumps.

Tuning suggestions:

- Jitter, shaking, or action spikes: increase `inference_smoothing_window` from 8 to 10 or 12; reduce `max_action_increment` from 0.06 to 0.04.
- Motion is too slow, lags behind, or grasps too late: reduce `inference_smoothing_window` to 4 or 2; increase `max_action_increment` to 0.08 or 0.10.
- Obvious stalling: reduce `inference_smoothing_window` first, then check whether BPU/PyTorch inference is stable at 30 FPS.
- Sudden jumps only at chunk boundaries: tune `max_action_increment` downward first.
- High-frequency jitter through the whole trajectory: increase `inference_smoothing_window` or `smooth_loss_weight` first.

Try these three presets first:

```python
# Baseline, current default
self.inference_smoothing_window = 8
self.max_action_increment = 0.06
self.smooth_loss_kernel_size = 11
self.smooth_loss_weight = 1.0
```

```python
# More stable, for clear jitter when slower motion is acceptable
self.inference_smoothing_window = 10
self.max_action_increment = 0.04
self.smooth_loss_kernel_size = 11
self.smooth_loss_weight = 1.5
```

```python
# More responsive, for timing-sensitive grasping or over-smoothed actions
self.inference_smoothing_window = 4
self.max_action_increment = 0.08
self.smooth_loss_kernel_size = 7
self.smooth_loss_weight = 0.5
```

### 4. ACT Chunk Parameter Interaction

ACT defaults:

```text
lerobot/common/policies/act/configuration_act.py
chunk_size = 100
n_action_steps = 100
temporal_ensemble_coeff = None
```

To replan more frequently, reduce `n_action_steps` in the training command. It must not be greater than `chunk_size`:

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=taoz/so100_total \
  --policy.type=act \
  --policy.chunk_size=100 \
  --policy.n_action_steps=50 \
  --output_dir=/root/gpufree-data/outputs/train/act_so100_total \
  --job_name=act_so100_total \
  --policy.device=cuda \
  --wandb.enable=false \
  --dataset.root=/root/Desktop/lerobot/so100_total
```

Practical guidance:

- `n_action_steps=100`: better continuity, slower correction.
- `n_action_steps=50`: easier correction, suitable when object positions vary slightly.
- `n_action_steps=25`: more responsive, but may increase jitter if the model is unstable.

`temporal_ensemble_coeff=0.01` is a common ACT temporal ensembling setting, but when enabled, `n_action_steps` must be 1. This codebase already has action queue smoothing, so tune the existing three smoothing layers first. Only try temporal ensembling when you intentionally want to predict every frame and ensemble actions over time:

```bash
--policy.n_action_steps=1 \
--policy.temporal_ensemble_coeff=0.01
```

### 5. Recommended Tuning Order

1. Train and run PyTorch real-robot evaluation with default parameters, then record 5 to 10 eval episodes.
2. If the training data itself is jittery, tune `window_radius` in `lerobot_dataset.py`, then recollect or resave the dataset.
3. If the data is clean but model output is jittery, tune `smooth_loss_weight` and `smooth_loss_kernel_size`, then retrain.
4. If jitter happens only during execution or at chunk boundaries, tune `inference_smoothing_window` and `max_action_increment` without retraining first.
5. Change only one or two parameters at a time and save eval videos so the cause of each change remains clear.

## Post-Training Evaluation

Before using BPU, evaluate the PyTorch checkpoint directly on the robot:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a pencil and put it in the box." \
  --control.repo_id=taoz/eval_act_so100_total \
  --control.tags='["so100","eval"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=40 \
  --control.reset_time_s=10 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.policy.path=/root/gpufree-data/outputs/train/act_so100_total/checkpoints/last/pretrained_model
```

If PyTorch evaluation reliably completes the task, continue to BPU export and board-side deployment.

### Use the Open-Source Model for Evaluation

To test the already trained `tidy_desktop` model from this project, point `--control.policy.path` to the Hugging Face model repository:

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Tidy the desktop." \
  --control.repo_id=taoz/eval_tidy_desktop_local \
  --control.tags='["so100","eval","tidy_desktop"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=40 \
  --control.reset_time_s=10 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.policy.path=taoz-member/eval_tidy_desktop
```

For offline execution, download the model first:

```bash
huggingface-cli download \
  taoz-member/eval_tidy_desktop \
  --local-dir /root/Desktop/lerobot/models/eval_tidy_desktop
```

Then use the local path:

```bash
--control.policy.path=/root/Desktop/lerobot/models/eval_tidy_desktop
```

## Export ACT to BPU

The `rdk_LeRobot_tools` export script splits ACT into a vision encoder and transformer layers, exports ONNX, and generates OpenExplorer compilation configs plus `build_all.sh`.

### Update Export Config

Edit:

```text
rdk_LeRobot_tools/bpu_export_config.yaml
```

Recommended config:

```yaml
dataset:
  repo_id: "taoz/so100_total"
  root: "/root/Desktop/lerobot/so100_total"

policy:
  type: "act"
  device: "cpu"

wandb:
  enable: false

act_path: "/root/gpufree-data/outputs/train/act_so100_total/checkpoints/last/pretrained_model"
export_path: "bpu_export_output"
cal_num: 100
onnx_sim: true
type: "nash-e"
combine_jobs: 6
```

Notes:

- `dataset.root` must point to the collected dataset and is used to generate calibration data.
- `act_path` points to the trained ACT checkpoint. The directory should contain `config.json`, model weights, and training config.
- RDK S100/S100P uses the Nash architecture, so `type: "nash-e"` is usually used.
- `cal_num` is the number of calibration samples for quantization. Calibration data quality matters more than blindly increasing the number.

### Run Export

```bash
cd /root/Desktop/lerobot/rdk_LeRobot_tools
python export_bpu_actpolicy.py --config bpu_export_config.yaml
```

On success, this generates:

```text
rdk_LeRobot_tools/bpu_export_output/
  build_all.sh
  BPU_ACTPolicy_VisionEncoder/
  BPU_ACTPolicy_TransformerLayers/
  bpu_output/
```

`bpu_output` usually contains:

```text
BPU_ACTPolicy_VisionEncoder.hbm
BPU_ACTPolicy_TransformerLayers.hbm
action_mean.npy
action_std.npy
action_mean_unnormalize.npy
action_std_unnormalize.npy
<camera_name>_mean.npy
<camera_name>_std.npy
new_actions.npy
```

## OpenExplorer Compilation

ONNX-to-HBM compilation must be done inside the D-Robotics/OpenExplorer toolchain Docker environment. Inside the toolchain container, run:

```bash
cd /open_explorer
bash build_all.sh
```

Or, if the project directory is mounted to `/workspace`:

```bash
cd /workspace/rdk_LeRobot_tools/bpu_export_output
bash build_all.sh
```

After compilation, confirm that `.hbm` files exist under `bpu_output`:

```bash
ls -lh bpu_output
```

Expected files for RDK S100/S100P:

```text
BPU_ACTPolicy_VisionEncoder.hbm
BPU_ACTPolicy_TransformerLayers.hbm
```

## RDK Board Deployment

Copy the compiled `bpu_output` directory to the RDK board, for example:

```text
/root/Desktop/lerobot/bpu_output
```

Then run on the RDK board:

```bash
cd /root/Desktop/lerobot/rdk_LeRobot_tools
python bpu_control_robot.py \
  --bpu-act-path ../bpu_output \
  --fps 30 \
  --inference-time 60
```

### Required SO100 Check Before Deployment

The local `rdk_LeRobot_tools/bpu_control_robot.py` currently creates the robot with:

```python
robot = make_robot("so101")
```

For SO100 deployment, change it to:

```python
robot = make_robot("so100")
```

Otherwise, board-side inference will use the SO101 configuration, which can cause mismatched ports, calibration directories, and camera settings.

## Troubleshooting

### 1. Data Was Recorded but Training Cannot Find It

The recording command does not explicitly set `--control.root`, while the training command expects:

```text
/root/Desktop/lerobot/so100_total
```

Confirm where the dataset was actually saved. If it is not in this path, copy it there or add this argument during the next recording run:

```bash
--control.root=/root/Desktop/lerobot/so100_total
```

### 2. The Arm Does Not Move or Serial Ports Cannot Be Opened

Check:

```bash
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0 /dev/ttyUSB1
```

Then confirm that the leader/follower ports in `So100RobotConfig` are not swapped.

### 3. Teleoperation Direction Is Wrong or Joints Are Offset

Recalibrate first. If the issue remains, check motor IDs, assembly direction, whether the leader gears were removed, and whether cables are blocking motion.

### 4. Camera View Is Wrong or BPU Export Misses Camera Normalization Files

Make sure collection, training, export, and inference use the same `So100RobotConfig.cameras` names and indices. If the dataset camera names are `laptop` and `phone`, the BPU output should include the corresponding mean/std files.

### 5. BPU Compilation Cannot Find `hb_compile`

The current shell is not inside the OpenExplorer toolchain environment. Enter the D-Robotics toolchain Docker container and run `bash build_all.sh` again.

### 6. BPU Inference Starts but Actions Are Wrong

Check:

- Whether `bpu_control_robot.py` has been changed to `make_robot("so100")`.
- Whether `bpu_output` was generated from the same ACT checkpoint.
- Whether `dataset.root` matches the training dataset.
- Whether the board-side camera order matches the camera order used during training.

## Recommended Practice

- Record one test episode before each collection session and replay it immediately.
- Too many failed trajectories can noticeably hurt ACT training. Fewer clean trajectories are better than more noisy ones.
- Fix camera positions before training. Do not change viewpoints after training.
- Evaluate the PyTorch checkpoint on the real robot before BPU export to confirm the model itself works.
- Every time `configs.py` is changed, record the port mapping, camera indices, and date for easy rollback.

