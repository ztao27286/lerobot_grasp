# SO100 + LeRobot + RDK 全流程说明

## 开源资源与参考

本项目相关数据集和训练模型已在 Hugging Face 开源，可用于复现实验、对比训练效果或作为数据格式参考。

作者本人的开源数据集：

- [`taoz-member/tidy_desktop`](https://huggingface.co/datasets/taoz-member/tidy_desktop)：桌面整理任务数据集。
- [`taoz-member/grab_blocks`](https://huggingface.co/datasets/taoz-member/grab_blocks)：抓取积木任务数据集。

作者本人的开源训练模型：

- [`taoz-member/eval_tidy_desktop`](https://huggingface.co/taoz-member/eval_tidy_desktop)：`tidy_desktop` 任务对应的训练/评估模型。

## 目录结构

当前目录大致包含：

```text
lerobot_grasp/
  lerobot/                  LeRobot 主代码
  examples/                 官方/本地示例，SO100 文档在 examples/10_use_so100.md
  docs/                     LeRobot 文档
  media/                    示例图片、教程素材
  rdk_LeRobot_tools/        RDK S100/S100P BPU 导出和板端推理工具
  bpu_output/               已有 BPU 推理产物示例
  pyproject.toml            Python 包配置
```

`rdk_LeRobot_tools` stable 版本主要面向较旧的 LeRobot/v2.1 数据集流程，并且当前主要验证的是 ACT 模型在 RDK S100 上的部署。使用 SO100 时，采集和训练命令使用 `--robot.type=so100` 与 ACT 策略；后续 BPU 推理脚本如果仍默认 `so101`，需要手动改成 `so100`。

## 总体流程

1. 准备开发机/RDK 环境。
2. 配置 SO100 电机 ID、机械臂端口和摄像头。
3. 校准主手和从手。
4. 遥操作检查机械臂与相机是否正常。
5. 采集 `taoz/so100_total` 数据集。
6. 在带 NVIDIA GPU 的开发机上训练 ACT。
7. 导出 ACT 模型为 ONNX，并用 OpenExplorer 工具链编译为 BPU 模型。
8. 把 `bpu_output` 拷贝到 RDK 板端，运行 BPU 推理。

## 环境准备

建议使用 Python 3.10。开发机用于训练和导出，RDK 板端用于标定、采集、遥操作和最终推理。

### 开发机或训练服务器

```bash
cd /root/Desktop/lerobot

conda create -y -n lerobot python=3.10
conda activate lerobot

pip install -e ".[feetech]"
pip install onnx onnxsim termcolor tqdm
```

如果遇到 LeRobot 版本和数据集格式兼容问题，优先使用 D-Robotics 的 LeRobot fork，或确认 `datasets==2.19.0`：

```bash
pip install "datasets==2.19.0"
```

训练 ACT 需要 NVIDIA GPU 和可用的 CUDA 版 PyTorch。确认方式：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

### RDK 板端

```bash
cd /root/Desktop/lerobot

pip install -e ".[feetech]"
pip install hbm-runtime
```

板端如果只做采集和遥操作，`hbm-runtime` 不是必须；如果要跑 BPU 推理，则必须安装。

## SO100 硬件配置

### 查找机械臂串口

先把 leader/follower 两个总线转接板连接到机器，然后运行：

```bash
python lerobot/scripts/find_motors_bus_port.py
```

按提示拔插 USB，记录主手和从手对应端口，例如：

```text
leader:   /dev/ttyUSB0
follower: /dev/ttyUSB1
```

Linux/RDK 上如果没有串口权限，可以临时执行：

```bash
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1
```

### 配置电机 ID

SO100 主手和从手各 6 个 Feetech `sts3215` 电机，ID 通常为 1 到 6。每次只连接一个电机，然后按实际端口依次设置：

```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/ttyUSB0 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```

把 `--ID` 依次改为 `2`、`3`、`4`、`5`、`6`，主手和从手都要设置。建议给电机贴标签，例如 `L1-L6`、`F1-F6`，后面排查会轻松很多。

### 修改 SO100 端口配置

打开：

```text
lerobot/common/robot_devices/robots/configs.py
```

找到 `So100RobotConfig`，把 `leader_arms` 和 `follower_arms` 中的 `port` 改成实际端口：

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

## 摄像头配置

连接摄像头后查看索引：

```bash
python lerobot/common/robot_devices/cameras/opencv.py \
  --images-dir outputs/images_from_opencv_cameras
```

查看生成的图片，确认 `camera_index=0`、`camera_index=1` 分别对应哪个视角。随后在 `So100RobotConfig` 的 `cameras` 中更新：

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

相机名字会写入数据集，也会影响后续 BPU 导出产物中的归一化文件名。训练、导出、板端推理最好使用同一套相机命名。

## 标定

标定前确认机械臂没有卡线、急停或明显限位风险。建议先不接任务物体，只验证机械臂本体。

### 标定从手

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'
```

### 标定主手

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'
```

标定会写入 `.cache/calibration/so100`。如果换了机械臂、重装结构、换了端口映射或动作明显错位，建议重新标定。

## 遥操作检查

先不打开相机，仅检查主从手跟随：

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

再打开相机和数据流，确认图像、关节状态和动作同步：

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=teleoperate \
  --control.fps=30
```

如果画面打不开，先回到摄像头配置步骤确认索引。若终端提示 headless 环境，键盘快捷键和窗口显示可能不可用。

## 采集数据集

下面是本项目使用的采集命令，默认 `taoz/so100_total`、60 条 episode、每条 40 秒、不开启上传：

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

采集时常用快捷键：

- 右方向键：提前结束当前录制阶段，进入 reset；在 reset 阶段按下则提前进入下一条。
- 左方向键：放弃当前 episode 并重新录制。
- Esc：停止整个采集任务并保存已经完成的数据。

注意：上面的命令没有显式写 `--control.root`。如果 LeRobot 使用默认本地缓存路径，数据可能保存到 Hugging Face/LeRobot 默认缓存目录，而不是 `/root/Desktop/lerobot/so100_total`。训练命令会从 `/root/Desktop/lerobot/so100_total` 读取数据，所以采集完成后请确认数据集最终位于该路径；如果不在，请把数据集复制过去，或者下次采集时额外加：

```bash
--control.root=/root/Desktop/lerobot/so100_total
```

为了保持你的原始采集命令不变，上面的参数没有直接加入正式命令。

## 数据集检查

采集完成后建议先做三件事：

1. 检查 episode 数量是否接近 60。
2. 抽查视频画面是否清晰、视角是否固定、物体是否在画面内。
3. 回放一条 episode，确认动作和图像同步。

可本地可视化：

```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id taoz/so100_total \
  --local-files-only 1
```

也可以回放第 0 条 episode：

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=taoz/so100_total \
  --control.episode=0
```

如果数据集在指定本地路径，必要时补充对应的 root 参数。

## 使用开源数据集

如果不重新采集，也可以直接参考 Hugging Face 上本作者已经开源的数据集。在线训练时可把 `--dataset.repo_id` 换成目标数据集：

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=taoz-member/tidy_desktop \
  --policy.type=act \
  --output_dir=/root/gpufree-data/outputs/train/act_tidy_desktop \
  --job_name=act_tidy_desktop \
  --policy.device=cuda \
  --wandb.enable=false
```

另一个开源数据集训练示例：

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=taoz-member/grab_blocks \
  --policy.type=act \
  --output_dir=/root/gpufree-data/outputs/train/act_grab_blocks \
  --job_name=act_grab_blocks \
  --policy.device=cuda \
  --wandb.enable=false
```

如果网络不稳定，建议先下载到本地目录，再用 `--dataset.root` 指向本地路径：

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

注意：开源数据集如果是 v3.0 结构，而当前 RDK 导出链路需要 v2.1，请先按下一节转换后再用于 BPU 导出。

## 数据集 v3.0 转 v2.1

`rdk_LeRobot_tools` stable 版本主要按 LeRobot v2.1 数据集结构验证。若你用较新的 LeRobot 采集到了 v3.0 数据集，建议先转成 v2.1，再用于训练、ACT 导出和 BPU 编译。

本工程已经带有降级脚本：

```text
lerobot/common/datasets/v30/convert_dataset_v30_to_v21.py
```

常用转换命令：

```bash
python lerobot/common/datasets/v30/convert_dataset_v30_to_v21.py \
  --input-dir /root/Desktop/lerobot/so100_total_v30 \
  --output-dir /root/Desktop/lerobot/so100_total \
  --overwrite
```

如果只想先快速验证 parquet 和 metadata 能不能转，可以临时跳过视频：

```bash
python lerobot/common/datasets/v30/convert_dataset_v30_to_v21.py \
  --input-dir /root/Desktop/lerobot/so100_total_v30 \
  --output-dir /root/Desktop/lerobot/so100_total_v21_test \
  --skip-videos \
  --overwrite
```

脚本做的主要改动：

- 把 v3.0 的 `data/chunk-xxx/file-xxx.parquet` 拆回 v2.1 的单 episode 文件：`data/chunk-xxx/episode_000000.parquet`。
- 把 v3.0 的 `meta/tasks.parquet` 转成 v2.1 的 `meta/tasks.jsonl`。
- 把 v3.0 的 `meta/episodes/chunk-xxx/file-xxx.parquet` 转成 `meta/episodes.jsonl`。
- 从 v3.0 episode metadata 里的 `stats/...` 字段重建 `meta/episodes_stats.jsonl`。
- 把 v3.0 合并视频 `videos/{video_key}/chunk-xxx/file-xxx.mp4` 拆成 v2.1 的 `videos/chunk-xxx/{video_key}/episode_000000.mp4`。
- 重写 `meta/info.json`，把 `codebase_version` 改为 `v2.1`，并更新 `total_episodes`、`total_frames`、`total_tasks`、`total_videos`、`data_path`、`video_path` 等字段。
- 复制数据集卡片文件，例如 `README.md` 和 `.gitattributes`。

转换后建议检查：

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

确认输出里 `codebase_version` 是 `v2.1`，并且 data parquet 数量等于 episode 数量。后续训练命令继续使用：

```bash
--dataset.root=/root/Desktop/lerobot/so100_total
```

## 训练 ACT 策略

训练建议在带 NVIDIA GPU 的开发机或训练服务器上执行。下面是本项目使用的训练命令：

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

参数说明：

- `--dataset.repo_id=taoz/so100_total`：数据集 ID。即使使用本地路径，也建议保持和采集时一致。
- `--dataset.root=/root/Desktop/lerobot/so100_total`：训练读取的本地数据集目录。
- `--policy.type=act`：使用 ACT 策略。
- `--output_dir=/root/gpufree-data/outputs/train/act_so100_total`：训练输出目录。
- `--job_name=act_so100_total`：任务名。
- `--policy.device=cuda`：使用 NVIDIA GPU。
- `--wandb.enable=false`：关闭 Weights & Biases。

训练过程中重点观察：

- loss 是否整体下降。
- GPU 显存是否稳定。
- 数据读取是否报缺帧、缺相机、缺字段。
- 输出目录是否生成 `checkpoints`。

恢复训练示例：

```bash
python lerobot/scripts/train.py \
  --config_path=/root/gpufree-data/outputs/train/act_so100_total/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

训练完成后，常用的模型路径是：

```text
/root/gpufree-data/outputs/train/act_so100_total/checkpoints/last/pretrained_model
```

如果你想使用某个固定 step 的 checkpoint，把 `last` 换成实际目录名。

## 轨迹平滑调优

本工程里已经加了三层轨迹平滑逻辑：数据保存时平滑、训练时平滑损失、推理时动作队列平滑。它们分别处理不同阶段的问题，不建议一次把所有参数大幅调高。

### 1. 数据集保存时平滑

位置：

```text
lerobot/common/datasets/lerobot_dataset.py:836
```

核心逻辑：

```python
def _smooth_action_sequence(actions: list, window_radius: int = 5) -> list:
    ...

if "action" in episode_buffer:
    episode_buffer["action"] = self._smooth_action_sequence(episode_buffer["action"])
```

作用：在每条 episode 保存前，对 `action` 做滑动平均。默认 `window_radius=5`，实际窗口大小是 `2 * 5 + 1 = 11` 帧。30 FPS 下大约覆盖 0.37 秒。代码会保留开头和结尾各 5 帧原始动作，避免 episode 边界被均值拉偏。

调参建议：

- `window_radius=3`：轻微平滑，适合夹爪动作、快速抓取、轨迹本身比较干净的数据。
- `window_radius=5`：当前默认值，适合 SO100 抓笔放盒子这类中速任务。
- `window_radius=8`：强平滑，适合遥操作抖动明显的数据，但可能让抓取、放置瞬间变慢。
- `window_radius=0` 或去掉 `save_episode` 里的调用：保留原始动作，不做数据级平滑。

如果发现夹爪开合被“抹平”、抓取时机变晚，优先把 `window_radius` 从 5 降到 3。若机械臂大关节抖动明显但夹爪需要快，可以进一步改成只平滑前 5 个关节、跳过夹爪维度。

### 2. 训练时平滑损失

位置：

```text
lerobot/common/policies/act/modeling_act.py:79
```

当前参数：

```python
self.smooth_loss_kernel_size = 11
self.smooth_loss_weight = 1.0
```

作用：训练时让模型输出的 action chunk 更接近滑动平均轨迹。它不会直接修改数据集，而是给预测动作增加一个平滑正则项：

```text
总 loss = ACT 原始 loss + smooth_loss * smooth_loss_weight
```

调参建议：

- `smooth_loss_weight=0.2-0.5`：动作需要更灵活，减少过度平滑。
- `smooth_loss_weight=1.0`：当前默认值，先用这个作为基线。
- `smooth_loss_weight=1.5-2.0`：模型 rollout 抖动明显时再增加。
- `smooth_loss_kernel_size=7`：响应更快，平滑弱一些。
- `smooth_loss_kernel_size=11`：当前默认值。
- `smooth_loss_kernel_size=15`：平滑更强，但容易拖慢动作变化。

注意：`smooth_loss_kernel_size` 最好用奇数。偶数在代码里会自动减 1。若训练 loss 正常下降但实机动作拖泥带水，不要继续加大 `smooth_loss_weight`，应减小它。

### 3. 推理时动作队列平滑

位置：

```text
lerobot/common/policies/act/modeling_act.py:79
```

当前参数：

```python
self.inference_smoothing_window = 8
self.max_action_increment = 0.06
```

作用：

- `inference_smoothing_window`：对预测动作队列做滑动平均。默认 8，对应窗口大小 `17`。
- `max_action_increment`：限制前后动作跳变。如果新 chunk 的第一帧和上一次执行动作差太大，代码会插入过渡动作，避免机械臂突然跳。

调参建议：

- 抖动、甩动、动作有尖峰：把 `inference_smoothing_window` 从 8 增加到 10 或 12；把 `max_action_increment` 从 0.06 降到 0.04。
- 动作太慢、跟不上、抓取晚：把 `inference_smoothing_window` 降到 4 或 2；把 `max_action_increment` 提到 0.08 或 0.10。
- 出现明显卡顿：优先减小 `inference_smoothing_window`，再检查 BPU/PyTorch 推理帧率是否稳定到 30 FPS。
- 只在 chunk 切换时突然抖一下：优先调小 `max_action_increment`。
- 整条轨迹一直有高频抖动：优先调大 `inference_smoothing_window` 或 `smooth_loss_weight`。

推荐先按下面三档试：

```python
# 基线，当前默认
self.inference_smoothing_window = 8
self.max_action_increment = 0.06
self.smooth_loss_kernel_size = 11
self.smooth_loss_weight = 1.0
```

```python
# 更稳，适合抖动明显但允许慢一点
self.inference_smoothing_window = 10
self.max_action_increment = 0.04
self.smooth_loss_kernel_size = 11
self.smooth_loss_weight = 1.5
```

```python
# 更灵活，适合抓取时机要求高或动作被抹平
self.inference_smoothing_window = 4
self.max_action_increment = 0.08
self.smooth_loss_kernel_size = 7
self.smooth_loss_weight = 0.5
```

### 4. ACT chunk 参数联动

ACT 默认：

```text
lerobot/common/policies/act/configuration_act.py
chunk_size = 100
n_action_steps = 100
temporal_ensemble_coeff = None
```

如果要更频繁重规划，可以在训练命令中减小 `n_action_steps`，但不能大于 `chunk_size`：

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

经验上：

- `n_action_steps=100`：动作连续性更好，但模型纠偏慢。
- `n_action_steps=50`：更容易纠偏，适合物体位置有轻微变化的任务。
- `n_action_steps=25`：响应更快，但若模型不稳可能增加抖动。

`temporal_ensemble_coeff=0.01` 是 ACT 常见的时序集成设置，但启用它时 `n_action_steps` 必须为 1。当前这份代码已经有动作队列平滑，建议先调现有三层平滑；只有在你明确想每帧重新预测并做时序集成时，再尝试：

```bash
--policy.n_action_steps=1 \
--policy.temporal_ensemble_coeff=0.01
```

### 5. 推荐调试顺序

1. 先用默认参数训练和 PyTorch 实机评估，记录 5 到 10 条 eval episode。
2. 如果训练数据本身抖，先调 `lerobot_dataset.py` 的 `window_radius`，重新采集或重新保存数据。
3. 如果数据干净但模型输出抖，调 `smooth_loss_weight` 和 `smooth_loss_kernel_size`，重新训练。
4. 如果只是在执行时抖或 chunk 切换时跳，调 `inference_smoothing_window` 和 `max_action_increment`，先不重训。
5. 每次只改一到两个参数，并保存 eval 视频，避免不知道是哪一项带来的变化。

## 训练后评估

可以先不用 BPU，直接用 PyTorch checkpoint 在机器人上录制评估数据：

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

如果 PyTorch 评估已经能稳定完成任务，再进入 BPU 导出和板端部署。

### 使用开源模型评估

如果想直接测试本作者已训练完毕的 `tidy_desktop` 模型，可以把 `--control.policy.path` 指向 Hugging Face 模型仓库：

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

如果需要离线运行，也可以先下载模型：

```bash
huggingface-cli download \
  taoz-member/eval_tidy_desktop \
  --local-dir /root/Desktop/lerobot/models/eval_tidy_desktop
```

然后改成本地路径：

```bash
--control.policy.path=/root/Desktop/lerobot/models/eval_tidy_desktop
```

## 导出 ACT 到 BPU

`rdk_LeRobot_tools` 的导出脚本会把 ACT 拆成视觉编码器和 Transformer 部分，先导出 ONNX，再生成 OpenExplorer 编译配置和 `build_all.sh`。

### 修改导出配置

编辑：

```text
rdk_LeRobot_tools/bpu_export_config.yaml
```

建议改成类似下面这样：

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

说明：

- `dataset.root` 必须能读取到采集数据，用于生成校准数据。
- `act_path` 指向训练好的 ACT checkpoint，目录中应包含 `config.json`、权重文件和训练配置。
- RDK S100/S100P 使用 Nash 架构，通常选择 `type: "nash-e"`。
- `cal_num` 是用于量化校准的数据量，数据质量比盲目增大数量更重要。

### 运行导出

```bash
cd /root/Desktop/lerobot/rdk_LeRobot_tools
python export_bpu_actpolicy.py --config bpu_export_config.yaml
```

成功后会生成：

```text
rdk_LeRobot_tools/bpu_export_output/
  build_all.sh
  BPU_ACTPolicy_VisionEncoder/
  BPU_ACTPolicy_TransformerLayers/
  bpu_output/
```

`bpu_output` 中通常包含：

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

## OpenExplorer 编译

ONNX 到 HBM 的编译需要在 D-Robotics/OpenExplorer 工具链 Docker 环境中完成。进入工具链容器后执行：

```bash
cd /open_explorer
bash build_all.sh
```

或者如果你把工程目录挂载到 `/workspace`：

```bash
cd /workspace/rdk_LeRobot_tools/bpu_export_output
bash build_all.sh
```

编译成功后，确认 `bpu_output` 下存在 `.hbm` 文件：

```bash
ls -lh bpu_output
```

RDK S100/S100P 预期看到：

```text
BPU_ACTPolicy_VisionEncoder.hbm
BPU_ACTPolicy_TransformerLayers.hbm
```

## RDK 板端部署

把编译完成的 `bpu_output` 拷贝到 RDK 板端，例如：

```text
/root/Desktop/lerobot/bpu_output
```

然后在 RDK 上执行：

```bash
cd /root/Desktop/lerobot/rdk_LeRobot_tools
python bpu_control_robot.py \
  --bpu-act-path ../bpu_output \
  --fps 30 \
  --inference-time 60
```

### SO100 部署前必须检查

当前本地 `rdk_LeRobot_tools/bpu_control_robot.py` 中机器人创建代码是：

```python
robot = make_robot("so101")
```

如果你部署的是 SO100，请先把它改成：

```python
robot = make_robot("so100")
```

否则板端推理会按 SO101 的配置连接机械臂，容易出现端口、校准目录、相机配置不一致。

## 常见问题

### 1. 采集成功但训练找不到数据

你的采集命令没有显式指定 `--control.root`，而训练命令指定了：

```text
/root/Desktop/lerobot/so100_total
```

请确认数据实际保存位置。如果不在这个路径，把数据集复制过去，或重新采集时增加：

```bash
--control.root=/root/Desktop/lerobot/so100_total
```

### 2. 机械臂不动或串口打不开

检查：

```bash
ls /dev/ttyUSB*
sudo chmod 666 /dev/ttyUSB0 /dev/ttyUSB1
```

再确认 `So100RobotConfig` 中 leader/follower 端口没有写反。

### 3. 遥操作方向怪、关节错位

优先重新标定。若仍然异常，检查电机 ID、装配方向、主手是否去齿轮、线缆是否卡住。

### 4. 相机画面错位或导出缺少相机归一化文件

确保采集、训练、导出、推理使用同一套 `So100RobotConfig.cameras` 名称和索引。数据集中相机名如果是 `laptop`、`phone`，BPU 输出也应该有对应的均值/方差文件。

### 5. BPU 编译找不到 `hb_compile`

说明当前 shell 不在 OpenExplorer 工具链环境里。请进入 D-Robotics 提供的工具链 Docker，再执行 `bash build_all.sh`。

### 6. BPU 推理能启动但动作不对

检查四点：

- `bpu_control_robot.py` 是否已改为 `make_robot("so100")`。
- `bpu_output` 是否来自同一个 ACT checkpoint。
- `dataset.root` 是否和训练数据一致。
- 板端相机顺序和训练时相机顺序是否一致。

## 推荐工作习惯

- 每次采集前先录 1 条测试 episode，并马上回放。
- 数据集中失败轨迹太多会明显影响 ACT，宁可少一点也要干净。
- 训练前固定好摄像头位置，不要训练后再换视角。
- BPU 导出前先用 PyTorch checkpoint 做实机评估，确认模型本身可用。
- 每次改 `configs.py` 后记录端口、相机索引和日期，方便回滚。
