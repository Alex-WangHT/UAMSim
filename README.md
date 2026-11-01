# UAMSim — 空中机械臂验证与对比框架

UAMSim 负责 MuJoCo 模型、任务、扰动、通信接口、指标、实验结果、录像与回放。PX4 飞控算法和自定义 ROS 2 包是独立工程，框架不构建 PX4、不固定 ROS 节点、不依赖 ROS 2 安装。

| 工程 | 职责 |
| --- | --- |
| UAMSim（本仓库） | 仿真、可重复评测、基线对比、状态记录与视频回放 |
| UAMFlight（同级独立工程） | 外部飞控算法、PX4 构建配置、机型及 SITL 启动 |
| UAMROS2（同级独立工程） | 自定义 ROS 发布/订阅、MAVLink 适配、机械臂与 CANopen 接口 |

独立工程无需随本仓库安装；可以替换成你自己的实现。当前参考部署把三个工程放在同一台机器的 `ros2-learning` 容器中。

## 安装与基线对比

```bash
python -m pip install -e '.[dev,sitl]'
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 python -m unittest discover -s tests -v
python -m examples.benchmark --controllers pid geometric --tasks hover tracking arm_motion --seeds 0 1 2
```

Python >= 3.10；无界面渲染需要 EGL/Mesa，视频导出需要 FFmpeg。`sitl` 是可选的 MAVLink 依赖，纯基线评测无需安装。PID/几何控制器作为验证用参考基线保留。

## 接入外部工程

实验配置属于你自己的集成工程。配置中的 `start`、`check`、`wait` 分别表示启动服务、执行验收和等待服务结束；命令使用参数数组，框架不拼接 shell 命令。

```bash
python -m examples.run_experiment --config /path/to/your/case.json --output logs/candidate-001
python -m examples.run_experiment --compare logs/reference/summary.json logs/candidate-001/summary.json --output logs/comparison.json
```

比较要求相同 `case_id`、不同 `label`，失败实验也保留。使用者负责保证同一 case 下模型、任务、种子和指标含义一致。配置规范见 [实验与接口契约](docs/EXPERIMENT_CONTRACT.md)。

## 录像与回放

```bash
python -m examples.px4_sitl_server --arm --sensors --duration 90 --record logs/run/recording
# 另行启动你自己的飞控和 ROS 工程；完成后可独立导出视频
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 python -m examples.replay_recording logs/run/recording --output logs/run/flight.mp4
```

生成 MP4 与带拖动、暂停、倍速的 HTML 播放页面。离线回放无需 PX4/ROS，须使用录制时相同的 MuJoCo 版本。详见 [录像机制](docs/RECORDING_AND_REPLAY.md)。

本机完整用法放在工程同级 `../docs/UAMSim使用说明.md`。该同级文档及 UAMFlight、UAMROS2 不属于此 Git 仓库；本仓库 README 和接口文档足以接入其他外部工程。

旧的三线程 API 保留兼容；新增验证链路采用单线程物理状态所有者，外部控制器经通信边界下发命令。历史 API 说明见 [DOCUMENT.md](docs/DOCUMENT.md)。
