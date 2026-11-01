# 视频录像与状态回放

录像和回放全部在 `ros2-learning` 内运行。外部工程配置的参考验收会记录完整的起飞、悬停、
机械臂运动和降落，结束后生成 `flight.mp4` 与浏览器回放页面 `flight.html`。

## 记录内容

运行时每个控制步保存 MuJoCo 的完整 integration state（当前 SITL 为 250 Hz），
同时保存最终编译模型，包括机械臂、起落架、相机、材质及纹理。
录制目录结构：

```text
recording/
  model.mjb       编译模型，自包含，不依赖原 MJCF/资源路径
  states.bin      小端 float64：时间戳 + mjSTATE_INTEGRATION 状态
  manifest.json   格式、版本、维数、样本数、时间范围、校验和与完成标记
```

运行时只复制/写入状态，视频渲染与编码在飞行验收结束后进行，避免额外编码负载干扰
ROS/PX4 控制周期。录制文件为顺序流，回放使用内存映射，不要求把整段状态加载进内存。
模型和状态都校验 SHA-256；必须使用记录时相同版本的 MuJoCo（当前 3.16.0）。

这是**已保存状态的回放**：恢复模型状态并计算画面，不重新执行控制器或物理积分，
不需要启动 PX4、ROS 或任何网络节点。它与评测模块中的“控制指令重新仿真回放”用途不同。
视频时间按仿真时间轴取样，默认 20 fps、640×480、H.264/yuv420p；墙钟运行较慢不会
改变导出视频的运动速度。非记录时刻采用前一份状态，最大时间偏差约一个记录周期。

## 自动记录与播放

```bash
source /opt/ros/jazzy/setup.bash
source /opt/uamsim/UAMROS2/install/setup.bash
cd /opt/uamsim/project
/opt/venv/bin/python -m examples.run_experiment \
  --config /opt/uamsim/UAMROS2/cases/local_reference.json --output logs/new-run
```

输出目录由脚本打印。用浏览器打开该目录的 `flight.html`，支持播放/暂停、时间轴拖动、
全屏和 0.5×/1×/2×速度。也可直接用任意 MP4 播放器打开 `flight.mp4`。
浏览器页面与 MP4 必须放在同一目录中；可以将这两个文件复制到 Windows 后播放。

## 手动录制

在原 SITL 启动命令后追加一个全新的录制目录：

```bash
cd /opt/uamsim/project
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 /opt/venv/bin/python -m examples.px4_sitl_server \
  --host 127.0.0.1 --arm --sensors --duration 90 --record logs/my-flight/recording
```

然后按同机部署指南启动本地 PX4 和 ROS 2。录制目录已存在时拒绝覆盖。
仿真异常退出时保存 `completed=false`；硬退出导致的截断文件不能当作完整录像回放。
当前单次录制不支持 reset，时间必须严格递增；新回合应建立新目录。

## 独立回放与重新导出

无须重新起飞，在容器内执行：

```bash
cd /opt/uamsim/project
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 /opt/venv/bin/python -m examples.replay_recording \
  logs/my-flight/recording --output logs/my-flight/overview.mp4
```

截取第 10–30 秒，以半速播放：

```bash
MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 /opt/venv/bin/python -m examples.replay_recording \
  logs/my-flight/recording --output logs/my-flight/slow-motion.mp4 \
  --start 10 --end 30 --speed 0.5 --fps 30
```

`--camera overview` 为固定第三人称整机视角；如果录制模型包含传感器相机，
可使用 `--camera uam/front_camera` 从机载相机视角重新导出。输出宽高须为偶数。
画面左上角显示原仿真时间与导出速度。输出 MP4/HTML 已存在时拒绝覆盖；
编码失败的临时文件使用 `.partial.mp4` 后缀，不标记为成功录像。

已经正常关闭但标记未完成的录制，可以明确添加 `--allow-incomplete` 查看已有部分；
仍须通过版本、模型/状态校验和、尺寸及时间戳检查。

代码接口：`StateRecorder.capture(data)`、`StateRecorder.close(completed=...)`；
`StateReplay.seek(sim_time)` 支持按时间定位，`close()` 或上下文管理器释放映射文件。
通过 `from UAMSim.recording import StateRecorder, StateReplay, export_video` 使用这些接口。
录像没有音轨。当前回放不伪造 ROS 话题，也不向电机或 PX4 重新发送控制命令。

2026-11-01 实测输出位于 `logs/local-recording-01`：`flight.mp4` / `flight.html` 为
90 秒整机录像，`onboard-slow.mp4` / `onboard-slow.html` 为机载视角半速片段。
两个 MP4 均经 ffprobe 核验帧数和时长；状态精确定位、校验失败拒绝、非完整回合标记、
裁剪/慢速帧数与播放器生成等 4 项专项测试通过，完整回归共 79 项通过。
