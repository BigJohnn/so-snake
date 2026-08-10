# so-snake

一个基于 SO-ARM100 Follower 的主动感知抓放项目。目标是让机械臂理解主人的意图,
例如「找到桌面上的红色方块,放在框里」,并通过 **主动观察 → 抓取 → 搬运 → 放置 → 验证**
的闭环完成任务。

数据通过 Nintendo Pro 手柄遥操作采集。整体设计见 [`image.png`](image.png)(蓝图草稿)。

## 硬件

| 项 | 配置 |
|---|---|
| 机械臂 | SO-ARM100(**SO-100 旧版**,非 SO-101) |
| 自由度 | 5 DoF + gripper |
| 腕部相机 | 普通 USB/UVC |
| 第三人称相机 | 普通 USB/UVC(仅用于标注/调试/蒸馏,策略不吃) |
| 遥操作 | Nintendo Switch Pro 手柄 |
| 训练机 | RTX 4090 24G |

## 开发环境

优先使用仓库内最小环境。它只安装离线 Gate 需要的依赖,方便迁移到新机器或 macOS;
硬件采集/训练再切到 lerobot 环境。

```bash
# 推荐: uv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# 没有 uv 时
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
PYTHONPATH=src scripts/check_teleop_loop.py --steps 300
```

最小环境只保证这些离线路径:

| 依赖 | 来源 | 用途 |
|---|---|---|
| numpy | base dependency | FK/Jacobian/5D IK/atlas/Mock 闭环 |
| pytest | `.[dev]` | Gate A/B 与 Mock 闭环测试 |

可选能力按需安装,不要让它们阻塞最小环境:

```bash
uv pip install -e ".[dev,sim]"   # MuJoCo 仿真/离屏相机
```

MuJoCo 仿真(含离屏相机)在 macOS Apple Silicon(MBP M1/M2/…)与 Linux 上通用。
唯一的平台差异是交互式 viewer:macOS 要求 passive viewer 独占主线程,须由
`.[sim]` 附带的 `mjpython` 启动。`scripts/view_pro_controller_sim.py` 会在 macOS
上用普通 `python` 启动时自动 re-exec 到 `mjpython`,无需手动切换。

viewer 的遥操作输入也自动选路:`--source auto`(默认)先尝试真手柄,缺 lerobot 或
没插手柄(macOS 常态)时回退到内置 scripted 波形,让仿真照样动;`--source pro`
强制真手柄,`--source scripted` 强制波形。想在本机用真手柄,装 `teleop` 附加层。
注意:Switch Pro 的 `NintendoTeleop` 只在 lab 的 `linkage-x/lerobot` fork 的 `box`
分支上,huggingface 上游 main 没有,所以 `teleop` 附加层锁定该 fork 分支:

```bash
# GIT_LFS_SKIP_SMUDGE=1 跳过 lerobot 仓库缺失的 LFS 测试图片,否则 git-lfs smudge 会让安装失败
# 需要对 linkage-x/lerobot 的 SSH 访问权限
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev,sim,teleop]"   # 真手柄遥操作(拉取 torch 等,较重)
PYTHONPATH=src .venv/bin/python scripts/view_pro_controller_sim.py                  # 自动选路
PYTHONPATH=src .venv/bin/python scripts/view_pro_controller_sim.py --source scripted # 无手柄看仿真
```

| 依赖 | 安装层 | 用途 |
|---|---|---|
| mujoco | `.[sim]` | URDF/STL 仿真、解析 Jacobian 交叉验证、离屏相机;macOS viewer 经 `mjpython` |
| lerobot + hidapi | `.[teleop]` | 本机真 Switch Pro 手柄遥操作(lerobot=linkage-x fork `box` 分支);viewer `--source pro` |
| placo | lab/lerobot 环境 | 独立 FK 交叉验证 |
| lerobot | lab/lerobot 环境 | SOFollower、NintendoTeleop、LeRobotDataset、训练 |
| feetech-servo-sdk | lab/lerobot 环境 | STS3215 舵机通讯 |
| hidapi | lab/lerobot 环境 | Pro 手柄读取 |
| torch | lab/lerobot 环境 | 训练 |

`~/Codes/lerobot/.venv` 仍是当前硬件/数据采集的完整环境,但不应作为跑本仓库离线
测试的前置条件:

```bash
/home/hanyu/Codes/lerobot/.venv/bin/python scripts/check_kinematics_agreement.py
```

注意:系统环境里可能有 ROS pytest 插件,会在缺依赖或 hook 不匹配时污染测试收集。
本仓库 Gate 使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`。

## 真机 preflight

上真机前先在插着臂的机器上跑 preflight。除 `--probe` 外全部不碰硬件;`--probe`
只读地打开舵机总线(ping + 读当前位置),**不上力矩、不发目标位、不会让臂动**:

```bash
# 安全检查:依赖 / 关节契约 / 串口 / 手柄 / 标定文件是否存在
PYTHONPATH=src .venv/bin/python scripts/preflight_real_arm.py --port <PORT>
# 追加只读总线探测(臂需上电插好;不会动臂)
PYTHONPATH=src .venv/bin/python scripts/preflight_real_arm.py --port <PORT> --probe
```

真机需要 `.[teleop]` 附加层(含 `feetech-servo-sdk` 提供 `scservo_sdk`)。macOS 上
串口通常是 `/dev/cu.usbmodem*`。首次使用必须先标定(会手动移动臂过全程),标定与
第一次发力矩/运动都是操作者手动步骤,脚本不代劳。舵机 id 已是 1–6(preflight 探测
确认),无需 `lerobot-setup-motors`,直接标定即可:

```bash
# 交互式:先把每个关节摆到量程中点回车,再把所有关节各自过一遍全程,回车结束
lerobot-calibrate --robot.type=so100_follower \
    --robot.port=/dev/cu.usbmodem58760434321 --robot.id=so_snake
```

标定文件写到 `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`;
要重标定就先删掉它。官方教程:<https://huggingface.co/docs/lerobot/en/so100>
(SO-100 与 SO-101 标定流程通用,视频见 SO-101 文档)。

### 关节坐标映射(lerobot ↔ URDF)

lerobot 标定后的度数(零位=量程中点)和本仓运动学用的 **URDF 约定不一致**,直接
驱动会撞机。用 `scripts/map_joint_frames.py` 建立并核对每关节 `q_urdf = sign·q_lero
+ offset` 的映射,全程只读、不发运动:

```bash
PYTHONPATH=src .venv/bin/python scripts/map_joint_frames.py draft            # 零运动:从标定文件算 offset
PYTHONPATH=src .venv/bin/python scripts/map_joint_frames.py signs --port <PORT>  # 手推到硬限位定 sign
PYTHONPATH=src .venv/bin/python scripts/map_joint_frames.py check --port <PORT>  # 手扶核对映射
```

映射写到 `assets/so100_joint_map.json`。`SOFollowerBackend(joint_map=...)` 读时
lero→URDF、写时 URDF→lero(精确双射,读回即写不会跳)。

### 真机遥操

映射就绪后用 `scripts/teleop_real_arm.py`,它把
`NintendoProSource → TeleopLoop → SOFollowerBackend` 接到真机,并加保守安全默认:
`max_relative_target`(每步硬件钳位,默认 5°)、只在按住 clutch(ZL)时运动、退出时
落力矩。首次务必**清空工作区、手放急停旁、小幅慢速起步**:

```bash
PYTHONPATH=src .venv/bin/python scripts/teleop_real_arm.py \
    --port <PORT> --max-relative-target 5 --steps 600
```

`SOFollowerBackend` 用 lerobot 的 `SOFollowerRobotConfig`,`max_relative_target` 是
后端硬件层钳位,和回路自己的 `max_joint_step_deg`(6°/步)叠加。

## 录制与回放

一条 episode 是一个目录:`meta.json`(录制条件 + 配置快照 + 指标)加 `frames.npz`
(每个控制步一行)。用 npz 而不是 parquet,是因为 numpy 是本仓唯一的基础依赖 ——
录制是整条链路里最不能因为环境原因失败的一环,人和臂都挪开之后那条演示就补不回来了。
`LeRobotDataset` 仍是训练格式,转换器该待在装了 lerobot 的训练机上,还没写。

列名就是 `TeleopLoop` 文档里那套 dataset layout:`action.raw.*`(设备原样上报)、
`action.task.*`(策略的训练目标)、`action.joint.*`(发给舵机的)、
`observation.state.*`、`diagnostics.*`。三条动作流都存,是故意的冗余:IK 改了之后,
旧数据要能在新解算器上重放并对比。

```bash
PYTHONPATH=src .venv/bin/python scripts/record_episode.py --backend mujoco --steps 600 \
    --task "把红色方块放进框里"
PYTHONPATH=src .venv/bin/python scripts/replay_episode.py --list
PYTHONPATH=src .venv/bin/python scripts/replay_episode.py --id ep_... --check      # 只检查,不动
PYTHONPATH=src .venv/bin/python scripts/replay_episode.py --id ep_... --backend mujoco --mode task
```

有相机时每个视角多一个 `<role>.mp4`,**每个控制步写一帧**,所以 video 帧 *i* 就是
`frames.npz` 的第 *i* 行 —— 相机没有新帧就重写上一帧(计入 `stale`),队列满了才丢
(计入 `dropped`),两个数都写进 `meta.json`,因为一条视频比 `n_steps` 短就意味着对齐
断了,这件事必须从元数据看得出来,而不是等训练时才发现。

编码器**不设全局默认,按机器探测选**。实测两路 1080p30:`hevc_videotoolbox` 要 0.30
核、1.67 Mbps;`libsvtav1` 要 1.44 核、0.52 Mbps,画质相同(43.0 / 42.8 dB)。硬编码
省 4.8 倍 CPU,软编码省 3.2 倍磁盘,所以选哪个取决于这台机器缺什么 —— 核数少于
`hw_core_threshold`(默认 8)就用硬编码,否则用软编码。`VideoConfig.codec` 可以直接
点名覆盖。选中的编码器**和理由**都写进 `meta.json`。

探测方式是**真编几帧**,不是构造一下编码器对象:`av.codec.Codec(name, "w")` 只说明
它被编译进来了,没有 NVIDIA 驱动的机器上 NVENC 照样构造成功,然后在录制第一帧时炸。
这跟 `probe_gl_backend` 是同一个道理。

回放有两种模式,回答的是不同问题:

| 模式 | 问题 | 偏差说明什么 |
|---|---|---|
| `joint` | 机械臂能不能复现当时的动作 | 硬件、伺服、场景 —— 不是解算 |
| `task` | 今天的控制器会不会做同样的事 | 改了 projector / atlas / IK 之后的回归 |

录下来的东西本身不是一串安全指令,所以回放前会:静态检查(关节顺序、超出当前限位的
指令、该速度会不会超限速,`--check` 单独跑),先限速无 IK 地**走到第一帧**,按
**deg/s** 而不是 deg/步 限速(否则 2× 速度会让臂真的快一倍而每步检查照样通过),按
**当前**配置的关节限位钳位,以及在 MuJoCo 下逐帧检查网格离地。

## Web GUI

```bash
scripts/run_gui.sh          # 装前端依赖(仅首次)、按需构建、起服务、开浏览器
```

分步版本,或者想自己控制的时候:

```bash
cd tools/gui/frontend && npm install && npm run build
PYTHONPATH=src .venv/bin/python scripts/serve_gui.py     # http://localhost:8770
```

前后端都在本机一个进程里:网关(标准库 `http.server` + numpy,没有 web 框架)自己发
构建好的前端。四个页面 —— 遥操作/录制、数据集、回放、进度。细节见
[`tools/gui/README.md`](tools/gui/README.md)。

默认只绑 `127.0.0.1`;`--host 0.0.0.0` 会把「让真臂动」的按钮暴露到网络上。

真实 USB 相机在遥操作页「相机」一栏扫描并指派给腕部 / 第三人称视角,预览优先显示真
相机,没指派才回落到 MuJoCo 仿真相机。采集走 lerobot 的 `OpenCVCamera`(cv2 是
lerobot 的硬依赖,真机路径上本来就有),读帧是非阻塞 peek,实测不影响控制回路。

**扫描给的是每个设备的缩略图,不是名字,按画面认。** macOS 上 OpenCV 的 index 既不
对应 `system_profiler` 的顺序,也不对应 ffmpeg 的 AVFoundation 列表 —— 同一批相机三
种枚举给出三种顺序,任何按位置推出来的名字都是猜的,而猜错是静默的(内置摄像头和 USB
相机都是 1080p、都正常出帧)。细节见
[`tools/gui/README.md`](tools/gui/README.md) 的「真实相机」一节。

仿真相机预览要能渲染。启动时会打出选了哪个 GL 后端(egl → glfw → osmesa,子进程里
真渲染一帧决定),`MUJOCO_GL` 已 export 则跳过探测。撞到 `EGLDeviceEXT` 报错、或者
预览花屏/撕裂,见 [`tools/gui/README.md`](tools/gui/README.md) 的 GL 一节 —— 两个坑
都有记录,后者(GL context 的线程亲和性)在改 `SimPreview` 之前必读。

## 与 lerobot 的关系

so-snake 是**独立仓库**,依赖 lerobot 而不修改它。复用其现成件:

| 蓝图模块 | lerobot 现成件 |
|---|---|
| M4 执行 | `robots.so_follower.SOFollower` |
| 遥操作 | `teleoperators.nintendo.NintendoTeleop` (`NintendoController.PRO`) |
| M3 IK/安全 | 本仓库 `TaskFrame` / `FeasibilityAtlas` / `TaskIK5D`; lerobot 只接收安全 joint command |
| 数据引擎 | `datasets.LeRobotDataset` |

蓝图的核心增量 —— M1 几何教师、M2 三头 VLA(模式/观察意图/动作)、M3 可行性投影、
M5 任务验证 —— lerobot 没有,由本仓库实现。

## 当前状态

**阶段 0(基础搭建)基本完成,手柄→真机遥操已端到端跑通。** SO-100 已接入
(macOS Apple Silicon),从 Pro 手柄到真机的整条链路实测通过:方向正确、松开
clutch 即停(无运动尾巴)、夹爪可控、退出卸力。真机接入的关键决策与踩坑见
[`docs/real_arm_bringup.md`](docs/real_arm_bringup.md)。

- [x] 环境勘察,确认 lerobot 现成件可用
- [x] SO-100 URDF 落地 + TCP 坐标系补全(见 [`docs/so100_vs_so101.md`](docs/so100_vs_so101.md);上游来源已逐字节核对)
- [x] FK/IK 离线验证通过(round-trip p95 = 0.078 mm / 0.004°)
- [x] `T_world_base` 坐标系约定(零位时臂指向 −Y,已修正为 +X 朝前)
- [x] 工作空间实测重推(继承的盒子只有 84% 可达,新盒子 100%)
- [x] Mock 机器人后端 + 30Hz 闭环,离线跑通(位置误差中位 0.010 mm)
- [x] **迁移到 5 维任务空间** —— `(x, y, z, pitch, roll)`,位置锚定 chart,atlas pitch clamp,5D DLS IK
- [x] MuJoCo 3.6 仿真模型 + 三方运动学互校(ArmChain / placo / MuJoCo)
- [x] **MuJoCo viewer 支持 macOS Apple Silicon**:自动 re-exec 到 `mjpython`;手柄源自动选路(真 Pro 手柄,否则内置 scripted 波形)
- [x] **真机接入**:只读 preflight、STS3215 标定、lerobot↔URDF 关节映射(`map_joint_frames.py` + `JointFrameMap`)、`SOFollowerBackend`、关节空间 move-to-start、`teleop_real_arm.py`
- [x] **手柄→真机遥操跑通**:方向正确、松手无尾巴、夹爪 3× 速、退出落力矩、分层安全(先回 start / settle 中止 / 每步硬件钳位)
- [x] Gate 与默认闭环 desk check: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` 88 passed; `scripts/check_teleop_loop.py --steps 300` PASS
- [x] Episode 录制 + joint/task 双模式回放,含静态检查与限速/离地保护
- [x] 本机 Web GUI:遥操作、录制、回放、进度看板
- [x] **真实 USB 双相机接入 GUI**:按缩略图指派视角(编号不可信,见上)、两路实时预览优先于仿真相机;采集在 lerobot 线程上,读帧 0.0014 ms,回路无损
- [x] **相机帧写入 episode**:每个视角一个 mp4,编码在独立线程上,每控制步一帧保证 video 帧 i == npz 行 i;编码器按机器探测选(缺 CPU 用硬编,缺磁盘用软编),选中结果与理由写进 `meta.json`
- [x] **数据集页双路视频回看**:两路相机与轨迹曲线共用游标,按帧号对齐(不是时间戳 —— 实测 19.2 s 的 take 视频文件只有 16.7 s,按时间对齐片尾会差 2.5 s)
- [ ] LeRobotDataset 导出
- [ ] **待精修**:TCP 实测校准、offset 精修(标定欠扫的 pan/wrist_flex)、clutch/atlas 手感调参、相机外参

### 5 维任务空间迁移

实测证明 SO-100 的末端可控空间不是 SE(3),而是五维流形
`(x, y, z, pitch, roll)` —— 固定位置时可控姿态切空间的秩恒为 2,TCP 的 yaw
由位置唯一决定。控制链路已经迁移到 5D task target;旧的 6D target、Euler RPY box、
orientation weight、placo IK retry 和 feasibility feedback 都已删除。

- 结论与证据:[`docs/five_dof_orientation.md`](docs/five_dof_orientation.md)
- 冻结的接口契约与实施顺序:[`docs/plan_5dof_task_space.md`](docs/plan_5dof_task_space.md)

## 目录结构

```
assets/urdf/so100/     SO-100 URDF 与网格(so100.urdf 已补 TCP,so100_original.urdf 为上游原件)
assets/atlas/          SO-100 可行性图集
assets/mujoco/         MuJoCo XML 模型
assets/so100_joint_map.json    lerobot↔URDF 每关节映射(本臂标定产物;map_joint_frames.py 生成)
assets/so100_start_pose.json   记录的 start pose(本臂;move_to_start / teleop 起始)
docs/                  设计决策记录
scripts/               可复现的分析与验证脚本
src/so_snake/          M0~M5 模块实现
tools/gui/frontend/    Web GUI 前端(React + Vite;后端在 src/so_snake/gui/)
data/episodes/         录制的 episode(不入版本库)
```

`assets/so100_*.json` 是**这台臂**的标定/记录产物(舵机装配相关),换臂或重标定需重新生成。

## 脚本

离线 / 仿真:

| 脚本 | 用途 | 需要硬件 |
|---|---|---|
| `scripts/run_gui.sh` | 一键起 GUI:前端依赖/构建按需跑,再起 `serve_gui.py` | 否 |
| `scripts/serve_gui.py` | 本机 Web GUI(遥操作 / 录制 / 回放 / 进度) | 否 |
| `scripts/record_episode.py` | 录制一条 episode 到 `data/episodes/` | 视 backend |
| `scripts/replay_episode.py` | 回放 episode(`--check` 只检查不动) | 视 backend |
| `scripts/check_kinematics_agreement.py` | ArmChain / placo / MuJoCo 三方 FK/Jacobian 互校 | 否 |
| `scripts/check_teleop_loop.py` | MockFollower + ScriptedSource 默认闭环 Gate | 否 |
| `scripts/check_kinematics.py` | 旧 FK/IK round-trip 验证(待归档) | 否 |
| `scripts/build_feasibility_atlas.py` | 构建 position-conditioned pitch/roll atlas | 否 |
| `scripts/build_mujoco_model.py` | 从 URDF/STL 生成 MuJoCo XML | 否 |
| `scripts/compare_so100_so101.py` | SO-100/SO-101 运动学不变量比较 | 否 |
| `scripts/derive_so100_tcp.py` | 从 SO-101 官方 TCP 推导 SO-100 TCP | 否 |
| `scripts/view_pro_controller_sim.py` | MuJoCo viewer 遥操(macOS 自动 mjpython;手柄可选) | 否(需 `.[sim]`;真手柄需 `.[teleop]`) |
| `scripts/check_pro_controller_sim.py` | 真手柄 → MuJoCo/Mock 无头闭环 | 手柄 |

`compare_so100_so101.py` 与 `derive_so100_tcp.py` 需要上游仓库:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/TheRobotStudio/SO-ARM100.git /tmp/soarm
(cd /tmp/soarm && git sparse-checkout set Simulation)
export SOARM_UPSTREAM=/tmp/soarm
```

真机(需 `.[teleop]` + 臂上电插好,详见「真机 preflight」与 `docs/real_arm_bringup.md`):

| 脚本 | 用途 | 动臂? |
|---|---|---|
| `scripts/preflight_real_arm.py` | 依赖/关节契约/串口/手柄/标定检查;`--probe` 只读探测舵机 | 否(全程不动臂) |
| `scripts/map_joint_frames.py` | 建立并核对 lerobot↔URDF 关节映射(`draft`/`signs`/`check`) | 否(手推,不上力矩) |
| `scripts/move_to_start.py` | 关节空间移动到记录的 start pose | 是 |
| `scripts/teleop_real_arm.py` | Pro 手柄 → TeleopLoop → 真机(先回 start,分层安全) | 是 |
