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
# 安全检查:依赖 / 关节契约 / 串口 / 手柄 / 相机 / 标定文件是否存在
PYTHONPATH=src .venv/bin/python scripts/preflight_real_arm.py
# 追加只读总线探测(臂需上电插好;不会动臂)
PYTHONPATH=src .venv/bin/python scripts/preflight_real_arm.py --probe
```

真机需要 `.[teleop]` 附加层(含 `feetech-servo-sdk` 提供 `scservo_sdk`)。

### 串口自动检测

**串口不用自己填。**所有驱动真臂的脚本(preflight / teleop / move_to_start /
map_joint_frames / record / replay)和 GUI 都会自己找:按 USB 厂商:产品号认驱动板的
桥接芯片(本机是 WCH CH343 `1a86:55d3`),名字形状只作兜底;macOS 的
`Bluetooth-Incoming-Port` 和 debug console 直接排除 —— 它们在每台 mac 上都在,打开
蓝牙那个还会卡住。

只有一个候选口时**不打开任何设备**就定下来,所以自动检测不会打扰正在跑的会话;插了
两个 USB 串口适配器时才退回到只读 ping(问舵机型号,不上力矩、不发目标位),谁答话
就是谁;还是分不出来就报错并列出候选,要求 `--port` 点名。

```bash
PYTHONPATH=src .venv/bin/python scripts/scan_devices.py          # 串口 + 相机,都列出来
PYTHONPATH=src .venv/bin/python scripts/scan_devices.py --probe  # 顺带只读 ping 每个候选口
export SO_SNAKE_ARM_PORT=/dev/cu.usbmodem58760434321             # 检测不对时的长期覆盖
```

macOS 上串口通常是 `/dev/cu.usbmodem*`(只用 `cu.`,`tty.` 会等载波信号)。首次使用必须先标定(会手动移动臂过全程),标定与
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
PYTHONPATH=src .venv/bin/python scripts/map_joint_frames.py signs  # 手推到硬限位定 sign
PYTHONPATH=src .venv/bin/python scripts/map_joint_frames.py check  # 手扶核对映射
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
    --max-relative-target 5 --steps 600
```

`SOFollowerBackend` 用 lerobot 的 `SOFollowerRobotConfig`,`max_relative_target` 是
后端硬件层钳位,和回路自己的 `max_joint_step_deg`(6°/步)叠加。

## 回路频率:配的 30 Hz,以前跑 26 Hz

**已经修好了,现在是实测 30.00 Hz。**结论先放这里,因为找它的过程里排除掉的东西比找到
的那个更有用。

一句话:**不是活干得慢,是 `time.sleep` 还得晚。**

每步的活加起来只有约 **5 ms**,占 33 ms 预算的 15%,全是实测:

| 环节 | 实测 | 说明 |
|---|---|---|
| 舵机总线读 `sync_read` | **1.49 ms** | 6 个舵机一包,1 Mbps |
| `send_action` 里再读一次 | 1.49 ms | `max_relative_target` 要拿当前位置来钳位,lerobot 自己也标了 `/!\ Slower fps expected` |
| `sync_write` 下发 | ~0.5 ms | 单向,不等回包 |
| 5D IK + FK | **0.36 ms** | `ik.solve` 0.274,`task_pose` 0.049 |
| Switch Pro 手柄 `get_action` | **1.37 ms** | hidapi 非阻塞,1 ms 超时 |
| **合计** | **≈ 5 ms** | |

相机被完全排除了:**0 路相机的 take 和 2 路相机的 take,周期都是 37.6–37.9 ms**,一模
一样。采集在 lerobot 自己的线程上,编码也在独立线程上。

真正的原因是 macOS 的 timer coalescing —— 内核会把线程的唤醒时刻往外取整,好把多个唤醒
凑到一起、让核多睡一会儿,而它加的余量**和请求的睡眠时长成正比**:

| 请求 | 实际返回 | 多睡 |
|---|---|---|
| 1.0 ms | 1.46 ms | +0.5 ms |
| 5.0 ms | 6.19 ms | +1.2 ms |
| 28.3 ms | 32.42 ms | **+4.1 ms** |
| 33.3 ms | 37.32 ms | **+4.0 ms** |

`sleep(33.3 ms)` 回来是 37.3 ms —— 和修复前录的那批 take 实测的周期中位数 **37.7 ms**
几乎一样。
而旧代码按 `period - 本步已用时间` 睡,是从**本次迭代开头**算的,于是每一次多睡的 4 ms
都被永久记账,回路只会往后掉、掉了就再也补不回来。

周期分布也印证这是稳态而不是偶发卡顿:p50 37.7 ms、p90 39.5 ms,**只有 0.3% 的步超过
60 ms**。不是「偶尔卡一下拉低了均值」,是每一步都稳定慢 4 ms。

修法在 [`src/so_snake/pacing.py`](src/so_snake/pacing.py),两件事:

* **睡到一个按周期递推的绝对 deadline**,不是从迭代开头算剩余 —— 晚了的一步由下一步的
  短等待还回来,栅格自己收敛;
* **最后 6 ms 自旋不睡** —— 这才是真正守住频率的部分,因为 `sleep` 晚回来的那部分,
  靠 `sleep` 本身是没法要求它别晚的。代价是每 33 ms 里最多 6 ms 单核自旋。

外加一条:**补偿最多补一个周期**。一步要是卡了 400 ms(本机 p99.9 就是 404 ms,USB 抖
一下就有),deadline 会落在很远的过去,不封顶的话接下来十几步会一路不等待地冲出去,
按比操作者当初快得多的速率去打总线。超过一个周期就认赔,栅格从当下重开。

回路和回放共用这个 `RateKeeper`。回放那边还顺带修了一个反向的错:它按
`meta.control_hz`(配置值 30)定速,而 take 实际是 26 —— 两个 bug 原来**刚好互相抵消**,
所以回放速度看着是对的。只修其中一个会让回放快 15%,所以两个一起修:现在按
`episode.playback_hz`(实测值)播。

### 还有一个把 30 Hz 藏起来的东西:按录制键那 700 ms

回路修好之后,第一条新 take 试算出来仍然是 **28.2 Hz**,不是 30。原因和上面无关:

`n_steps / duration_s` 是**均值**,而每条 take 里有且只有一次大卡顿 —— **第 1 步 711 ms**。
根因是按下录制时要选编码器,而 `select_encoder` 是**真的编三帧** 1080p 来验证,实测
**678 ms**;它在 `start_recording` 持有的会话锁里跑,而控制回路每步写遥测也要这把锁。
于是每条 take 的第一帧必然停一次。292 步里的这一步(**0.3%**)把均值从 **30.1 拽到
28.2**(**6%** 的误差),而导出会拿这个数当整条 take 的时间栅格。

两头都修了:

* **帧率改测「步周期的中位数」**(`1 / median(dt)`),不再用均值 —— 中位数报的是另外
  291 步真正的周期,也就是策略学到的每一对相邻帧真实的间隔。修复前录的那批不受影响(两种
  算法都给 26 Hz,而且中位数版本的离散度更小:26.27–26.62 对 25.84–26.93);
* **编码器探测结果缓存,并在开会话时预热**(不是按录制时)。按下录制的代价从
  **678 ms 降到 0.05 ms**。探测按 key 加锁,所以就算刚开会话就按录制,也是等同一次
  探测的结果,不会重复探一遍。

改完:同一条 take 测出 **30.01 Hz**,与 30 的偏差 **0.0%**。

> 修复前录的 take 仍然是 26 Hz 的(盘上现在 44 条全是这批),新录的是 30 Hz。两批
> 不能混进同一个数据集 ——
> 见上面「帧率是量出来的」。筛选会自动拦(`fps_tolerance` 8%,26 vs 30 差 15%),
> 并且报告会明说这是「另一批」而不是「一条坏 take」。

### 一条 take 的帧率是录出来的,不是导出来的

**8-16 之前录的每一条 take 都是 26 Hz,导出只能如实报 26。**试算某个 task 出来是 26 而
不是 30,先看它是什么时候录的:

| take | 录制时间 | 步周期中位数 | 实测 |
|---|---|---|---|
| `ep_20260810_211649` | 08-10 13:16 | 37.91 ms | 26.38 Hz |
| `ep_20260810_232308`(`把牛牛放在胶带上`) | 08-10 15:23 | 37.90 ms | 26.39 Hz |
| `ep_20260812_213956` | 08-12 13:39 | 37.56 ms | 26.62 Hz |
| `ep_20260816_143651` | **08-16 06:36** | **33.33 ms** | **30.01 Hz** |

分界线正好在修复那天。37.9 ms 也对得上第一性原理的账:约 5 ms 干活 + `sleep(28.3)`
实际睡 32.4 ms = 37.4 ms,剩下的约 0.5 ms 是这条 take 带着两路相机而基准测试没带。

所以 `把牛牛放在胶带上`(只有 1 条,08-10 录的)导出必然是 26 Hz。**重导不会变高,
只能重录。**试算报告现在会直接这么说:

```
rate: takes ran 26.39-26.39 Hz, worst deviation from 26 Hz is 1.5%
  ^ 1/1 were recorded against a configured 30 Hz and did not hold it. That is
    baked into those takes -- the export reports what the arm actually did, so
    re-exporting cannot raise it. Re-record them to get the configured rate.
```

## 录制与回放

一条 episode 是一个目录:`meta.json`(录制条件 + 配置快照 + 指标)加 `frames.npz`
(每个控制步一行)。用 npz 而不是 parquet,是因为 numpy 是本仓唯一的基础依赖 ——
录制是整条链路里最不能因为环境原因失败的一环,人和臂都挪开之后那条演示就补不回来了。
`LeRobotDataset` 仍是训练格式,转换见下面「导出训练集」—— 它对 lerobot 的 import 是
惰性的,录制路径不会因此多一个依赖。

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

相机在命令行上用 `--camera <role>=<device>` 指派(`third_person` / `wrist`),
`scripts/scan_devices.py` 会把每个 index 的缩略图写到 `data/device_scan/`,**按画面认
相机**:macOS 上 OpenCV 的 index 既不对应设备名也不在重插后保持不变,认错了录出来的
episode 看着没问题,要等有人看视频才发现。只插了一台相机时可以写 `=auto`,多于一台
`auto` 会拒绝并列出候选 —— 这里宁可不认也不能猜。GUI 的「扫描相机」按钮同样是按缩略图选。

```bash
PYTHONPATH=src .venv/bin/python scripts/record_episode.py --backend real --source pro \
    --camera third_person=0 --camera wrist=3 --task "把红色方块放进框里"
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

## 导出训练集(LeRobotDataset)

**GUI 里有按钮**:在「录制」页审完一批原始 take,翻到「导出训练集」面板:选技能 →
试算 → 导出。导出跑在后台线程上,能取消,关掉浏览器再打开会接回正在跑的那次。臂在动
的时候会拒绝导出 —— 转码两路视频是本仓最重的活,而回路现在靠自旋守住 30 Hz,两者
抢核。

「训练集」页是导出的归宿:每个数据集一个条目,带 manifest、缓存的校验结果、
「重新校验」按钮和「在机械臂上回放」按钮。回放走的是和录制 take 回放同一套
安全层(限速趋近、deg/s 钳位、关节/工作区限位、MuJoCo 网格干涉),所以一次回放
既能验证导出是对的,也能确认策略确实能驱动这台臂。校验和回放都通过 HTTP
(`/api/export/datasets`、`/api/export/verify`、`/api/replay/dataset`),不需要
起终端。CLI `scripts/replay_lerobot_dataset.py` 走的是同一个 `SessionManager.start_dataset_replay`。

**没有 `export.json` 的数据集依然能跑**(foreign dataset、老 export、被人手动
删过 manifest 的):`episode_from_dataset` 直接读 lerobot 的 `meta/info.json`
拿 fps 和 action space,replay 跟有 manifest 时一模一样;`verify` 会跑
round-trip / 时间轴 / 视频帧数,但 **不会**跟源 take 比对(没有 source
mapping),verdict 标 PARTIAL 而不是 OK。完整校验要做的事:在「录制」页用「覆盖
同名数据集」重新导一次,这会写新的 `export.json`。

**`issues` 和 `skipped` 的分界**(校验查的三件事见下面「导出的数据可回放,而且是
验过的」):`issues` 是「检查跑了,数据集没过」—— 行和录制对不上、时间戳和
`frame_index / fps` 对不上、manifest 和 parquet 对集数对不上。这些参照物都在数据集
**内部**,对不上就是这堆字节有毛病,拦训。`skipped` 是「检查跑不了」,因为数据集
**外面**的东西不在:没有 `export.json`、没传 store、或者源 take 导完之后被删了。
删一个 take 不改动数据集的任何一个字节,所以这种情况标 PARTIAL(琥珀)而不是
FAILED —— 一个会因为**别的目录**变了而翻红的 verdict,根本不是在讲这个数据集:同
一份导出会在还留着 take 的工位上是绿的、在从来没有过 take 的训练机上是红的,这种
verdict 没法拿出来引用。这类数据集照样能训,丢掉的是「能和录制对账」和「能再导一
次」。缺口会点名到 take,而 `episodes_compared`(GUI 里那格「源比对 N/M 集」)说明
误差数字覆盖了多少集 —— 0/1 集下的 `0.00e+0` 不是「全对」,是根本没比。

命令行同样一套:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_lerobot_dataset.py --list-tasks
PYTHONPATH=src .venv/bin/python scripts/export_lerobot_dataset.py --task "牛牛抓放" --dry-run
PYTHONPATH=src .venv/bin/python scripts/export_lerobot_dataset.py --task "牛牛抓放" \
    --repo-id so_snake/niuniu_pick_place --out data/lerobot/niuniu_pick_place
```

**按 task 标签选**,一次导一个技能 —— 一个 store 里放着几种任务,混着训出来的策略学的
是它们的平均。`--dry-run` 走完除了解码和写盘之外的全部流程(筛选、测帧率、算动作统计),
先跑它:一条 take 视频对不齐,在这里发现最便宜。

**目标目录已存在会拒绝**(默认行为):`FileExistsError`,消息里说明那里有
多少条 / 多少帧 / 多大。覆盖要显式 `--overwrite`(GUI:导出面板的「覆盖同名
数据集」复选框,会弹一次确认)。这是 destructive 操作,默认拦截是为了不让一个
写错的 `--out` 把别人的数据集擦掉。

### 映射:绝对 5D 流形位姿 → 同一张图上的增量

是的,现在就是这个。两边都活在 `so_snake.m3_safety.task_pose` 定义的**同一张图**
`(x, y, z, pitch, roll)` 上,数据集里不出现 SE(3)、四元数或旋转矩阵。

记 `q_t` 为总线读回的关节角,`Φ` 为进入这张图的正向映射(FK + 解析到图坐标,
`TaskIK5D.task_pose`):

```
p_t = Φ(q_t)                   臂实际到达的 5D 位姿
c_t = action.task.target[t]    回路当时下发的 5D 目标

state[t]  = ( p_t ,  g_t 实测 )
action[t] = ( c_t ⊖ p_{t-1} ,  g_t 指令 )
```

**更新公式**(rollout 每步跑的那个逆):

```
c_t = p_{t-1} ⊕ action[t][0:5]
g_t = action[t][5]
```

`⊕` / `⊖` 逐分量作用,只在两个角度分量上和普通加减不同:

```
(a ⊕ b)_i = a_i + b_i             i ∈ {x, y, z}       米
(a ⊕ b)_i = wrap( a_i + b_i )     i ∈ {pitch, roll}   弧度
(a ⊖ b)_i = a_i − b_i             i ∈ {x, y, z}
(a ⊖ b)_i = wrap( a_i − b_i )     i ∈ {pitch, roll}

wrap(θ) = (θ + π) mod 2π − π      折进 (−π, π]
```

真机 rollout 时 `p_{t-1}` **不是从数据集读的,是当场测的**:

```
c_t = Φ( q_实测 ) ⊕ π_θ(观测)[0:5]
```

这正是锚点选「到达位姿」的全部理由。`⊕` 只有一份实现 ——
`so_snake/data/export.py` 里的 `apply_action`,训练和 rollout 因此不可能对不上。

三点刻意不是:

* **不是切空间/指数映射的步长。**`⊖` 就是图坐标的直接差,不是李代数 log。这里成立,
  是因为这张图在臂能到的地方处处正则 —— `psi` 是**位置**的函数而不是工具自身方位角的
  函数,就是为了夹爪垂直向下时不退化(见 `task_pose.py`)。单步位移很小,图在这个尺度
  上光滑,坐标差和真正的测地步长的差别远小于伺服滞后。
* **夹爪不是增量。**action 里是绝对角,state 里是**实测角**,故意是两个数。在第一条
  format v2 take 上实测两者最大差 **10.2°** —— 夹爪堵在物体上,指令角看不出来。
* **锚点不是上一条指令。**是 `p_{t-1}` 不是 `c_{t-1}`,这条的理由见
  [`docs/act_baseline.md`](docs/act_baseline.md)(「整步零动作」占比 25% → 0.1%)。

单位是米 / 弧度 / 度混在同一个向量里 —— 难看,但故意:这是本仓其它每一层已经在用的
单位,数据集偷偷换算会让所有打印出来的诊断和真臂对不上。

实测校验(`ep_20260812_213956`,552 行):`state[:, :5] == Φ(q_实测)` 与
`action[:, :5] == c_t ⊖ p_{t-1}` 精确成立,`p_{t-1} ⊕ action[t]` 还原 `c_t` 误差
**1.8e-7**(float32 存储)。每次导出都会重跑这个检查,见下。

### 导出的数据可回放,而且是验过的

写完不等于能用。**写盘那一刻,所有致命故障看起来都像成功**:parquet 页脚没写(原来根本
没调 `LeRobotDataset.finalize()`,官方注释写明「不调就加载不了」)、某路视频少一帧、
时间轴按一个谁也没跑过的帧率生成。这些只有**把盘上的东西读回来**才看得见。

所以导出默认跟一次读回校验(GUI 里自动跑,CLI 里 `--no-verify` 才关掉),它重新打开
parquet、manifest 和 mp4,问三件事:

1. **行还是那些行吗** —— 盘上的 state/action 对比源 episode 现算一遍;
2. **还能反解吗** —— 用 `apply_action` 把盘上的行还原成当初下发的 5D target。这是
   rollout 依赖的那条契约,也是「可回放」和「能读」的区别;
3. **时间轴是真的吗** —— timestamp 对 `frame_index / fps`,以及每行每路各有一帧可解码
   的视频。

实测一份 1672 行的导出:target 还原误差 **0.010 µm / 12 µdeg**(float32 舍入,不是
契约误差),timestamp 与栅格差 **0.9 µs**,两路视频各 1672 帧 = 1672 行。把其中一路
视频截到 500 帧,校验立刻判 **NOT REPLAYABLE**。

每份导出还会在数据集根目录写一个 `export.json`,按数据集 episode 顺序记下源 take 的
id —— lerobot 的元数据没地方放这个,没有它导出就是一扇单向门:没法核对、没法重导、
也没法告诉操作者该回去补录哪一条。

校验结果本身也落盘(数据集目录里的 `verify.json`),因为解完整份数据集的视频要几十秒
到几分钟,而「训练集」页要给每个数据集显示一个 badge,不可能每次开页都重算全部。这份
缓存靠两样东西保持诚实:一个版本号 —— 检查改了含义,旧 verdict 就丢掉重算,而不是当
成少了个字段迁移过来(绿色 badge 背后是一次从没跑过的检查,是这里最坏的结果);以及
数据集的最新 mtime —— 之后又被写过的,标「过期」而不是继续算绿。算 mtime 时**跳过
`verify.json` 自己**:它是**关于**数据集的,不是数据集的一部分。这一条不是洁癖 ——
文件里记的 mtime 是写它之前读的,把它自己算进去,这次写入就成了目录里最新的东西,于是
每份 verdict 刚算完就是「过期」,重新校验永远清不掉。

**真的放到臂上回放:**

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_lerobot_dataset.py \
    --dataset data/lerobot/niuniu_pick_place --list
PYTHONPATH=src .venv/bin/python scripts/replay_lerobot_dataset.py \
    --dataset data/lerobot/niuniu_pick_place --episode 0 --backend mujoco
```

它把导出的一条 episode 还原成和录制 take 一样的 `Episode`,交给**同一个**
`EpisodeReplayer` 播 —— 同样的限速趋近、同样的 deg/s 钳位、同样的关节限位、同样的网格
离地检查,安全层一行都没有复制。只有 task 模式:数据集刻意不带关节流(策略是在任务空间
训的),关节由今天的 IK 从 target 解出来,这本来就是 task 模式在做的事。实测一条 558 步
的导出 episode 在 mock 上跑完:任务位置误差 p95 **0.0034 mm**,IK 收敛 99.1%。

录制格式是刻意冗余的(三条动作流都存),训练集则必须挑一个。三个选择都是数据逼出来的,
决策记录见 [`docs/act_baseline.md`](docs/act_baseline.md)(含实测证据、M1 训练实测数据、
以及明天从哪接着做),代码里的理由写在
[`src/so_snake/data/export.py`](src/so_snake/data/export.py) 的模块头:

**state 用臂真正到达的位姿,不是它被要求到达的。**`observation.state.task_pose` 名字
像观测但不是:它是 **IK 解**的正运动学,所以它和 `action.task.target` 的距离是解算残差
—— 本机实测 1e-6。臂离目标的真实距离比这大三个数量级:**中位 9.6 mm、p95 41 mm、
p95 pitch 10°**,全是负载下的伺服滞后。导出时用
`FK(observation.state.joints_deg)` 现算,这才是能报告舵机堵转、物体脱手、碰撞的那一路。

**action 是从「到达的位姿」起步的增量,不是从上一条指令起步的。**增量必须锚在某个东西
上,锚在哪里决定误差会不会累积:锚在上一个 *target* 上,策略就是个开环积分器,系统性
低估会一路走掉再也回不来;锚在**到达的位姿**上,每一步都重新参考测量值,rollout 自己
纠正自己。

    action[t] = target[t] - FK(measured joints)[t-1]
    rollout:   target = FK(measured now) + action

两边带着同一份伺服滞后,策略于是复现遥操作指令原本带的那个提前量。**滞后比单步位移
还大**是这件事在这台臂上要紧的原因:p95 上滞后贡献 41 mm、操作者意图只有 5 mm,忽略
它不是小近似。实测也印证了:锚在测量值上,「整步零动作」的比例从 25% 掉到 **0.1%**。

`--action-space absolute` 导出 `target[t]` 本身,作对照组 —— delta 的 rollout 要是漂了,
它回答漂移是不是动作空间造成的。夹爪两种模式下都是绝对角(实测只在 2° 和 90° 两个值
之间跳,delta 会让它整条 episode 预测零、然后要求它精确命中一次 88° 跃变)。

**帧率是量出来的,不是从配置读的。**盘上这 44 条 take,回路配的是 30 Hz,实际跑
26.1 Hz(成因和修复见下面「回路频率」),而录制把**配置值**写进了 mp4 头,所以视频声称
比它记录的 take 短 15%。按配置值导出会训出一个动作块横跨 3.3 s 意图、却在 2.9 s 墙钟里
放完的策略 —— 比它学过的每一条示范都快 15%,而这条臂的跟踪滞后本来就是动作里最大的
一项。所以帧率是**量两层中位数**:每条 take 取自己**步周期的中位数**(`1 / median(dt)`,
不是 `n_steps / duration_s` —— 为什么见下面「一条 take 的帧率是录出来的」),再对选中
的这批 take 取中位数(中位而非均值:一条跑飞的 take 应该被筛掉,而不是把全体的时间栅格
拽走)。视频**按帧号**读出来再按这个帧率重编码 ——
这同时消掉了头信息里的那个谎,lerobot 是按时间戳 seek 这些文件的,头和行栅格对不上不是
外观问题。

回路修好之后新录的 take 会是 30 Hz,但**帧率照样量**:26 Hz 的旧 take 和 30 Hz 的新
take 差 15%,放不进同一条时间栅格,而这正是筛选要拦下来的事(`fps_tolerance` 默认 8%,
所以混选会被自动拒掉,不会静悄悄训出一个半快半慢的数据集)。**这两批要分开导。**

实测导出(本机当前 44 条 take):42 条 `牛牛抓放` 里 40 条可用 / 19491 帧 / 26 Hz,
各 take 26.27–26.59 Hz,**与 26 Hz 最大偏差 2.3%**;另 2 条缺第三人称相机,导出时拒绝。
这些数字是**当下盘上的快照** —— take 会录会删,写死在文档里的计数活不过两次会话。

### 在 MacBook Pro M1 上训练

**可以,实测跑得动。**ACT(ResNet18 ×2 相机,52M 参数)在 M1 Pro / 16 GB 上用 MPS:

| 分辨率 | 训练一步 | 20k 步 | 100k 步 | 冷启动一次推理 |
|---|---|---|---|---|
| 240×320 | **304 ms** | **1.7 h** | 8.4 h | 22 ms |
| 360×480 | 593 ms | 3.3 h | 16.5 h | 36 ms |
| 480×640 | 953 ms | 5.3 h | 26.5 h | 56 ms |

导出默认 **240×320**,因为 26 Hz 的控制周期是 38 ms,而 ACT 每 `n_action_steps` 步要
重新规划一次整块:480×640 的 56 ms 塞不进一个周期,240×320 的 22 ms 塞得进。真实训练
实测 316 ms/步(`updt_s 0.313`、`data_s 0.005`),两路视频的解码被 dataloader 完全
overlap 掉了,不是瓶颈;显存约 1 GB。

```bash
HF_LEROBOT_HOME=data/lerobot .venv/bin/lerobot-train \
    --dataset.repo_id=so_snake/niuniu_pick_place \
    --dataset.root=data/lerobot/niuniu_pick_place \
    --policy.type=act --policy.device=mps --policy.push_to_hub=false \
    --output_dir=outputs/act_niuniu --steps=20000 --batch_size=8
```

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

#### 「扫描不出腕部相机」通常不是没扫到

腕部相机**在列表里**,只是那张缩略图什么都看不出来,于是在一排五个里被当成没扫到。
本机实测:它开得了、出 1920×1080、曝光正常(**对比度 69,和其它设备一样健康**),
只是拉普拉斯方差只有 **4.3**,而同型号的第三人称那台是 **128**。

**这不是故障。**腕部相机对焦在**夹爪距离**上 —— 那是它唯一有用的距离,策略要看的是
夹爪合拢时的物体,不是房间另一头。面前没东西的时候它拍到的就是一片糊,这是镜头在
干它该干的活。所以:

* **失焦不阻止任何事。**开相机、录制、导出,没有一条路径会因为这个数值拒绝设备 ——
  代码里从来没有过这样的门,也不会加。
* 扫描测这个数值,是为了回答扫描本来要回答的问题:**这张缩略图是哪台相机**。细节太少
  的图在一排五个里看着像空位,操作者就会以为它没扫到。
* 所以列表里只标一个中性的 `·`、边框改虚线,横幅用蓝色写「有 N 个相机画面细节很少,
  不好按图认 —— 它们**在**列表里,能选也能录」,并附一句:腕部相机本来就该是糊的,
  不用去拧镜头。想确认是哪台,**把手放到镜头前几厘米再扫一次**。

(早先这里写的是「拧对焦环」,那是错的 —— 照做会把一台设置正确的相机弄坏。)

锁屏的 iPhone(Continuity Camera)会被单独判成「画面全黑」(对比度≈0),和细节少
是两回事,但同样不阻止任何事。

顺带修掉一个静默的坑:GUI 里「隐藏这个设备」原来按 **index** 存在 localStorage 里,
而 macOS 的 index 会挪位 —— 曾经把 index 2 当内置摄像头隐藏掉,重插之后 index 2 变成
腕部相机,它就**真的从列表里消失了**,操作者还在找一个界面故意不显示的相机。现在隐藏
记录会连同当时的设备指纹一起存,设备一变就整份作废并提示,宁可多点一次也不能藏错。

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

**数据引擎这条线到「能训」为止是通的**:录制 → 双路视频回看 → 按 task 导出
LeRobotDataset → 读回校验 → 把导出的数据放回臂上回放,都在 GUI 里一条路走完;
state/action 契约定死了([`docs/act_baseline.md`](docs/act_baseline.md)),ACT 在这台
M1 上实测训得动(240×320 双相机 316 ms/步)。**缺的是闭环**:rollout 执行器还没写
(策略 → 5D target → 现有 atlas/IK/限速安全层 → 真臂),所以没有一个策略真正驱动过
这条臂;数据量也还只有一个任务的 40 条可训 take,物体位置变化不足。这两件事就是下面
未打勾的部分,`进度` 页(`src/so_snake/gui/roadmap.py`)是同一份账。

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
- [x] Gate 与默认闭环 desk check: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` 248 passed; `scripts/check_teleop_loop.py --steps 300` PASS(IK 位置误差 p95 0.0001 mm、pitch p95 0.0002°、收敛 100%、最大单步 0.91°)
- [x] Episode 录制 + joint/task 双模式回放,含静态检查与限速/离地保护
- [x] 本机 Web GUI:遥操作、录制、回放、进度看板
- [x] **真实 USB 双相机接入 GUI**:按缩略图指派视角(编号不可信,见上)、两路实时预览优先于仿真相机;采集在 lerobot 线程上,读帧 0.0014 ms,回路无损
- [x] **相机帧写入 episode**:每个视角一个 mp4,编码在独立线程上,每控制步一帧保证 video 帧 i == npz 行 i;编码器按机器探测选(缺 CPU 用硬编,缺磁盘用软编),选中结果与理由写进 `meta.json`
- [x] **录制页双路视频回看**:两路相机与轨迹曲线共用游标,按帧号对齐(不是时间戳 —— 实测 19.2 s 的 take 视频文件只有 16.7 s,按时间对齐片尾会差 2.5 s)
- [x] **修掉第二路相机的「跳帧播放」**:第二路从来没人调过 `play()`,它只靠 `onTimeUpdate` 里的 seek 前进,而浏览器把 `timeupdate` 限到 ~4 Hz —— 于是主画面 30 fps、腕部画面 4 fps 一顿一顿。现在第二路跟随第一路的 play/pause/seek/倍速,帧号校正只是校正
- [x] **修掉帧号「1, 9, 17」跳着走**:同一个 `timeupdate` 4 Hz 的根因 —— 30 fps ÷ 4 Hz = 每次跳 ~8 帧。`currentTime` 本身是连续的,粗的只是事件,所以改成播放时用 `requestAnimationFrame` 采样,索引变了才回调。配套把 `SeriesPlot` 里按样本数计费的部分(1200 点 path 字符串 ×5 图)`useMemo` 掉,否则光标每帧移动会重建它们
- [x] **LeRobotDataset 导出**:按 task 选、5D manifold state + manifold 增量 action(锚在测量位姿上,所以 rollout 自纠)、帧率量出来而不是读配置(rollout 与示范同速,当前批偏差 2.3%)、两路相机;**GUI 里一个按钮**(试算 → 后台导出 → 可取消),臂在动时拒绝
- [x] **导出后读回校验,证明可回放**:重新打开盘上的 parquet/manifest/mp4,查行是否一致、`apply_action` 是否还能反解出当初的 5D target、每行每路是否各有一帧。实测 1672 行还原误差 0.010 µm、时间轴差 0.9 µs;顺带发现原来**从没调过 `LeRobotDataset.finalize()`**(不调 parquet 页脚不写,数据集加载不了)
- [x] **导出的数据能真的放回臂上**:「训练集」页有「在机械臂上回放」按钮,跟 `scripts/replay_lerobot_dataset.py` 走同一个 `SessionManager.start_dataset_replay`(安全层零复制)。558 步实测跑完,任务位置误差 p95 0.0034 mm
- [x] **原始 take 和导出训练集分成两页**:`data/episodes/` 下的叫「录制」(审 raw take,审完翻到底导出),`data/lerobot/` 下的叫「训练集」(每个数据集带 manifest + 缓存校验结果,可以重新校验、可以回放)。同一页混着展示会让人把只对导出成立的校验 verdict 误读成对 take 的;分开之后每页的 verdict 是什么就是什么
- [x] **没有 `export.json` 的数据集也能用**:`episode_from_dataset` 直接读 lerobot 的 `meta/info.json` 拿 fps 和 action space,replay 不再被 manifest 卡住;`verify` 在这种情况下跑 round-trip 和时间轴,但**不会**跟源 take 比对(没有 source mapping),verdict 标 PARTIAL 而不是 OK。本仓历史里那条 `niuniu_pick_place`(10 条 / 4221 帧, 26 Hz)就是这样 — 用现在的代码可以回放,PARTIAL 校验过(round-trip / 时间轴 / 视频帧数全过)
- [x] **覆盖同名数据集是显式 destructive 操作**:`export()` 默认拒绝写进已存在的目录,错误里说明那里有几条 / 几帧 / 多大。覆盖要 `--overwrite`(CLI)或「覆盖同名数据集」复选框(GUI,弹确认)。这是 destructive 操作,默认拦截是为了不让一个写错的 `--out` 把别人的数据集擦掉
- [x] **回路真的跑到 30 Hz**(原来 26.3):不是活慢(每步 ≈5 ms),是 `time.sleep` 在 macOS 上多睡 4 ms,且旧代码从迭代开头算剩余把每次超时都永久记账。改成递推绝对 deadline + 6 ms 自旋尾,补偿封顶一个周期。回放同步修(原来两个 bug 互相抵消)
- [x] **帧率改按「步周期中位数」测**,不再用 `n_steps / duration_s`:按录制键会触发一次 ~700 ms 的编码器探测,一条 292 步的 take 里这一步就把均值从 30.1 拽到 28.2(0.3% 的步造成 6% 的误差)。探测结果现在缓存并在开会话时预热,按下录制的代价从 678 ms 降到 0.05 ms
- [x] **录制补上夹爪实测角**(format v2):总线本来就读到了,v1 把它切掉了 —— 指令角看不出夹爪堵在物体上
- [x] **M1 Pro 上 ACT 训练实测可行**:240×320 双相机 316 ms/步,20k 步 1.7 h
- [x] **给未标注的 33 条打 task 标签**:盘上 44 条 take,`牛牛抓放` 42 条,其中 40 条能过筛 = 19491 帧 ≈ 12.4 分钟;另 2 条缺第三人称相机(计数是快照,take 会录会删)
- [x] **细节少的相机不再被当成「没扫到」**:扫描测细节量,把这类设备标出来而不是藏起来(腕部那台实测 4.3 vs 同型号 128)。**不阻止任何事** —— 腕部相机对焦在夹爪距离上,静止时本来就糊,这是对的;标注只为了在一排缩略图里认出它是谁。隐藏记录连设备指纹一起存,插拔后作废
- [x] **校验分清了「没过」和「没跑」**:源 take 在导出之后被删掉,原来判 FAILED(「不能直接拿去训」),现在判 PARTIAL —— 删一个 take 不改数据集的任何一个字节,verdict 一旦会因为别的目录变了而翻红,它就不是在讲这个数据集了。同时补上覆盖率:报告和 GUI 都写明「源比对 N/M 集」,0/1 集下的 `0.00e+0` 不能读成「全对」。`VERDICT_VERSION` 升到 2,旧的 verdict 会被丢掉重算(v1 在这件事上会说 FAILED)
- [x] **缓存的校验结果不再自己把自己判过期**:`verify.json` 里记的 mtime 是**写它之前**读的,而它自己就落在数据集目录里 —— 于是这次写入成了目录里最新的东西,每份 verdict 刚算完就显示「过期」,重新校验也清不掉(库里三个数据集全挂着 `过期`)。现在算目录 mtime 时跳过 verdict 文件本身:verdict 是**关于**数据集的,不是它的一部分。顺带校验一次不再把数据集顶到列表最前(排序就用这个数)
- [x] **两个库页面的时间不再骗人**:数据集列表把 epoch **秒**当毫秒传进格式化函数,三条数据集全显示 `01-21`(1970 年 1 月 21 日);两个 `shortTime` 都走 `toISOString()`,本地 22:36 显示成 14:36,「今天」还提前 8 小时翻页。录制页同一个毛病更刺眼:take 的 `created_at` 是 UTC,而 take **id** 是本地时间,同一条 take 挂着两个差 8 小时的钟
- [ ] rollout 执行器(策略 → 5D target → 现有 atlas/IK/限速安全层 → 真臂)
- [ ] 补录到 ~50 条并变化物体位置(现有 40 条可训 take / 19491 帧 ≈ 12.4 分钟,还缺位置变化与真机 rollout 验证)。录之前值得先把最早那 10 条验证集上量到的两件事在这 40 条上重量一遍:26% 的步钉在 pitch 限位上、atlas 在 56% 的步上钳了 pitch;x 贴着工作区下界跑(0.170–0.219 m 对下界 0.17,3–13% 的步触发钳位)—— 策略会把这两样一起学走。这两个数字目前只在那 10 条上量过
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
assets/so100_start_pose.json   记录的 start pose(本臂;move_to_start / teleop 起始,
                               也是 GUI「归位」的目标)。用
                               `scripts/move_to_start.py --capture` 或 GUI 上的
                               「记录当前位姿为归位点」按钮写入
docs/                  设计决策记录
scripts/               可复现的分析与验证脚本
src/so_snake/          M0~M5 模块实现
tools/gui/frontend/    Web GUI 前端(React + Vite;后端在 src/so_snake/gui/)
data/episodes/         录制的 episode(不入版本库)
data/lerobot/          导出的 LeRobotDataset(不入版本库)
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
| `scripts/export_lerobot_dataset.py` | 按 task 导出 `LeRobotDataset`(`--dry-run` 只筛选不写,`--verify` 只校验已有数据集) | 否(需 lerobot) |
| `scripts/replay_lerobot_dataset.py` | 把导出的数据集放回臂上(`--check` 只检查不动) | 视 backend(需 lerobot) |
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
