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

使用 `~/Codes/lerobot/.venv`(uv 管理)。这是唯一齐全的环境:

```bash
/home/hanyu/Codes/lerobot/.venv/bin/python scripts/check_kinematics.py
```

| 依赖 | 版本 | 用途 |
|---|---|---|
| lerobot | 0.4.5 (editable) | 机器人驱动、遥操作、数据集、训练框架 |
| placo | 0.9.16 | 独立 FK 交叉验证 |
| feetech-servo-sdk | 1.0.0 | STS3215 舵机通讯 |
| hidapi | 0.14.0.post4 | Pro 手柄读取 |
| torch | 2.10.0 | — |
| mujoco | 3.6 | URDF/STL 仿真、解析 Jacobian 交叉验证、离屏相机 |

> conda `base` 缺 placo / feetech-servo-sdk;conda `lerobot` env 是 0.4.1 老版本且缺得更多。不要用。

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

**阶段 0(基础搭建)进行中。** 硬件暂不在手边,策略是先写完整真机代码,配合
Mock 后端离线验证整条链路,回家把后端切成真机即可。

- [x] 环境勘察,确认 lerobot 现成件可用
- [x] SO-100 URDF 落地 + TCP 坐标系补全(见 [`docs/so100_vs_so101.md`](docs/so100_vs_so101.md))
- [x] FK/IK 离线验证通过(round-trip p95 = 0.078 mm / 0.004°)
- [x] `T_world_base` 坐标系约定(零位时臂指向 −Y,已修正为 +X 朝前)
- [x] 工作空间实测重推(继承的盒子只有 84% 可达,新盒子 100%)
- [x] Mock 机器人后端 + 30Hz 闭环,离线跑通(位置误差中位 0.010 mm)
- [x] **迁移到 5 维任务空间** —— `(x, y, z, pitch, roll)`,位置锚定 chart,atlas pitch clamp,5D DLS IK
- [x] MuJoCo 3.6 仿真模型 + 三方运动学互校(ArmChain / placo / MuJoCo)
- [x] Gate A/B 与默认闭环 desk check: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` 34 passed; `scripts/check_teleop_loop.py --steps 300` PASS
- [ ] 双相机采集 + LeRobotDataset 落盘
- [ ] **回家实测**:舵机 ID/标定、TCP 实测校准、相机外参、clutch 手感调参

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
docs/                  设计决策记录
scripts/               可复现的分析与验证脚本
src/so_snake/          M0~M5 模块实现
assets/atlas/          SO-100 可行性图集
assets/mujoco/         MuJoCo XML 模型
```

## 脚本

| 脚本 | 用途 | 需要硬件 |
|---|---|---|
| `scripts/check_kinematics_agreement.py` | ArmChain / placo / MuJoCo 三方 FK/Jacobian 互校 | 否 |
| `scripts/check_teleop_loop.py` | MockFollower + ScriptedSource 默认闭环 Gate | 否 |
| `scripts/check_kinematics.py` | 旧 FK/IK round-trip 验证(待归档) | 否 |
| `scripts/build_feasibility_atlas.py` | 构建 position-conditioned pitch/roll atlas | 否 |
| `scripts/build_mujoco_model.py` | 从 URDF/STL 生成 MuJoCo XML | 否 |
| `scripts/compare_so100_so101.py` | SO-100/SO-101 运动学不变量比较 | 否 |
| `scripts/derive_so100_tcp.py` | 从 SO-101 官方 TCP 推导 SO-100 TCP | 否 |

后两个需要上游仓库:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/TheRobotStudio/SO-ARM100.git /tmp/soarm
(cd /tmp/soarm && git sparse-checkout set Simulation)
export SOARM_UPSTREAM=/tmp/soarm
```
