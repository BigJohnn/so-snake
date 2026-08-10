# so-snake GUI

浏览器里的遥操作、录制与回放。前后端都跑在本机一个进程里。

```
浏览器 ──HTTP──▶ scripts/serve_gui.py
                    └─ so_snake.gui.server      HTTP + 静态文件
                       └─ so_snake.gui.session  SessionManager(同一时刻只有一件事在驱动机械臂)
                          ├─ TeleopLoop         30 Hz 控制回路
                          ├─ EpisodeRecorder    写 data/episodes/
                          └─ EpisodeReplayer    回放
```

## 起服务

```bash
scripts/run_gui.sh                       # 一键:依赖 → 构建 → 起服务 → 开浏览器
scripts/run_gui.sh --port 9000           # 认识的参数之外,原样转给 serve_gui.py
scripts/run_gui.sh --skip-build          # 确定 dist 是新的,别等 npm
scripts/run_gui.sh --no-open             # 不动浏览器
```

`npm install` 只在 `node_modules` 缺失、或 `package-lock.json` 比
`node_modules/.package-lock.json` 新时跑;`npm run build` 只在 `dist/index.html`
缺失、或前端源文件比它新时跑。所以重复执行的代价就是起一个进程 —— 拿它当日常入口
即可,不用自己判断该不该重新构建。

分步版本(CI、或者想自己控制每一步):

```bash
cd tools/gui/frontend && npm install && npm run build   # 只需一次(改前端后重跑)
PYTHONPATH=src .venv/bin/python scripts/serve_gui.py              # http://localhost:8770
```

网关自己就把 `frontend/dist` 发出去,所以整套东西是**一个进程**。默认只绑
`127.0.0.1`。`--host 0.0.0.0` 会把「让真机械臂动起来」的按钮暴露到网络上 ——
只在你能看见那条臂、且信任那个网络的时候才这么做。

开发前端时用 Vite,它把 `/api` 代理到网关:

```bash
PYTHONPATH=src .venv/bin/python scripts/serve_gui.py &     # 后端
cd tools/gui/frontend && npm run dev             # http://localhost:5173
```

注意用哪个解释器。`python` 在这台机器上是 miniconda base(mujoco 3.2.5、PyOpenGL
3.1.0),不是仓库的 `.venv`。仓库其它命令一律用 `.venv/bin/python`,这里也一样。

## 真实相机

遥操作页左侧「相机」一栏点**扫描相机**,把扫出来的设备指派给「第三人称」和「腕部」,
再启动会话。预览窗口优先显示真相机,没有指派才回落到 MuJoCo 的仿真相机。

几件必须知道的事:

**扫描是个按钮,不是自动的。** 列举要逐个打开设备并真取一帧才算数 —— `isOpened()`
不够,macOS 上锁屏的 iPhone 会以 Continuity Camera 的身份开得好好的然后一帧都不给。
所以它慢(每个设备一秒上下),而且在 macOS 上第一次会弹系统摄像头权限。这两件事都不
该发生在页面加载里。会话运行中扫描会被拒(409):扫描要打开的正是会话占着的设备。

**没扫到任何相机,先查权限。** 被拒绝的进程看到的是「没有设备」,不是报错。
系统设置 → 隐私与安全性 → 摄像头。

**按画面认相机,不要按编号。** 扫描结果给的是每个设备**当前看到的缩略图**,没有名字。
这不是偷懒 —— macOS 上 OpenCV 的 index 空间和 `system_profiler` 的顺序不一致,和
ffmpeg 的 AVFoundation 列表也不一致,同一批相机三种枚举给出三种顺序:

```
OpenCV index         0=DECXIN(工作区)  1=DECXIN  2=FaceTime  3=OBS 虚拟相机
system_profiler      0=OBS  1=FaceTime  2=DECXIN  3=DECXIN
ffmpeg avfoundation  0=OBS  1=FaceTime  2=DECXIN  3=DECXIN
```

早期版本把 `system_profiler` 的顺序按位置套到 OpenCV index 上,于是**把内置
FaceTime 相机报成了 DECXIN**,而且下游没有任何东西能发现 —— 两者都是 1080p、都正常
出帧,录出来的 episode 要等人打开看才知道拍错了。所以名字整个去掉了,靠看。

这也是区分两个同型号相机的唯一办法:这台机器上两个 DECXIN 型号字符串完全相同,能
区分它们的只有各自拍到的画面。换 USB 口或重启后编号会变,重扫一次。

### 怎么确定哪个是哪个

**Linux(实验室 / Orin):有稳定 id,直接用。** 扫描会优先给出
`/dev/v4l/by-id/usb-..._-video-index0` 这样的符号链接,udev 用厂商、型号、序列号**和
物理 USB 口路径**拼出来,所以两个同型号相机拿到不同的链接,而且重启、别的设备增减都
不影响。OpenCV 接受路径和接受编号一样,`CameraSpec.index_or_path` 两者都存,所以这样
配好的 rig 拔插之后依然有效。菜单上带 🔒 的就是这种。换到别的 USB 口链接会变 —— 那是
诚实的行为,因为你确实改了物理接线。

**macOS:没有可靠的程序化办法,别找了。** uniqueID 是存在的(`0x1300001bcf2cd1`,编的
是 USB 位置),但 OpenCV 不接受按 uniqueID 打开,试过返回 False。而任何枚举都不共享
OpenCV 的 index 空间 —— 这台机器上四种枚举给出四种顺序:

```
OpenCV 实际         0=DECXIN(工作区)  1=DECXIN  2=FaceTime  3=OBS
system_profiler     0=OBS  1=FaceTime  2=DECXIN  3=DECXIN
ffmpeg avfoundation 0=OBS  1=FaceTime  2=DECXIN  3=DECXIN
AVFoundation(PyObjC) 0=FaceTime 1=OBS  2=DECXIN  3=DECXIN
```

能凑出一个今天对得上的置换,但那是拟合一次观测,下个 OpenCV 版本或下台机器就不成立
——而且猜错是静默的。所以:

1. **看缩略图**(默认做法)
2. **拔插法**,要确认时最可靠:扫描 → 拔掉一个相机 → 重扫,消失的那个就是它。零代码,
   100% 准确
3. **减少变量**:退出 OBS(它的虚拟相机占一个编号)、关掉 iPhone 连续互通。设备少了,
   编号就不容易在两次会话之间变

**不向相机请求分辨率。** lerobot 的 `_configure_capture_settings` 会在设备没有采纳
请求的宽高时抛错,而这两个同型号相机里有一个根本不接受 640x480,始终吐 1920x1080。
所以一律按设备原生分辨率采,预览侧再等比缩放加黑边(不拉伸 —— 画面构图是操作者用来
判断相机对没对准的依据,压扁了会误导)。

**读帧不会拖慢控制回路。** 采集在 lerobot 自己的线程上,`read_latest` 只是看一眼它
已经放下的那一帧,实测 0.0014 ms/次。两个 1080p 相机开着时回路 p05 25.9 Hz,不开时
23.8 Hz —— 差异在噪声内。超过 500 ms 的帧当作没有,宁可空着也不显示一张不知道已经
冻住了的图。

## 视频编码

录制时每个视角写一个 `<role>.mp4` 到 episode 目录里。编码在自己的线程上,前面挡一个
有界队列 —— 编 1080p 一帧是毫秒级,而回路整个预算只有 33 ms,所以控制线程只负责把帧
丢进队列。队列有界是因为无界的那个版本会把「编码器跟不上」变成「录制途中吃光内存」:
1080p RGB 一帧 6 MB,积压一分钟就是 11 GB。

**每个控制步写一帧**,即使相机没有新画面。这样 video 帧 *i* 就是 `frames.npz` 的第
*i* 行。相机没新帧就重写上一帧(`stale`),队列满了才真丢(`dropped`),两个计数都进
`meta.json`。跳过一帧会让之后每一帧都错位,而这种错位在训练之前完全看不出来。

编码器**按机器探测选**,没有全局默认。规则和实测依据见 `so_snake/config.py` 的
`VideoConfig`;`codec="auto"` 时,核数少于 `hw_core_threshold`(默认 8)用硬编码,
否则用软编码。要点名就设 `VideoConfig.codec`,点名的也会先验证 —— 指定一个跑不了的
编码器应该报错,而不是被悄悄换掉。

探测是**真编几帧**而不是构造编码器对象。`av.codec.Codec(name, "w")` 只说明它被编译
进来了:很多 FFmpeg 构建里有 NVENC 而机器上没有 NVIDIA 驱动,构造成功、编码失败,
失败点落在一条 take 的第一帧上。同 `probe_gl_backend` 的道理,只是弱一档 —— GL 选错
会让 `import mujoco` 直接抛异常所以必须开子进程,编码器抛的是能接住的异常,留在本
进程里就够。

**`q:v` 是个陷阱。** 它不是编码器选项,是 ffmpeg 命令行的简写,由 ffmpeg.c 翻译成
`AV_CODEC_FLAG_QSCALE` + `global_quality = N × FF_QP2LAMBDA`,VideoToolbox 读的是那个
**标志**。从 PyAV 的 options 字典塞 `"q:v"` 匹配不上任何 AVOption,被静默丢弃 ——
质量 40 和 25 会产出字节完全相同的文件,都停在编码器默认的 ~9 Mbps 上。
(lerobot 的 `datasets/video_utils.py` 目前就有这个 bug。)`data/video.py` 显式设了
标志和 `global_quality`,这才是硬编码可用的前提。

HEVC 还要显式打 `hvc1` tag:VideoToolbox 默认打 `hev1`,QuickTime / 预览 / Safari 对
它支持很差。torchcodec 不在乎,但你双击一个 episode 想看一眼的时候在乎。

## 回看:按帧号对齐,不是按时间戳

数据集页把两路相机和轨迹曲线放在一起,共用一个游标:点曲线定位视频,播放视频时游标
跟着走。

**对齐用的是帧号,不是时间戳**,这一点必须清楚。录制时每个控制步写一帧,所以 video 帧
*i* 就是 `frames.npz` 的第 *i* 行 —— 但文件的帧率写的是配置里的 `control_hz`(30),
而回路实跑只有 26 Hz 左右。实测 `ep_20260810_232308`:501 帧,录制时长 **19.22 s**,
视频文件时长 **16.70 s**,差 2.52 s。按时间戳对齐的话,片尾会错开两秒半 —— 对一条抓取
演示来说这是致命的。帧号才是两边真正共享的坐标。

视频通过 `/api/episode/video?id=...&camera=...` 提供,**实现了 Range 请求**。这不是可选
的:`<video>` 元素拿不到 byte range 就没法拖动进度条,而一条只能从头播的 episode 视频
对回看毫无用处。标准库的 `SimpleHTTPRequestHandler` 不做 Range,所以是自己写的。

## GL 后端 / 相机预览

启动时网关会打出它选了哪个,以及凭什么:

```
  MUJOCO_GL egl     — verified by rendering a test frame
  MUJOCO_GL glfw    — verified by rendering a test frame
  MUJOCO_GL (none)  — no working GL backend (egl: ...; glfw: ...; osmesa: ...)
```

选择是**在子进程里真渲染一帧**决定的,不是靠 import 检查。这不是讲究,是被迫的:

MuJoCo 在 **import 时**读 `MUJOCO_GL`,并立刻 import 对应的 GL context 模块。所以
选错的代价不是「预览黑掉」,而是 `import mujoco` 直接抛异常,整个仿真 backend 一起
没了。两种独立的挂法都在这台机器上撞到过:

1. `AttributeError: module 'OpenGL.EGL' has no attribute 'EGLDeviceEXT'` ——
   PyOpenGL 太老(3.1.0 没这属性,3.1.10 有),MuJoCo 的 EGL 路径要用它。
2. 属性在,但 EGL **驱动**不支持 `PLATFORM_DEVICE`,renderer 构造失败。

而这两件事都没法在本进程里查。MuJoCo 会在 import `OpenGL.EGL` 之前自己设
`PYOPENGL_PLATFORM=egl`;任何东西抢先 import 了那个模块 —— 包括一个好心检查
`EGLDeviceEXT` 在不在的探针 —— 都会让 PyOpenGL 绑到 GLX 平台上(只要 `DISPLAY`
有值),之后 MuJoCo 的 EGL 设备查询就失败。**探针会破坏它要探的东西**,而且
PyOpenGL 的平台绑定一旦做了就撤不回来。

所以顺序是 egl → glfw → osmesa,谁能真渲染出一帧就用谁,一次子进程启动、几百毫秒。
自己 export 过 `MUJOCO_GL` 就完全跳过探测 —— 那是操作者已经做过的决定:

```bash
MUJOCO_GL=glfw   ... scripts/serve_gui.py   # 有显示器
MUJOCO_GL=egl    ... scripts/serve_gui.py   # 无头
pip install -U PyOpenGL                     # 修 EGL 本身
```

### GL context 属于线程,不是属于「同一时刻一个调用者」

改 `SimPreview` 之前必读。`ThreadingHTTPServer` 每个请求开一条新线程,而 MuJoCo 的
renderer 把 GL context 绑在**构造它的那条线程**上。于是「谁请求就在谁的线程上渲染」
会落在一条不持有 context 的线程上:

- **GLFW**:只 warn 一句 `GLX: Failed to make context current`,**不抛异常**,MuJoCo
  照样去 `mjr_readPixels` —— 读回来是撕裂、品红/绿噪声、底部噪声带。构造 renderer
  时甚至能直接 **segfault** 掉整个进程。
- **EGL**:严格,抛 `EGLError: EGL_BAD_ACCESS`。

复现很干净:同一个静止不动的场景,从新线程渲染 6/6 帧全坏(frame mean 0.00,参考
值 91.23)。表现是**偶发**,因为请求线程渲染完就退出,下一条能不能绑上取决于上一条
死线程的绑定有没有被驱动释放;100 ms 的帧缓存又把大部分请求挡掉了。

**加锁解决不了这个。** 互斥保证的是「同一时刻只有一个渲染」,这里要的性质是线程
亲和性 —— 给一个线程亲和的资源加锁并不会让它变成线程安全的。所以所有 GL 调用
(构造、绘制、close)都提交到 `SimPreview` 自己的一条 `max_workers=1` 长驻线程,
由它独占 context;调用方拿到的仍是一个普通的阻塞方法。

## 页面

| 页面 | 做什么 |
|---|---|
| 遥操作 / 录制 | 选 backend/source 起会话;实时看回路频率、IK 误差、限幅标志、关节指令 vs 实测、仿真相机;一次会话里连续录多条 |
| 数据集 | 列出 `data/episodes/`,两路相机与轨迹曲线共用一个游标对齐回看,补标注,删除 |
| 回放 | joint / task 两种模式回放到 mock / mujoco / real,带静态检查、接近首帧、限速与偏差统计 |
| 进度 | 蓝图各模块的状态与 TODO,数据来自 `so_snake/gui/roadmap.py` |

## 设计上的几条约定

**网关不持有机器人状态。** 全部在 `SessionManager` 里,CLI 脚本驱动的也是同一套
对象。handler 一旦开始自己记「臂是不是在动」,这唯一不能有两个来源的事实就有了
两个来源。

**同一时刻只有一件事驱动机械臂。** 遥操作、回放、归位都走同一个 worker 线程和同一个
`_mode`,不是靠各个 endpoint 互相检查。并发的回放会和控制回路抢这条臂,在真机上
这不是事后能调的 bug。

**每个写操作都是 POST 并返回新快照。** 前端不用猜自己那一下点出了什么,也不用轮询
去发现。

**失败是 4xx/5xx 带原文,不是 200 带标志位。** 被拒绝的回放要拒绝得响亮,而且报的
是机器人代码真正抛的那句话。

**相机预览不能拖慢控制回路,也不能在调用方的线程上画。** 渲染跑在 `SimPreview` 自己
的 `MjData` 上,backend 锁只在拷 `qpos` 的那一瞬间持有:直接锁着 `sim.data` 渲染时
实测控制步掉到 12 Hz(预算 33 ms,一次渲染几十 ms),改完之后同样压力下 p05 是
29.8 Hz。而那次 memcpy 本身也发生在 GL 线程里(作为提交任务的一部分),这样拷贝和
绘制天然有序,不需要第二把锁 —— 原因见上面「GL context 属于线程」。

## 依赖

后端只用标准库 + numpy —— 没有 web 框架。PNG 编码是 `zlib` 上的三十行,免掉了
Pillow/opencv。前端是 React + Vite,只有这两个运行时依赖,图表是自己画的 SVG。
