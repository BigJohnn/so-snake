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

## 串口:留空即自动检测

真机 backend 的「串口」框留空就行,网关按 USB 桥接芯片认驱动板(见仓库 README 的
「串口自动检测」)。框下面列出的是它认为可能是机械臂的口,点一下即填;`✓` 是它选中
的那个。`/api/ports` 只读设备表、不打开任何口,所以会话跑着也能刷新。

## 归位之后不卸力

**归位到位后力矩保持着,状态是 `held`(已归位 · 保持力矩),可以直接点「启动遥操作」。**
这条是有理由的:真机在归位结束时卸力,重力立刻把大臂和肘带下去,操作者刚要到的那个
位姿在他能用之前就没了 —— 于是「归位」这个动作除了让臂晃一下之外什么也没留下。

`held` 状态下:

- **backend 不重建,直接接管。** 同一个 `SOFollowerBackend` 交给新会话,力矩全程没断,
  遥操作从 home 位姿起步。真机上这也是唯一可行的做法:串口被握着,再开一个只会失败,
  而要腾出来就得先 disconnect —— 那正是要避免的掉力矩。
- **只有「停止 / 卸力」会松开它。** 卸力是操作者要求的动作,不是某个动作做完的副作用。
- **换一条臂会被拒绝**(换了 backend / 端口),提示先停止卸力;只改每步钳位是允许的,
  它直接写到在线的 backend 上(lerobot 每次 `send_action` 都读这个配置)。
- **相机可以扫描**:归位不打开任何相机,没有设备冲突。

## 归位点:遥操作途中就能改

归位走的是 `assets/so100_start_pose.json` 里记录的关节角,没有这个文件才回落到
`TeleopConfig.home_joints_deg`。**会话卡上的「记录当前位姿为归位点」按钮直接读当前
关节角写进去** —— 遥操作途中飞到你希望每条 take 都从那里开始的位置,按一下就行,
下一次归位(包括 take 之间的自动归位)就去那儿。

- 关节角是**从臂上读的**,不是拿最近一帧遥测凑的:遥测里手爪是*指令*值、其余是实测,
  一个各部分来源不同的位姿等于没人真正看过它。代价是一次总线读,由 `LockedBackend`
  和控制回路串起来。
- **写入前后各校验一次关节限位。** 现在存下的越限位姿,是以后某次无人看管的归位一头
  撞进限位钳的原因;而文件是能手改、也会被下一个 clone 仓库的人读到的,所以读的时候
  再查一遍。越限就拒绝并**回落到配置的 home**,理由打到日志和界面上 —— 因为「JSON 坏了
  所以不给你归位」会让操作者没法把一条通电的臂停下来。
- 工作区盒子只记录、不强制:盒子是**遥操作**的限幅,从盒外折叠的位姿起步再飞进去是
  正常需求(本机现有的 start pose 就在盒外)。界面上给一条提示,仅此而已。
- 命令行同一件事:`scripts/move_to_start.py --capture`(不发运动指令;注意 lerobot 的
  `configure()` 在 `connect()` 末尾会重新上力矩,所以读的那一两秒臂是握住的,读完显式
  卸力)。

## 一条一条录:定长 + 自动归位

录制卡上两个数:**每条帧数**和**目标条数**。

- 录满设定帧数**自动保存**并停止。手动停的 take 长度取决于操作者的反应时间,而这点
  方差会原样进数据集;定长把它去掉。填 0 就是老行为(一直录到手动停)。
- 一条录完(保存或丢弃都算)**自动归位**,归位期间模式是 `homing`,「开始录制」按钮
  是灰的,到位后回到 `teleop` 等着 —— **不会自动开下一条**,要操作者再按一次。中间这
  段时间正好用来把场景摆回去。
- 归位跑在控制回路的 step 回调里,也就是拥有这条臂的那个线程上,期间回路是停住的;
  走完之后 `sync_target_to_arm()` 把回路的目标重新对到臂当前的位置,否则下一步会把
  臂直接从 home 拽回 take 结束时的目标。
- **目标条数只是计数提示**,到数了不会停任何东西,只在状态栏和日志里显示 `[3/10]`。

### 录完之后:保留 / 丢弃

自停的那一条会**回来等一个判断**:录制卡上出现「刚录完 ep_xxx · N 帧 · X s / 这条要吗」
加「保留」「丢弃」两个按钮,正好卡在自动归位那段时间里 —— 那也是操作者刚看完这条抓得
成不成的时刻。

- **数据在按按钮之前就已经落盘了。** recorder 是边录边写(帧和视频都是),这正是崩溃、
  拔线、磁盘满的时候还能留下这条 take 的原因。所以「丢弃」是一次**删除**,不是「决定
  不写」。
- 丢弃会把它从批次计数里**减掉**,`8 / 10` 才是八条真能拿去训练的。
- 不选也行:直接按「开始录制」下一条,上一条按保留处理(日志里写明,免得读成「忘了」)。
- **手动按「保存这条」停的不会再问一遍** —— 那本来就是一次判断。定长跑完只是帧数用完了,
  不是。

## 回放:到不了第一帧,不等于回放不了

真机回放的第一步是**限速、无 IK 地走到该条的第一帧**(`move_to_joints`),走到了才开始
放。这里踩过一个坑,值得写下来:

判「走到了」的容差原本是 **1.0°**,而这条臂根本做不到。STS3215 在位置模式下是个比例
控制器,lerobot 的 `configure()` 还把增益减半(`P_Coefficient` 32 → 16,注释写的是
"to avoid shakiness"),于是每个受力的关节都会**稳态偏一点**。从当晚录的 take 里直接
量(`observation.state.joints_deg` 减 `action.joint.commanded_deg`,取臂静止在起始位姿
那几十帧):

```
shoulder_pan  2.7    shoulder_lift  1.2    elbow_flex  0.8
wrist_flex    1.1    wrist_roll     0.9        (度,均值)
```

肩部差 2.7°,比容差大一倍多,所以 `move_to_joints` **永远**报不成功;而回放当时把这个
当致命错误直接 return,`SessionManager` 随即 teardown → `disconnect()` → 卸力。现象就是
**臂走到起始位姿、卸力、一帧没放**。用当晚那条 441 帧的 take 在同样偏置的伺服模型上复现:
旧设置放了 0/441,新设置 441/441。

改法不是把容差调大了事,而是把两件事分开:

- `TeleopConfig.joint_settle_tol_deg`(3.0°)= **伺服能站住的精度**,是量出来的,不是偏好;
  归位、move_to_start、回放接近全都从这里取值。
- `TeleopConfig.joint_stuck_deg`(8.0°)= **有东西挡着**。只有超过这个才中止回放,并且报的是
  「哪几个关节、差多少、是不是已经不动了」,不再是一句「did not reach the first frame」。
- 中间地带照常回放,并在日志和界面上说明「起点差 X°,前几帧会补上」—— 回放本身是限速的,
  这点残差它自己就吃掉了。
- `move_to_joints` 现在还会**检测停滞**(1 秒内最好成绩没改善 0.15°),不再对着一个够不到的
  目标推满 200 步(7 秒)再放弃。

**真机回放结束后也保持力矩**(状态 `held`),理由和归位一样:轨迹放完不是操作者要求卸力,
而在最后一帧的位置上撒手,臂会直接掉下去。仿真 backend 不受影响,跑完照常回到 idle。

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

录制页把两路相机和轨迹曲线放在一起,共用一个游标:点曲线定位视频,播放视频时游标
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
| 遥操作 / 录制 | 选 backend/source 起会话(串口留空自动检测);归位后保持力矩可直接接着遥操作;实时看回路频率、IK 误差、限幅标志、关节指令 vs 实测、相机;定长录制,一条录完自动归位后等下一次开录 |
| 数据集 | 列出 `data/episodes/`,两路相机与轨迹曲线共用一个游标对齐回看,补标注,删除 |
| 回放 | joint / task 两种模式回放到 mock / mujoco / real,带静态检查、接近首帧、限速与偏差统计 |
| 进度 | 蓝图各模块的状态与 TODO,数据来自 `so_snake/gui/roadmap.py` |

## 设计上的几条约定

**网关不持有机器人状态。** 全部在 `SessionManager` 里,CLI 脚本驱动的也是同一套
对象。handler 一旦开始自己记「臂是不是在动」,这唯一不能有两个来源的事实就有了
两个来源。

**同一时刻只有一件事驱动机械臂。** 遥操作、回放、归位都走同一个 worker 线程和同一个
`_mode`,不是靠各个 endpoint 互相检查。并发的回放会和控制回路抢这条臂,在真机上
这不是事后能调的 bug。取数据的 take 之间那次归位也走这条路 —— 它跑在控制回路自己的
线程里,而不是响应 HTTP 的那条线程里。

**卸力是操作者要求的,不是动作做完的副作用。** 归位结束停在 `held`(力矩保持),
唯一的出口是「停止 / 卸力」。反过来的默认值听着更安全,实际是让机械臂在归位结束的
一瞬间自己掉下来。

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
