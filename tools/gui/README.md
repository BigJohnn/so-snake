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
| 数据集 | 列出 `data/episodes/`,看单条的轨迹曲线和指标,补标注,删除 |
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
