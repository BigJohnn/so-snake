# The ACT baseline: what the training set commits to, and why

**Implemented decision, one open item.** The export exists
(`scripts/export_lerobot_dataset.py`), a dataset has been written, and ACT
trains on this laptop. The rollout runner is not written. Companion to the
README's "导出训练集 / 在 MacBook Pro M1 上训练" sections.

Continues [`plan_5dof_task_space.md`](plan_5dof_task_space.md), whose frozen
contract already said the policy action is `Δ(x, y, z, pitch, roll)` + gripper.
It did not say what the `Δ` is measured *from*, and that turns out to be the
decision that matters. This document settles the anchor, and everything else the
recorded format left open until a training set had to be built from it.

None of this required re-recording. All three action streams are on disk, so the
anchor was a choice available at export time — which is the redundancy argued for
in `episode.py` paying for itself the first time it was asked to.

## The measurement that decided the anchor

The obvious anchor is the previous target: `action.task.delta` is already a
recorded column, and it is exactly `target[t] - target[t-1]`.

The reason not to use it starts with a number that looks like good news.
`observation.state.task_pose` sits 1e-6 away from `action.task.target` across
every take on this bench. That reads as near-perfect tracking. It is not
tracking at all: **that column is the forward kinematics of the IK solution**,
so its distance to the target is the solver residual. The name is misleading and
should be read as `command.task.pose`.

Recomputing the pose from the joints the bus actually returned —
`FK(observation.state.joints_deg)`, over the ten `牛牛抓放` takes — gives the
real distance between where the arm was told to be and where it was:

| quantity | median | p95 | max |
|---|---|---|---|
| position error | **9.6 mm** | 41 mm | 94 mm |
| pitch error | 2.7 deg | 10.1 deg | 23 deg |
| roll error | 0.01 deg | 1.2 deg | 13 deg |

Against a per-step motion whose p95 is 2–7 mm. **The standing servo lag is
larger than one step's commanded displacement**, by a factor of five at p95. An
action space that anchors on the previous command does not merely ignore a small
term; it ignores the largest one.

## The contract

| | |
|---|---|
| state | `FK(observation.state.joints_deg)` as `(x, y, z, pitch, roll)`, + gripper angle — 6 |
| action, `delta` | `target[t] - FK(measured joints)[t-1]`, + **absolute** gripper — 6 |
| action, `absolute` | `target[t]`, + absolute gripper — 6 |
| rollout inverse | `target = FK(measured now) + action` (`export.apply_action`) |
| angular dims | pitch and roll wrapped to `(-pi, pi]` on both sides |
| fps | measured: `1 / median(step period)`, then median over the selection |
| cameras | `third_person` and `wrist`, both fed to the policy |
| resolution | 240x320 |

### The mapping, exactly

Yes — the export maps an **absolute pose on the 5D manifold** to a **delta in the
same 5D manifold coordinates**. Both sides live in the one chart
`(x, y, z, pitch, roll)` that `so_snake.m3_safety.task_pose` defines; nothing
here touches SE(3), and no quaternion or rotation matrix appears in the dataset.

Write `q_t` for the joint angles the bus reported at step `t`, and

```
    Φ : R^5 → M ,   Φ(q) = (x, y, z, pitch, roll)
```

for the forward map into the chart (`TaskIK5D.task_pose`, i.e. forward
kinematics followed by the chart's resolution of the tool frame). Let

```
    p_t = Φ(q_t)                    the pose the arm REACHED at step t
    c_t = action.task.target[t]     the 5D target the loop COMMANDED at step t
    g_t                             gripper angle, degrees
```

Then one exported row is

```
    state[t]  = ( p_t ,  g_t^measured )                        ∈ R^6
    action[t] = ( c_t ⊖ p_{t-1} ,  g_t^commanded )             ∈ R^6
```

and the **update formula** — the inverse, which the rollout runs once per step —
is

```
    c_t = p_{t-1} ⊕ action[t][0:5]
    g_t = action[t][5]
```

where `⊕` and `⊖` act componentwise on the chart and differ from ordinary
addition only on the two angular coordinates:

```
    (a ⊕ b)_i = a_i + b_i                        i ∈ {x, y, z}      metres
    (a ⊕ b)_i = wrap( a_i + b_i )                i ∈ {pitch, roll}  radians
    (a ⊖ b)_i = a_i - b_i                        i ∈ {x, y, z}
    (a ⊖ b)_i = wrap( a_i - b_i )                i ∈ {pitch, roll}

    wrap(θ) = (θ + π) mod 2π − π                 folded into (−π, π]
```

At rollout `p_{t-1}` is not read from the dataset — it is measured, right then:

```
    c_t = Φ( q_measured ) ⊕ π_θ(observation)[0:5]
```

which is the whole reason the anchor is the reached pose. `apply_action` in
`so_snake/data/export.py` **is** `⊕`, and it is the only implementation of it;
training and rollout cannot drift apart because there is one function.

Three things this is deliberately **not**:

* **Not a tangent-space / exponential-map step.** `⊖` is a plain difference of
  chart coordinates, not a Lie-algebra log. That is sound here because the chart
  is regular everywhere the arm can go — `psi` is a function of *position*, not
  of the tool's own azimuth, precisely so it does not go singular when the
  gripper points straight down (see `task_pose.py`). A per-step motion is small
  and the chart is smooth over it, so the coordinate difference and the true
  geodesic step agree to well under the servo lag.
* **Not a delta on the gripper.** It is absolute in the action and measured in
  the state; those are two different numbers on purpose. Measured on the first
  format-v2 take, they diverge by up to **10.2°** — the jaws stalling on the
  object, which the commanded angle cannot show.
* **Not anchored on the previous command.** `p_{t-1}`, not `c_{t-1}`. This is
  the decision the rest of this section is about.

Units are metres, radians, degrees — in that order, mixed within the vector.
That is ugly and it is on purpose: they are the units every other layer of this
repository already uses, and a dataset that silently rescaled them would make
every printed diagnostic disagree with the arm.

Verified on `ep_20260812_213956` (552 rows): `state[:, :5] == Φ(q_measured)` and
`action[:, :5] == c_t ⊖ p_{t-1}` hold exactly, and `p_{t-1} ⊕ action[t]`
reconstructs `c_t` to **1.8e-7** (float32 storage). The exporter re-checks this
on every run — see `verify`.

### Why the reached pose, on both sides

Anchored on the previous target, the policy is an open-loop integrator: nothing
in the loop ever compares it to the arm, so a systematic under-prediction walks
away and never comes back. Anchored on the reached pose, every step re-references
the measurement and a rollout corrects itself.

Both sides then carry the same standing lag, so the policy reproduces the lead
the teleoperator's commands carried instead of trailing it. This is why the lag
being large is not itself a problem — it is a repeatable function of velocity,
and it appears identically in training and at rollout.

It also fixes a distribution the target-anchored version cannot:

| anchor | steps whose manifold action is all-zero |
|---|---|
| previous target | **25.1 %** (per-dim zeros 78–92 %) |
| reached pose | **0.1 %** |

The operator moves in bursts and the clutch gates the rest, so target-to-target
deltas are mostly exactly zero. Under L1 the minimiser of that is a policy that
outputs zero and does not move. Anchoring on the measurement makes every step
informative, because the lag is never zero.

`--action-space absolute` exports `target[t]` as the control. If a delta rollout
drifts, it is what says whether the drift came from the action space.

### Why the gripper stays absolute

It is a two-state signal — these takes only ever hold 2 deg or 90 deg and pass
through the middle. A delta on it would spend the episode predicting zero and
then have to hit an 88 deg jump exactly once, which is the worst case for a
chunked L1 objective. As *state* it is the measured angle, the only channel that
reports the jaws stalling on an object.

### Why the frame rate is measured

**Superseded in part, 2026-08-16 — the loop now holds 30 Hz.** What follows
still describes the takes recorded before that, which are the training set as it
stands: every one of the 44 takes on the bench is from that batch. See "The two
rate batches" below.

The loop was configured for 30 Hz and held 26.1. Recording wrote the
*configured* rate into the mp4 headers, so the videos claim to be 15 % shorter
than the takes they came from — the README's "19.2 s take, 16.7 s video" is this.

Exporting at the configured rate trains a policy whose action chunk spans 3.3 s
of intent and then replays it over 2.9 s of wall clock: a rollout 15 % faster
than every demonstration behind it, on an arm whose tracking lag is already the
largest term in the action.

So the rate is measured. It is the **median step period**, `1 / median(dt)`,
taken per take and then the median again over the selection. Two medians, for
two different reasons:

* *Within* a take, because the mean is not robust and there is reliably one
  outlier. Choosing a video encoder costs ~700 ms and used to run on the first
  frame of a take, under the lock the control loop needs. On a 292-step take
  that one step pulled `n_steps / duration_s` from 30.1 Hz down to **28.2** —
  a 6 % error in the grid all 292 frames get laid on, caused by 0.3 % of them.
  The median reports the period the other 291 actually ran at. (The stall
  itself is fixed too: the probe is cached and warmed at session start.)
* *Across* takes, because one take that ran at a different speed should be
  screened out, not allowed to drag the grid every other take is then measured
  against.

### The two rate batches

The pacing fix splits the store in two, and they cannot share a dataset:

| batch | takes | measured | why |
|---|---|---|---|
| ≤ 2026-08-12 | 43 | **26.3 Hz** | `time.sleep` overshoot, booked permanently by `period - elapsed` pacing |
| ≥ 2026-08-16 | new | **30.0 Hz** | `so_snake.pacing.RateKeeper` |

26 and 30 differ by 15 %, far outside `fps_tolerance` (8 %), so a selection
spanning both is caught by the screening rather than silently averaged — the
minority batch is listed as skipped with `ran at 30.01 Hz, dataset is 26 Hz`,
and the report says out loud that this is a second batch and not a bad take.

**Export the two separately.** Widening the tolerance to force them together is
the one thing not to do: it would train a policy on demonstrations whose action
chunks span two different amounts of wall clock.

Frames are read out of the videos **by index** and re-encoded at that rate, which
is also what removes the header lie: lerobot seeks these files by timestamp
(`frame_index / fps`, tolerance 1e-4), so a header disagreeing with the row grid
is not cosmetic. Do not export the `t` column as the timestamp — it is real wall
clock, disagrees with the grid, and will fail that tolerance check.

Measured on the exported set: takes ran 25.84–26.17 Hz, worst deviation from
26 Hz is **0.7 %**; a 441-step take ends at timestamp 16.923 s against a recorded
duration of 16.916 s.

## The format v2 fix

`teleop/loop.py` read all six joints off the bus and sliced `measured[:5]`, so
the gripper's *measured* angle was never written. Commanded and measured part
company exactly when it matters: closing on an object stalls the servo short of
the commanded angle, and that gap is the only proprioceptive evidence that
something is held.

v2 adds `observation.state.gripper_deg`. The 18 takes recorded before it cannot
be repaired; `Episode.measured_gripper_deg()` falls back to the commanded angle
**and returns a flag saying so**, which the export report prints per episode. A
consumer that needs the real thing can refuse rather than train on a substitute
believing it was the observation.

## Training on the MacBook Pro M1

Verified on this machine, not estimated. ACT (ResNet18 x2 cameras, 52 M params),
M1 Pro / 16 GB, MPS, batch 8:

| resolution | train step | 20k steps | 100k steps | cold chunk inference |
|---|---|---|---|---|
| **240x320** | **304 ms** | **1.7 h** | 8.4 h | **22 ms** |
| 360x480 | 593 ms | 3.3 h | 16.5 h | 36 ms |
| 480x640 | 953 ms | 5.3 h | 26.5 h | 56 ms |

240x320 is the default because the control period at 26 Hz is 38 ms and ACT
re-plans a whole chunk every `n_action_steps`: 480x640's 56 ms does not fit in a
period, 240x320's 22 ms does.

A real run on the exported dataset measured **316 ms/step** with
`updt_s 0.313, data_s 0.005` — decoding two video streams is fully overlapped by
the dataloader and is not the bottleneck. ~1 GB of MPS memory. 26 MB on disk for
10 takes.

```bash
HF_LEROBOT_HOME=data/lerobot .venv/bin/lerobot-train \
    --dataset.repo_id=so_snake/niuniu_pick_place \
    --dataset.root=data/lerobot/niuniu_pick_place \
    --policy.type=act --policy.device=mps --policy.push_to_hub=false \
    --output_dir=outputs/act_niuniu --steps=20000 --batch_size=8
```

## What the data looks like, and what is wrong with it

Exportable `牛牛抓放` set after labelling the unlabelled batch: **40 takes,
19491 frames, ~12.4 minutes at 26 Hz** (42 labelled, 2 rejected for having no
third-person camera; 44 takes on the bench in total). Gripper spans 2–90 deg,
closed 38 % of steps, 84 open/close crossings. This is now close to the LeRobot
SO-100 tutorial baseline's 50 takes in count, but still narrow in object
placement. Counts are a snapshot of the bench, not a fixed property of the
project — takes get recorded and deleted between sessions, so re-run
`--dry-run` rather than trusting this line.

Two things about the original 10-take validation set are worth checking against
the larger labelled batch before recording more, because a policy will learn
both faithfully:

**Pitch is nearly dead.** 26 % of steps are pinned exactly at the −1.5707 limit,
the atlas clamped pitch on 56.5 % of steps (per-take, up to 86 %), and the whole
range spans 0.30 rad with std 0.0497. The demonstrations contain a lot of "the
operator asked to rotate and the projector ate it". Worth a look at the atlas
tuning or the clutch feel before the next batch.

**x sits on the workspace floor.** The target range is 0.170–0.219 m against a
`pos_min_m[0]` of 0.17, with the workspace clamp firing on 3–13 % of steps.

## Where to pick up

1. **Write the rollout runner.** `export.apply_action()` is already its half of
   the action-space contract and is covered by a round-trip test — an anchor that
   disagrees between training and rollout is silent and shows up only as an arm
   that creeps. `EpisodeReplayer` (`data/replay.py`) is structurally the same
   thing with the joint targets coming from a file instead of a network, so its
   safety layer (rate limit in deg/s, joint clamp against *current* limits,
   move-to-first-frame, mesh clearance) is reusable whole. Hold the last command
   across the 22 ms re-planning stall rather than blocking the loop.
2. **Then record to ~50 usable takes** with the object in varied positions.

Do not change the state/action contract above without re-exporting; the anchor
is not recoverable from a trained checkpoint.
