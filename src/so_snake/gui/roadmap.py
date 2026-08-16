"""The status board: what the blueprint asks for, and where each piece stands.

This is the same information as the README's checklist and `docs/`, restated in
a form a page can render. It lives in the repository rather than in the frontend
because it is a claim about *this code* -- when M0 perception lands, the person
who lands it edits this file next to the module they wrote, and the board is
right. A copy in TypeScript would be right for one release.

Statuses are deliberately only three. "done" means there is code and a gate that
covers it; "partial" means it runs but something named in `blockers` is missing;
"todo" means nothing has been built. Anything measured belongs in `evidence`,
with the number, because "IK works" and "round-trip p95 = 0.078 mm" are not the
same claim.
"""

from __future__ import annotations

from typing import Any

STATUSES = ("done", "partial", "todo")


def _item(
    key: str,
    title: str,
    status: str,
    detail: str,
    *,
    module: str = "",
    evidence: str = "",
    blockers: tuple[str, ...] = (),
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    return {
        "key": key,
        "title": title,
        "status": status,
        "detail": detail,
        "module": module,
        "evidence": evidence,
        "blockers": list(blockers),
    }


ROADMAP: tuple[dict[str, Any], ...] = (
    {
        "group": "M1 — geometry & kinematics",
        "items": (
            _item(
                "urdf", "SO-100 URDF + TCP frame", "done",
                "The upstream URDF with the gripper TCP frame added; SO-100, not SO-101.",
                module="assets/urdf/so100/so100.urdf",
                evidence="docs/so100_vs_so101.md",
            ),
            _item(
                "fk-ik", "FK / 5D task IK", "done",
                "Damped least squares over the five task coordinates, no yaw objective.",
                module="so_snake.m3_safety.ik5d",
                evidence="round-trip p95 0.078 mm / 0.004 deg",
            ),
            _item(
                "cross-check", "Three-way kinematics agreement", "done",
                "ArmChain, placo and MuJoCo compared on FK and Jacobians.",
                module="scripts/check_kinematics_agreement.py",
                evidence="agree to 1e-9 m",
            ),
        ),
    },
    {
        "group": "M3 — feasibility & safety",
        "items": (
            _item(
                "task-space", "5D task space (x, y, z, pitch, roll)", "done",
                "The arm's controllable manifold is five-dimensional; the action space says so.",
                module="so_snake.m3_safety.task_pose",
                evidence="docs/five_dof_orientation.md",
            ),
            _item(
                "atlas", "Position-conditioned feasibility atlas", "done",
                "Per-voxel pitch interval and roll feasibility, applied to the target before IK.",
                module="so_snake.m3_safety.atlas",
                evidence="built from 40 M samples",
            ),
            _item(
                "clearance", "Mesh clearance guard", "done",
                "In MuJoCo, the lowest moving mesh point is checked before every write.",
                module="so_snake.sim.mujoco_arm",
            ),
            _item(
                "tcp-cal", "Measured TCP calibration", "todo",
                "The TCP is derived from the SO-101 drawing, not measured on this arm.",
                blockers=("needs the arm and a measurement jig",),
            ),
        ),
    },
    {
        "group": "M4 — execution & teleoperation",
        "items": (
            _item(
                "backends", "Mock / MuJoCo / real backends", "done",
                "One Protocol, three implementations; the loop cannot tell them apart.",
                module="so_snake.m4_execution.backends",
            ),
            _item(
                "loop", "30 Hz teleoperation loop", "done",
                "Clutch retarget, workspace clamp, atlas, 5D IK, joint safety, backend.",
                module="so_snake.teleop.loop",
                evidence="scripts/check_teleop_loop.py PASS",
            ),
            _item(
                "pacing", "The loop actually holds 30 Hz", "done",
                "It used to run at 26.3. Not a slow step -- a step costs about 5 ms of "
                "the 33 ms budget (bus sync_read 1.5 ms twice, sync_write 0.5, IK 0.3, "
                "controller poll 1.4) -- but time.sleep returning late: macOS coalesces "
                "timer wake-ups and adds slack proportional to the sleep, ~4 ms on a "
                "33 ms period. Pacing to `period - elapsed` from the top of the "
                "iteration then booked every overshoot permanently. Now paced to a "
                "deadline that advances by one period, sleeping to within 6 ms and "
                "spinning the rest, with repayment capped at one period so a stall "
                "cannot buy a burst of unpaced commands.",
                module="so_snake.pacing",
                evidence="mock backend measured 30.00 Hz over 200 steps (was 26.8); "
                         "the 44 takes on this bench were all recorded before the fix "
                         "and sit at 26.1 Hz median (25.8-26.9), which is why the "
                         "exporter measures the rate instead of reading the config",
            ),
            _item(
                "joint-map", "lerobot ↔ URDF joint map", "done",
                "Exact bijection, so reading back and writing straight out does not move the arm.",
                module="so_snake.m4_execution.joint_map",
            ),
            _item(
                "real-teleop", "Real-arm teleoperation", "partial",
                "The script and the safety layering exist and the arm has been driven; "
                "the clutch feel and the start pose still need tuning against the hardware.",
                module="scripts/teleop_real_arm.py",
                blockers=("clutch tuning on hardware",),
            ),
        ),
    },
    {
        "group": "Data engine",
        "items": (
            _item(
                "record", "Episode recording", "done",
                "All three action streams -- raw device, projected task, executed joints -- "
                "plus the config snapshot that gives them meaning.",
                module="so_snake.data.recorder",
            ),
            _item(
                "replay-joint", "Joint-mode replay", "done",
                "Send the recorded commands back out; approach, rate cap and clearance guard.",
                module="so_snake.data.replay",
            ),
            _item(
                "replay-task", "Task-mode replay", "done",
                "Re-solve the recorded 5D targets through today's IK, to regression-test "
                "a solver change against real operator input without re-recording.",
                module="so_snake.data.replay",
            ),
            _item(
                "cameras", "Wrist + third-person capture", "done",
                "Real USB cameras are enumerated with names, assigned to roles by the "
                "operator, previewed live, and written into episodes as one mp4 per role. "
                "Encoding runs on its own thread behind a bounded queue, and one video "
                "frame is written per control step so video frame i is row i. The encoder "
                "is chosen per machine -- hardware where CPU is scarce, software where "
                "disk is -- verified by encoding before it is used, and both the choice "
                "and its reason go into meta.json.",
                module="so_snake.data.video",
                evidence="501-step take, two 1080p cameras: 501 frames each, 0 dropped, "
                         "0 stale, matching n_steps exactly; loop 26.5 Hz median while "
                         "encoding with both previews polling vs 26.7 Hz idle",
            ),
            _item(
                "review", "Reviewing a take against its video", "done",
                "The dataset page plays both cameras beside the trajectory plots, sharing a "
                "cursor. Alignment is by frame index, not timestamp: the recorder writes one "
                "frame per control step, but the file carries the configured control_hz while "
                "the loop runs slower, so on a 19.2 s take the video is 16.7 s and a "
                "timestamp-aligned cursor would be 2.5 s out by the end. Clicking a plot seeks "
                "the video; the video's own clock drives the cursor while it plays. The "
                "second camera follows the first's transport -- play, pause, seek, rate "
                "-- and the index nudge is only a correction on top. It used to be the "
                "only thing moving it: nobody called play() on the follower, so it "
                "advanced solely via the seek in the timeupdate handler, which browsers "
                "throttle to ~4 Hz. The wrist view was a 4 fps slideshow beside a 30 fps "
                "scene view, which is the one camera that shows whether the jaws closed.",
                module="so_snake.gui.server",
                evidence="ep_20260810_232308: 501 video frames == 501 rows on both cameras; "
                         "byte-range requests answered 206 so the player can seek",
            ),
            _item(
                "lerobot-export", "LeRobotDataset export", "done",
                "Task-selected export to LeRobotDataset with reached-pose state, "
                "measurement-anchored 5D delta action, absolute gripper, measured "
                "dataset fps and two camera streams. Driven from a button on the "
                "recordings page -- dry run, then a background job with progress -- or "
                "from the CLI. The lerobot import is lazy, so recording keeps its "
                "numpy-only dependency surface.",
                module="so_snake.data.export",
                evidence="牛牛抓放 dry-run: 40 usable takes / 19491 frames at 26 Hz; "
                         "takes ran 26.27-26.59 Hz, worst deviation 2.3%",
            ),
            _item(
                "export-verify", "Exports are read back before they are trusted", "done",
                "Every export writes an export.json naming its source takes in dataset "
                "order, then reopens the parquet, the manifest and the video files and "
                "checks three things: the rows match a fresh conversion, apply_action "
                "still inverts them to the commanded 5D targets, and there is one "
                "decodable video frame per row per camera. This is not the dry run "
                "repeated -- it is the only check that can see a missing parquet "
                "footer (LeRobotDataset.finalize was never being called), a video a "
                "frame short, or a timestamp grid built from a rate nothing ran at. "
                "All three look like success at the moment of writing.\n\n"
                "Failure and gap are separate verdicts. An issue is a check that ran "
                "and the dataset lost, against a reference inside it -- the rows, the "
                "timestamp grid, the manifest's own episode count -- and blocks "
                "training. A skip is a check that could not run because something "
                "outside the dataset was absent: no export.json, no store, or a source "
                "take since deleted. That last one used to be an issue, which was "
                "wrong: deleting a take changes no byte of the dataset, and a verdict "
                "that turns red because another directory changed is not a statement "
                "about the dataset -- the same export would read green on the bench "
                "that still holds the takes and red on the training box that never "
                "had them. Those read PARTIAL, with the unresolvable takes named and "
                "episodes_compared saying how much of the dataset the error figures "
                "actually cover.",
                module="so_snake.data.export",
                evidence="1672-row export reconstructs its targets to 0.010 um / "
                         "12 udeg, timestamps within 0.9 us of frame_index/fps, "
                         "1672 video frames on both cameras; a video truncated to 500 "
                         "frames is rejected; a one-episode export whose source take "
                         "was deleted afterwards reads PARTIAL at 0/1 compared, not "
                         "FAILED",
            ),
            _item(
                "verdict-cache", "The verify verdict is cached, and says when to distrust it",
                "done",
                "Verifying decodes every frame of every video -- minutes on a "
                "20 000-frame dataset -- so the answer is stored as verify.json inside "
                "the dataset it is about, and the library view shows a badge per "
                "dataset without re-reading them all on every page load. Two things "
                "keep the cache honest: a version stamp, so a verdict computed by "
                "checks that have since changed meaning is discarded rather than "
                "migrated, and the dataset's newest mtime, so bytes rewritten since "
                "read as stale rather than as still-good.\n\n"
                "The mtime deliberately excludes verify.json itself. Counting it made "
                "the check self-defeating -- the mtime goes into the file before the "
                "file is written, so the write became the newest thing under the "
                "dataset and every verdict came back \"changed since verified\" within "
                "a second of being computed. Every dataset in the library showed 过期 "
                "and re-verifying could not clear it.",
                module="so_snake.gui.exporter",
                evidence="tests/test_gui_datasets.py covers a fresh verdict not being "
                         "stale, a genuine rewrite being stale, and a verdict from "
                         "another schema version being discarded",
            ),
            _item(
                "dataset-replay", "Replaying an exported dataset onto the arm", "done",
                "An exported episode is rebuilt into the same Episode shape a recorded "
                "take has, and played by the same EpisodeReplayer -- same rate-limited "
                "approach, same deg/s clamp, same joint limits, same mesh clearance "
                "check. None of that safety logic is duplicated. Task mode only: the "
                "dataset carries no joint stream by design, so joints are solved from "
                "the targets, which is what task-mode replay does anyway. Triggered "
                "from the datasets page (one click, same rig controls as a take "
                "replay) or from scripts/replay_lerobot_dataset.py -- the two paths "
                "go through the same `SessionManager.start_dataset_replay`.",
                module="so_snake.data.export",
                evidence="558-step exported episode replayed on the mock arm: "
                         "completed, task position error p95 0.0034 mm, IK converged 99.1%",
            ),
            _item(
                "raw-vs-exported", "Raw takes and exported datasets live on different pages",
                "done",
                "What `data/episodes/` and `data/lerobot/` have in common stops at "
                "both being JSON-shaped. A page that mixed them invited reading a "
                "verdict that only applied to the export as if it spoke about the "
                "take. The recordings page is now labelled \"录制\" and shows raw "
                "takes; a new \"训练集\" page lists the exported datasets, shows the "
                "manifest + cached verify verdict, and re-runs verify / replays any "
                "dataset onto the arm. Dataset rows wear an amber rail in the library; "
                "the recordings page does not.",
                module="so_snake.gui.server",
                evidence="/api/export/datasets, /api/export/verify, /api/replay/dataset "
                         "wired through the gateway; tests/test_gui_datasets.py "
                         "covers list, verify, replay, path-traversal and the busy-arm "
                         "refusal",
            ),
            _item(
                "task-labels", "Task-labelled training batch", "done",
                "The previously unlabelled batch was assigned the 牛牛抓放 task label, "
                "and export screening selects by task. A count here is a snapshot of "
                "what is on the bench, not a fixed claim -- takes get recorded and "
                "deleted between sessions.",
                module="so_snake.data.store",
                evidence="44 takes on disk: 42 labelled 牛牛抓放 (40 pass export "
                         "screening, 2 have no third-person camera), 2 one-off labels; "
                         "40 usable takes = 19491 frames = 12.4 min at 26 Hz",
            ),
            _item(
                "qc", "Automatic take review", "todo",
                "Flag takes with clamped stretches, held commands or a lost clutch, so a "
                "bad demonstration is caught at the bench and not in training.",
            ),
        ),
    },
    {
        # Separate from "M2 three-headed VLA" on purpose. This is the baseline
        # policy the training set was built for, and it is the piece with the
        # nearest open item -- keeping it inside a single "M2 todo" row hid both
        # the settled contract and the one thing that is actually missing.
        "group": "ACT baseline",
        "items": (
            _item(
                "act-contract", "State / action contract for training", "done",
                "The recorded format is deliberately redundant -- three action "
                "streams -- and a training set has to pick one. State is "
                "FK(measured joints), not the IK solution's own forward kinematics "
                "(that column's distance to the target is the solver residual, 1e-6 "
                "on this bench; the arm's real distance is 9.6 mm median). The delta "
                "action is anchored on that reached pose rather than on the previous "
                "target, so a rollout re-references the measurement every step "
                "instead of integrating open-loop. The gripper stays absolute in both "
                "spaces. `export.apply_action` is the inverse, so training and rollout "
                "cannot drift apart.",
                module="so_snake.data.export",
                evidence="docs/act_baseline.md; anchoring on the measurement drops "
                         "all-zero action steps from 25% to 0.1%",
            ),
            _item(
                "act-train", "ACT trains on this laptop", "done",
                "Measured on the machine, not estimated: ACT (ResNet18 x2 cameras, "
                "52 M params) on M1 Pro / 16 GB via MPS. 240x320 is the export default "
                "because the 26 Hz control period is 38 ms and ACT re-plans a whole "
                "chunk every n_action_steps -- 480x640's 56 ms does not fit in a "
                "period, 240x320's 22 ms does.",
                evidence="316 ms/step on the exported dataset (updt_s 0.313, "
                         "data_s 0.005 -- two video streams fully overlapped by the "
                         "dataloader), ~1 GB MPS; 20k steps in 1.7 h",
            ),
            _item(
                "rollout", "Rollout runner (policy → arm)", "todo",
                "Policy output through apply_action into a 5D target, then the "
                "existing atlas / IK / rate-limit / clearance layer onto the arm. "
                "EpisodeReplayer is structurally the same loop with the targets "
                "coming from a network instead of a file, so its safety layer is "
                "reusable whole rather than re-derived; the new part is holding the "
                "last command across the ~22 ms re-planning stall instead of blocking "
                "the loop.",
                blockers=("not written",),
                evidence="docs/act_baseline.md 'Where to pick up'",
            ),
            _item(
                "more-takes", "Wider training set", "todo",
                "40 usable takes of one task is close to the LeRobot SO-100 tutorial "
                "baseline in count but narrow in object placement. Two properties of "
                "the original 10-take validation set are worth re-checking on the "
                "larger batch before recording more, because a policy learns both "
                "faithfully: 26% of steps sat pinned at the pitch limit with the atlas "
                "clamping pitch on 56% of them, and x rode the workspace floor "
                "(0.170-0.219 m against a 0.17 m minimum, clamp firing on 3-13% of "
                "steps). Neither has been re-measured over the 40.",
                blockers=("needs the arm, and varied object placement",),
                evidence="40 takes / 19491 frames / 12.4 min at 26 Hz today; the "
                         "pitch/x figures above are from the 10-take set",
            ),
        ),
    },
    {
        "group": "M0 / M2 / M5 — the rest of the blueprint",
        "items": (
            _item(
                "perception", "M0 perception", "todo",
                "Wrist and third-person observation into the policy's input.",
                module="so_snake.m0_perception",
            ),
            _item(
                "policy", "M2 three-headed VLA", "todo",
                "Mode, observation intent and action heads.",
                module="so_snake.m2_policy",
            ),
            _item(
                "verification", "M5 task verification", "todo",
                "Did the cube end up in the box?",
                module="so_snake.m5_verification",
            ),
        ),
    },
)


def roadmap_payload() -> dict[str, Any]:
    groups = [{"group": g["group"], "items": list(g["items"])} for g in ROADMAP]
    items = [item for group in groups for item in group["items"]]
    counts = {status: sum(1 for i in items if i["status"] == status) for status in STATUSES}
    return {"groups": groups, "counts": counts, "total": len(items)}
