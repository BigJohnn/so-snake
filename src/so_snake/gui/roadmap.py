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
                evidence="322-step take, two 1080p cameras: 322 frames each, 0 dropped, "
                         "0 stale; loop 26.6 Hz median while encoding vs 26.7 Hz idle",
            ),
            _item(
                "lerobot-export", "LeRobotDataset export", "todo",
                "Episodes are npz so recording never fails for a missing dependency. "
                "Converting to LeRobotDataset belongs on the training machine.",
                blockers=("converter not written",),
            ),
            _item(
                "qc", "Automatic take review", "todo",
                "Flag takes with clamped stretches, held commands or a lost clutch, so a "
                "bad demonstration is caught at the bench and not in training.",
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
