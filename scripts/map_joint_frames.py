#!/usr/bin/env python
"""Establish the joint-frame map between lerobot's calibration and the URDF.

so-snake's kinematics/safety are in the **official SO-ARM100 URDF** convention
(zeros at CAD assembly pose). lerobot's runtime calibration reports **degrees
centred on the recorded range midpoint** -- a different zero, and possibly a
flipped direction. The two are related, per joint, by an affine map with unit
gain:

    q_urdf_deg = sign * q_lerobot_deg + offset_deg,   sign in {+1, -1}

This is exact: lerobot DEGREES mode is `(raw - mid) * 360/4095` with
`mid=(range_min+range_max)/2`, so both frames are linear in the raw encoder tick
with the same |slope|; they differ only by a zero (`offset`) and a mounting
direction (`sign`). Therefore:

  * `offset` is recoverable with NO motion: the recorded range is symmetric about
    lerobot 0, so `offset ~= (urdf_lower + urdf_upper)/2` (the URDF midpoint).
  * `sign` is NOT recoverable from data -- it depends on how each servo is
    physically mounted. It needs one physical observation per joint.

The tool is READ-ONLY: it never enables torque and never commands motion. In
`verify` you gently hand-move the (limp, torque-off) arm while it prints live
numbers; a wrong `sign` shows up as the mapped URDF angle leaving its limits as
you sweep an asymmetric joint (shoulder_lift, elbow_flex, wrist_flex).

Workflow (all read-only; the arm is never powered/moved by the tool):

    # 1. Draft the map from the existing calibration file (no hardware, no motion):
    PYTHONPATH=src .venv/bin/python scripts/map_joint_frames.py draft

    # 2. Capture signs: move each joint (torque OFF, by hand) to its two HARD
    #    STOPS -- no direction to judge -- and answer one visible comparison:
    PYTHONPATH=src .venv/bin/python scripts/map_joint_frames.py signs

    # 3. Confirm: hand-move the limp arm and watch the mapped URDF angles + FK TCP:
    PYTHONPATH=src .venv/bin/python scripts/map_joint_frames.py check

Steps 2 and 3 auto-detect the arm's serial port; `--port` overrides it.

The map is written to assets/so100_joint_map.json. Wiring it into
SOFollowerBackend is a separate, explicit step -- this tool only produces and
checks it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from so_snake.config import ARM_JOINTS, GRIPPER_JOINT, JOINT_LIMITS_DEG, REPO_ROOT
from so_snake.devices import DeviceDetectionError, detect_arm_port

MAP_PATH = REPO_ROOT / "assets" / "so100_joint_map.json"
CAL_PATH = (
    Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/so_follower"
)
CONVENTION = "q_urdf_deg = sign * q_lerobot_deg + offset_deg"
STS3215_MAX_RES = 4095  # model_resolution - 1, used by lerobot DEGREES mode


def _load_calibration(robot_id: str) -> dict:
    fpath = CAL_PATH / f"{robot_id}.json"
    if not fpath.is_file():
        raise SystemExit(
            f"no calibration file at {fpath}\n"
            f"calibrate first: lerobot-calibrate --robot.type=so100_follower "
            f"--robot.port=<PORT> --robot.id={robot_id}"
        )
    return json.loads(fpath.read_text())


def _draft(robot_id: str, flips: set[str]) -> dict:
    """Offsets from the calibration file (no motion); signs provisional (+1 or flipped)."""
    cal = _load_calibration(robot_id)
    joints: dict[str, dict] = {}
    for name in ARM_JOINTS:
        lo, hi = JOINT_LIMITS_DEG[name]
        urdf_mid = (lo + hi) / 2.0
        c = cal[name]
        # lerobot DEGREES half-span at the recorded stops, in degrees.
        lero_half_span = (c["range_max"] - c["range_min"]) / 2.0 * 360.0 / STS3215_MAX_RES
        urdf_half_span = (hi - lo) / 2.0
        joints[name] = {
            "type": "degrees_affine",
            "sign": -1 if name in flips else 1,
            "offset_deg": round(urdf_mid, 4),
            # How far the physical range (lerobot) differs from the URDF range.
            # Large => the mechanical stop is not the URDF limit, so offset from
            # the midpoint carries up to ~half this error. Flagged as low confidence.
            "span_residual_deg": round(lero_half_span - urdf_half_span, 3) * 2,
        }
    # Gripper: lerobot reports it 0..100 (RANGE_0_100), not degrees. Map linearly
    # onto the URDF gripper range. Lower safety stakes; kept for completeness.
    glo, ghi = JOINT_LIMITS_DEG[GRIPPER_JOINT]
    joints[GRIPPER_JOINT] = {
        "type": "range_0_100_to_deg",
        "urdf_min_deg": glo,
        "urdf_max_deg": ghi,
    }
    return {
        "convention": CONVENTION,
        "robot_id": robot_id,
        "arm_joints": list(ARM_JOINTS),
        "gripper_joint": GRIPPER_JOINT,
        "joints": joints,
        "note": "signs are PROVISIONAL until confirmed with `verify`; see script docstring",
    }


def _apply(mapping: dict, name: str, lero_value: float) -> float:
    j = mapping["joints"][name]
    if j["type"] == "degrees_affine":
        return j["sign"] * lero_value + j["offset_deg"]
    # gripper 0..100 -> deg
    frac = np.clip(lero_value, 0.0, 100.0) / 100.0
    return j["urdf_min_deg"] + frac * (j["urdf_max_deg"] - j["urdf_min_deg"])


def cmd_draft(args) -> int:
    flips = {s.strip() for s in (args.flip or "").split(",") if s.strip()}
    unknown = flips - set(ARM_JOINTS)
    if unknown:
        raise SystemExit(f"--flip: unknown joint(s) {unknown}; valid: {ARM_JOINTS}")
    mapping = _draft(args.id, flips)
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"wrote {MAP_PATH}")
    print(f"convention: {CONVENTION}")
    print(f"{'joint':14} {'sign':>4} {'offset_deg':>11} {'span_resid':>11}  {'URDF range':>18}")
    for name in ARM_JOINTS:
        j = mapping["joints"][name]
        lo, hi = JOINT_LIMITS_DEG[name]
        flag = "  <- LOW CONFIDENCE (stop != URDF limit)" if abs(j["span_residual_deg"]) > 6 else ""
        print(
            f"{name:14} {j['sign']:>4} {j['offset_deg']:>11.3f} "
            f"{j['span_residual_deg']:>11.2f}  [{lo:7.2f},{hi:7.2f}]{flag}"
        )
    if flips:
        print(f"flipped sign for: {sorted(flips)}")
    print("\nSigns above are PROVISIONAL (+1). Capture the real signs by nudging each")
    print("joint (torque OFF), which needs the arm plugged in:")
    print(f"  PYTHONPATH=src {sys.executable} {Path(__file__).name} signs")
    return 0


def _read_lerobot_degrees(port: str, robot_id: str):
    """Open the bus read-only and return (arm_degs dict, gripper_0_100). No motion."""
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    robot = SOFollower(SOFollowerRobotConfig(port=port, id=robot_id))
    bus = robot.bus
    bus.connect()
    try:
        deg = bus.sync_read("Present_Position", normalize=True)
    finally:
        bus.disconnect(disable_torque=False)  # read-only; preserve torque state
    return deg


def _stop_specs() -> dict[str, dict]:
    """Per joint, an unambiguous fact about ITS OWN driven link at the two stops.

    Each question is about the segment the joint directly drives -- the upper arm
    for shoulder_lift, the forearm for elbow_flex, the tool for wrist_flex -- never
    the gripper's world height. A driven segment's orientation is a monotonic
    function of its own joint, so it is not confounded by the others (the gripper's
    height is, and even doubles back over shoulder_lift's range). Answers are
    computed here from the per-link FK frames so the tool knows the URDF side.
    """
    from so_snake.kinematics import ArmChain, rotation_log
    from so_snake.config import TeleopConfig

    chain = ArmChain()
    home = np.array(TeleopConfig().home_joints_deg, dtype=float)

    def frames_at(i: int, val: float) -> dict:
        q = home.copy()
        q[i] = val
        return chain.fk_all(q)

    def seg(frames: dict, a: str, b: str) -> np.ndarray:
        v = frames[b][:3, 3] - frames[a][:3, 3]
        return v / (np.linalg.norm(v) + 1e-12)

    # Per joint: the driven-link feature (higher value = the described state) and
    # the plain-language description of that state.
    def feature(name: str, f: dict) -> float:
        if name == "shoulder_pan":
            return seg(f, "upper_arm", "lower_arm")[1]  # arm swung to the left (+y)
        if name == "shoulder_lift":
            return seg(f, "upper_arm", "lower_arm")[0]  # upper arm toward front (+x)
        if name == "elbow_flex":
            return float(np.dot(seg(f, "upper_arm", "lower_arm"), seg(f, "lower_arm", "wrist")))
        if name == "wrist_flex":
            return seg(f, "wrist", "gripper_frame_link")[2]  # tool tilted up (+z), arm at home
        raise AssertionError(name)

    questions = {
        "shoulder_pan": "the whole arm swung to YOUR LEFT",
        "shoulder_lift": "the UPPER ARM (big segment) pointing toward the FRONT of the base (vs behind it)",
        "elbow_flex": "the forearm roughly STRAIGHT, in line with the upper arm (vs folded back against it)",
        "wrist_flex": "the gripper/tool cocked UPWARD relative to the forearm (vs downward)",
    }

    specs: dict[str, dict] = {}
    for i, name in enumerate(ARM_JOINTS):
        lo, hi = JOINT_LIMITS_DEG[name]
        if name == "wrist_roll":
            # Continuous joint: no hard stop. Determine which visual rotation
            # sense is +URDF from the axis direction (config-independent).
            t0 = chain.fk(home.copy())
            q = home.copy()
            q[i] += 5.0
            axis = rotation_log(chain.fk(q)[:3, :3] @ t0[:3, :3].T)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            toolward = float(np.dot(axis, t0[:3, 2]))
            specs[name] = {"type": "roll", "cw_is_plus": toolward < 0.0}
            continue
        fl, fh = frames_at(i, lo), frames_at(i, hi)
        specs[name] = {
            "type": "stop",
            "question": questions[name],
            # "described state" is true at the higher-feature stop; is that the URDF upper limit?
            "cond_is_upper": feature(name, fh) > feature(name, fl),
            "urdf_span": hi - lo,
        }
    return specs


def _ask_int(prompt: str, choices: tuple[int, ...]) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) in choices:
            return int(raw)
        print(f"    please type one of {choices}.")


def cmd_signs(args) -> int:
    """Read-only sign capture by hard stops: no direction to judge, no motion commanded."""
    if not MAP_PATH.is_file():
        raise SystemExit(f"no map at {MAP_PATH}; run `draft` first")
    mapping = json.loads(MAP_PATH.read_text())
    specs = _stop_specs()

    print("READ-ONLY sign capture. The arm is limp (torque OFF); nothing is powered.")
    print("You move each joint by hand to its HARD STOPS (until it won't turn) --")
    print("there is no direction to judge -- then answer one question about that")
    print("joint's OWN segment (upper arm / forearm / tool), not the gripper's height.")
    print("Keep the OTHER joints roughly at the home/middle pose while you do each one.\n")
    results: dict[str, dict] = {}
    try:
        for name in ARM_JOINTS:
            spec = specs[name]
            if spec["type"] == "roll":
                input(f"[{name}] Point the gripper toward your FACE, then ENTER for the before reading...")
                before = _read_lerobot_degrees(args.port, args.id)[name]
                while True:
                    input("    Spin the wrist-roll CLOCKWISE (as it faces you) ~60 deg, then ENTER...")
                    after = _read_lerobot_degrees(args.port, args.id)[name]
                    d = after - before
                    if abs(d) < 15.0:
                        print(f"    only {d:+.1f} deg -- spin it further and retry.")
                        continue
                    motion_urdf = 1 if spec["cw_is_plus"] else -1  # CW == +URDF?
                    sign = motion_urdf * (1 if d > 0 else -1)
                    results[name] = {"sign": sign, "note": f"roll dlero={d:+.1f}"}
                    print(f"    {name}: sign {sign:+d}\n")
                    break
                continue

            # Hinge joint: capture both hard stops, then a comparison question.
            while True:
                input(f"[{name}] Move {name} to ONE stop (until it won't turn), then ENTER...")
                a = _read_lerobot_degrees(args.port, args.id)[name]
                input(f"    Now move {name} to the OTHER stop, then ENTER...")
                b = _read_lerobot_degrees(args.port, args.id)[name]
                span = abs(b - a)
                if span < 0.5 * spec["urdf_span"]:
                    print(f"    stops only {span:.0f} deg apart (expected ~{spec['urdf_span']:.0f}); "
                          "make sure you hit both hard stops. Retrying.")
                    continue
                which = _ask_int(f"    At which stop was {spec['question']}?  [1=first / 2=second]: ", (1, 2))
                lero_cond, lero_other = (a, b) if which == 1 else (b, a)
                if spec["cond_is_upper"]:
                    lero_upper, lero_lower = lero_cond, lero_other
                else:
                    lero_lower, lero_upper = lero_cond, lero_other
                sign = 1 if lero_upper > lero_lower else -1
                resid = span - spec["urdf_span"]
                results[name] = {"sign": sign, "note": f"span={span:.0f} (URDF {spec['urdf_span']:.0f}, resid {resid:+.0f})"}
                print(f"    {name}: sign {sign:+d}  ({results[name]['note']})\n")
                break
    except KeyboardInterrupt:
        print("\naborted; map not updated.")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for name, r in results.items():
        mapping["joints"][name]["sign"] = r["sign"]
    mapping["note"] = "signs captured via hard-stop comparison; verify with `check`"
    MAP_PATH.write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"updated signs in {MAP_PATH}:")
    for name in ARM_JOINTS:
        print(f"  {name:14} sign {mapping['joints'][name]['sign']:+d}   {results[name]['note']}")
    print(f"\nNow confirm: {Path(__file__).name} check --port {args.port}")
    return 0


def cmd_check(args) -> int:
    """Passive read-only live table: apply the map, show URDF angles + FK TCP."""
    if not MAP_PATH.is_file():
        raise SystemExit(f"no map at {MAP_PATH}; run `draft` first")
    mapping = json.loads(MAP_PATH.read_text())
    from so_snake.kinematics import ArmChain

    chain = ArmChain()
    lo_arr = np.array([JOINT_LIMITS_DEG[j][0] for j in ARM_JOINTS])
    hi_arr = np.array([JOINT_LIMITS_DEG[j][1] for j in ARM_JOINTS])

    print("READ-ONLY check. Arm stays limp (torque OFF); no motion is commanded.")
    print("Hand-move the arm; the mapped URDF angles should stay in [lo,hi] and the")
    print("FK TCP should match where the gripper actually is. Ctrl-C to stop.\n")
    ever_out = np.zeros(len(ARM_JOINTS), dtype=bool)
    try:
        while True:
            deg = _read_lerobot_degrees(args.port, args.id)
            urdf = np.array([_apply(mapping, j, deg[j]) for j in ARM_JOINTS])
            in_range = (urdf >= lo_arr - 1.0) & (urdf <= hi_arr + 1.0)
            ever_out |= ~in_range
            tcp = chain.fk(urdf)[:3, 3]
            print("\x1b[2J\x1b[H", end="")
            print(f"{mapping['convention']}   (Ctrl-C to stop)")
            print(f"{'joint':14} {'lero':>8} {'sign':>4} {'->urdf':>8} {'[lo,hi]':>18} {'in?':>4}")
            for i, name in enumerate(ARM_JOINTS):
                s = mapping["joints"][name]["sign"]
                print(
                    f"{name:14} {deg[name]:8.2f} {s:>+4d} {urdf[i]:8.2f} "
                    f"[{lo_arr[i]:7.1f},{hi_arr[i]:6.1f}] {'OK' if in_range[i] else 'OUT':>4}"
                )
            print(f"\nFK TCP (m): x={tcp[0]:+.3f} y={tcp[1]:+.3f} z={tcp[2]:+.3f}")
            print(f"gripper: lero {deg[GRIPPER_JOINT]:.1f} -> {_apply(mapping, GRIPPER_JOINT, deg[GRIPPER_JOINT]):.1f} deg")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("\nsummary:")
    for i, name in enumerate(ARM_JOINTS):
        note = "  <- left range during sweep; check offset (elbow physical range > URDF)" if ever_out[i] else ""
        print(f"  {name:14} sign {mapping['joints'][name]['sign']:+d}{note}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_draft = sub.add_parser("draft", help="write the map from the calibration file (no hardware)")
    p_draft.add_argument("--id", default="so_snake", help="lerobot robot id")
    p_draft.add_argument("--flip", default="", help="comma-separated joints to flip sign, e.g. elbow_flex,wrist_flex")
    p_draft.set_defaults(func=cmd_draft)

    p_signs = sub.add_parser("signs", help="guided read-only sign capture; nudge each joint as prompted")
    p_signs.add_argument("--port", default="", help="arm serial port; auto-detected when omitted")
    p_signs.add_argument("--id", default="so_snake", help="lerobot robot id")
    p_signs.set_defaults(func=cmd_signs)

    p_check = sub.add_parser("check", help="passive read-only live table; hand-move to confirm the map")
    p_check.add_argument("--port", default="", help="arm serial port; auto-detected when omitted")
    p_check.add_argument("--id", default="so_snake", help="lerobot robot id")
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    if hasattr(args, "port"):
        # Every hardware subcommand takes the same port, so it is resolved once
        # here rather than in each of them. `draft` has no --port and no bus.
        try:
            args.port = detect_arm_port(args.port)
        except DeviceDetectionError as exc:
            print(f"{exc}", file=sys.stderr)
            return 2
        print(f"arm port: {args.port}\n")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
