#!/usr/bin/env python
"""Serve the so-snake GUI: teleoperation, episode recording, and replay in a browser.

Front and back end both run on this host. The gateway serves the built frontend
itself, so one process is the whole thing:

    cd tools/gui/frontend && npm install && npm run build
    PYTHONPATH=src python scripts/serve_gui.py

While developing the frontend, run Vite instead and let it proxy `/api` here:

    PYTHONPATH=src python scripts/serve_gui.py &
    cd tools/gui/frontend && npm run dev

Binds to localhost by default. `--host 0.0.0.0` exposes it to the network, which
also exposes a button that moves a real robot arm -- only do that on a network
you control, and only when you can see the arm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from so_snake.data import DEFAULT_EPISODE_ROOT
from so_snake.gui.server import DEFAULT_FRONTEND_DIST, serve


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--frontend-dist", type=Path, default=DEFAULT_FRONTEND_DIST)
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: serving on {args.host} — anyone who can reach this host can "
              "command the arm.", file=sys.stderr)

    serve(
        args.host,
        args.port,
        episode_root=args.episode_root,
        frontend_dist=args.frontend_dist,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
