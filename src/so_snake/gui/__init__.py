"""The local web GUI: a JSON API over the teleoperation, recording and replay path.

`session` owns the arm and enforces that one thing drives it at a time, `server`
exposes that over HTTP and serves the built frontend, `roadmap` is the status
board, and `preview` turns MuJoCo frames into PNGs without an image library.

Launch it with `scripts/serve_gui.py`. The frontend source is in
`tools/gui/frontend`.
"""

from .roadmap import roadmap_payload
from .server import Gateway, GuiServer, serve
from .session import SessionManager

__all__ = ["Gateway", "GuiServer", "SessionManager", "roadmap_payload", "serve"]
