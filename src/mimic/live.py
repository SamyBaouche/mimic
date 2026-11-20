"""Live view: Chrome's built-in screencast (CDP) streams JPEG frames of the agent's page.

The browser stays headless (no window to cover, no focus issues); the dashboard shows the
frames. Chrome only sends a frame when the page changes, so an idle page costs nothing.
"""

from __future__ import annotations

from playwright.sync_api import Page


class Screencast:
    def __init__(self, page: Page, publish, max_width: int = 1100, max_height: int = 760):
        self.page = page
        self.publish = publish
        self.max_width = max_width
        self.max_height = max_height
        self.frames = 0
        self._cdp = None

    def start(self) -> Screencast:
        try:
            self._cdp = self.page.context.new_cdp_session(self.page)
        except Exception:  # noqa: BLE001 - not Chromium: no live view, the run still works
            return self
        self._cdp.on("Page.screencastFrame", self._on_frame)
        self._cdp.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": 60,
                "maxWidth": self.max_width,
                "maxHeight": self.max_height,
                "everyNthFrame": 1,
            },
        )
        return self

    def _on_frame(self, params: dict) -> None:
        self.frames += 1
        self.publish({"type": "frame", "data": params["data"]})
        try:
            self._cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
        except Exception:  # noqa: BLE001 - page closing
            pass

    def stop(self) -> None:
        if self._cdp is not None:
            try:
                self._cdp.send("Page.stopScreencast")
            except Exception:  # noqa: BLE001
                pass
