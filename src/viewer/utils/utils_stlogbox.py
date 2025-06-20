from datetime import datetime

class StreamlitJobLogger:
    def __init__(self, committed_box, live_box):
        self.committed_box = committed_box
        self.live_box = live_box
        self.committed_lines = []
        self.live_buffer = ""

    def _timestamp(self):
        return datetime.now().strftime("%H:%M:%S")

    def commit(self, text: str):
        """Commit the full text (e.g. job output) to permanent log."""
        for line in text.strip().splitlines():
            self.committed_lines.append(f"[{self._timestamp()}] {line}")
        self._refresh_committed()

    def info(self, msg: str):
        """Quick single-line commit."""
        self.committed_lines.append(f"[{self._timestamp()}] INFO: {msg}")
        self._refresh_committed()

    def error(self, msg: str):
        self.committed_lines.append(f"[{self._timestamp()}] ERROR: {msg}")
        self._refresh_committed()

    def update_live(self, live_text: str):
        """Update live log for current job polling result."""
        self.live_buffer = live_text
        self.live_box.code(self.live_buffer, language="log")

    def clear_live(self):
        self.live_buffer = ""
        self.live_box.empty()

    def _refresh_committed(self):
        self.committed_box.code("\n".join(self.committed_lines), language="log")

