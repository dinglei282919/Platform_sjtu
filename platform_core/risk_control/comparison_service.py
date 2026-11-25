"""Registry shared by the independently owned CSTR comparison sessions."""
from .cstr import SAMPLE_MIN
from .session_lifecycle import SessionRegistry


def sample_bands(rows):
    """Half-open exposed intervals in the supplied sampling window, in minutes."""
    bands = []
    previous = None
    for row in rows:
        if row["phase"] == "active":
            end = round((row["seq"] + 1) * SAMPLE_MIN, 10)
            if previous is not None and previous["phase"] == "active" and previous.get("segment") == row.get("segment"):
                bands[-1][1] = end
            else:
                bands.append([row["t"], end])
        previous = row
    return bands


class ComparisonService(SessionRegistry):
    missing_session = "对照会话不存在或已释放"

    def create(self, settings=None, *, lease=False, auto_start=True):
        config = self.config_type.parse(settings)
        return self._create_session(self.session_type, config, lease=lease, auto_start=auto_start)
