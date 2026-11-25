"""Session ownership and lease rules shared by the four live CSTR simulations."""
from threading import RLock


def lease_action(leased, connections, idle_seconds, status):
    """Only disconnected, leased sessions auto-pause or expire."""
    if leased and connections == 0:
        if idle_seconds > 120:
            return "expire"
        if idle_seconds > 20 and status == "running":
            return "pause"
    return None


class SessionRegistry:
    """Own the live sessions and serialize registration and ownership checks."""
    missing_session = "会话不存在或已释放"

    def __init__(self, root):
        self.root = root
        self.sessions = {}
        self.lock = RLock()

    def _create_session(self, session_type, *args, **kwargs):
        with self.lock:
            self.sessions = {key: session for key, session in self.sessions.items() if not session.closed}
            session = session_type(self.root, *args, **kwargs)
            self.sessions[session.id] = session
            return session

    def session(self, identifier, owner=None, control=False):
        with self.lock:
            session = self.sessions.get(identifier)
            if session is None or session.closed:
                raise KeyError(self.missing_session)
            if control and owner != session.owner:
                raise PermissionError("此会话处于只读模式")
            return session

    def release(self, session):
        with self.lock:
            self.sessions.pop(session.id, None)
        session.close()

    def close(self):
        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for session in sessions:
            session.close()
