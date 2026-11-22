"""Registry shared by the independently owned CSTR comparison sessions."""
from threading import RLock


class ComparisonService:
    def __init__(self, root):
        self.root, self.sessions, self.lock = root, {}, RLock()

    def create(self, settings=None, *, lease=False, auto_start=True):
        config = self.config_type.parse(settings)
        with self.lock:
            self.sessions = {k: s for k, s in self.sessions.items() if not s.closed}
            session = self.session_type(self.root, config, lease=lease, auto_start=auto_start)
            self.sessions[session.id] = session
        return session

    def session(self, identifier, owner=None, control=False):
        with self.lock:
            s = self.sessions.get(identifier)
            if s is None or s.closed:
                raise KeyError("对照会话不存在或已释放")
            if control and s.owner != owner:
                raise PermissionError("此会话处于只读模式")
            return s

    def release(self, session):
        with self.lock:
            self.sessions.pop(session.id, None)
        session.close()

    def close(self):
        with self.lock:
            current = list(self.sessions.values())
            self.sessions.clear()
        for s in current:
            s.close()
