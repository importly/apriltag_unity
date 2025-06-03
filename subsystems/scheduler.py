class Subsystem:
    """Base class for robot subsystems."""

    def periodic(self):
        """Called periodically by the scheduler."""
        pass

    def close(self):
        """Optional cleanup when shutting down."""
        pass


class CommandScheduler:
    """Very small scheduler that calls each subsystem's periodic method."""

    def __init__(self):
        self._subsystems = []
        self._running = False

    def register(self, subsystem: Subsystem):
        self._subsystems.append(subsystem)

    def run(self):
        for subsystem in list(self._subsystems):
            subsystem.periodic()

    def shutdown(self):
        for subsystem in list(self._subsystems):
            subsystem.close()
        self._subsystems.clear()
