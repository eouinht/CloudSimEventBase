from log.logger import SimLogger

logger = SimLogger.get_logger("VM")

class VM:
    def __init__(self, vm_id, created_at, memory, duration, vm_util):
        self.id = vm_id
        self.created_at = created_at
        self.memory = memory
        self.duration = duration
        self.vm_util = vm_util
        
        self.pm_id = None
        self.start_time = created_at
        self.end_time = created_at + duration

    def is_active(self, t):
        active = self.created_at <= t < (self.created_at + self.duration)
        # logger.debug(f"VM {self.id} active at t={t}: {active}")
        return active

    def cpu_demand(self, t):
        idx = t - self.start_time
        if idx < 0:
            return 0.0

        if idx >= len(self.vm_util):
            logger.warning(
                f"VM {self.id} vm_util length < duration "
                f"(idx={idx}, len={len(self.vm_util)})"
            )
            return 0.0

        return self.vm_util[idx]