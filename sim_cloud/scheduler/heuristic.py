from log.logger import SimLogger
from scheduler.base import BaseScheduler
from events.vm_events import MigrateVMEvent, EmbededVMEvent
import random

logger = SimLogger.get_logger("SCHEDULER")

class HeuristicScheduler(BaseScheduler):
    def __init__(self,
                 cpu_threshold = 0.8,
                 min_interval = 1
                 ):
        
        """_summary_

        Args:
            cpu_threshold (float, optional): hotspot threshold (0–1). Defaults to 0.8.
            min_interval (int, optional): min time between scheduling decisions. Defaults to 1.
        """
        
        self.cpu_threshold = cpu_threshold
        self.min_interval = min_interval
        self.last_decision_time = -1
        
        
    def on_time_step(self, state):
        t =  state.current_time
        feq = t - self.last_decision_time
        if feq < self.min_interval:
            return []

        events = []
        overloaded_pms = []
        underloaded_pms = []
        
        for pm in state.pms.values():
            cpu_used = 0.0
            for vm in pm.vms:
                if vm.is_active(t):
                    cpu_used += vm.cpu_demand(t)
            
            cpu_ratio = cpu_used / pm.cpu_cap if pm.cpu_cap > 0 else 0
            if cpu_ratio >= self.cpu_threshold:
                overloaded_pms.append((pm, cpu_ratio))
            else:
                underloaded_pms.append((pm, cpu_ratio))
        
        if not overloaded_pms:
            logger.debug("No overloaded PMs at this timestep")
            return
              
        src_pm, _ = max(overloaded_pms, key=lambda x: x[1])
        

        dst_pm, _ = min(underloaded_pms, key=lambda x: x[1])
        active_vms = [v for v in src_pm.vms if v.is_active(t)]
        if not active_vms:
            return []
        vm = max(active_vms, key=lambda v: v.cpu_demand(t))
        
        event = MigrateVMEvent(
            time=t,
            vm_id=vm.id,
            target_pm_id=dst_pm.id
        )
        logger.info(
            f"[Heuristic] migrate VM {vm.id} "
            f"{src_pm.id} -> {dst_pm.id} at t={t}"
        )
        events.append(event)
        return events

class FCFSScheduler(BaseScheduler):
    """
    First-Come-First-Serve Scheduler

    - Duyệt VM theo thứ tự đến (pending_vms)
    - Với mỗi VM, duyệt PM theo thứ tự ID
    - PM nào chứa được thì embed
    """
    def __init__(self):
        super().__init__()
        
    def on_time_step(self, state):
        t = state.current_time

        if not state.pending_vms:
            return []

        min_time = min(vm.created_at for vm in state.pending_vms.values())
        
        batch = [
            vm for vm in state.pending_vms.values()
            if vm.created_at == min_time
        ]
        
        pms = list(state.pms.values())
        random.shuffle(pms)
        
        events = []
        
        
        for vm in batch:
            placed = False

            # duyệt PM theo thứ tự
            for pm in pms:
                
                if pm.can_host(vm, t):
                    
                    logger.info(
                        f"FCFS embed VM {vm.id} -> PM {pm.id} at t={t}"
                    )

                    events.append( 
                        EmbededVMEvent(
                            time=t,
                            vm=vm,
                            pm_id=pm.id
                        )
                    )
                    
                    placed = True
                    break  # xong VM này

            # FCFS nghiêm ngặt: VM đầu không đặt được thì dừng
            if not placed:
                logger.debug(
                    f"FCFS: VM {vm.id} cannot be placed at t={t}"
                )
                break
            
        return events