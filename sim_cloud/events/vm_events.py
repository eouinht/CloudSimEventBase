from abc import ABC, abstractmethod
from log.logger import SimLogger

logger = SimLogger.get_logger("VM_EVENT")


class VMEvent(ABC):
    """
    Base class for all VM-related events.
    """
    def __init__(self, time: int):
        self.time = time

    @abstractmethod
    def apply(self, state):
        """
        Apply event to simulation state via dispatcher logic.
        """
        pass

class CreateVMEvent(VMEvent):
    def __init__(self, time: int, vm):
        super().__init__(time)
        self.vm = vm
        
    def apply(self, state):
        vm = self.vm
        vm_id = vm.id
        if vm_id not in state.pending_vms:
            state.add_pending_vm(self.vm) 
            
        # logger.info(
        #     f"VM {self.vm.id} created (pending) at t={self.time}"
        # )
        return True
    
class EmbededVMEvent(VMEvent):
    def __init__(self, time: int, vm, pm_id: str):
        super().__init__(time)
        self.vm = vm 
        self.pm_id = pm_id
        
    def apply(self, state):
        vm = self.vm
        vm_id = vm.id
        
        if vm_id not in state.pending_vms:
            logger.debug(
                f"Embed ignored: VM {vm_id} is no longer pending"
            )
            return False
   
        
        
        pm = state.get_pm(self.pm_id)
        
        if pm is None:
            logger.error(
                f"Embed failed: PM {self.pm_id} does not exist"
            )
            return False
        # if not pm.can_host(self.vm, self.time):
        #     return False        
        state.place_vm(self.vm, self.pm_id)
        return True
               
class FinishVMEvent(VMEvent):
    def __init__(self, time: int, vm_id: str):
        super().__init__(time)
        self.vm_id = vm_id

    def apply(self, state):
        # Case 1: Vm in pending_vms
        # vm = state.pending_vms.pop(self.vm_id, None)
        # if vm is not None:
        #     state.finished_vms[self.vm_id] = vm
        #     logger.info(
        #         f"VM {self.vm_id} finished while pending at t={self.time}"
        #     )
        #     return True
        
        # Case: Vm in running_vms
        
        if self.vm_id in state.running_vms:
            vm = state.running_vms[self.vm_id]
            pm = state.get_pm(vm.pm_id)
            if pm is not None:
                pm.remove_finished_vms(self.time)
            state.finish_vm(self.vm_id)
            # logger.info(
            #     f"VM {self.vm_id} finished (running → finished) at t={self.time}"
            # )
            return True

        logger.warning(
            f"Finish VM ignored: VM {self.vm_id} not found in any state"
        )
        return False

class MigrateVMEvent(VMEvent):
    def __init__(self, time: int, vm_id: str, target_pm_id: str):
        super().__init__(time)
        self.vm_id = vm_id
        self.target_pm_id = target_pm_id

    def apply(self, state):
        vm = state.running_vms.get(self.vm_id)
        if vm is None:
            logger.warning(f"MigrateVM failed: VM {self.vm_id} not found")
            return False

        src_pm = state.get_pm(vm.pm_id)
        dst_pm = state.get_pm(self.target_pm_id)

        if dst_pm is None:
            logger.error(f"MigrateVM failed: target PM {self.target_pm_id} not found")
            return False

        if not dst_pm.can_host(vm):
            logger.warning(
                f"MigrateVM rejected: PM {self.target_pm_id} insufficient resource for VM {vm.id}"
            )
            return False

        if src_pm is not None:
            src_pm.remove_vm(vm)

        dst_pm.add_vm(vm)
        vm.pm_id = self.target_pm_id

        # logger.info(
        #     f"VM {vm.id} migrated {src_pm.id if src_pm else 'None'} -> "
        #     f"{self.target_pm_id} at t={self.time}"
        # )
        return True
