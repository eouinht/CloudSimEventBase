
from log.logger import SimLogger

logger = SimLogger.get_logger("STATE")

class SimulationState:
    # current_time
    # pms: Dict[pm_id, PM]
    # running_vms: Dict[vm_id, VM]
    # finished_vms
    def __init__(self):
        self.current_time=0
        self.pms={}         # pm_id -> pm
        self.running_vms={} # vm_id -> VM(already placed)
        self.finished_vms={}# vm_id -> VM(waiting for placement)
        self.pending_vms={} # vm_id -> VM(done -> wait to remove)
        
    def set_time(self, t: int):
        self.current_time = t        
        logger.debug(f"Advance simulation time to t={t}")
        
    def add_pm(self, pm):
        if pm.id in self.pms:
            logger.error(f"PM {pm.id} already exists")
            raise ValueError(f"Duplicate PM id {pm.id}")
        self.pms[pm.id] = pm
        logger.info(f"PM added: {pm.id}")
        
    def get_pm(self, pm_id):
        return self.pms.get(pm_id, None)
    
        
    def add_pending_vm(self, vm):        
        
        if vm.id in self.pending_vms:
            raise ValueError(f"VM {vm.id} already exists in pend list")
        if vm.id in self.running_vms:
            raise ValueError(f"VM {vm.id} already exists in run list")
        
        self.pending_vms[vm.id] = vm
        
        
    def place_vm(self, vm, pm_id):
        
        
        vm_check = vm.id
        print(f"VM-CHECK {vm_check}") 
        
        if vm_check not in self.pending_vms:
            raise ValueError(f"Vm {vm_check} does not exist")
        
        pm = self.get_pm(pm_id)
        
        if pm is None:
            raise ValueError(f"PM {pm_id} does not exist")
        
        
        vm.start_time = self.current_time
        vm.end_time = self.current_time + vm.duration
        vm.pm_id = pm_id
        
        pm.add_vm(vm)
        print("herrrrrreeeeee")
        
        del self.pending_vms[vm_check]
        
        self.running_vms[vm.id] = vm
        
        logger.info(f"VM {vm.id} placed on PM {pm_id}")
    # def start_vm(self, vm):
    #     if vm.id in self.running_vms:
    #         return
    #     self.pending_vms.remove(vm)
    #     self.running_vms[vm.id] = vm    
                    
    # def add_vm(self, vm):
    #     if vm.id in self.running_vms:
    #         logger.error(f"VM {vm.id} already running")
    #         raise ValueError(f"Duplicate VM id {vm.id}")

    #     self.running_vms[vm.id] = vm
    #     logger.info(f"VM registered: {vm.id}")
        
    def finish_vm(self, vm_id):
        vm = self.running_vms.pop(vm_id, None)
        if vm is None:
            logger.warning(f"Finish VM failed: {vm_id} not found")
            return

        self.finished_vms[vm_id] = vm
        logger.info(f"VM finished: {vm_id}")
    
    def snapshot(self):
        """
        Lightweight snapshot for logging / debugging.
        """
        return {
            "time": self.current_time,
            "num_pms": len(self.pms),
            "num_running_vms": len(self.running_vms),
            "num_finished_vms": len(self.finished_vms),
        }