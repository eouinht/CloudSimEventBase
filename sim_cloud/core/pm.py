
from log.logger import SimLogger

logger = SimLogger.get_logger("PM")

class PM:
    def __init__(self, pm_id, cpu_cap, mem_cap):
        self.id = pm_id
        self.cpu_cap = cpu_cap
        self.mem_cap = mem_cap
        self.vms = []
        logger.info(
            f"Init PM {self.id} | CPU_CAP={self.cpu_cap} | MEM_CAP={self.mem_cap}"
        )
        
    def can_host(self, vm, t):
        # 1. Tính CPU và Memory đang sử dụng tại thời điểm t
        cpu_used = 0.0
        mem_used = 0.0
        
        for running_vm in self.vms:
            # print(f" isactive {running_vm.is_active(t)}")
            if running_vm.is_active(t):
                cpu_used += running_vm.cpu_demand(t)
                mem_used += running_vm.memory

        # 2. Nhu cầu tài nguyên của VM mới tại thời điểm t
        vm_cpu = vm.cpu_demand(t)
        vm_mem = vm.memory

        total_cpu_after = cpu_used + vm_cpu
        
        # print(f"=================== cpu_used: {cpu_used} ===================")
        
        if total_cpu_after > self.cpu_cap: 
            # logger.warning(
            #     f"[REJECT] PM {self.id} CPU overload | "
            #     f"need={total_cpu_after:.3f}, cap={self.cpu_cap}"
            # )
            return False
        
        total_mem_after = mem_used + vm_mem
        if total_mem_after > self.mem_cap:
            # logger.warning(
            #     f"[REJECT] PM {self.id} MEM overload | "
            #     f"need={total_mem_after:.3f}, cap={self.mem_cap}"
            # )
            return False
        
        logger.debug(
            f"[ACCEPTABLE] PM {self.id} can host VM {vm.id} at t={t}"
        )
        return True
    
    def add_vm(self, vm):
        
        print ("AAAAAAAAAAAAAAAAAAAAAAdd successfullyyyyyyyyyyyyyyyyy")
        vm.pm_id = self.id
        self.vms.append(vm)
        

    def remove_finished_vms(self, t):
        active_vms = []

        for vm in self.vms:
            if vm.is_active(t):
                active_vms.append(vm)
            else:
                logger.info(
                    f"[FINISH] VM {vm.id} removed from PM {self.id} at t={t}"
                )
        
        self.vms = active_vms
