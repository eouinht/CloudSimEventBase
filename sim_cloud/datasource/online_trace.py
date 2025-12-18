import json
import uuid
from log.logger import SimLogger

from .base import TraceAdapter
from core.vm import VM
from events.vm_events import CreateVMEvent, FinishVMEvent, EmbededVMEvent
import os

logger = SimLogger.get_logger("ONLINE_TRACE")

def parse_num_pms(filename: str)-> int:
    name = os.path.basename(filename)
    x = name.split("_")[0]
    return int(x)

class OnlineTraceAdapter(TraceAdapter):
    """
    Adapter for VM Online Scheduling dataset.
    Dataset → Event stream
    """

    def __init__(self, trace_file: str):
        self.trace_file = trace_file
        self.num_pms = parse_num_pms(trace_file)
    
    def load_events(self):    
        
        logger.info(f"Loaded online trace from {self.trace_file}")
        logger.info(f"Number of PMs = {self.num_pms}")
        
        with open(self.trace_file, "r") as f:
            data = json.load(f)
            
        events = []
        vm_counter = 0

        for rec in data:
            vm_id = f"vm_{vm_counter}"
            vm_counter += 1
            created_at = rec["created_at_point"]
            memory = rec["memory"]
            duration = rec["duration_point"]
            vm_util = rec["vm_util"]
            
            vm = VM(
                vm_id=vm_id,
                created_at=created_at,
                memory=memory,
                duration=duration,
                vm_util=vm_util)
            
            # Creat event
            events.append(
                CreateVMEvent(
                    time=created_at,
                    vm=vm
                )
            )
            
            # Finish event
            events.append(
                FinishVMEvent(
                    time=created_at + duration,
                    vm_id=vm_id, 
                )
            )
        logger.info(f"Generated {len(events)} events from trace")
        return events
    def _create_vm(self, record: dict) -> VM:
        """
        Convert dataset record → VM object
        """

        vm = VM(
            vm_id=str(uuid.uuid4()),
            memory=record["memory"],
            util_trace=record["vm_util"],
            start_time=record["created_at_point"],
            duration=record["duration_point"]
        )

        return vm
