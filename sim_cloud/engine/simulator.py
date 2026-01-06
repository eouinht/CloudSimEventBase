import heapq
from log.logger import SimLogger

logger = SimLogger.get_logger("SIMULATOR")
class Simulator:
    """
        manage event queue
        update sim time
        scheduler
        call dispatcher
    """
    def __init__(self, state, dispatcher, scheduler = None):
        self.state = state
        self.dispatcher = dispatcher
        self.scheduler = scheduler
        
        # Min-heap: (time, seg, event)
        self.event_queue = []
        self.seq = 0
        
    def add_event(self, event):
        heapq.heappush(
            self.event_queue,
            (event.time, self.seq, event)
        )
        self.seq += 1
        
    def run(self):
        
        logger.info("Simulation started")

        
        while self.event_queue:
            
            time, _, _ = self.event_queue[0]
            self.state.set_time(time)        
            logger.info(f" ========================= Time = {time} ==========================")
          
            # ===== PHASE 1: lấy toàn bộ event cùng time =====
            same_time_events = []
            while self.event_queue and self.event_queue[0][0] == time:
                _, _, event = heapq.heappop(self.event_queue)
                same_time_events.append(event)
            
            # ===== PHASE 2: dispatch event =====    
            for event in same_time_events:
                self.dispatcher.dispatch(event)
                
            # ===== PHASE 3: scheduling (chỉ 1 lần) =====   
            if self.scheduler:
                new_event = self.scheduler.on_time_step(self.state) 
                for e in new_event:
                    self.add_event(e)
            
        logger.info("Simulation finished")  