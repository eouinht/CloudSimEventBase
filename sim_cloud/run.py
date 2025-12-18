
import json
import glob
from core.state import SimulationState
from core.pm import PM
from engine.simulator import Simulator
from engine.dispatcher import Dispatcher
from scheduler.heuristic import HeuristicScheduler, FCFSScheduler
from datasource.online_trace import OnlineTraceAdapter
from log.logger import SimLogger

logger = SimLogger.get_logger("RUN")

DATASET_DIR = "datasource/"
PM_CPU_CAP = 128
PM_MEM_CAP = 2400


# Load dataset
json_files = sorted(glob.glob(f"{DATASET_DIR}/*.json"))
if not json_files:
    raise RuntimeError("No dataset files found")
dataset_file = json_files[0]   # chạy 1 file trước
logger.info(f"Using dataset: {dataset_file}")

# Init state
state = SimulationState()

# X-1.json -> state
def init_pms(filename):
    basename = filename.split("/")[-1]
    x = int(basename.split("_")[0])
    
    logger.info(f"Initializing {x} PMs")
    
    for i in range(x):
        pm_id = f"PM_{i}"
        pm = PM(
            pm_id=pm_id,
            cpu_cap=PM_CPU_CAP,
            mem_cap=PM_MEM_CAP
        )
        state.add_pm(pm)


init_pms(dataset_file)        

# Adapter -> events
adapter = OnlineTraceAdapter(dataset_file) 
events = adapter.load_events()

# Dispatcher -> scheduler -> Simulator
dispatcher = Dispatcher(state)

# scheduler = HeuristicScheduler(
#     cpu_threshold=0.8,
#     min_interval=1
# )

scheduler = FCFSScheduler()

simulator = Simulator(
    state=state,
    dispatcher=dispatcher,
    scheduler=scheduler
)

# Add events
for event in events:
    simulator.add_event(event)

# Run
simulator.run()

logger.info("Simulation done")
logger.info(f"Final snapshot: {state.snapshot()}")
