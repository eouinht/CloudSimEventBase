
# import json, torch
# import glob
# from core.state import SimulationState
# from core.pm import PM
# from engine.simulator import Simulator
# from engine.dispatcher import Dispatcher
# from scheduler.heuristic import HeuristicScheduler, FCFSScheduler
# from datasource.online_trace import OnlineTraceAdapter
# from log.logger import SimLogger
# from models.components.pt_tranformer import CloudTransformer

# logger = SimLogger.get_logger("RUN")

# DATASET_DIR = "datasource/"
# PM_CPU_CAP = 44 
# PM_MEM_CAP = 178

# encoder = CloudTransformer(input_dim=3, d_model=64)
# encoder.eval()

# def process_step(state, target_vm):
#     raw_features = state.get_feature_matrix(state.current_time)
#     input_tensor = raw_features.unsqueeze(0)
#     sp = get_dynamic_sp(state)
#     with torch.no_grad():
#         state_embedding = encoder(
#             input_tensor,
#             split_point=sp
#         )
        
    
# def get_dynamic_sp(state):
#     """
#     Xác định điểm phân tách (split_point) dựa trên quy tắc 40 phút của Huawei Cloud.
#     Nhóm 1: Các PM có nguy cơ biến động cao (chứa VM mới chạy < 40p).
#     Nhóm 2: Các PM đã ổn định.
#     """
#     # Sắp xếp danh sách PM theo ID để đảm bảo thứ tự nhất quán cho Transformer
#     pms = sorted(state.pms.values(), key=lambda x: x.id)
    
#     risk_pms_count = 0
#     # 40 phút tương đương với 8 mốc thời gian (nếu mỗi mốc là 5 phút theo bài báo)
#     threshold_ticks = 8 

#     for pm in pms:
#         # Kiểm tra nếu PM có bất kỳ VM nào mới được khởi tạo gần đây
#         has_young_vm = any((state.current_time - vm.start_time) < threshold_ticks for vm in pm.running_vms.values())
#         if has_young_vm:
#             risk_pms_count += 1
            
#     # Đảm bảo split_point nằm trong khoảng hợp lệ [1, tổng_số_pm - 1]
#     # Nếu không có PM nào rủi ro hoặc tất cả đều rủi ro, ta chia đôi mặc định
#     if risk_pms_count == 0 or risk_pms_count == len(pms):
#         return len(pms) // 2
        
#     return risk_pms_count


# # Load dataset
# json_files = sorted(glob.glob(f"{DATASET_DIR}/*.json"))
# if not json_files:
#     raise RuntimeError("No dataset files found")
# dataset_file = json_files[0]   # chạy 1 file trước
# logger.info(f"Using dataset: {dataset_file}")

# # Init state
# state = SimulationState()

# # X-1.json -> state
# def init_pms(filename):
#     basename = filename.split("/")[-1]
#     x = int(basename.split("_")[0])
    
#     logger.info(f"Initializing {x} PMs")
    
#     for i in range(x):
#         pm_id = f"PM_{i}"
#         pm = PM(
#             pm_id=pm_id,
#             cpu_cap=PM_CPU_CAP,
#             mem_cap=PM_MEM_CAP
#         )
#         state.add_pm(pm)

# init_pms(dataset_file)        

# # Adapter -> events
# adapter = OnlineTraceAdapter(dataset_file) 
# events = adapter.load_events()

# # Dispatcher -> scheduler -> Simulator
# dispatcher = Dispatcher(state)

# # scheduler = HeuristicScheduler(
# #     cpu_threshold=0.8,
# #     min_interval=1
# # )

# scheduler = FCFSScheduler()

# simulator = Simulator(
#     state=state,
#     dispatcher=dispatcher,
#     scheduler=scheduler
# )

# # Add events
# for event in events:
#     simulator.add_event(event)

# # Run
# simulator.run()

# logger.info("Simulation done")
# logger.info(f"Final snapshot: {state.snapshot()}")


import torch
import glob
from core.state import SimulationState
from core.pm import PM
from engine.simulator import Simulator
from engine.dispatcher import Dispatcher
from scheduler.heuristic import FCFSScheduler
from datasource.online_trace import OnlineTraceAdapter
from log.logger import SimLogger
from models.components.pt_tranformer import CloudTransformer

logger = SimLogger.get_logger("RUN")

# --- CONFIGURATION ---
CONFIG = {
    "DATASET_DIR": "datasource/",
    "PM_CPU_CAP": 44,
    "PM_MEM_CAP": 178,
    "EMBED_DIM": 64,
    "INPUT_DIM": 3,
    "STABILITY_THRESHOLD": 8  # 40 phút / 5 phút mỗi tick
}

class AgentManager:
    """Quản lý Module 1 (Encoder) và các logic xử lý Tensor liên quan."""
    def __init__(self, config):
        self.encoder = CloudTransformer(
            input_dim=config["INPUT_DIM"], 
            d_model=config["EMBED_DIM"]
        )
        self.encoder.eval()
        self.threshold = config["STABILITY_THRESHOLD"]

    def get_dynamic_sp(self, state):
        """Xác định split_point dựa trên đặc tính tải động của PMs."""
        pms = sorted(state.pms.values(), key=lambda x: x.id)
        risk_count = 0
        
        for pm in pms:
            # Kiểm tra trạng thái "trẻ" của VM để phân loại nhóm rủi ro
            is_risky = any((state.current_time - vm.start_time) < self.threshold 
                           for vm in pm.running_vms.values())
            if is_risky:
                risk_count += 1
        
        # Logic dự phòng nếu không có sự phân cấp rõ ràng
        if risk_count == 0 or risk_count == len(pms):
            return len(pms) // 2
        return risk_count

    def observe_and_encode(self, state):
        """Quan sát snapshot và tạo ra State Representation thông qua Transformer."""
        raw_features = state.get_feature_matrix(state.current_time)
        input_tensor = raw_features.unsqueeze(0) # Thêm chiều Batch
        sp = self.get_dynamic_sp(state)
        
        with torch.no_grad():
            # Thực hiện Attention phân nhánh (Bifurcated Attention)
            embedding = self.encoder(input_tensor, split_point=sp)
        return embedding

# --- SIMULATION SETUP ---
def setup_environment(filename):
    """Khởi tạo thực thể Simulation từ file dataset."""
    state = SimulationState()
    basename = filename.split("/")[-1]
    num_pms = int(basename.split("_")[0])
    
    logger.info(f"Initializing Cluster: {num_pms} PMs")
    for i in range(num_pms):
        state.add_pm(PM(f"PM_{i}", CONFIG["PM_CPU_CAP"], CONFIG["PM_MEM_CAP"]))
    
    return state

def main():
    # 1. Prepare Data
    json_files = sorted(glob.glob(f"{CONFIG['DATASET_DIR']}/*.json"))
    if not json_files: raise RuntimeError("Empty dataset")
    target_file = json_files[0]

    # 2. Init Components
    state = setup_environment(target_file)
    agent = AgentManager(CONFIG)
    adapter = OnlineTraceAdapter(target_file)
    dispatcher = Dispatcher(state)
    scheduler = FCFSScheduler() # Hoặc nâng cấp thành SmartScheduler(agent, scheduler)

    simulator = Simulator(state, dispatcher, scheduler)

    # 3. Load & Run
    for event in adapter.load_events():
        # Ở đây bạn có thể can thiệp: trước khi event được xử lý, agent sẽ quan sát
        # state_rep = agent.observe_and_encode(state)
        simulator.add_event(event)

    logger.info("Starting Simulation...")
    simulator.run()
    
    logger.info(f"Final Report: {state.snapshot()}")

if __name__ == "__main__":
    main()