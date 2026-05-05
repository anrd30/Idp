import torch
import numpy as np
import taichi as ti
from ndt_engine import BatchedTaichiWaveSolver
from ndt_dsp import NDTSignalProcessor
from ndt_model import NDTNet
import time
from tqdm import tqdm

def run_statistical_comparison(total_tests=1000, batch_size=100):
    ti.init(arch=ti.gpu)
    res = 256
    max_steps = 800
    grid_res = 50
    
    solver = BatchedTaichiWaveSolver(batch_size=batch_size, res=res)
    dsp = NDTSignalProcessor(fs=1000)
    chirp_t, chirp_sig = dsp.generate_chirp(100, 300, 0.05)
    
    # Load CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NDTNet().to(device)
    try:
        model.load_state_dict(torch.load("ndt_cnn_model.pth", map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    num_batches = total_tests // batch_size
    tdoa_errors = []
    cnn_errors = []
    
    # Coordinate grid for max detection
    x_coords = np.linspace(0, 1, grid_res)
    y_coords = np.linspace(0, 1, grid_res)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    print(f"--- Starting Statistical Comparison ({total_tests} tests) ---")
    start_time = time.time()
    
    for b_idx in range(num_batches):
        solver.reset()
        
        # Setup batch
        gt_positions = []
        sensor_coords = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        
        for i in range(batch_size):
            gt_x, gt_y = np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7)
            gt_positions.append((gt_x, gt_y))
            solver.set_batch_params(i, 0.45) # Wave velocity
            solver.set_defect(i, gt_x, gt_y, 5.0)
            for s in range(4):
                solver.set_sensor_pos(i, s, sensor_coords[s][0], sensor_coords[s][1])
        
        # Run Simulation
        batch_sensor_data = np.zeros((batch_size, 4, max_steps), dtype=np.float32)
        for t in range(max_steps):
            if t < len(chirp_sig):
                solver.inject_source(chirp_sig[t] * 5.0)
            solver.step()
            batch_sensor_data[:, :, t] = solver.get_sensor_data()
            
        # 1. AI CNN Inference (Batch processing is very fast)
        with torch.no_grad():
            x_tensor = torch.from_numpy(batch_sensor_data).to(device)
            cnn_outputs = model(x_tensor).cpu().numpy() # (batch, 1, 50, 50)
            
        # 2. Sequential Analysis (TDOA and Error Calculation)
        for i in range(batch_size):
            gt_x, gt_y = gt_positions[i]
            
            # CNN Localization
            prob_map_cnn = cnn_outputs[i].squeeze()
            max_idx_cnn = np.argmax(prob_map_cnn)
            cnn_x = X_grid.flatten()[max_idx_cnn]
            cnn_y = Y_grid.flatten()[max_idx_cnn]
            cnn_errors.append(np.sqrt((cnn_x - gt_x)**2 + (cnn_y - gt_y)**2))
            
            # Classic TDOA Localization
            unit_velocity = 0.45 / res
            arrival_times = [0.0] * 4
            for s in range(1, 4):
                raw = batch_sensor_data[i, s, :]
                comp, env = dsp.process_signal(raw, chirp_sig)
                dist_direct = np.sqrt((sensor_coords[s][0] - sensor_coords[0][0])**2 + (sensor_coords[s][1] - sensor_coords[0][1])**2)
                direct_arrival_steps = dist_direct / unit_velocity
                blank_window = int(direct_arrival_steps + len(chirp_sig) * 0.8)
                env_masked = env.copy()
                if blank_window < len(env_masked): env_masked[:blank_window] = 0.0
                threshold = np.max(env_masked) * 0.5
                peaks = np.where(env_masked > threshold)[0]
                if peaks.size > 0: arrival_times[s] = peaks[0] * unit_velocity
            
            _, _, prob_map_tdoa = dsp.locate_defect(sensor_coords, arrival_times, 1.0, grid_res)
            max_idx_tdoa = np.argmax(prob_map_tdoa)
            tdoa_x = X_grid.flatten()[max_idx_tdoa]
            tdoa_y = Y_grid.flatten()[max_idx_tdoa]
            tdoa_errors.append(np.sqrt((tdoa_x - gt_x)**2 + (tdoa_y - gt_y)**2))
            
        print(f"Batch {b_idx+1}/{num_batches} complete...")

    end_time = time.time()
    
    mean_tdoa = np.mean(tdoa_errors)
    mean_cnn = np.mean(cnn_errors)
    std_tdoa = np.std(tdoa_errors)
    std_cnn = np.std(cnn_errors)
    improvement = (mean_tdoa - mean_cnn) / mean_tdoa * 100
    
    print("\n--- STATISTICAL RESULTS (N=1000) ---")
    print(f"Execution Time: {end_time - start_time:.1f} seconds")
    print(f"Mean TDOA Error: {mean_tdoa:.6f} (±{std_tdoa:.4f})")
    print(f"Mean AI CNN Error: {mean_cnn:.6f} (±{std_cnn:.4f})")
    print(f"Average Improvement: {improvement:.1f}%")
    
    if improvement > 0:
        print(f"\nConclusion: AI CNN significantly outperforms TDOA in mean accuracy.")
    else:
        print(f"\nConclusion: TDOA remains more accurate (check dataset/training).")

if __name__ == "__main__":
    run_statistical_comparison(total_tests=1000, batch_size=100)
