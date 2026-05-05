import taichi as ti
import numpy as np
import time
import os
from ndt_engine import BatchedTaichiWaveSolver
from ndt_dsp import NDTSignalProcessor

def generate_target_heatmap(defect_list, grid_res=50):
    heatmap = np.zeros((grid_res, grid_res), dtype=np.float32)
    sigma = 1.5
    for defect in defect_list:
        cx, cy = defect[0] * grid_res, defect[1] * grid_res
        for y in range(grid_res):
            for x in range(grid_res):
                dist_sq = (x - cx)**2 + (y - cy)**2
                heatmap[y, x] = max(heatmap[y, x], np.exp(-dist_sq / (2 * sigma**2)))
    return heatmap

def run_dataset_generation(total_sims=10000, batch_size=200, output_dir="cnn_dataset"):
    print(f"Starting CNN Dataset Generation: {total_sims} simulations")
    os.makedirs(output_dir, exist_ok=True)
    ti.init(arch=ti.gpu)
    res, max_steps, grid_res = 256, 800, 50
    solver = BatchedTaichiWaveSolver(batch_size=batch_size, res=res)
    dsp = NDTSignalProcessor(fs=1000)
    chirp_t, chirp_sig = dsp.generate_chirp(100, 300, 0.05)
    num_batches = total_sims // batch_size
    start_time = time.time()
    for batch_idx in range(num_batches):
        solver.reset()
        batch_heatmaps = np.zeros((batch_size, grid_res, grid_res), dtype=np.float32)
        for b in range(batch_size):
            wave_vel = np.random.uniform(0.3, 0.5)
            solver.set_batch_params(b, wave_vel)
            sensor_coords = [[np.random.uniform(0.05, 0.2), np.random.uniform(0.05, 0.2)],
                             [np.random.uniform(0.8, 0.95), np.random.uniform(0.05, 0.2)],
                             [np.random.uniform(0.05, 0.2), np.random.uniform(0.8, 0.95)],
                             [np.random.uniform(0.8, 0.95), np.random.uniform(0.8, 0.95)]]
            for s in range(4): solver.set_sensor_pos(b, s, sensor_coords[s][0], sensor_coords[s][1])
            num_defects = np.random.choice([0, 1, 2, 3])
            defect_list = []
            for _ in range(num_defects):
                cx, cy = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
                defect_list.append((cx, cy))
                solver.set_defect(b, cx, cy, 5.0)
            batch_heatmaps[b] = generate_target_heatmap(defect_list, grid_res)
        sensor_data = np.zeros((batch_size, 4, max_steps), dtype=np.float32)
        for t_step in range(max_steps):
            if t_step < len(chirp_sig): solver.inject_source(chirp_sig[t_step] * 5.0)
            solver.step()
            sensor_data[:, :, t_step] = solver.get_sensor_data()
        np.savez_compressed(os.path.join(output_dir, f"batch_{batch_idx:04d}.npz"), X=sensor_data, Y=batch_heatmaps)
        if (batch_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Batch {batch_idx+1}/{num_batches} saved. Rate: {((batch_idx + 1) * batch_size / elapsed):.1f} sims/sec")
    print(f"\n--- GENERATION COMPLETE ---")

if __name__ == '__main__':
    run_dataset_generation(total_sims=100000, batch_size=200)
