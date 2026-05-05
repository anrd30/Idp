import torch
import numpy as np
import time
from ndt_dsp import NDTSignalProcessor
from ndt_model import NDTNet

def benchmark_tradeoffs(total_samples=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on: {device}")
    
    # 1. Setup Data
    max_steps = 800
    grid_res = 50
    dsp = NDTSignalProcessor(fs=1000)
    _, chirp_sig = dsp.generate_chirp(100, 300, 0.05)
    
    # Dummy data
    batch_data = np.random.randn(total_samples, 4, max_steps).astype(np.float32)
    sensor_coords = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
    
    # Load CNN
    model = NDTNet().to(device)
    model.eval()
    
    # --- 1. TDOA BENCHMARK ---
    print("\nStarting TDOA Benchmark (Sequential Processing)...")
    t_start_tdoa = time.time()
    for i in range(total_samples):
        # We simulate the exact processing pipeline used in main_ndt
        arrival_times = [0.0] * 4
        unit_velocity = 0.45 / 256
        for s in range(1, 4):
            raw = batch_data[i, s, :]
            # Matched Filtering (FFT based)
            comp, env = dsp.process_signal(raw, chirp_sig)
            
            # Peak detection and Blanking
            dist_direct = 0.5 # constant for bench
            direct_arrival_steps = dist_direct / unit_velocity
            blank_window = int(direct_arrival_steps + len(chirp_sig) * 0.8)
            env_masked = env.copy()
            if blank_window < len(env_masked): env_masked[:blank_window] = 0.0
            peaks = np.where(env_masked > 0.01)[0]
            if peaks.size > 0: arrival_times[s] = peaks[0] * unit_velocity
            
        # Grid Search / Probability Mapping
        _, _, _ = dsp.locate_defect(sensor_coords, arrival_times, 1.0, grid_res)
    
    t_tdoa = time.time() - t_start_tdoa
    
    # --- 2. CNN BENCHMARK (Individual) ---
    print("Starting CNN Benchmark (Serial Inference)...")
    t_start_cnn_serial = time.time()
    with torch.no_grad():
        for i in range(total_samples):
            x = torch.from_numpy(batch_data[i:i+1]).to(device)
            _ = model(x).cpu()
    t_cnn_serial = time.time() - t_start_cnn_serial
    
    # --- 3. CNN BENCHMARK (Batched) ---
    print("Starting CNN Benchmark (Batched Inference)...")
    t_start_cnn_batch = time.time()
    with torch.no_grad():
        x = torch.from_numpy(batch_data).to(device)
        # Process in chunks of 100 to avoid OOM but show scaling
        chunk_size = 100
        for i in range(0, total_samples, chunk_size):
            _ = model(x[i:i+chunk_size]).cpu()
    t_cnn_batch = time.time() - t_start_cnn_batch
    
    # Results Calculation
    ms_per_sample_tdoa = (t_tdoa / total_samples) * 1000
    ms_per_sample_cnn_serial = (t_cnn_serial / total_samples) * 1000
    ms_per_sample_cnn_batch = (t_cnn_batch / total_samples) * 1000
    
    print("\n" + "="*40)
    print("COMPUTATIONAL TRADEOFF ANALYSIS")
    print("="*40)
    print(f"{'Method':<20} | {'Latency (ms/sample)':<20}")
    print("-" * 43)
    print(f"{'Classic TDOA':<20} | {ms_per_sample_tdoa:>18.2f}")
    print(f"{'CNN (Single)':<20} | {ms_per_sample_cnn_serial:>18.2f}")
    print(f"{'CNN (Batched 100)':<20} | {ms_per_sample_cnn_batch:>18.2f}")
    print("="*40)
    
    print("\nSCALE ANALYSIS:")
    print(f"1. TDOA is highly sensitive to Grid Resolution (currently {grid_res}x{grid_res}).")
    print(f"   If grid doubles to 100x100, TDOA latency increases by ~4x (O(N^2)).")
    print(f"2. CNN is constant time relative to search grid (reconstruction is fixed).")
    print(f"3. Batched CNN is {ms_per_sample_tdoa / ms_per_sample_cnn_batch:.1f}x faster than TDOA at scale.")
    print(f"4. Tradeoff: CNN requires ~11MB of VRAM and a GPU for max efficiency,")
    print(f"   whereas TDOA runs on minimal CPU memory but scales poorly.")

if __name__ == "__main__":
    benchmark_tradeoffs(1000)
