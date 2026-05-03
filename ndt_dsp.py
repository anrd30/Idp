import numpy as np
from scipy.signal import chirp, correlate, hilbert

class NDTSignalProcessor:
    def __init__(self, fs=100000):
        self.fs = fs

    def generate_chirp(self, f0, f1, duration, method='quadratic'):
        t = np.linspace(0, duration, int(self.fs * duration))
        signal = chirp(t, f0=f0, f1=f1, t1=duration, method=method)
        # Apply Hanning window for better spectral properties
        signal *= np.hanning(len(signal))
        return t, signal

    def process_signal(self, raw_signal, reference_chirp):
        # Matched Filter (Cross-correlation)
        # Normalize for consistency
        ref = reference_chirp / np.max(np.abs(reference_chirp))
        compressed = correlate(raw_signal, ref, mode='same')
        
        # Analytic signal via Hilbert transform for envelope detection
        analytic_signal = hilbert(compressed)
        envelope = np.abs(analytic_signal)
        
        return compressed, envelope

    def locate_defect(self, sensor_positions, arrival_times, wave_speed, grid_res):
        """
        Estimate defect position using TDOA (Time Difference of Arrival).
        For simplicity in this 2D demo, we use a grid-search 'likelihood' map.
        """
        x = np.linspace(0, 1, grid_res)
        y = np.linspace(0, 1, grid_res)
        X, Y = np.meshgrid(x, y)
        
        # likelihood map: points that satisfy the distance equation
        # dist(Source, Point) + dist(Point, Sensor[i]) = WaveSpeed * Time[i]
        prob_map = np.zeros_like(X)
        
        # Sensor 0 is our source. 
        s0 = sensor_positions[0]
        
        # We only use sensors that actually received an echo (arrival_times[i] > source_time)
        valid_sensors = 0
        for i in range(1, len(sensor_positions)):
            t_arrival = arrival_times[i]
            if t_arrival > 0:
                # Expected travel distance
                total_dist = t_arrival * wave_speed
                
                # Calculate theoretical distance for every point on the grid
                d_from_source = np.sqrt((X - s0[0])**2 + (Y - s0[1])**2)
                d_to_sensor = np.sqrt((X - sensor_positions[i][0])**2 + (Y - sensor_positions[i][1])**2)
                
                # Difference between actual travel and theoretical travel
                error = np.abs((d_from_source + d_to_sensor) - total_dist)
                
                # Accumulate 'probability' (Gaussian kernel on the error)
                prob_map += np.exp(- (error**2) / 0.005)
                valid_sensors += 1
        
        if valid_sensors > 0:
            prob_map /= valid_sensors
            
        return X, Y, prob_map
