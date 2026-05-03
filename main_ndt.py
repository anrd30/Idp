import dearpygui.dearpygui as dpg
import numpy as np
import time
from ndt_engine import TaichiWaveSolver
from ndt_dsp import NDTSignalProcessor

class NDTApp:
    def __init__(self):
        self.res = 256
        self.solver = TaichiWaveSolver(res=self.res)
        self.dsp = NDTSignalProcessor(fs=1000) # Simplified fs for sim scale
        
        # State
        self.running = False
        self.t_step = 0
        self.max_steps = 800
        self.sensor_data = [[] for _ in range(4)]
        self.defect_pos = [0.6, 0.5]
        self.sensor_coords = [
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]
        ]
        self.wave_velocity = 0.45
        self.active_sensor = 0
        
        # Signal Generation: Shorter pulse (0.05s) for better spatial resolution
        self.chirp_t, self.chirp_sig = self.dsp.generate_chirp(100, 300, 0.05)
        
        self.setup_dpg()

    def setup_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title='Ultra-HD Solar NDT Pro Simulator', width=1280, height=800)
        
        # Texture Registry for Wavefield
        with dpg.texture_registry(show=False):
            initial_data = np.zeros((self.res, self.res, 4), dtype=np.float32)
            dpg.add_dynamic_texture(width=self.res, height=self.res, default_value=initial_data.flatten(), tag="wavefield_tex")
            
            # Probability Heatmap Texture
            initial_prob = np.zeros((50, 50, 4), dtype=np.float32)
            dpg.add_dynamic_texture(width=50, height=50, default_value=initial_prob.flatten(), tag="prob_tex")

        with dpg.window(label="Industrial Control Panel", width=300, height=800, pos=(0,0), no_move=True, no_close=True):
            with dpg.collapsing_header(label="PHYSICAL PARAMETERS", default_open=True):
                dpg.add_slider_float(label="Wave Velocity", default_value=0.45, min_value=0.1, max_value=0.6, callback=self.update_params, tag="vel_slider")
                dpg.add_slider_float(label="Defect X", default_value=0.6, min_value=0.1, max_value=0.9, callback=self.update_params, tag="dx_slider")
                dpg.add_slider_float(label="Defect Y", default_value=0.5, min_value=0.1, max_value=0.9, callback=self.update_params, tag="dy_slider")
            
            with dpg.collapsing_header(label="GEOMETRY CONTROLS", default_open=True):
                for i in range(4):
                    with dpg.tree_node(label=f"Sensor {i}"):
                        dpg.add_slider_float(label="X", default_value=self.sensor_coords[i][0], min_value=0.05, max_value=0.95, callback=self.update_params, tag=f"sx_{i}")
                        dpg.add_slider_float(label="Y", default_value=self.sensor_coords[i][1], min_value=0.05, max_value=0.95, callback=self.update_params, tag=f"sy_{i}")

            with dpg.collapsing_header(label="EXPERIMENT CONTROLS", default_open=True):
                dpg.add_button(label="INJECT PULSE & START", width=-1, height=40, callback=self.start_sim)
                dpg.add_button(label="RESET SIMULATION", width=-1, callback=self.reset_sim)
            
            with dpg.collapsing_header(label="SENSOR ANALYSIS", default_open=True):
                dpg.add_combo(label="Active Channel", items=["Sensor 0", "Sensor 1", "Sensor 2", "Sensor 3"], default_value="Sensor 0", callback=self.change_sensor)
                for i in range(4):
                    dpg.add_text(f"Sensor {i}: Active", tag=f"s_status_{i}", color=(100, 100, 255))
            
            dpg.add_spacer(height=20)
            dpg.add_text("DIAGNOSTICS", color=(0, 255, 0))
            self.fps_text = dpg.add_text("FPS: 0")

        with dpg.window(label="Main Visualization Deck", width=980, height=800, pos=(300,0), no_move=True, no_close=True):
            with dpg.group(horizontal=True):
                # Wavefield View
                with dpg.child_window(width=500, height=500, border=True):
                    dpg.add_text("REAL-TIME SURFACE WAVEFIELD (Taichi FDTD)")
                    with dpg.drawlist(width=480, height=480):
                        dpg.draw_image("wavefield_tex", [0, 0], [480, 480])
                        # Overlays for sensors and defect
                        self.overlay_tag = dpg.add_draw_node()
                # Probability Map
                with dpg.child_window(width=450, height=500, border=True):
                    dpg.add_text("DEFECT PROBABILITY HEATMAP (TDOA)")
                    with dpg.drawlist(width=400, height=400):
                        dpg.draw_image("prob_tex", [0, 0], [400, 400])
                    
                    with dpg.plot(label="Localization Analysis", height=150, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, label="X (normalized)", tag="x_axis_p")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Y (normalized)", tag="y_axis_p")
                        dpg.add_scatter_series([self.defect_pos[0]], [self.defect_pos[1]], label="Ground Truth", parent="y_axis_p", tag="truth_marker")
                        dpg.add_scatter_series([], [], label="Estimated Defect", parent="y_axis_p", tag="est_marker")
                        dpg.add_plot_legend()

            # A-Scan Display

            with dpg.child_window(width=950, height=250, border=True):
                dpg.add_text("INDUSTRIAL A-SCAN (Pulse Compression & Hilbert Envelope)")
                with dpg.plot(label="Signal Analysis", height=180, width=-1):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Sample Index", tag="x_axis_s")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude (Normalized)", tag="y_axis_s")
                    dpg.add_line_series([], [], label="Raw Compressed", parent="y_axis_s", tag="raw_plot")
                    dpg.add_line_series([], [], label="Hilbert Envelope", parent="y_axis_s", tag="env_plot")
                    dpg.add_plot_legend()

        self.update_params()
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def change_sensor(self, sender, app_data):
        self.active_sensor = int(app_data.split(" ")[1])
        self.reset_plots()

    def update_params(self):
        # Update Defect
        self.defect_pos = [dpg.get_value("dx_slider"), dpg.get_value("dy_slider")]
        self.wave_velocity = dpg.get_value("vel_slider")
        self.solver.c_norm = self.wave_velocity
        self.solver.set_defect(self.defect_pos[0], self.defect_pos[1], 5.0)
        
        # Update Sensors
        for i in range(4):
            self.sensor_coords[i] = [dpg.get_value(f"sx_{i}"), dpg.get_value(f"sy_{i}")]
            self.solver.set_sensor_pos(i, self.sensor_coords[i][0], self.sensor_coords[i][1])
            
        dpg.set_value("truth_marker", [[self.defect_pos[0]], [self.defect_pos[1]]])
        self.refresh_overlays()

    def refresh_overlays(self):
        dpg.delete_item(self.overlay_tag, children_only=True)
        # Draw Sensors
        for i in range(4):
            color = (0, 255, 255) if i == self.active_sensor else (100, 100, 255)
            # Map normalized (0-1) to drawlist (0-480)
            px = self.sensor_coords[i][1] * 480 # DPG draws Y then X in some contexts, adjust to match Taichi
            py = self.sensor_coords[i][0] * 480
            dpg.draw_circle([px, py], 6, color=color, fill=color, parent=self.overlay_tag)
            dpg.draw_text([px+8, py-8], f"S{i}", size=14, color=(255, 255, 255), parent=self.overlay_tag)

    def start_sim(self):
        self.reset_sim()
        self.running = True

    def reset_sim(self):
        self.running = False
        self.t_step = 0
        self.solver.reset()
        self.update_params()
        self.sensor_data = [[] for _ in range(4)]
        self.reset_plots()

    def reset_plots(self):
        dpg.set_value("raw_plot", [[], []])
        dpg.set_value("env_plot", [[], []])
        dpg.set_value("prob_tex", np.zeros((50, 50, 4)).flatten())
        dpg.set_value("est_marker", [[], []])

    def update_heatmap_texture(self, prob_map):
        # Create a RGBA texture from the 0-1 probability map
        # Use 'Viridis' like colors: dark purple to bright yellow
        grid_res = prob_map.shape[0]
        tex_data = np.zeros((grid_res, grid_res, 4), dtype=np.float32)
        
        # Simple color ramp
        norm_prob = prob_map / (np.max(prob_map) + 1e-6)
        tex_data[:, :, 0] = norm_prob * 0.9 # Red
        tex_data[:, :, 1] = norm_prob * 0.8 # Green
        tex_data[:, :, 2] = 0.2 + norm_prob * 0.1 # Blue
        tex_data[:, :, 3] = 1.0 # Alpha
        
        dpg.set_value("prob_tex", tex_data.flatten())

    def perform_localization(self):
        arrival_times = [0.0] * 4 # Index 0 is source
        # Wave speed in normalized units (0-1) per step
        # FDTD speed is c_norm pixels/step, resolution is res
        unit_velocity = self.wave_velocity / self.res 
        
        # Analyze each sensor for echo peaks
        for i in range(1, 4):
            raw = np.array(self.sensor_data[i])
            if len(raw) > 0:
                comp, env = self.dsp.process_signal(raw, self.chirp_sig)
                
                # --- DIRECT PATH BLANKING ---
                # Calculate time of arrival for direct wave (in steps)
                dist_direct = np.sqrt(
                    (self.sensor_coords[i][0] - self.sensor_coords[0][0])**2 + 
                    (self.sensor_coords[i][1] - self.sensor_coords[0][1])**2
                )
                direct_arrival_steps = dist_direct / unit_velocity
                
                # Blank the direct pulse (arrival time + pulse duration)
                # Pulse duration in samples is chirp_sig length
                blank_window = int(direct_arrival_steps + len(self.chirp_sig) * 0.8)
                
                env_masked = env.copy()
                if blank_window < len(env_masked):
                    env_masked[:blank_window] = 0.0
                
                # Peak detection: Look for the first significant peak after blanking
                threshold = np.max(env_masked) * 0.5
                peaks = np.where(env_masked > threshold)[0]
                
                if peaks.size > 0:
                    # Echo arrival in normalized units
                    arrival_times[i] = peaks[0] * unit_velocity
        
        # Call DSP localization with corrected speed
        grid_res = 50
        X, Y, prob_map = self.dsp.locate_defect(self.sensor_coords, arrival_times, 1.0, grid_res)
        
        # Update Heatmap Texture
        self.update_heatmap_texture(prob_map)
        
        # Update Estimated Marker (Point of max probability)
        max_idx = np.argmax(prob_map)
        est_y = Y.flatten()[max_idx]
        est_x = X.flatten()[max_idx]
        dpg.set_value("est_marker", [[est_x], [est_y]])

    def run(self):
        last_time = time.time()
        while dpg.is_dearpygui_running():
            if self.running and self.t_step < self.max_steps:
                # Physics Step
                for _ in range(4): # Sub-steps for visual speed
                    if self.t_step < len(self.chirp_sig):
                        # Increased injection power for better SNR
                        self.solver.inject_source(self.chirp_sig[self.t_step] * 5.0)
                    
                    self.solver.step()
                    s_vals = self.solver.get_sensor_data()
                    for i in range(4):
                        self.sensor_data[i].append(s_vals[i])
                    
                    self.t_step += 1
                
                # Update Wavefield Texture
                frame = self.solver.get_frame()
                dpg.set_value("wavefield_tex", frame.flatten())
                
                # Check for completion
                if self.t_step >= self.max_steps:
                    self.running = False
                    self.perform_localization()

                # DSP Update (every 20 steps to keep UI fluid)
                if self.t_step % 20 == 0:
                    try:
                        raw = np.array(self.sensor_data[self.active_sensor])
                        if len(raw) > 50:
                            comp, env = self.dsp.process_signal(raw, self.chirp_sig)
                            
                            # Normalize for visualization
                            max_val = np.max(env) if np.max(env) > 0 else 1.0
                            comp_norm = comp / max_val
                            env_norm = env / max_val
                            
                            indices = np.arange(len(env_norm))
                            dpg.set_value("raw_plot", [indices.tolist(), comp_norm.tolist()])
                            dpg.set_value("env_plot", [indices.tolist(), env_norm.tolist()])
                            
                            # Auto-fit axes
                            dpg.fit_axis_data("x_axis_s")
                            dpg.fit_axis_data("y_axis_s")
                    except Exception as dsp_err:
                        pass

            # UI Update
            curr_time = time.time()
            fps = 1.0 / (curr_time - last_time + 1e-6)
            dpg.set_value(self.fps_text, f"FPS: {fps:.1f}")
            last_time = curr_time
            
            dpg.render_dearpygui_frame()
        
        dpg.destroy_context()

if __name__ == "__main__":
    import taichi as ti
    ti.init(arch=ti.cpu)
    try:
        app = NDTApp()
        app.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
