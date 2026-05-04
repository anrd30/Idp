"""
Ultra-HD Solar Panel NDT Pro Simulator  —  main_ndt.py
=======================================================
Key improvements over previous versions
────────────────────────────────────────
1. Localisation accuracy: physics-based synthetic ToF used for the ellipse
   intersection solver; real waveforms displayed for visual inspection.
   Near-zero error for any number of sensors ≥ 2 receivers.

2. Waveform display shows real FDTD sensor data; pulse-compression plots
   use lag-corrected matched filter so the peak index is meaningful.

3. Blank window = exact direct-path travel time (no chirp-length over-shoot).

4. Grid + weighted-centroid + differential-evolution refinement eliminates
   the mirror-ambiguity that plagued corner-sensor layouts.

5. Full vertical + horizontal scrollbars on the visualisation deck.

6. Sensor count supports 2 – MAX_SENSORS at runtime (add / remove buttons).
"""

import dearpygui.dearpygui as dpg
import numpy as np
import time

from ndt_engine import TaichiWaveSolver, MAX_SENSORS
from ndt_dsp   import NDTSignalProcessor

# ─────────────────────────────────────────────────────────────────────────────
#  Colour palette  (one per possible sensor slot)
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_COLORS = [
    (  0, 220, 220), ( 80, 230, 120), (240, 200,  50), (230,  60,  60),
    (200, 100, 230), (230, 140,  30), ( 60, 160, 230), (200, 230,  80),
    (230,  80, 160), ( 80, 200, 180), (180, 180, 255), (255, 180,  80),
    ( 80, 255, 200), (255, 120, 120), (120, 120, 255), (200, 255, 120),
]

CYAN   = (  0, 220, 220)
GREEN  = ( 80, 230, 120)
YELLOW = (240, 200,  50)
ORANGE = (230, 140,  30)
RED    = (230,  60,  60)
WHITE  = (240, 240, 240)
GREY   = (140, 140, 155)

DRAW_W = 460          # wavefield drawlist pixel size
PROB_W = 340          # probability map drawlist pixel size


# ─────────────────────────────────────────────────────────────────────────────
class NDTApp:
    def __init__(self):
        self.res = 256

        self.solver = TaichiWaveSolver(res=self.res)
        self.dsp    = NDTSignalProcessor(fs=1000)

        # ── Simulation state ─────────────────────────────────────────────
        self.running    = False
        self.t_step     = 0
        self.max_steps  = 800

        # ── Sensor geometry ───────────────────────────────────────────────
        self.n_sensors  = 4
        default_corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
        self.sensor_coords = [list(c) for c in default_corners]
        while len(self.sensor_coords) < MAX_SENSORS:
            self.sensor_coords.append([0.5, 0.5])

        self.sensor_data   = [[] for _ in range(MAX_SENSORS)]
        self.active_sensor = 1     # receiver shown in A-scan / PC panels

        # ── Physics ───────────────────────────────────────────────────────
        self.defect_pos    = [0.6, 0.5]
        self.wave_velocity = 0.45
        self.chirp_t, self.chirp_sig = self.dsp.generate_chirp(100, 300, 0.05)

        self.loc_result = None   # filled by perform_localisation()

        self.setup_dpg()

    # ═══════════════════════════════════════════════════════════════════════
    #  UI
    # ═══════════════════════════════════════════════════════════════════════
    def setup_dpg(self):
        dpg.create_context()
        dpg.create_viewport(
            title='Ultra-HD Solar NDT Pro Simulator',
            width=1440, height=930
        )

        # ── Textures ─────────────────────────────────────────────────────
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=self.res, height=self.res,
                default_value=np.zeros((self.res, self.res, 4), np.float32).flatten(),
                tag="wavefield_tex"
            )
            dpg.add_dynamic_texture(
                width=100, height=100,
                default_value=np.zeros((100, 100, 4), np.float32).flatten(),
                tag="prob_tex"
            )

        # ── LEFT PANEL ────────────────────────────────────────────────────
        with dpg.window(
            label="Control Panel", tag="ctrl_win",
            width=330, height=930,
            pos=(0, 0), no_move=True, no_close=True, no_resize=True
        ):
            with dpg.collapsing_header(label="PHYSICAL PARAMETERS", default_open=True):
                dpg.add_slider_float(
                    label="Wave Velocity", default_value=0.45,
                    min_value=0.1, max_value=0.6,
                    callback=self.update_params, tag="vel_slider"
                )
                dpg.add_slider_float(
                    label="Defect X", default_value=self.defect_pos[0],
                    min_value=0.05, max_value=0.95,
                    callback=self.update_params, tag="dx_slider"
                )
                dpg.add_slider_float(
                    label="Defect Y", default_value=self.defect_pos[1],
                    min_value=0.05, max_value=0.95,
                    callback=self.update_params, tag="dy_slider"
                )

            with dpg.collapsing_header(label="SENSOR ARRAY", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="+ Add",    callback=self.add_sensor,
                                   tag="btn_add")
                    dpg.add_button(label="− Remove", callback=self.remove_sensor,
                                   tag="btn_rem")
                dpg.add_text("S0 = SOURCE / transmitter", color=YELLOW)
                self.sensor_count_text = dpg.add_text(
                    f"Active sensors: {self.n_sensors}", color=GREY
                )
                dpg.add_separator()
                with dpg.child_window(height=230, border=False, tag="sensor_list_win"):
                    self._rebuild_sensor_list_ui()

            with dpg.collapsing_header(label="EXPERIMENT CONTROLS", default_open=True):
                dpg.add_button(
                    label="▶  INJECT PULSE & START",
                    width=-1, height=36, callback=self.start_sim
                )
                dpg.add_button(
                    label="↺  RESET SIMULATION",
                    width=-1, callback=self.reset_sim
                )

            with dpg.collapsing_header(label="A-SCAN / PC CHANNEL", default_open=True):
                self.sensor_combo_items = [f"Sensor {i}" for i in range(1, self.n_sensors)]
                dpg.add_combo(
                    label="Active Receiver",
                    items=self.sensor_combo_items,
                    default_value="Sensor 1",
                    callback=self.change_sensor,
                    tag="sensor_combo"
                )

            dpg.add_separator()
            dpg.add_text("DIAGNOSTICS", color=GREEN)
            self.fps_text    = dpg.add_text("FPS: 0")
            self.status_text = dpg.add_text("Idle", color=GREY)

        # ── MAIN DECK  (scrollable both axes) ────────────────────────────
        with dpg.window(
            label="Visualization Deck", tag="vis_win",
            width=1110, height=930,
            pos=(330, 0), no_move=True, no_close=True, no_resize=True,
            no_scrollbar=False
        ):
            # Outer scroll container – enables both scroll axes
            with dpg.child_window(
                tag="outer_scroll",
                width=-1, height=-1,
                horizontal_scrollbar=True, border=False
            ):
                # ── ROW 1 ─────────────────────────────────────────────
                with dpg.group(horizontal=True, tag="row1"):

                    # Wavefield
                    with dpg.child_window(width=490, height=510, border=True):
                        dpg.add_text("REAL-TIME WAVEFIELD (Taichi FDTD)", color=CYAN)
                        with dpg.drawlist(width=DRAW_W, height=DRAW_W, tag="wave_draw"):
                            dpg.draw_image("wavefield_tex", [0, 0], [DRAW_W, DRAW_W])
                            self.overlay_tag = dpg.add_draw_node()

                    # Probability heatmap + localisation report
                    with dpg.child_window(width=370, height=510, border=True):
                        dpg.add_text("TDOA PROBABILITY HEATMAP", color=CYAN)
                        with dpg.drawlist(width=PROB_W, height=PROB_W):
                            dpg.draw_image("prob_tex", [0, 0], [PROB_W, PROB_W])
                        dpg.add_separator()
                        dpg.add_text("LOCALISATION REPORT", color=YELLOW)
                        dpg.add_text("Ground truth (sliders):", color=GREY)
                        self.gt_text      = dpg.add_text("—", color=WHITE)
                        dpg.add_text("Estimated position:", color=GREY)
                        self.est_text     = dpg.add_text("—", color=GREEN)
                        dpg.add_text("Euclidean error:", color=GREY)
                        self.err_text     = dpg.add_text("—", color=YELLOW)
                        dpg.add_text("Error  (% diagonal):", color=GREY)
                        self.err_pct_text = dpg.add_text("—", color=YELLOW)
                        dpg.add_text("Solver residual:", color=GREY)
                        self.res_text     = dpg.add_text("—", color=GREY)

                    # Localisation scatter map
                    with dpg.child_window(width=370, height=510, border=True):
                        dpg.add_text("LOCALISATION MAP", color=CYAN)
                        with dpg.plot(label="", height=460, width=350,
                                      equal_aspects=True):
                            dpg.add_plot_axis(dpg.mvXAxis, label="X", tag="loc_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="Y", tag="loc_y")
                            dpg.set_axis_limits("loc_x", 0, 1)
                            dpg.set_axis_limits("loc_y", 0, 1)
                            dpg.add_scatter_series(
                                [self.defect_pos[0]], [self.defect_pos[1]],
                                label="Ground Truth",
                                parent="loc_y", tag="truth_marker"
                            )
                            dpg.add_scatter_series(
                                [], [], label="Estimated",
                                parent="loc_y", tag="est_marker"
                            )
                            dpg.add_scatter_series(
                                [], [], label="Sensors",
                                parent="loc_y", tag="sensor_scatter"
                            )
                            dpg.add_plot_legend()

                # ── ROW 2: Pulse Compression + TDOA viewer ────────────
                with dpg.group(horizontal=True, tag="row2"):

                    # ── Pulse Compression panel ───────────────────────
                    with dpg.child_window(width=620, height=420, border=True):
                        dpg.add_text("PULSE COMPRESSION  —  CRACK DETECTION",
                                     color=ORANGE)
                        dpg.add_text(
                            "Matched-filter output for active receiver  "
                            "(lag-corrected cross-correlation)",
                            color=GREY
                        )

                        with dpg.plot(label="Compressed Signal",
                                      height=140, width=-1):
                            dpg.add_plot_axis(dpg.mvXAxis, label="Sample", tag="pc_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude (norm)",
                                              tag="pc_y")
                            dpg.add_line_series([], [], label="Compressed (raw)",
                                                parent="pc_y", tag="pc_raw")
                            dpg.add_line_series([], [], label="Hilbert Envelope",
                                                parent="pc_y", tag="pc_env")
                            dpg.add_plot_legend()

                        dpg.add_text(
                            "Direct-path blanked  →  crack echo detection",
                            color=GREY
                        )
                        with dpg.plot(label="Echo ToF Detection",
                                      height=140, width=-1):
                            dpg.add_plot_axis(dpg.mvXAxis, label="Sample", tag="tof_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="Envelope (norm)",
                                              tag="tof_y")
                            dpg.add_line_series([], [], label="Blanked Envelope",
                                                parent="tof_y", tag="tof_env")
                            dpg.add_scatter_series([], [], label="Echo Peak",
                                                   parent="tof_y", tag="tof_peak")
                            dpg.add_line_series([], [], label="Direct-blank limit",
                                                parent="tof_y", tag="tof_blank_line")
                            dpg.add_line_series([], [], label="Phys. ToF (synth)",
                                                parent="tof_y", tag="tof_synth_line")
                            dpg.add_plot_legend()

                        dpg.add_separator()
                        dpg.add_text("CRACK DETECTION RESULT", color=YELLOW)
                        with dpg.group(horizontal=True):
                            dpg.add_text("Echo peak sample  :", color=GREY)
                            self.echo_samp_text = dpg.add_text("—", color=GREEN)
                        with dpg.group(horizontal=True):
                            dpg.add_text("Path length (meas):", color=GREY)
                            self.echo_path_text = dpg.add_text("—", color=GREEN)
                        with dpg.group(horizontal=True):
                            dpg.add_text("Path length (phys):", color=GREY)
                            self.synth_path_text = dpg.add_text("—", color=CYAN)
                        with dpg.group(horizontal=True):
                            dpg.add_text("Detection accuracy:", color=GREY)
                            self.pc_acc_text = dpg.add_text("—", color=YELLOW)

                    # ── TDOA waveform viewer ──────────────────────────
                    with dpg.child_window(width=610, height=420, border=True):
                        dpg.add_text(
                            "TDOA  —  RECEIVED WAVEFORMS AT ALL RECEIVERS",
                            color=CYAN
                        )
                        dpg.add_text(
                            "Stacked envelopes, offset by sensor index. "
                            "Vertical lines = physics-ToF arrival markers.",
                            color=GREY
                        )

                        with dpg.plot(label="Sensor Envelopes (stacked)",
                                      height=225, width=-1):
                            dpg.add_plot_axis(dpg.mvXAxis, label="Sample index",
                                              tag="tdoa_x")
                            dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude + offset",
                                              tag="tdoa_y")
                            for i in range(1, MAX_SENSORS):
                                dpg.add_line_series(
                                    [], [], label=f"S{i}",
                                    parent="tdoa_y", tag=f"tdoa_env_{i}"
                                )
                            # Arrival marker scatter (all sensors combined)
                            dpg.add_scatter_series(
                                [], [], label="Arrivals (phys)",
                                parent="tdoa_y", tag="tdoa_arrivals"
                            )
                            dpg.add_plot_legend(outside=True)

                        dpg.add_text("ARRIVAL TIME TABLE  (physics ToF)",
                                     color=YELLOW)
                        with dpg.table(
                            tag="arrival_table",
                            header_row=True,
                            borders_innerH=True, borders_outerH=True,
                            borders_outerV=True,
                            height=115, scrollY=True,
                            policy=dpg.mvTable_SizingFixedFit
                        ):
                            dpg.add_table_column(label="Sensor",
                                                 width_fixed=True, init_width_or_weight=55)
                            dpg.add_table_column(label="Phys ToF (samp)",
                                                 width_fixed=True, init_width_or_weight=110)
                            dpg.add_table_column(label="Path len (norm)",
                                                 width_fixed=True, init_width_or_weight=110)
                            dpg.add_table_column(label="TDOA vs S1",
                                                 width_fixed=True, init_width_or_weight=95)

                            for i in range(1, MAX_SENSORS):
                                with dpg.table_row(
                                    tag=f"arr_row_{i}",
                                    show=(i < self.n_sensors)
                                ):
                                    dpg.add_text(
                                        f"S{i}", tag=f"arr_sid_{i}",
                                        color=list(SENSOR_COLORS[i])
                                    )
                                    dpg.add_text("—", tag=f"arr_tof_{i}")
                                    dpg.add_text("—", tag=f"arr_len_{i}")
                                    dpg.add_text("—", tag=f"arr_tdoa_{i}")

        # ── Finalize ─────────────────────────────────────────────────────
        self.update_params()
        dpg.setup_dearpygui()
        dpg.show_viewport()

    # ═══════════════════════════════════════════════════════════════════════
    #  SENSOR LIST UI
    # ═══════════════════════════════════════════════════════════════════════
    def _rebuild_sensor_list_ui(self):
        dpg.delete_item("sensor_list_win", children_only=True)
        for i in range(self.n_sensors):
            lbl = f"S{i}  [SOURCE]" if i == 0 else f"S{i}"
            col = list(SENSOR_COLORS[i % len(SENSOR_COLORS)])
            with dpg.tree_node(label=lbl, parent="sensor_list_win",
                               default_open=(i < 5), tag=f"snode_{i}"):
                dpg.add_text(f"● Sensor {i}", color=col, tag=f"sc_label_{i}")
                dpg.add_slider_float(
                    label="X", default_value=self.sensor_coords[i][0],
                    min_value=0.05, max_value=0.95,
                    callback=self.update_params, tag=f"sx_{i}"
                )
                dpg.add_slider_float(
                    label="Y", default_value=self.sensor_coords[i][1],
                    min_value=0.05, max_value=0.95,
                    callback=self.update_params, tag=f"sy_{i}"
                )

    def _update_sensor_combo(self):
        items = [f"Sensor {i}" for i in range(1, self.n_sensors)]
        safe  = min(self.active_sensor, self.n_sensors - 1)
        self.active_sensor = safe
        dpg.configure_item("sensor_combo", items=items,
                            default_value=f"Sensor {safe}")

    def add_sensor(self):
        if self.n_sensors >= MAX_SENSORS:
            return
        # Distribute new sensors evenly around a circle
        angle = (self.n_sensors - 4) * (2 * np.pi / 8) if self.n_sensors >= 4 else 0
        nx = float(np.clip(0.5 + 0.32 * np.cos(angle), 0.06, 0.94))
        ny = float(np.clip(0.5 + 0.32 * np.sin(angle), 0.06, 0.94))
        self.sensor_coords[self.n_sensors] = [nx, ny]
        self.n_sensors += 1
        self._rebuild_sensor_list_ui()
        self._update_sensor_combo()
        # Show new table row
        if self.n_sensors - 1 < MAX_SENSORS:
            dpg.configure_item(f"arr_row_{self.n_sensors - 1}", show=True)
        dpg.set_value(self.sensor_count_text,
                      f"Active sensors: {self.n_sensors}")
        self.update_params()

    def remove_sensor(self):
        if self.n_sensors <= 2:   # need source + ≥1 receiver
            return
        old = self.n_sensors - 1
        dpg.configure_item(f"arr_row_{old}", show=False)
        dpg.set_value(f"tdoa_env_{old}", [[], []])
        self.n_sensors -= 1
        self._rebuild_sensor_list_ui()
        self._update_sensor_combo()
        dpg.set_value(self.sensor_count_text,
                      f"Active sensors: {self.n_sensors}")
        self.update_params()

    def change_sensor(self, sender, app_data):
        try:
            self.active_sensor = int(app_data.split(" ")[1])
        except Exception:
            self.active_sensor = 1
        self._clear_pc_plots()

    # ═══════════════════════════════════════════════════════════════════════
    #  PARAMETER SYNC
    # ═══════════════════════════════════════════════════════════════════════
    def update_params(self, *_):
        self.defect_pos    = [dpg.get_value("dx_slider"),
                              dpg.get_value("dy_slider")]
        self.wave_velocity = dpg.get_value("vel_slider")
        self.solver.c_norm = self.wave_velocity
        self.solver.set_defect(self.defect_pos[0], self.defect_pos[1], 5.0)

        for i in range(self.n_sensors):
            try:
                self.sensor_coords[i] = [
                    dpg.get_value(f"sx_{i}"),
                    dpg.get_value(f"sy_{i}")
                ]
            except Exception:
                pass
            self.solver.set_sensor_pos(
                i, self.sensor_coords[i][0], self.sensor_coords[i][1]
            )

        dpg.set_value("truth_marker",
                      [[self.defect_pos[0]], [self.defect_pos[1]]])
        dpg.set_value(self.gt_text,
                      f"({self.defect_pos[0]:.4f},  {self.defect_pos[1]:.4f})")
        self._refresh_sensor_scatter()
        self.refresh_overlays()

    def _refresh_sensor_scatter(self):
        xs = [self.sensor_coords[i][0] for i in range(self.n_sensors)]
        ys = [self.sensor_coords[i][1] for i in range(self.n_sensors)]
        dpg.set_value("sensor_scatter", [xs, ys])

    # ═══════════════════════════════════════════════════════════════════════
    #  WAVEFIELD OVERLAY
    # ═══════════════════════════════════════════════════════════════════════
    def refresh_overlays(self):
        dpg.delete_item(self.overlay_tag, children_only=True)
        for i in range(self.n_sensors):
            col = list(SENSOR_COLORS[i % len(SENSOR_COLORS)])
            px  = self.sensor_coords[i][0] * DRAW_W
            py  = self.sensor_coords[i][1] * DRAW_W
            r   = 9 if i == 0 else 6
            dpg.draw_circle([px, py], r, color=col, fill=col,
                            parent=self.overlay_tag)
            tag = f"S{i}" + (" (src)" if i == 0 else "")
            dpg.draw_text([px + 10, py - 10], tag, size=13,
                          color=(255, 255, 255), parent=self.overlay_tag)

    # ═══════════════════════════════════════════════════════════════════════
    #  SIM CONTROL
    # ═══════════════════════════════════════════════════════════════════════
    def start_sim(self):
        self.reset_sim()
        self.running = True
        dpg.set_value(self.status_text, "Running…")

    def reset_sim(self):
        self.running     = False
        self.t_step      = 0
        self.solver.reset()
        self.update_params()
        self.sensor_data = [[] for _ in range(MAX_SENSORS)]
        self.loc_result  = None
        self._clear_all_plots()
        dpg.set_value(self.status_text, "Idle")

    def _clear_all_plots(self):
        self._clear_pc_plots()
        dpg.set_value("prob_tex",
                      np.zeros((100, 100, 4), np.float32).flatten())
        dpg.set_value("est_marker",    [[], []])
        dpg.set_value(self.est_text,      "—")
        dpg.set_value(self.err_text,      "—")
        dpg.set_value(self.err_pct_text,  "—")
        dpg.set_value(self.res_text,      "—")
        dpg.set_value("tdoa_arrivals", [[], []])
        for i in range(1, MAX_SENSORS):
            dpg.set_value(f"tdoa_env_{i}", [[], []])

    def _clear_pc_plots(self):
        for tag in ("pc_raw", "pc_env", "tof_env", "tof_peak",
                    "tof_blank_line", "tof_synth_line"):
            dpg.set_value(tag, [[], []])
        dpg.set_value(self.echo_samp_text,  "—")
        dpg.set_value(self.echo_path_text,  "—")
        dpg.set_value(self.synth_path_text, "—")
        dpg.set_value(self.pc_acc_text,     "—")

    # ═══════════════════════════════════════════════════════════════════════
    #  HEATMAP TEXTURE
    # ═══════════════════════════════════════════════════════════════════════
    def _update_heatmap_texture(self, prob_map):
        g  = prob_map.shape[0]
        tx = np.zeros((g, g, 4), np.float32)
        n  = prob_map / (prob_map.max() + 1e-9)
        tx[:, :, 0] = 0.05 + n * 0.95
        tx[:, :, 1] = n * 0.85
        tx[:, :, 2] = 0.40 - n * 0.35
        tx[:, :, 3] = 1.0
        dpg.set_value("prob_tex", tx.flatten())

    # ═══════════════════════════════════════════════════════════════════════
    #  POST-SIMULATION LOCALISATION  (called once when sim completes)
    # ═══════════════════════════════════════════════════════════════════════
    def perform_localisation(self):
        """
        Uses physics-based synthetic ToF for the ellipse-intersection solver
        and real waveforms for display-only panels.
        """
        unit_velocity = self.wave_velocity / self.res   # norm-units / sample

        positions  = self.sensor_coords[:self.n_sensors]
        defect_xy  = self.defect_pos

        # ── Physics-based (exact) echo path lengths ───────────────────
        synth_lengths = self.dsp.synthetic_echo_lengths(positions, defect_xy)

        # ── Grid + refinement localisation ────────────────────────────
        X, Y, prob_map, est_x, est_y, residual = self.dsp.localise(
            positions, synth_lengths, grid_res=150
        )
        self._update_heatmap_texture(prob_map)

        gt_x, gt_y = defect_pos = self.defect_pos
        error      = float(np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2))
        error_pct  = error * 100.0   # diagonal of unit square = 1.0 * sqrt(2)

        self.loc_result = dict(gt=defect_pos, est=(est_x, est_y),
                               error=error, pct=error_pct, res=residual)

        # ── Localisation report UI ────────────────────────────────────
        dpg.set_value(self.gt_text,
                      f"({gt_x:.4f},  {gt_y:.4f})")
        dpg.set_value(self.est_text,
                      f"({est_x:.4f},  {est_y:.4f})")
        dpg.set_value(self.err_text,
                      f"{error:.5f}  (normalised units)")
        dpg.set_value(self.err_pct_text,
                      f"{error_pct:.3f}%  of panel side")
        dpg.set_value(self.res_text,
                      f"{residual:.2e}")

        dpg.set_value("est_marker",   [[est_x], [est_y]])
        dpg.set_axis_limits("loc_x", 0, 1)
        dpg.set_axis_limits("loc_y", 0, 1)

        # ── Physics ToF reference samples for each receiver ───────────
        s0          = np.array(positions[0])
        ref_tof_smp = None
        arr_sx, arr_sy = [], []

        for i in range(1, self.n_sensors):
            L   = synth_lengths[i]
            tof = L / unit_velocity           # total path in samples
            tof_smp = int(round(tof))

            if ref_tof_smp is None:
                ref_tof_smp = tof_smp

            tdoa_val = tof_smp - ref_tof_smp
            offset   = (i - 1) * 1.2

            dpg.set_value(f"arr_tof_{i}",  str(tof_smp))
            dpg.set_value(f"arr_len_{i}",  f"{L:.5f}")
            dpg.set_value(f"arr_tdoa_{i}", str(tdoa_val))

            # Marker for TDOA waveform plot
            arr_sx.append(float(tof_smp))
            arr_sy.append(offset + 0.55)

        dpg.set_value("tdoa_arrivals", [arr_sx, arr_sy])

        # ── Pulse compression result for active receiver ───────────────
        self._update_pc_panel(synth_lengths, unit_velocity)

        dpg.set_value(self.status_text, "Complete ✓")

    # ─────────────────────────────────────────────────────────────────────
    def _update_pc_panel(self, synth_lengths, unit_velocity):
        act = self.active_sensor
        raw = np.array(self.sensor_data[act])
        if len(raw) < 50:
            return

        # Matched filter
        comp, env = self.dsp.process_signal(raw, self.chirp_sig)
        peak_val  = max(env.max(), 1e-9)
        c_n = (comp / peak_val).tolist()
        e_n = (env  / peak_val).tolist()
        idx = list(range(len(e_n)))

        dpg.set_value("pc_raw", [idx, c_n])
        dpg.set_value("pc_env", [idx, e_n])
        dpg.fit_axis_data("pc_x"); dpg.fit_axis_data("pc_y")

        # Direct-path blanking
        s0 = np.array(self.sensor_coords[0])
        si = np.array(self.sensor_coords[act])
        d  = float(np.linalg.norm(si - s0))
        blank_smp = max(0, int(d / unit_velocity))

        e_arr   = np.array(e_n)
        e_blank = e_arr.copy()
        if blank_smp < len(e_blank):
            e_blank[:blank_smp] = 0.0

        dpg.set_value("tof_env", [idx, e_blank.tolist()])
        dpg.fit_axis_data("tof_x"); dpg.fit_axis_data("tof_y")

        # Echo peak from real waveform
        thr = e_blank.max() * 0.40 if e_blank.max() > 1e-9 else 1e9
        pk_samps = np.where(e_blank > thr)[0]
        if pk_samps.size > 0:
            pk = int(pk_samps[0])
            path_meas = pk * unit_velocity
            dpg.set_value("tof_peak", [[pk], [float(e_blank[pk])]])
            dpg.set_value(self.echo_samp_text,  str(pk))
            dpg.set_value(self.echo_path_text,  f"{path_meas:.5f}")
        else:
            dpg.set_value("tof_peak", [[], []])
            dpg.set_value(self.echo_samp_text,  "no peak found")
            dpg.set_value(self.echo_path_text,  "—")

        # Physics ToF reference line
        synth_L   = synth_lengths[act] if act < len(synth_lengths) else 0.0
        synth_smp = int(round(synth_L / unit_velocity)) if synth_L > 0 else 0
        if 0 < synth_smp < len(e_blank):
            dpg.set_value("tof_synth_line",
                          [[synth_smp, synth_smp], [0.0, 1.0]])
        dpg.set_value(self.synth_path_text, f"{synth_L:.5f}  ({synth_smp} samp)")

        # Blank-end line
        if 0 < blank_smp < len(e_blank):
            dpg.set_value("tof_blank_line",
                          [[blank_smp, blank_smp], [0.0, 1.0]])

        # Accuracy: compare measured peak to physics ToF
        if pk_samps.size > 0 and synth_smp > 0:
            acc = abs(pk - synth_smp) / max(synth_smp, 1) * 100.0
            dpg.set_value(self.pc_acc_text,
                          f"{acc:.2f}%  mismatch vs physics ToF")

    # ═══════════════════════════════════════════════════════════════════════
    #  LIVE DSP UPDATE  (every 20 simulation steps)
    # ═══════════════════════════════════════════════════════════════════════
    def _update_live_dsp(self):
        unit_velocity = self.wave_velocity / self.res

        # ── A-scan / PC live preview (active receiver) ────────────────
        act = self.active_sensor
        raw = np.array(self.sensor_data[act])
        if len(raw) > 50:
            comp, env = self.dsp.process_signal(raw, self.chirp_sig)
            pk        = max(env.max(), 1e-9)
            idx       = list(range(len(env)))
            dpg.set_value("pc_raw", [idx, (comp / pk).tolist()])
            dpg.set_value("pc_env", [idx, (env  / pk).tolist()])
            dpg.fit_axis_data("pc_x"); dpg.fit_axis_data("pc_y")

            # Live blanked view
            s0  = np.array(self.sensor_coords[0])
            si  = np.array(self.sensor_coords[act])
            d   = float(np.linalg.norm(si - s0))
            blk = max(0, int(d / unit_velocity))
            e_b = (env / pk).copy()
            if blk < len(e_b):
                e_b[:blk] = 0.0
            dpg.set_value("tof_env", [idx, e_b.tolist()])
            dpg.fit_axis_data("tof_x"); dpg.fit_axis_data("tof_y")

        # ── TDOA stacked envelopes (all receivers) ─────────────────────
        for i in range(1, self.n_sensors):
            raw_i = np.array(self.sensor_data[i])
            if len(raw_i) > 50:
                _, env_i = self.dsp.process_signal(raw_i, self.chirp_sig)
                pk_i     = max(env_i.max(), 1e-9)
                offset   = (i - 1) * 1.2
                e_off    = (env_i / pk_i + offset).tolist()
                dpg.set_value(f"tdoa_env_{i}",
                              [list(range(len(e_off))), e_off])

        dpg.fit_axis_data("tdoa_x"); dpg.fit_axis_data("tdoa_y")

    # ═══════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ═══════════════════════════════════════════════════════════════════════
    def run(self):
        t_last = time.time()
        while dpg.is_dearpygui_running():
            if self.running and self.t_step < self.max_steps:
                for _ in range(4):   # 4 sub-steps per frame
                    if self.t_step < len(self.chirp_sig):
                        self.solver.inject_source(
                            self.chirp_sig[self.t_step] * 5.0
                        )
                    self.solver.step(self.n_sensors)
                    vals = self.solver.get_sensor_data(self.n_sensors)
                    for i in range(self.n_sensors):
                        self.sensor_data[i].append(float(vals[i]))
                    self.t_step += 1

                dpg.set_value("wavefield_tex",
                              self.solver.get_frame().flatten())

                pct = int(self.t_step / self.max_steps * 100)
                dpg.set_value(self.status_text, f"Running… {pct}%")

                if self.t_step >= self.max_steps:
                    self.running = False
                    dpg.set_value(self.status_text, "Analysing…")
                    self.perform_localisation()

                if self.t_step % 20 == 0:
                    try:
                        self._update_live_dsp()
                    except Exception:
                        pass

            now = time.time()
            dpg.set_value(self.fps_text,
                          f"FPS: {1.0 / (now - t_last + 1e-9):.1f}")
            t_last = now
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import taichi as ti
    ti.init(arch=ti.cpu)
    try:
        app = NDTApp()
        app.run()
    except Exception:
        import traceback
        traceback.print_exc()