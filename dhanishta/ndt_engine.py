"""
ndt_engine.py  —  Taichi FDTD wave solver
ti.init() must be called by the main application before instantiating.
"""
import taichi as ti
import numpy as np

MAX_SENSORS = 16


@ti.data_oriented
class TaichiWaveSolver:
    def __init__(self, res=256, c_norm=0.45, damping=0.992):
        self.res     = res
        self.nx      = res
        self.ny      = res
        self.c_norm  = c_norm
        self.damping = damping

        # Three-buffer FDTD fields
        self.u      = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.u_prev = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.u_next = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.mask   = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))

        # RGBA display buffer
        self.pixels = ti.Vector.field(4, dtype=ti.f32, shape=(self.nx, self.ny))

        # Sensor storage — always MAX_SENSORS slots
        self.sensors_pos  = ti.Vector.field(2, dtype=ti.i32, shape=MAX_SENSORS)
        self.sensor_vals  = ti.field(dtype=ti.f32, shape=MAX_SENSORS)

        # Default four corners
        init = np.zeros((MAX_SENSORS, 2), dtype=np.int32)
        init[0] = [20,        20]
        init[1] = [res - 20,  20]
        init[2] = [20,        res - 20]
        init[3] = [res - 20,  res - 20]
        for k in range(4, MAX_SENSORS):
            init[k] = [res // 2, res // 2]
        self.sensors_pos.from_numpy(init)

        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.u.fill(0.0)
        self.u_prev.fill(0.0)
        self.u_next.fill(0.0)
        self.mask.fill(1.0)
        self.sensor_vals.fill(0.0)

    def step(self, n_active: int):
        self.compute_next()
        self.update_buffers()
        self.capture_sensors(n_active)

    def get_sensor_data(self, n_active: int):
        return self.sensor_vals.to_numpy()[:n_active].copy()

    def get_frame(self):
        self.update_pixels()
        return self.pixels.to_numpy()

    # ------------------------------------------------------------------ #
    @ti.kernel
    def capture_sensors(self, n: ti.i32):
        for i in range(n):
            sx = self.sensors_pos[i][0]
            sy = self.sensors_pos[i][1]
            self.sensor_vals[i] = self.u[sx, sy]

    @ti.kernel
    def set_defect(self, cx: ti.f32, cy: ti.f32, radius: ti.f32):
        gcx = cx * self.res
        gcy = cy * self.res
        for i, j in self.mask:
            dist = ti.sqrt(
                (ti.cast(i, ti.f32) - gcy) ** 2 +
                (ti.cast(j, ti.f32) - gcx) ** 2
            )
            self.mask[i, j] = 0.0 if dist <= radius else 1.0

    @ti.kernel
    def compute_next(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            lap = (self.u[i+1, j] + self.u[i-1, j] +
                   self.u[i, j+1] + self.u[i, j-1] -
                   4.0 * self.u[i, j])
            uv  = 2.0 * self.u[i, j] - self.u_prev[i, j] + (self.c_norm ** 2) * lap

            ed  = ti.min(ti.min(i, self.nx - 1 - i), ti.min(j, self.ny - 1 - j))
            dmp = self.damping
            if ed < 20:
                dmp *= ed / 20.0

            self.u_next[i, j] = uv * dmp * self.mask[i, j]

    @ti.kernel
    def update_buffers(self):
        for i, j in self.u:
            self.u_prev[i, j] = self.u[i, j]
            self.u[i, j]      = self.u_next[i, j]

    @ti.kernel
    def set_sensor_pos(self, idx: ti.i32, x: ti.f32, y: ti.f32):
        self.sensors_pos[idx] = ti.Vector([
            ti.cast(x * self.res, ti.i32),
            ti.cast(y * self.res, ti.i32)
        ])

    @ti.kernel
    def inject_source(self, val: ti.f32):
        sx = self.sensors_pos[0][0]
        sy = self.sensors_pos[0][1]
        self.u[sx, sy] += val

    @ti.kernel
    def update_pixels(self):
        for i, j in self.pixels:
            val = self.u[i, j]
            v   = ti.max(-1.0, ti.min(1.0, val * 5.0))

            r = ti.cast(0.05, ti.f32)
            g = ti.cast(0.05, ti.f32)
            b = ti.cast(0.10, ti.f32)

            if v > 0.0:
                r = 0.05 + v * 0.95
                g = 0.05 + v * 0.25
                b = 0.10
            else:
                nv = -v
                r  = 0.05
                g  = 0.05 + nv * 0.45
                b  = 0.10 + nv * 0.90

            if self.mask[i, j] == 0.0:
                self.pixels[i, j] = ti.Vector([1.0, 1.0, 1.0, 1.0])
            else:
                self.pixels[i, j] = ti.Vector([r, g, b, 1.0])