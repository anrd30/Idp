import taichi as ti
import numpy as np

# ti.init() should be called by the main application

@ti.data_oriented
class TaichiWaveSolver:
    def __init__(self, res=256, c_norm=0.45, damping=0.992):
        self.res = res
        self.nx = res
        self.ny = res
        
        # Simulation Fields (Three-buffer FDTD)
        self.u = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.u_prev = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.u_next = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.mask = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        
        # Display/Buffer Field
        self.pixels = ti.Vector.field(4, dtype=ti.f32, shape=(self.nx, self.ny))
        
        # Physics Params
        self.c_norm = c_norm
        self.damping = damping
        
        # Sensors
        self.sensor_count = 4
        self.sensors_pos = ti.Vector.field(2, dtype=ti.i32, shape=self.sensor_count)
        self.sensors_pos.from_numpy(np.array([
            [20, 20], [self.res-20, 20], [20, self.res-20], [self.res-20, self.res-20]
        ], dtype=np.int32))
        self.sensor_vals = ti.field(dtype=ti.f32, shape=self.sensor_count)
        
        self.reset()

    def reset(self):
        self.u.fill(0.0)
        self.u_prev.fill(0.0)
        self.u_next.fill(0.0)
        self.mask.fill(1.0)
        self.sensor_vals.fill(0.0)

    @ti.kernel
    def capture_sensors(self):
        for i in range(self.sensor_count):
            sx, sy = self.sensors_pos[i]
            self.sensor_vals[i] = self.u[sx, sy]

    @ti.kernel
    def set_defect(self, cx: ti.f32, cy: ti.f32, radius: ti.f32):
        gcx, gcy = cx * self.res, cy * self.res
        for i, j in self.mask:
            dist = ti.sqrt((ti.cast(i, ti.f32) - gcy)**2 + (ti.cast(j, ti.f32) - gcx)**2)
            if dist <= radius:
                self.mask[i, j] = 0.0
            else:
                self.mask[i, j] = 1.0

    @ti.kernel
    def compute_next(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            laplacian = (
                self.u[i+1, j] + self.u[i-1, j] +
                self.u[i, j+1] + self.u[i, j-1] -
                4.0 * self.u[i, j]
            )
            
            u_val = 2.0 * self.u[i, j] - self.u_prev[i, j] + (self.c_norm**2) * laplacian
            
            # Absorbing Boundary condition (simplified)
            edge_dist = ti.min(ti.min(i, self.nx-1-i), ti.min(j, self.ny-1-j))
            local_damping = self.damping
            if edge_dist < 20:
                local_damping *= (edge_dist / 20.0)
            
            self.u_next[i, j] = u_val * local_damping * self.mask[i, j]

    @ti.kernel
    def update_buffers(self):
        for i, j in self.u:
            self.u_prev[i, j] = self.u[i, j]
            self.u[i, j] = self.u_next[i, j]

    @ti.kernel
    def set_sensor_pos(self, idx: ti.i32, x: ti.f32, y: ti.f32):
        self.sensors_pos[idx] = ti.Vector([ti.cast(x * self.res, ti.i32), ti.cast(y * self.res, ti.i32)])

    def step(self):
        self.compute_next()
        self.update_buffers()
        self.capture_sensors()

    @ti.kernel
    def inject_source(self, val: ti.f32):
        sx, sy = self.sensors_pos[0]
        self.u[sx, sy] += val

    @ti.kernel
    def update_pixels(self):
        for i, j in self.pixels:
            val = self.u[i, j]
            # Diverging Professional Colormap (Blue-Grey-Red)
            # Normalize and clamp for stability in visualization
            v = ti.max(-1.0, ti.min(1.0, val * 5.0))
            
            r, g, b = 0.1, 0.1, 0.1 # Dark grey background
            if v > 0:
                r = 0.1 + v * 0.9
                g = 0.1 + v * 0.2
                b = 0.1
            else:
                r = 0.1
                g = 0.1 + (-v) * 0.4
                b = 0.1 + (-v) * 0.9
            
            if self.mask[i, j] == 0.0:
                # Highlight defect
                self.pixels[i, j] = ti.Vector([1.0, 1.0, 1.0, 1.0])
            else:
                self.pixels[i, j] = ti.Vector([r, g, b, 1.0])

    def get_sensor_data(self):
        return self.sensor_vals.to_numpy()

    def get_frame(self):
        self.update_pixels()
        return self.pixels.to_numpy()
