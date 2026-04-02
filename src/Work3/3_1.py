import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# GUI 绘制数据缓冲池
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_SEGMENTS + 1))

# 原版De Casteljau算法（保留）
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i+1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

# 修复版：三次均匀B样条曲线
def cubic_bspline_curve(control_points):
    n = len(control_points)
    if n < 4:
        return np.zeros((1, 2), dtype=np.float32)
    
    # 三次B样条基矩阵
    M = np.array([
        [-1,  3, -3,  1],
        [ 3, -6,  3,  0],
        [-3,  0,  3,  0],
        [ 1,  4,  1,  0]
    ], dtype=np.float32) / 6.0
    
    curve = []
    segments = n - 3
    samples_per_seg = NUM_SEGMENTS // segments if segments > 0 else 0
    
    for i in range(segments):
        # 取4个连续控制点
        p0, p1, p2, p3 = control_points[i:i+4]
        P = np.array([p0, p1, p2, p3], dtype=np.float32)
        # 分段采样
        for j in range(samples_per_seg):
            t = j / samples_per_seg
            T = np.array([t**3, t**2, t, 1], dtype=np.float32)
            pt = T @ M @ P
            curve.append(pt)
    return np.array(curve, dtype=np.float32)

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

# 反走样GPU绘制内核
@ti.kernel
def draw_curve_antialiased(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        fx = pt[0] * WIDTH
        fy = pt[1] * HEIGHT

        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                x = int(fx + dx)
                y = int(fy + dy)
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    dist = ti.sqrt((fx - x)**2 + (fy - y)**2)
                    alpha = ti.exp(-dist * dist * 1.0)
                    pixels[x, y] = ti.max(pixels[x, y], ti.Vector([0.0, 0.8, 0.2]) * alpha)

def main():
    window = ti.ui.Window("Bezier / B-Spline (AntiAliased)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points = []
    use_bspline = False  # 局部变量，直接赋值

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)
            elif e.key == 'c':
                control_points = []
                print("Canvas cleared.")
            elif e.key == 'b':
                # 直接赋值，去掉nonlocal
                use_bspline = not use_bspline
                print(f"Switched to {'B-Spline' if use_bspline else 'Bezier'} mode")

        clear_pixels()
        current_count = len(control_points)

        if current_count >= 2:
            if not use_bspline:
                # 贝塞尔曲线
                curve_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
                for t_int in range(NUM_SEGMENTS + 1):
                    t = t_int / NUM_SEGMENTS
                    curve_np[t_int] = de_casteljau(control_points, t)
            else:
                # B样条曲线
                curve_np = cubic_bspline_curve(control_points)

            curve_points_field.from_numpy(curve_np)
            draw_curve_antialiased(len(curve_np))

        canvas.set_image(pixels)

        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.006, color=(1.0, 0.0, 0.0))

            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i+1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5,0.5,0.5))

        window.show()

if __name__ == '__main__':
    main()