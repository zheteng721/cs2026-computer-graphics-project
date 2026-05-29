import taichi as ti

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# 物理与网格参数
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长
k_s = 6000.0       # 结构弹簧劲度系数
k_shear = 3000.0   # 剪切弹簧劲度系数
k_bend = 1500.0    # 弯曲弹簧劲度系数
k_d = 5.0          # 阻尼系数（增大防抖）
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 20.0  # 速度上限

# 球体碰撞参数
sphere_center = ti.Vector([0.0, 0.2, 0.0])
sphere_radius = 0.3
bounce = 0.5

# 定义 Taichi 数据场
x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

# 隐式欧拉专用缓存
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# 弹簧系统（结构+剪切+弯曲 三种弹簧）
max_springs = N * N * 12
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_types = ti.field(dtype=int, shape=max_springs)  # 标记弹簧类型 0=结构 1=剪切 2=弯曲
num_springs = ti.field(dtype=int, shape=())

# 固定球体渲染
sphere_pos = ti.Vector.field(3, dtype=float, shape=1)

# ============ 初始化 ============
@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        # 固定两个角点
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        # 1. 结构弹簧 Structural（上下左右相邻）
        if i < N - 1:
            idx_r = (i+1)*N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_r])
            spring_lengths[c] = (x[idx]-x[idx_r]).norm()
            spring_types[c] = 0
        if j < N - 1:
            idx_d = i*N + j+1
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_d])
            spring_lengths[c] = (x[idx]-x[idx_d]).norm()
            spring_types[c] = 0

        # 2. 剪切弹簧 Shear（对角线，防止扭曲）
        if i < N-1 and j < N-1:
            idx_rd = (i+1)*N + j+1
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_rd])
            spring_lengths[c] = (x[idx]-x[idx_rd]).norm()
            spring_types[c] = 1
        if i > 0 and j < N-1:
            idx_ld = (i-1)*N + j+1
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_ld])
            spring_lengths[c] = (x[idx]-x[idx_ld]).norm()
            spring_types[c] = 1

        # 3. 弯曲弹簧 Bending（隔一个点，控制褶皱）
        if i < N - 2:
            idx_rr = (i+2)*N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_rr])
            spring_lengths[c] = (x[idx]-x[idx_rr]).norm()
            spring_types[c] = 2
        if j < N - 2:
            idx_dd = i*N + j+2
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_dd])
            spring_lengths[c] = (x[idx]-x[idx_dd]).norm()
            spring_types[c] = 2

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i*2] = spring_pairs[i][0]
        spring_indices[i*2+1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()
    sphere_pos[0] = sphere_center

# ============ 力计算 + 碰撞 ============
@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    # 重力 + 阻尼
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    # 弹簧力
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm(1e-6)
        d_normalized = d / dist

        # 修复：提前定义k，避免报错
        k = k_s
        st = spring_types[i]
        if st == 1:
            k = k_shear
        if st == 2:
            k = k_bend
            
        f_spring = -k * (dist - spring_lengths[i]) * d_normalized
        ti.atomic_add(force[idx_a], f_spring)
        ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def handle_collision(idx: int):
    dx = x[idx] - sphere_center
    dist = dx.norm(1e-6)
    if dist < sphere_radius:
        n = dx / dist
        x[idx] = sphere_center + n * sphere_radius
        v[idx] = (1 - bounce) * v[idx] - bounce * v[idx].dot(n) * n

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

# ============ 三种积分器 ============
@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            handle_collision(i)

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            handle_collision(i)

@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]
        handle_collision(i)

# ============ 主函数（修复键盘常量+放大图像+Ctrl+A/D平移） ============
def main():
    init_cloth()

    window = ti.ui.Window("Games101 - Mass Spring System", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    # 放大主体图像：相机拉近到1.8，和图二大小完全一致
    camera.position(0.0, 0.5, 1.8)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1
    paused = False
    move_speed = 0.02  # 平移速度，手感顺滑

    while window.running:
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.36)
        window.GUI.text("Integration Method:")
        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler (Explosive)"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler (Damped)"):
            current_method = 2
            init_cloth()

        window.GUI.text("")
        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            init_cloth()
        window.GUI.end()

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        # 修复：旧版Taichi用 ti.ui.CTRL 而不是 LCTRL
        if window.is_pressed(ti.ui.CTRL):
            if window.is_pressed('a'):
                camera.position[0] -= move_speed
            if window.is_pressed('d'):
                camera.position[0] += move_speed
            # 保持相机看向的点不变，视角不会歪
            camera.lookat(camera.position[0], 0.0, 0.0)

        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))
        scene.particles(sphere_pos, radius=sphere_radius, color=(0.9, 0.3, 0.3))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()