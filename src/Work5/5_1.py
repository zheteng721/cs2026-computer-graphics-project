import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())

MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

@ti.func
def normalize(v):
    return v / v.norm(1e-6)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    cosi = ti.max(-1.0, ti.min(1.0, I.dot(N)))
    etai = 1.0
    etat = eta
    n = N
    if cosi < 0:
        cosi = -cosi
    else:
        etai, etat = etat, etai
        n = -N
    eta_ratio = etai / etat
    k = 1.0 - eta_ratio**2 * (1.0 - cosi**2)
    R = ti.Vector([0.0, 0.0, 0.0])
    if k < 0:
        R = reflect(I, n)
    else:
        R = eta_ratio * I + (eta_ratio * cosi - ti.sqrt(k)) * n
    return R

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) * 0.5
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, y):
    t = -1.0
    if abs(rd.y) > 1e-6:
        t = (y - ro.y) / rd.y
    return t, ti.Vector([0.0, 1.0, 0.0])

@ti.func
def scene_intersect(ro, rd):
    min_t = 1e9
    n = ti.Vector([0.0, 0.0, 0.0])
    col = ti.Vector([0.0, 0.0, 0.0])
    mat = MAT_DIFFUSE

    t0, n0 = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t0 < min_t:
        min_t = t0
        n = n0
        col = ti.Vector([1.0, 1.0, 1.0])
        mat = MAT_GLASS

    t1, n1 = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t1 < min_t:
        min_t = t1
        n = n1
        col = ti.Vector([0.9, 0.9, 0.9])
        mat = MAT_MIRROR

    t2, n2 = intersect_plane(ro, rd, -1.0)
    if 0 < t2 < min_t:
        min_t = t2
        n = n2
        mat = MAT_DIFFUSE
        p = ro + rd * t2
        f = ti.floor(p.x * 2) + ti.floor(p.z * 2)
        col = ti.Vector([0.3, 0.3, 0.3]) if int(f) % 2 == 0 else ti.Vector([0.8, 0.8, 0.8])

    return min_t, n, col, mat

@ti.func
def trace(ro, rd, light, max_b):
    color = ti.Vector([0.0, 0.0, 0.0])
    thru = ti.Vector([1.0, 1.0, 1.0])
    ray_ro = ro
    ray_rd = rd
    inside = False

    for _ in range(max_b):
        t, N, col, mat = scene_intersect(ray_ro, ray_rd)
        if t > 1e8:
            color += thru * ti.Vector([0.05, 0.15, 0.2])
            break

        p = ray_ro + ray_rd * t

        if mat == MAT_GLASS:
            ray_ro = p + ray_rd * 1e-4
            if not inside:
                ray_rd = normalize(refract(ray_rd, N, 1.5))
                inside = True
            else:
                ray_rd = normalize(refract(ray_rd, -N, 1/1.5))
                inside = False
            thru *= col

        elif mat == MAT_MIRROR:
            ray_ro = p + N * 1e-4
            ray_rd = normalize(reflect(ray_rd, N))
            thru *= col

        elif mat == MAT_DIFFUSE:
            L = normalize(light - p)
            shadow_t, _, _, _ = scene_intersect(p + N*1e-4, L)
            shadow = 1.0 if (0 < shadow_t < (light-p).norm()) else 0.0
            amb = 0.2 * col
            diff = 0.8 * col * ti.max(0.0, N.dot(L)) * (1.0 - shadow)
            color += thru * (amb + diff)
            break
    return color

@ti.kernel
def render(aa: ti.i32):
    light = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    ro = ti.Vector([0.0, 1.0, 5.0])
    for i, j in pixels:
        c = ti.Vector([0.0, 0.0, 0.0])
        for _ in range(aa):
            rx = ti.random() - 0.5
            ry = ti.random() - 0.5
            u = (i + rx - res_x/2) / res_y * 2
            v = (j + ry - res_y/2) / res_y * 2
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))
            c += trace(ro, rd, light, max_bounces[None])
        avg_color = c / aa
        # 分量级裁剪，修复make_matrix错误
        r = avg_color[0]
        g = avg_color[1]
        b = avg_color[2]
        if r < 0.0: r = 0.0
        if r > 1.0: r = 1.0
        if g < 0.0: g = 0.0
        if g > 1.0: g = 1.0
        if b < 0.0: b = 0.0
        if b > 1.0: b = 1.0
        pixels[i,j] = ti.Vector([r, g, b])

def main():
    window = ti.ui.Window("Glass + AA", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    light_pos_x[None] = 2
    light_pos_y[None] = 4
    light_pos_z[None] = 3
    max_bounces[None] = 3

    while window.running:
        render(4)
        canvas.set_image(pixels)
        with gui.sub_window("Ctrl", 0.75,0.05,0.23,0.22):
            light_pos_x[None] = gui.slider_float("Light X", light_pos_x[None],-5,5)
            light_pos_y[None] = gui.slider_float("Light Y", light_pos_y[None],1,8)
            light_pos_z[None] = gui.slider_float("Light Z", light_pos_z[None],-5,5)
            max_bounces[None] = gui.slider_int("Bounce", max_bounces[None],1,5)
        window.show()

if __name__ == "__main__":
    main()