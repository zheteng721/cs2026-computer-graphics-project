import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 立方体顶点
vertices = np.array([
    [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
    [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
])

# 立方体的边
edges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
]

# 旋转矩阵：简单插值，不用scipy
def rotate_matrix(t):
    ang = t * np.pi/2
    c, s = np.cos(ang), np.sin(ang)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0,  1]
    ])

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)

lines = [ax.plot([],[],[], 'r-', lw=2)[0] for _ in edges]

def update(frame):
    t = frame / 50
    R = rotate_matrix(t)
    v_rot = vertices @ R.T
    
    for i, (a,b) in enumerate(edges):
        x = [v_rot[a,0], v_rot[b,0]]
        y = [v_rot[a,1], v_rot[b,1]]
        z = [v_rot[a,2], v_rot[b,2]]
        lines[i].set_data(x,y)
        lines[i].set_3d_properties(z)
    return lines

ani = animation.FuncAnimation(fig, update, frames=50, interval=30, blit=True)
plt.show()