import numpy as np
import visualize_utils
import open3d as o3d

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='depth pcd')
# vis.toggle_full_screen() #全屏
# 设置
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])  # 背景
opt.point_size = 1  # 点云大小


def add_coordinate_frame(pose):
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    axis_pcd = axis_pcd.transform(pose)
    vis.add_geometry(axis_pcd)


def add_points(points):
    '''
    points:[3,N]
    '''
    assert points.shape[0] == 3
    points_t = points.transpose()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_t)
    vis.add_geometry(point_cloud)


def add_plane(pose, radius_plane=20):
    # 绘制一个平面
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=radius_plane,
                                                    height=radius_plane,
                                                    depth=0.001)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.5, 0.5, 0.5])

    mesh_box = mesh_box.transform(pose)
    vis.add_geometry(mesh_box)


def add_line(polygon_points, color=None):
    # polygon_points: [N,3]
    if color is None:
        color = [1, 0, 0]
    N = polygon_points.shape[0]
    lines = []
    for i in range(1, N - 1):
        lines.append([i - 1, i])  # 连接的顺序
    lines.append([N - 1, 0])

    colors = [color for i in range(len(lines))]
    # 添加顶点，点云
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    points_pcd.paint_uniform_color([0, 0.3, 0])  # 点云颜色

    # 绘制线条
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(colors)  # 线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    vis.add_geometry(lines_pcd)
    vis.add_geometry(points_pcd)

#可视化坐标原点
add_coordinate_frame(np.eye(4))

# 地面平面
radius = 20
T_plane = np.eye(4)
T_plane[0, 3] = -radius / 2.
T_plane[1, 3] = -radius / 2.
add_plane(T_plane, radius_plane=radius)


T_vc = np.array([[1.65671625e-02, -3.32327391e-02, 9.99310319e-01, 2.17537799e+00],
             [9.99862347e-01, 3.52411779e-04, 1.65880340e-02, 2.80948292e-02],
             [-9.03434534e-04, -9.99447578e-01, -3.32223260e-02, 1.33101139e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], np.float32)
T_cv = np.linalg.inv(T_vc)

K = np.array([[758.5029907226562, 0.0, 886.4910888671875],
              [0.0, 732.5841674804688, 275.8777770996094],
              [0.0, 0.0, 1.0]],
             np.float32)
K_inv = np.linalg.inv(K)

# 可视化相机坐标轴
add_coordinate_frame(T_vc)

#图像大小
width = 1696
height = 540

# 生成相机坐标系下的归一化坐标
x_cam, y_cam = np.meshgrid(np.arange(0, width), np.arange(0, height))
x_cam, y_cam = x_cam.reshape(-1), y_cam.reshape(-1)
p_cam_pixel = np.vstack([x_cam, y_cam, np.ones([1, width * height], dtype=np.float32)])  # [3, H*W]
p_cam_norm = np.matmul(K_inv, p_cam_pixel)  # [3, H*W]
p_cam_norm_h = np.vstack([p_cam_norm, np.ones_like(x_cam)])  # [4, H*W]
# 变换到car坐标系
p_car_norm = np.matmul(T_vc, p_cam_norm_h)[:3, :]  # [3,H*W]

add_points(p_car_norm)

# 相机光心
O_car_cam = (T_vc[:3, 3]).reshape([3, 1])  # [3,1]
# 计算射线的方向向量
v_car = p_car_norm - O_car_cam
v_car = v_car / np.linalg.norm(v_car, ord=2, axis=0)  # 归一化

# 平面法向量
N_plane = np.array([0, 0, 1], np.float32).reshape([3, 1])
# 计算k
temp_1 = np.dot(N_plane.transpose(), -O_car_cam)
k = temp_1 / np.matmul(v_car.transpose(), N_plane)  # [H*W,1]
# 计算射线和平面的交点
p_car_inter = O_car_cam + np.squeeze(k, axis=1) * v_car  # [3,H*W]
# 将点变换到相机坐标系
p_cam_inter = np.matmul(T_cv, np.vstack([p_car_inter, np.ones([1, width * height])]))[:3, :]  # [4,H*W]

#####可视化

# 可视化相机到归一化平面的射线
for i in range(0, width * height, 1000):
    line_0 = np.hstack([O_car_cam, p_car_norm[:, i].reshape([3, 1])])
    add_line(line_0.transpose())

# 可视化相机到地面的射线
for i in range(0, width * height, 1000):
    if k[i, 0] <= 0:  # 表明无交点
        continue
    p1 = p_car_inter[:, i].reshape([3, 1])
    dist = np.linalg.norm(O_car_cam - p1)
    if dist > 20:
        continue
    line_0 = np.hstack([O_car_cam, p1])
    add_line(line_0.transpose(), color=[1, 1, 1])

# 获得深度
depth = p_cam_inter[2]  # [1,H*W]
print(depth.shape)
depth = depth.reshape([height,width])

print(depth.max())
print(depth.min())
print(depth[depth>0].min())

print(depth)

vis.run()
vis.destroy_window()
