# import calib_manager.manager
# import numpy as np
# import torch
#
# # 获取外参
# calib = calib_manager.CalibrationManager(vehicle="A12-001", date=20230718)
# sensor_graph = calib.sensor_graph
#
# cameras = calib.get_cameras()[1]
#
# T_vc = sensor_graph['camera-1']['body-2']  # v=body-2 , c=camera
#
# print(T_vc)
#
# width = 3
# height = 2
#
# x,y = np.meshgrid(np.arange(0, width), np.arange(0, height))
# x = x.reshape([-1])
# y = y.reshape([-1])
#
# x_reshape = x.reshape([height,width])
# y_reshape = y.reshape([height,width])
#
# print(x_reshape)
# print(y_reshape)
#
# #xyz = np.vstack([x, y, np.ones([1, width*height], dtype=np.float32)])  # [3, H*W]
# #xyz = np.vstack([x, y])  # [3, H*W]
# #xyz = xyz.transpose()
# #xyz = xyz.reshape([height,width,2])
#
# #print(xyz)
#
#
# #索引取值测试
# # A = torch.zeros([5,4],dtype=float)
# # xy_arr = np.array([[1,2],[2,3]])
# #
# # xy_t = torch.from_numpy(xy_arr)
# # xy_value = torch.from_numpy(np.array([6,7],dtype=np.float64))
# # A[xy_t[0,:],xy_t[1,:]] = xy_value
# # print(A)
#
# #向量堆叠测试
# A=np.array([1,2,3])
# B=np.array([4,5,6])
# C=np.array([7,8,9])
# L=[A,B,C]
#
# D=np.array(L)
# print(D.shape)
#
# print(D)
#
import numpy as np

# import cv2
# import numpy as np
#
# mask = cv2.imread("/home/cjq/data/mvs/lidar/20221020_5.78km_2022-11-29-13-55-40/semantic_maps_ori/1669701514.876.png",-1)
# #print(mask)
# print(mask.shape)
#
# mask = mask.reshape([-1])
# mask_unique = np.unique(mask)
#
# print(mask_unique)

#
# T_wl = np.matrix([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
#
# xyz_offset = T_wl[:3, 3]
# print(xyz_offset)
#
# T_wl[:3, 3] = T_wl[:3, 3] - xyz_offset
# print(T_wl)

W=3
H=2
xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, H))
xx, yy = xx.reshape(-1), yy.reshape(-1)

print(xx)
print(yy)


