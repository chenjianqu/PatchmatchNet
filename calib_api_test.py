import calib_manager.manager
import numpy as np


# 获取外参
calib = calib_manager.CalibrationManager(vehicle="A12-001", date=20230718)
sensor_graph = calib.sensor_graph

cameras = calib.get_cameras()[1]

T_vc = sensor_graph['camera-1']['body-2']  # v=body-2 , c=camera

print(T_vc)

width = 3
height = 2

x,y = np.meshgrid(np.arange(0, width), np.arange(0, height))
x = x.reshape([-1])
y = y.reshape([-1])

x_reshape = x.reshape([height,width])
y_reshape = y.reshape([height,width])

print(x_reshape)
print(y_reshape)

#xyz = np.vstack([x, y, np.ones([1, width*height], dtype=np.float32)])  # [3, H*W]
#xyz = np.vstack([x, y])  # [3, H*W]
#xyz = xyz.transpose()
#xyz = xyz.reshape([height,width,2])

#print(xyz)