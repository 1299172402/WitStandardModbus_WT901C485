import numpy as np
import math

class QuaternionConverter:
    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        """
        欧拉角转四元数 (ZYX顺序)
        """
        # 转换为弧度并取一半角度
        r = math.radians(roll) / 2
        p = math.radians(pitch) / 2
        y = math.radians(yaw) / 2
        
        # 计算四元数分量
        q_w = math.cos(r) * math.cos(p) * math.cos(y) + math.sin(r) * math.sin(p) * math.sin(y)
        q_x = math.sin(r) * math.cos(p) * math.cos(y) - math.cos(r) * math.sin(p) * math.sin(y)
        q_y = math.cos(r) * math.sin(p) * math.cos(y) + math.sin(r) * math.cos(p) * math.sin(y)
        q_z = math.cos(r) * math.cos(p) * math.sin(y) - math.sin(r) * math.sin(p) * math.cos(y)
        
        return np.array([q_w, q_x, q_y, q_z])
    
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        四元数转旋转矩阵
        """
        w, x, y, z = q
        
        # 归一化
        norm = np.linalg.norm(q)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # 构建旋转矩阵
        R = np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
        
        return R
    
    @staticmethod
    def rotation_matrix_to_wellbore_angles(R):
        """
        旋转矩阵转井眼角度
        """
        # 井斜角 (Inclination)
        inclination = math.degrees(math.acos(abs(R[2, 2])))
        
        # 方位角 (Azimuth)
        if abs(R[2, 2]) < 0.99999:  # 非垂直井
            azimuth = math.degrees(math.atan2(R[0, 2], R[1, 2]))
            if azimuth < 0:
                azimuth += 360
        else:
            azimuth = 0  # 垂直井中方位角任意
        
        # 重力工具面角 (Toolface)
        if inclination > 0.1:  # 非垂直井
            toolface = math.degrees(math.atan2(R[2, 0], R[2, 1]))
            if toolface < 0:
                toolface += 360
        else:
            toolface = 0  # 垂直井中工具面角无意义
        
        return azimuth, inclination, toolface

def attitude_to_wellbore_quaternion(roll, pitch, travel):
    """
    使用四元数进行姿态角到井眼角度的转换
    """
    converter = QuaternionConverter()
    
    # 步骤1: 欧拉角转四元数
    q = converter.euler_to_quaternion(roll, pitch, travel)
    
    # 步骤2: 四元数转旋转矩阵
    R = converter.quaternion_to_rotation_matrix(q)
    
    # 步骤3: 旋转矩阵转井眼角度
    azimuth, inclination, toolface = converter.rotation_matrix_to_wellbore_angles(R)
    

    ####### 注意：根据经验修正井斜角
    inclination = -2.4021*inclination+90.3764

    return azimuth, inclination, toolface, q


if __name__ == "__main__":
    roll, pitch, travel = -2.13811065971617, -16.7525039860491, -72.1968961819298
    azimuth, inclination, toolface, quaternion = attitude_to_wellbore_quaternion(
        roll, pitch, travel
    )
    print(f"输入姿态角: Roll={roll}°, Pitch={pitch}°, Travel={travel}°")
    print(f"四元数: [{quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f}]")
    print(f"(正确)方位角: {azimuth:.2f}°")
    print(f"(修正)井斜角: {inclination:.2f}°")
    print(f"(正确)重力工具面角: {toolface:.2f}°")
    
    # import pandas as pd

    # file = rf"D:\Desktop\data\DirRaw.xlsx"

    # df = pd.read_excel(file)

    # for index, row in df.iterrows():
    #     if index < 1:
    #         continue
    #     roll, pitch, travel = row.iloc[27], row.iloc[28], row.iloc[29]
    #     azimuth, inclination, toolface, quaternion = attitude_to_wellbore_quaternion(
    #         roll, pitch, travel
    #     )
    #     print(f"{azimuth} {inclination} {toolface} {quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}")