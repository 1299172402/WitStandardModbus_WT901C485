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
    def rotation_matrix_to_wellbore_angles(R, mag_x=None, mag_y=None, mag_z=None):
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
            gravity_toolface = math.degrees(math.atan2(R[2, 0], R[2, 1]))
            if gravity_toolface < 0:
                gravity_toolface += 360
        else:
            gravity_toolface = 0  # 垂直井中工具面角无意义
        
        # 磁工具面角
        magnetic_toolface = 0
        if mag_x is not None and mag_y is not None and mag_z is not None:
            # 将磁场矢量转换到井眼坐标系
            mag_vector = np.array([mag_x, mag_y, mag_z])
            # 应用旋转矩阵的逆变换（转置）
            mag_wellbore = R.T @ mag_vector
            
            # 计算磁工具面角
            if inclination > 0.1:  # 非垂直井
                # 在垂直于井眼轴的平面内计算磁北方向
                magnetic_toolface = math.degrees(math.atan2(mag_wellbore[0], mag_wellbore[1]))
                if magnetic_toolface < 0:
                    magnetic_toolface += 360
        
        return azimuth, inclination, gravity_toolface, magnetic_toolface


def attitude_to_wellbore_quaternion(roll, pitch, travel, mag_x=None, mag_y=None, mag_z=None):
    """
    使用四元数进行姿态角到井眼角度的转换
    """
    converter = QuaternionConverter()
    
    # 步骤1: 欧拉角转四元数
    q = converter.euler_to_quaternion(roll, pitch, travel)
    
    # 步骤2: 四元数转旋转矩阵
    R = converter.quaternion_to_rotation_matrix(q)
    
    # 步骤3: 旋转矩阵转井眼角度
    azimuth, inclination, gravity_toolface, magnetic_toolface = converter.rotation_matrix_to_wellbore_angles(
        R, mag_x, mag_y, mag_z
    )
    
    ####### 注意：根据经验修正井斜角
    inclination = -2.4021*inclination+90.3764

    return azimuth, inclination, gravity_toolface, magnetic_toolface, q


# 示例使用
roll, pitch, travel = -2.13811065971617, -16.7525039860491, -72.1968961819298
# 添加磁场数据（示例值，需要从传感器获取）
mag_x, mag_y, mag_z = -21289.0625, 9570.3125, 55273.4375  # 磁场强度分量

azimuth, inclination, gravity_toolface, magnetic_toolface, quaternion = attitude_to_wellbore_quaternion(
    roll, pitch, travel, mag_x, mag_y, mag_z
)

print(f"输入姿态角: Roll={roll}°, Pitch={pitch}°, Travel={travel}°")
print(f"磁场数据: Mx={mag_x}, My={mag_y}, Mz={mag_z}")
print(f"四元数: [{quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f}]")
print(f"方位角: {azimuth:.2f}°")
print(f"井斜角: {inclination:.2f}°")
print(f"重力工具面角: {gravity_toolface:.2f}°")
print(f"磁工具面角: {magnetic_toolface:.2f}°")



# ...existing code...

def analyze_rotation_matrix_example():
    """
    分析具体例子中的旋转矩阵
    """
    # 使用您的例子
    roll, pitch, travel = -2.13811065971617, -16.7525039860491, -72.1968961819298
    
    converter = QuaternionConverter()
    q = converter.euler_to_quaternion(roll, pitch, travel)
    R = converter.quaternion_to_rotation_matrix(q)
    
    print(f"输入: Roll={roll}°, Pitch={pitch}°, Travel={travel}°")
    print(f"四元数: {q}")
    print(f"旋转矩阵R:")
    print(f"{R}")
    print()
    
    # 分析矩阵含义
    print("矩阵分析:")
    print(f"R33 = {R[2,2]:.4f} -> 井斜角 = {-2.4021*math.degrees(math.acos(abs(R[2,2])))+90.3764:.2f}°")
    print(f"R13 = {R[0,2]:.4f}, R23 = {R[1,2]:.4f} -> 方位角 = {math.degrees(math.atan2(R[0,2], R[1,2])):.2f}°")
    print(f"R31 = {R[2,0]:.4f}, R32 = {R[2,1]:.4f} -> 工具面角 = {math.degrees(math.atan2(R[2,0], R[2,1])):.2f}°")

# 添加到现有代码后面
analyze_rotation_matrix_example()



if __name__ == "__main__":
    import pandas as pd

    file = rf"D:\Desktop\data\DirRaw.xlsx"

    df = pd.read_excel(file)

    for index, row in df.iterrows():
        if index < 1:
            continue
        roll, pitch, travel = row.iloc[27], row.iloc[28], row.iloc[29]
        mag_x, mag_y, mag_z = row.iloc[6], row.iloc[7], row.iloc[8]
        azimuth, inclination, gravity_toolface, magnetic_toolface, quaternion = attitude_to_wellbore_quaternion(
            roll, pitch, travel, mag_x, mag_y, mag_z
        )
        print(f"{azimuth} {inclination} {gravity_toolface} {magnetic_toolface} {quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}")



