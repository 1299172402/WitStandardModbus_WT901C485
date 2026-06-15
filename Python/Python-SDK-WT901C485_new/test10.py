import numpy as np
import math

class WellboreCalculator:
    def __init__(self):
        """井下工具姿态计算器"""
        pass
    
    def normalize_vector(self, vector):
        """向量归一化"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def acceleration_to_quaternion(self, ax, ay, az):
        """
        根据加速度计数据计算四元数
        假设加速度计测量的是重力加速度
        """
        # 归一化加速度向量
        acc = np.array([ax, ay, az])
        acc_norm = self.normalize_vector(acc)
        
        # 重力向量在地理坐标系中指向下方 (0, 0, -1)
        gravity_ref = np.array([0, 0, -1])
        
        # 计算旋转轴（叉积）
        rotation_axis = np.cross(gravity_ref, acc_norm)
        rotation_axis = self.normalize_vector(rotation_axis)
        
        # 计算旋转角度
        dot_product = np.dot(gravity_ref, acc_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = math.acos(dot_product)
        
        # 构造四元数
        if np.linalg.norm(rotation_axis) < 1e-6:  # 向量平行
            return np.array([1, 0, 0, 0])  # 单位四元数
        
        half_angle = angle / 2
        sin_half = math.sin(half_angle)
        cos_half = math.cos(half_angle)
        
        qw = cos_half
        qx = rotation_axis[0] * sin_half
        qy = rotation_axis[1] * sin_half
        qz = rotation_axis[2] * sin_half
        
        return np.array([qw, qx, qy, qz])
    
    def quaternion_multiply(self, q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def quaternion_conjugate(self, q):
        """四元数共轭"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def rotate_vector_by_quaternion(self, vector, q):
        """使用四元数旋转向量"""
        # 将向量转换为四元数形式 [0, x, y, z]
        v_quat = np.array([0, vector[0], vector[1], vector[2]])
        
        # 旋转公式: v' = q * v * q*
        q_conj = self.quaternion_conjugate(q)
        temp = self.quaternion_multiply(q, v_quat)
        result = self.quaternion_multiply(temp, q_conj)
        
        return result[1:4]  # 返回向量部分
    
    def calculate_wellbore_parameters(self, ax, ay, az, mx, my, mz):
        """
        计算井下工具参数
        
        参数:
        ax, ay, az: 加速度计三轴测量值
        mx, my, mz: 磁力计三轴测量值
        
        返回:
        gravity_toolface: 重力工具面角 (度)
        magnetic_toolface: 磁工具面角 (度)
        inclination: 井斜角 (度)
        azimuth: 方位角 (度)
        """
        
        # 1. 计算井斜角（倾斜角）
        acc_magnitude = math.sqrt(ax*ax + ay*ay + az*az)
        if acc_magnitude == 0:
            inclination = 0
        else:
            # 井斜角是重力向量与垂直向量的夹角
            inclination = math.acos(abs(az) / acc_magnitude)
        
        inclination_deg = math.degrees(inclination)
        
        # 2. 计算重力工具面角
        if abs(az) >= acc_magnitude - 1e-6:  # 接近垂直
            gravity_toolface_deg = 0
        else:
            gravity_toolface = math.atan2(ay, ax)
            gravity_toolface_deg = math.degrees(gravity_toolface)
        
        # 3. 处理磁力计数据
        mag = np.array([mx, my, mz])
        mag_norm = self.normalize_vector(mag)
        
        # 4. 计算方位角和磁工具面角
        if inclination_deg < 5:  # 接近垂直，特殊处理
            # 垂直井段，方位角从磁力计水平分量计算
            azimuth = math.atan2(my, mx)
            azimuth_deg = math.degrees(azimuth)
            magnetic_toolface_deg = 0
        else:
            # 倾斜井段
            # 建立井眼坐标系
            acc_norm = self.normalize_vector(np.array([ax, ay, az]))
            
            # 井轴方向（Z轴）
            well_axis = -acc_norm  # 重力反方向为井轴方向
            
            # 计算北向在井眼横截面上的投影
            # 假设北方向为 [1, 0, 0]（这里需要根据实际坐标系调整）
            north = np.array([1, 0, 0])
            
            # 计算井眼横截面上的东向和北向
            east_in_plane = north - np.dot(north, well_axis) * well_axis
            east_in_plane = self.normalize_vector(east_in_plane)
            
            north_in_plane = np.cross(well_axis, east_in_plane)
            north_in_plane = self.normalize_vector(north_in_plane)
            
            # 计算磁场在井眼横截面上的投影
            mag_in_plane = mag_norm - np.dot(mag_norm, well_axis) * well_axis
            mag_in_plane = self.normalize_vector(mag_in_plane)
            
            # 计算方位角（磁场投影与北向的夹角）
            cos_azimuth = np.dot(mag_in_plane, north_in_plane)
            sin_azimuth = np.dot(mag_in_plane, east_in_plane)
            azimuth = math.atan2(sin_azimuth, cos_azimuth)
            azimuth_deg = math.degrees(azimuth)
            
            # 计算磁工具面角
            # 工具面角是工具高边与重力高边的夹角
            gravity_high_side = np.array([-ax, -ay, 0])  # 重力高边方向
            gravity_high_side = self.normalize_vector(gravity_high_side)
            
            magnetic_toolface = math.atan2(
                np.dot(mag_in_plane, np.cross(gravity_high_side, well_axis)),
                np.dot(mag_in_plane, gravity_high_side)
            )
            magnetic_toolface_deg = math.degrees(magnetic_toolface)
        
        # 角度标准化到 [0, 360)
        def normalize_angle(angle):
            while angle < 0:
                angle += 360
            while angle >= 360:
                angle -= 360
            return angle
        
        gravity_toolface_deg = normalize_angle(gravity_toolface_deg)
        magnetic_toolface_deg = normalize_angle(magnetic_toolface_deg)
        azimuth_deg = normalize_angle(azimuth_deg)
        
        return {
            'gravity_toolface': gravity_toolface_deg,
            'magnetic_toolface': magnetic_toolface_deg,
            'inclination': inclination_deg,
            'azimuth': azimuth_deg
        }

# 使用示例
def main():
    calculator = WellboreCalculator()
    
    # 示例数据（请替换为实际测量值）
    ax, ay, az = -1.584962, -2.523489, 9.35654  # 加速度计数据 (m/s²)
    mx, my, mz = 7531.738281, 5157.470703, 48144.53125  # 磁力计数据 (Gauss 或 Tesla)
    
    results = calculator.calculate_wellbore_parameters(ax, ay, az, mx, my, mz)
    
    print("井下工具姿态参数:")
    print(f"重力工具面角: {results['gravity_toolface']:.2f}°")
    print(f"磁工具面角: {results['magnetic_toolface']:.2f}°")
    print(f"井斜角: {results['inclination']:.2f}°")
    print(f"方位角: {results['azimuth']:.2f}°")

if __name__ == "__main__":
    main()