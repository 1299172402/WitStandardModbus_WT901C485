import math

class AttitudeCalculator:
    def __init__(self, declination=0):
        """
        初始化姿态计算器
        
        参数:
        declination: 当地磁偏角（度）
        """
        self.declination = math.radians(declination)
    
    def calculate_attitude(self, ax, ay, az, mx, my, mz):
        """
        完整的姿态角计算
        
        参数:
        ax, ay, az: 加速度计读数 (m/s²)
        mx, my, mz: 磁力计读数 (任意单位，会自动归一化)
        
        返回:
        roll, pitch, yaw: 姿态角（度）
        """
        
        # 第一步：从加速度计计算Roll和Pitch
        roll, pitch = self.calculate_roll_pitch_from_accel(ax, ay, az)
        
        # 第二步：磁力计归一化
        m_norm = math.sqrt(mx*mx + my*my + mz*mz)
        if m_norm == 0:
            return None, None, None
            
        mx_norm = mx / m_norm
        my_norm = my / m_norm  
        mz_norm = mz / m_norm
        
        # 第三步：倾斜补偿
        mx_comp, my_comp = self.tilt_compensated_compass(
            mx_norm, my_norm, mz_norm, roll, pitch
        )
        
        # 第四步：计算方位角 - 修复版本
        yaw = self.calculate_yaw_from_compass_fixed(mx_comp, my_comp, self.declination)
        
        # 转换为角度
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    
    def calculate_roll_pitch_from_accel(self, ax, ay, az):
        """从加速度计计算Roll和Pitch"""
        g = math.sqrt(ax*ax + ay*ay + az*az)
        if g == 0:
            return 0, 0
            
        ax_norm = ax / g
        ay_norm = ay / g
        az_norm = az / g
        
        roll = math.atan2(ay_norm, az_norm)
        pitch = math.atan2(-ax_norm, math.sqrt(ay_norm*ay_norm + az_norm*az_norm))
        
        return roll, pitch
    
    def tilt_compensated_compass(self, mx, my, mz, roll, pitch):
        """磁力计倾斜补偿"""
        cos_roll = math.cos(roll)
        sin_roll = math.sin(roll)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        
        mx_comp = mx * cos_pitch + my * sin_roll * sin_pitch + mz * cos_roll * sin_pitch
        my_comp = my * cos_roll - mz * sin_roll
        
        return mx_comp, my_comp
    
    def calculate_yaw_from_compass_fixed(self, mx_comp, my_comp, declination):
        """
        修复版本的方位角计算
        
        常见的方位角定义：
        - 0° = 北方 (North)
        - 90° = 东方 (East) 
        - 180° = 南方 (South)
        - 270° = 西方 (West)
        """
        
        # 方法1：标准地理坐标系 (尝试这个)
        yaw = math.atan2(my_comp, mx_comp)
        
        # 方法2：如果方法1不对，试试这个
        # yaw = math.atan2(-my_comp, mx_comp)
        
        # 方法3：如果还不对，试试这个
        # yaw = math.atan2(mx_comp, my_comp)
        
        # 添加磁偏角校正
        yaw += declination
        
        # 归一化到[0, 2π]
        while yaw < 0:
            yaw += 2 * math.pi
        while yaw >= 2 * math.pi:
            yaw -= 2 * math.pi
            
        return yaw
    
    def calculate_yaw_from_compass(self, mx_comp, my_comp, declination):
        """原始版本（保留用于对比）"""
        yaw = math.atan2(-my_comp, mx_comp)
        yaw += declination
        
        # 归一化到[0, 2π]
        while yaw < 0:
            yaw += 2 * math.pi
        while yaw >= 2 * math.pi:
            yaw -= 2 * math.pi
            
        return yaw

# 调试用的测试函数
def debug_yaw_calculation():
    """调试方位角计算"""
    
    calc = AttitudeCalculator(declination=0)  # 先不考虑磁偏角
    
    # 测试已知方向的磁场数据
    test_cases = [
        # (mx, my, 期望的方位角度数)
        (1, 0, 0),      # 正北方向
        (0, 1, 90),     # 正东方向  
        (-1, 0, 180),   # 正南方向
        (0, -1, 270),   # 正西方向
        (0.707, 0.707, 45),   # 东北方向
    ]
    
    print("调试方位角计算:")
    print("mx\tmy\t期望角度\t原始方法\t修复方法")
    print("-" * 50)
    
    for mx, my, expected in test_cases:
        # 水平放置，无倾斜
        ax, ay, az = 0, 0, 9.8
        
        roll, pitch = calc.calculate_roll_pitch_from_accel(ax, ay, az)
        mx_comp, my_comp = calc.tilt_compensated_compass(mx, my, 0, roll, pitch)
        
        # 原始方法
        yaw_orig = calc.calculate_yaw_from_compass(mx_comp, my_comp, 0)
        yaw_orig_deg = math.degrees(yaw_orig)
        
        # 修复方法
        yaw_fixed = calc.calculate_yaw_from_compass_fixed(mx_comp, my_comp, 0)
        yaw_fixed_deg = math.degrees(yaw_fixed)
        
        print(f"{mx:.2f}\t{my:.2f}\t{expected}°\t\t{yaw_orig_deg:.1f}°\t\t{yaw_fixed_deg:.1f}°")

def test_attitude_calculation():
    """测试姿态计算"""
    
    # 创建计算器（假设磁偏角为5度）
    calc = AttitudeCalculator(declination=5.0)
    
    # 模拟传感器数据
    # 加速度计：设备向右倾斜30度，向前俯仰10度
    ax = -9.8 * math.sin(math.radians(-10))  # Pitch = -10度
    ay = 9.8 * math.cos(math.radians(-10)) * math.sin(math.radians(30))  # Roll = 30度
    az = 9.8 * math.cos(math.radians(-10)) * math.cos(math.radians(30))
    
    # 磁力计：假设指向北偏东45度
    mx = 0.707  # 北分量
    my = 0.707  # 东分量  
    mz = 0.2    # 垂直分量
    
    print(f"模拟加速度计: ax={ax:.2f}, ay={ay:.2f}, az={az:.2f}")
    print(f"模拟磁力计: mx={mx:.3f}, my={my:.3f}, mz={mz:.3f}")

    # 计算姿态角
    roll, pitch, yaw = calc.calculate_attitude(ax, ay, az, mx, my, mz)
    
    print(f"计算结果:")
    print(f"Roll (横滚): {roll:.1f}°")
    print(f"Pitch (俯仰): {pitch:.1f}°") 
    print(f"Yaw (方位): {yaw:.1f}°")
    print(f"期望Yaw约为: {45+5:.1f}° (45°磁方位 + 5°磁偏角)")

if __name__ == "__main__":
    # 先运行调试
    debug_yaw_calculation()
    print("\n" + "="*60 + "\n")
    # 再运行原测试
    test_attitude_calculation()