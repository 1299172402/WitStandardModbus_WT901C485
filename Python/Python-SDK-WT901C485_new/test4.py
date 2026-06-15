import math

def calculate_attitude_angles_general(ax, ay, az):
    """
    通用的姿态角计算（适用于复合旋转）
    
    参数:
    ax, ay, az: 三轴加速度值 (m/s²)
    
    返回:
    roll, pitch: 姿态角（度）
    """
    
    # 归一化重力向量
    g = math.sqrt(ax*ax + ay*ay + az*az)
    
    if g == 0:
        return 0, 0  # 避免除零错误
    
    ax_norm = ax / g
    ay_norm = ay / g  
    az_norm = az / g
    
    # 计算Pitch角（俯仰角）
    # 限制asin的输入范围在[-1, 1]
    sin_pitch = -ax_norm
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = math.asin(sin_pitch)
    
    # 计算Roll角（横滚角）
    # 使用atan2处理所有象限
    roll = math.atan2(ay_norm, az_norm)
    
    # 转换为角度
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    
    return roll_deg, pitch_deg

def calculate_attitude_robust(ax, ay, az):
    """
    更鲁棒的姿态角计算方法
    """
    
    # 归一化
    g = math.sqrt(ax*ax + ay*ay + az*az)
    if g < 0.1:  # 重力太小，可能是自由落体
        return None, None
        
    ax_norm = ax / g
    ay_norm = ay / g
    az_norm = az / g
    
    # 方法1：标准计算
    pitch = math.atan2(-ax_norm, math.sqrt(ay_norm*ay_norm + az_norm*az_norm))
    roll = math.atan2(ay_norm, az_norm)
    
    # 处理Pitch接近±90°的奇点情况
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) < 0.1:  # 接近90度
        print("警告：接近万向锁奇点")
        # 使用替代计算方法
        roll = math.atan2(ay_norm, az_norm)
    
    return math.degrees(roll), math.degrees(pitch)

# 测试复合旋转情况
def test_compound_rotation():
    """测试复合旋转的例子"""
    
    # 模拟"左下方低，右上方高"的姿态
    # 假设Roll = 30°, Pitch = -15°
    roll_actual = math.radians(30)   # 向右倾斜30度
    pitch_actual = math.radians(-15)  # 向前俯冲15度
    
    # 根据旋转矩阵计算理论加速度值
    g = 9.8
    ax_theory = -g * math.sin(pitch_actual)
    ay_theory = g * math.cos(pitch_actual) * math.sin(roll_actual)
    az_theory = g * math.cos(pitch_actual) * math.cos(roll_actual)
    
    print(f"理论姿态: Roll={math.degrees(roll_actual):.1f}°, Pitch={math.degrees(pitch_actual):.1f}°")
    print(f"理论加速度: ax={ax_theory:.2f}, ay={ay_theory:.2f}, az={az_theory:.2f}")
    
    # 使用我们的算法反推
    roll_calc, pitch_calc = calculate_attitude_robust(ax_theory, ay_theory, az_theory)
    
    print(f"计算姿态: Roll={roll_calc:.1f}°, Pitch={pitch_calc:.1f}°")
    print(f"误差: Roll={abs(roll_calc-math.degrees(roll_actual)):.3f}°, Pitch={abs(pitch_calc-math.degrees(pitch_actual)):.3f}°")

# 运行测试
test_compound_rotation()