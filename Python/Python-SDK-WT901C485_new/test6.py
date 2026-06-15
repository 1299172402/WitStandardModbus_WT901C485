import math

def calculate_inclination_angle(ax, ay, az):
    """
    计算井斜角
    
    参数:
    ax, ay, az: 三轴加速度计读数 (m/s²)
    
    返回:
    inclination: 井斜角（度）
    """
    # 计算重力加速度总量
    g_total = math.sqrt(ax*ax + ay*ay + az*az)
    
    # 避免除零错误
    if g_total == 0:
        return 0
    
    # 计算cos(θ)
    cos_theta = az / g_total
    
    # 限制cos值在[-1, 1]范围内，避免数值误差
    cos_theta = max(-1, min(1, cos_theta))
    
    # 计算井斜角（弧度）
    theta_rad = math.acos(cos_theta)
    
    # 转换为角度
    theta_deg = math.degrees(theta_rad)
    
    return theta_deg

# 测试计算
def test_inclination_calculation():
    # 给定数据
    ax = -0.28072
    ay = 0.225055  
    az = 9.80665
    
    inclination = calculate_inclination_angle(ax, ay, az)
    
    print(f"输入数据:")
    print(f"ax = {ax}")
    print(f"ay = {ay}")
    print(f"az = {az}")
    print(f"计算的井斜角: {inclination:.6f}°")
    
    # 验证计算过程
    g_total = math.sqrt(ax*ax + ay*ay + az*az)
    cos_theta = az / g_total
    print(f"\n计算过程:")
    print(f"g_total = √({ax}² + {ay}² + {az}²) = {g_total:.6f}")
    print(f"cos(θ) = {az} / {g_total} = {cos_theta:.6f}")
    print(f"θ = arccos({cos_theta:.6f}) = {inclination:.6f}°")

if __name__ == "__main__":
    test_inclination_calculation()