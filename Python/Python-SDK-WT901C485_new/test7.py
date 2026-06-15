import numpy as np

def calculate_survey_parameters(gx, gy, gz, hx, hy, hz):
    """
    计算旋转导向测井参数
    
    参数:
    gx, gy, gz: 三轴加速度计测量值 (m/s²)
    hx, hy, hz: 三轴磁力计测量值 (nT)
    
    返回:
    htfm, mtfm, gtb, htb, dipb, azmb
    """
    
    # 1. 计算总场强度
    gtb = np.sqrt(gx**2 + gy**2 + gz**2)
    htb = np.sqrt(hx**2 + hy**2 + hz**2)
    
    # 2. 计算井斜角
    dipb = np.arccos(gz / gtb) * 180 / np.pi
    
    # 3. 计算方位角
    azmb = np.arctan2(gy, gx) * 180 / np.pi
    if azmb < 0:
        azmb += 360
    
    # 4. 计算磁工具面角
    mtfm = np.arctan2(hy, hx) * 180 / np.pi
    if mtfm < 0:
        mtfm += 360
    
    # 5. 计算工具面角（简化计算）
    # 实际计算需要考虑磁偏角和复杂的坐标变换
    htfm = mtfm - azmb
    if htfm < 0:
        htfm += 360
    
    return htfm, mtfm, gtb, htb, dipb, azmb

# 验证第一行数据
gx, gy, gz = 1.47962, -2.57616, 9.35654
hx, hy, hz = 13916.01563, 32275.39063, 44726.5625

htfm, mtfm, gtb, htb, dipb, azmb = calculate_survey_parameters(gx, gy, gz, hx, hy, hz)
print(f"GTB: {gtb:.5f}")
print(f"HTB: {htb:.5f}")
print(f"DIPB: {dipb:.5f}")
print(f"AZMB: {azmb:.5f}")