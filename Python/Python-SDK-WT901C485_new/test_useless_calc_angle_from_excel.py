import device_model
import time

import math
import pandas as pd

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
        
        # 第四步：计算方位角
        yaw = self.calculate_yaw_from_compass(mx_comp, my_comp, self.declination)
        
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
    
    def calculate_yaw_from_compass(self, mx_comp, my_comp, declination):
        """从补偿后的磁力计数据计算方位角"""
        yaw = math.atan2(mx_comp, my_comp)
        yaw += declination
        
        # 归一化到[-π, π]，即[-180°, 180°]
        while yaw > math.pi:
            yaw -= 2 * math.pi
        while yaw <= -math.pi:
            yaw += 2 * math.pi
            
        return yaw

def calculate_angles(ax, ay, az, mx, my, mz):
    # print("=====")
    # print(f"Acc: ax={ax}, ay={ay}, az={az}")
    # print(f"Mag: mx={mx}, my={my}, mz={mz}")

    # 创建计算器（假设磁偏角为0度）
    calc = AttitudeCalculator(declination=0.0)

    # 计算姿态角
    roll, pitch, yaw = calc.calculate_attitude(ax, ay, az, mx, my, mz)
    
    return roll, pitch, yaw



# 数据更新事件  Data update event
def updateData(ax, ay, az, mx, my, mz):




    roll, pitch, yaw = calculate_angles(ax, ay, az, mx, my, mz)
    # print(f"Roll: {roll}°")
    # print(f"Pitch: {pitch}°")
    # print(f"Yaw: {yaw}°")

    # print(f"angx: {angx}°")
    # print(f"angy: {angy}°")
    # print(f"angz: {angz}°")

    # print(f"diff_angx:{roll - angx}")
    # print(f"diff_angy:{pitch - angy}")
    # print(f"diff_angz:{yaw - angz}")

    return roll, pitch, yaw




if __name__ == "__main__":

    file = rf"D:\Desktop\data\DirRaw.xlsx"

    # 读取Excel文件，假设数据在第一个工作表
    df = pd.read_excel(file)

    for index, row in df.iterrows():
        if index < 1:
            continue
        ax, ay, az, mx, my, mz = row.iloc[19], row.iloc[20], row.iloc[21], row.iloc[23], row.iloc[24], row.iloc[25]
        ht, mt, di, az = row.iloc[9], row.iloc[10], row.iloc[13], row.iloc[14]
        print("=====")
        print(f"Row {index}:")
        print(f"HT: {ht}, MT: {mt}, DI: {di}, AZ: {az}")
        print(f"Acc: ax={ax}, ay={ay}, az={az}")
        print(f"Mag: mx={mx}, my={my}, mz={mz}")
        roll, pitch, yaw = updateData(ax, ay, az, mx, my, mz)
        print(f"Roll: {roll}°")
        print(f"Pitch: {pitch}°")
        print(f"Yaw: {yaw}°")

        print("diff:")
        print(f"diff_roll:{roll - ht}")
        print(f"diff_pitch:{pitch - mt}")
        print(f"diff_yaw:{yaw - az}")
        print("\n")