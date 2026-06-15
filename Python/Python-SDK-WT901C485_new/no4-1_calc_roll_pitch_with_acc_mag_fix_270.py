import device_model
import time

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
        
        # 归一化到[0, 2π]
        while yaw < 0:
            yaw += 2 * math.pi
        while yaw >= 2 * math.pi:
            yaw -= 2 * math.pi
            
        return yaw

def calculate_angles(ax, ay, az, mx, my, mz):
    print("=====")
    print(f"Acc: ax={ax}, ay={ay}, az={az}")
    print(f"Mag: mx={mx}, my={my}, mz={mz}")

    # 创建计算器（假设磁偏角为0度）
    calc = AttitudeCalculator(declination=0.0)

    # 计算姿态角
    roll, pitch, yaw = calc.calculate_attitude(ax, ay, az, mx, my, mz)
    
    return roll, pitch, yaw



# 数据更新事件  Data update event
def updateData(DeviceModel):
    ax = DeviceModel.get(80, "AccX")
    ay = DeviceModel.get(80, "AccY")
    az = DeviceModel.get(80, "AccZ")
    mx = DeviceModel.get(80, "HX")
    my = DeviceModel.get(80, "HY")
    mz = DeviceModel.get(80, "HZ")
    print("-----")
    angx = DeviceModel.get(80, "AngX")
    angy = DeviceModel.get(80, "AngY")
    angz = DeviceModel.get(80, "AngZ")

    roll, pitch, yaw = calculate_angles(ax, ay, az, mx, my, mz)
    print(f"Roll: {roll}°")
    print(f"Pitch: {pitch}°")
    print(f"Yaw: {yaw}°")

    print(f"angx: {angx}°")
    print(f"angy: {angy}°")
    print(f"angz: {angz}°")

    print(f"diff_angx:{roll - angx}")
    print(f"diff_angy:{pitch - angy}")
    print(f"diff_angz:{yaw - angz}")



if __name__ == "__main__":
    # 读取的modbus地址列表 List of Modbus addresses read
    addrLis = [0x50]
    # 拿到设备模型 Get the device model
    device = device_model.DeviceModel(
        "HWT901B-485", "COM3", 230400, addrLis, updateData
    )
    # 开启设备 Turn on the device
    device.openDevice()
    # 开启轮询 Enable loop reading
    device.startLoopRead()

    # 如果 Ctrl+C，则关闭设备 If Ctrl+C, close the device
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        device.stopLoopRead()
        device.closeDevice()
