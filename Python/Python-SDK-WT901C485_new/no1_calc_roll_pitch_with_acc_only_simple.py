import device_model
import time

import math

def calculate_angles(ax, ay, az, mx, my, mz):
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
    
    return roll_deg, pitch_deg, 0



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
