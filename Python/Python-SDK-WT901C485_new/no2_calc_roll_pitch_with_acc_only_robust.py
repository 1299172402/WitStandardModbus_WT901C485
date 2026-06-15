import device_model
import time

import math

def calculate_angles(ax, ay, az, mx, my, mz):
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
    
    return math.degrees(roll), math.degrees(pitch), 0



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
