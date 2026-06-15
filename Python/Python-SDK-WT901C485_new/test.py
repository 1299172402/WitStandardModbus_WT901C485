import device_model
import time

import math


def calculate_attitude(ax, ay, az, mx, my, mz):
    """
    根据三轴加速度计和磁力计数据计算姿态角（俯仰角、滚转角、偏航角）。

    参数:
    ax, ay, az: 加速度计的三轴数据
    mx, my, mz: 磁力计的三轴数据

    返回:
    pitch: 俯仰角（单位：度）
    roll: 滚转角（单位：度）
    yaw: 偏航角（单位：度）
    """
    # 1. 计算俯仰角（Pitch）和滚转角（Roll）
    pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2))  # 俯仰角
    roll = math.atan2(ay, az)  # 滚转角

    # 2. 矫正磁力计数据
    mx2 = mx * math.cos(pitch) + mz * math.sin(pitch)
    my2 = (
        mx * math.sin(roll) * math.sin(pitch)
        + my * math.cos(roll)
        - mz * math.sin(roll) * math.cos(pitch)
    )

    # 3. 计算偏航角（Yaw）
    yaw = math.atan2(-my2, mx2)

    # 转换为角度
    pitch = math.degrees(pitch)
    roll = math.degrees(roll)
    yaw = math.degrees(yaw)

    return pitch, roll, yaw


def calculate_drilling_angles(ax, ay, az, mx, my, mz, declination=0):
    """
    计算井斜角、井斜方位角、重力工具面角和磁工具面角。

    参数:
    ax, ay, az: 加速度计的三轴数据
    mx, my, mz: 磁力计的三轴数据
    declination: 地磁偏角（单位：度），用于校正磁北到正北的偏差

    返回:
    inclination: 井斜角（单位：度）
    azimuth: 井斜方位角（单位：度）
    gravity_toolface: 重力工具面角（单位：度）
    magnetic_toolface: 磁工具面角（单位：度）
    """
    # 1. 计算井斜角（Inclination）
    inclination = math.acos(az / math.sqrt(ax**2 + ay**2 + az**2))
    inclination = math.degrees(inclination)

    # 2. 计算井斜方位角（Azimuth）
    azimuth = math.atan2(my, mx)
    azimuth = math.degrees(azimuth)
    azimuth = (azimuth + declination) % 360  # 校正地磁偏角并确保在 0-360 度范围内

    # 3. 计算重力工具面角（Gravity Toolface）
    gravity_toolface = math.atan2(ax, ay)
    gravity_toolface = math.degrees(gravity_toolface)
    gravity_toolface = gravity_toolface % 360  # 确保在 0-360 度范围内

    # 4. 计算磁工具面角（Magnetic Toolface）
    magnetic_toolface = math.atan2(mx * ay - my * ax, mx * ax + my * ay)
    magnetic_toolface = math.degrees(magnetic_toolface)
    magnetic_toolface = magnetic_toolface % 360  # 确保在 0-360 度范围内

    return inclination, azimuth, gravity_toolface, magnetic_toolface


def calculate_angles(
    accelerationX, accelerationY, accelerationZ, magneticX, magneticY, magneticZ
):
    # 俯仰角 (Pitch)
    pitch = math.atan2(-accelerationX, math.sqrt(accelerationY**2 + accelerationZ**2))
    pitch_deg = math.degrees(pitch)

    # 横滚角 (Roll)
    roll = math.atan2(accelerationY, accelerationZ)
    roll_deg = math.degrees(roll)

    # 航向角 (Yaw)
    mag_x = magneticX * math.cos(pitch) + magneticZ * math.sin(pitch)
    mag_y = (
        magneticX * math.sin(roll) * math.sin(pitch)
        + magneticY * math.cos(roll)
        - magneticZ * math.sin(roll) * math.cos(pitch)
    )
    yaw = math.atan2(-mag_y, mag_x)
    yaw_deg = math.degrees(yaw)

    return pitch_deg, roll_deg, yaw_deg


# 数据更新事件  Data update event
def updateData(DeviceModel):
    print(DeviceModel.deviceData)
    # 获得加速度x的值
    # print(DeviceModel.get(80, "AccX"))
    ax = DeviceModel.get(80, "AccX")
    ay = DeviceModel.get(80, "AccY")
    az = DeviceModel.get(80, "AccZ")
    mx = DeviceModel.get(80, "HX")
    my = DeviceModel.get(80, "HY")
    mz = DeviceModel.get(80, "HZ")

    # pitch, roll, yaw = calculate_attitude(ax, ay, az, mx, my, mz)
    # print(f"Pitch (俯仰角): {pitch:.2f}°")
    # print(f"Roll (滚转角): {roll:.2f}°")
    # print(f"Yaw (偏航角): {yaw:.2f}°")

    # # 示例数据（假设从传感器读取的值）
    # declination = 5.0  # 地磁偏角（单位：度）

    # # 计算钻井角度
    # inclination, azimuth, gravity_toolface, magnetic_toolface = (
    #     calculate_drilling_angles(ax, ay, az, mx, my, mz, declination)
    # )

    # print(f"Inclination (井斜角): {inclination:.2f}°")
    # print(f"Azimuth (井斜方位角): {azimuth:.2f}°")
    # print(f"Gravity Toolface (重力工具面角): {gravity_toolface:.2f}°")
    # print(f"Magnetic Toolface (磁工具面角): {magnetic_toolface:.2f}°")

    pitch, roll, yaw = calculate_angles(ax, ay, az, mx, my, mz)
    print(f"Pitch: {pitch:.2f}°")
    print(f"Roll: {roll:.2f}°")
    print(f"Yaw: {yaw:.2f}°")


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
