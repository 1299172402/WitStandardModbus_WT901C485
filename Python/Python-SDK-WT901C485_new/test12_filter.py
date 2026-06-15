import device_model
import time
from collections import deque
import math

class SensorFilter:
    def __init__(self, window_size=5, alpha=0.8):
        """
        传感器数据滤波器
        
        参数:
        window_size: 移动平均窗口大小
        alpha: 低通滤波器系数 (0-1, 越大响应越快)
        """
        self.window_size = window_size
        self.alpha = alpha
        
        # 移动平均缓冲区
        self.acc_buffer = {'x': deque(maxlen=window_size), 
                          'y': deque(maxlen=window_size), 
                          'z': deque(maxlen=window_size)}
        self.mag_buffer = {'x': deque(maxlen=window_size), 
                          'y': deque(maxlen=window_size), 
                          'z': deque(maxlen=window_size)}
        
        # 低通滤波器上一次值
        self.last_acc = {'x': None, 'y': None, 'z': None}
        self.last_mag = {'x': None, 'y': None, 'z': None}
    
    def moving_average_filter(self, ax, ay, az, mx, my, mz):
        """移动平均滤波"""
        # 添加新数据到缓冲区
        self.acc_buffer['x'].append(ax)
        self.acc_buffer['y'].append(ay)
        self.acc_buffer['z'].append(az)
        self.mag_buffer['x'].append(mx)
        self.mag_buffer['y'].append(my)
        self.mag_buffer['z'].append(mz)
        
        # 计算平均值
        ax_filtered = sum(self.acc_buffer['x']) / len(self.acc_buffer['x'])
        ay_filtered = sum(self.acc_buffer['y']) / len(self.acc_buffer['y'])
        az_filtered = sum(self.acc_buffer['z']) / len(self.acc_buffer['z'])
        mx_filtered = sum(self.mag_buffer['x']) / len(self.mag_buffer['x'])
        my_filtered = sum(self.mag_buffer['y']) / len(self.mag_buffer['y'])
        mz_filtered = sum(self.mag_buffer['z']) / len(self.mag_buffer['z'])
        
        return ax_filtered, ay_filtered, az_filtered, mx_filtered, my_filtered, mz_filtered
    
    def low_pass_filter(self, ax, ay, az, mx, my, mz):
        """低通滤波器"""
        # 初始化
        if self.last_acc['x'] is None:
            self.last_acc = {'x': ax, 'y': ay, 'z': az}
            self.last_mag = {'x': mx, 'y': my, 'z': mz}
            return ax, ay, az, mx, my, mz
        
        # 低通滤波公式: filtered = alpha * current + (1-alpha) * last
        ax_filtered = self.alpha * ax + (1 - self.alpha) * self.last_acc['x']
        ay_filtered = self.alpha * ay + (1 - self.alpha) * self.last_acc['y']
        az_filtered = self.alpha * az + (1 - self.alpha) * self.last_acc['z']
        mx_filtered = self.alpha * mx + (1 - self.alpha) * self.last_mag['x']
        my_filtered = self.alpha * my + (1 - self.alpha) * self.last_mag['y']
        mz_filtered = self.alpha * mz + (1 - self.alpha) * self.last_mag['z']
        
        # 更新上一次值
        self.last_acc = {'x': ax_filtered, 'y': ay_filtered, 'z': az_filtered}
        self.last_mag = {'x': mx_filtered, 'y': my_filtered, 'z': mz_filtered}
        
        return ax_filtered, ay_filtered, az_filtered, mx_filtered, my_filtered, mz_filtered


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


# 创建全局滤波器实例
sensor_filter = SensorFilter(window_size=5, alpha=0.8)


def calculate_angles(ax, ay, az, mx, my, mz):
    print("=====")
    print(f"Acc: ax={ax}, ay={ay}, az={az}")
    print(f"Mag: mx={mx}, my={my}, mz={mz}")

    # 应用滤波器 - 可以选择使用移动平均或低通滤波
    # 使用移动平均滤波
    # ax_f, ay_f, az_f, mx_f, my_f, mz_f = sensor_filter.moving_average_filter(ax, ay, az, mx, my, mz)
    
    # 或者使用低通滤波（取消注释下面这行，注释上面那行）
    ax_f, ay_f, az_f, mx_f, my_f, mz_f = sensor_filter.low_pass_filter(ax, ay, az, mx, my, mz)

    # 创建计算器（假设磁偏角为0度）
    calc = AttitudeCalculator(declination=0.0)

    # 使用滤波后的数据计算姿态角
    roll, pitch, yaw = calc.calculate_attitude(ax_f, ay_f, az_f, mx_f, my_f, mz_f)
    
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
