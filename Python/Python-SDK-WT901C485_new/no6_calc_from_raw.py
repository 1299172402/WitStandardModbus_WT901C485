# 从raw文件中计算三个角度 azimuth, inclination, toolface
import pandas as pd
import numpy as np
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
        m_norm = math.sqrt(mx * mx + my * my + mz * mz)
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
        g = math.sqrt(ax * ax + ay * ay + az * az)
        if g == 0:
            return 0, 0

        ax_norm = ax / g
        ay_norm = ay / g
        az_norm = az / g

        roll = math.atan2(ay_norm, az_norm)
        pitch = math.atan2(-ax_norm, math.sqrt(ay_norm * ay_norm + az_norm * az_norm))

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
    print("=====")
    print(f"Acc: ax={ax}, ay={ay}, az={az}")
    print(f"Mag: mx={mx}, my={my}, mz={mz}")

    # 创建计算器（假设磁偏角为0度）
    calc = AttitudeCalculator(declination=0.0)

    # 计算姿态角
    roll, pitch, yaw = calc.calculate_attitude(ax, ay, az, mx, my, mz)

    return roll, pitch, yaw


class QuaternionConverter:
    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        """
        欧拉角转四元数 (ZYX顺序)
        """
        # 转换为弧度并取一半角度
        r = math.radians(roll) / 2
        p = math.radians(pitch) / 2
        y = math.radians(yaw) / 2

        # 计算四元数分量
        q_w = math.cos(r) * math.cos(p) * math.cos(y) + math.sin(r) * math.sin(
            p
        ) * math.sin(y)
        q_x = math.sin(r) * math.cos(p) * math.cos(y) - math.cos(r) * math.sin(
            p
        ) * math.sin(y)
        q_y = math.cos(r) * math.sin(p) * math.cos(y) + math.sin(r) * math.cos(
            p
        ) * math.sin(y)
        q_z = math.cos(r) * math.cos(p) * math.sin(y) - math.sin(r) * math.sin(
            p
        ) * math.cos(y)

        return np.array([q_w, q_x, q_y, q_z])

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        四元数转旋转矩阵
        """
        w, x, y, z = q

        # 归一化
        norm = np.linalg.norm(q)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        # 构建旋转矩阵
        R = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )

        return R

    @staticmethod
    def rotation_matrix_to_wellbore_angles(R):
        """
        旋转矩阵转井眼角度
        """
        # 井斜角 (Inclination)
        inclination = math.degrees(math.acos(abs(R[2, 2])))

        # 方位角 (Azimuth)
        if abs(R[2, 2]) < 0.99999:  # 非垂直井
            azimuth = math.degrees(math.atan2(R[0, 2], R[1, 2]))
            if azimuth < 0:
                azimuth += 360
        else:
            azimuth = 0  # 垂直井中方位角任意

        # 重力工具面角 (Toolface)
        if inclination > 0.1:  # 非垂直井
            toolface = math.degrees(math.atan2(R[2, 0], R[2, 1]))
            if toolface < 0:
                toolface += 360
        else:
            toolface = 0  # 垂直井中工具面角无意义

        return azimuth, inclination, toolface


def attitude_to_wellbore_quaternion(roll, pitch, travel):
    """
    使用四元数进行姿态角到井眼角度的转换
    """
    converter = QuaternionConverter()

    # 步骤1: 欧拉角转四元数
    q = converter.euler_to_quaternion(roll, pitch, travel)

    # 步骤2: 四元数转旋转矩阵
    R = converter.quaternion_to_rotation_matrix(q)

    # 步骤3: 旋转矩阵转井眼角度
    azimuth, inclination, toolface = converter.rotation_matrix_to_wellbore_angles(R)

    ####### 注意：根据经验修正井斜角、方位角 ########
    inclination = (
        -2.4021 * inclination + 90.3764
    )  # 这里或许是90度整的旋转，但前面的系数并不确定
    azimuth = 360 - azimuth

    return azimuth, inclination, toolface, q


if __name__ == "__main__":
    file = rf"D:\Desktop\data\DirRaw.xlsx"

    df = pd.read_excel(file)

    for index, row in df.iterrows():
        if index < 1:
            continue
        ax, ay, az, mx, my, mz = (
            row.iloc[3],
            row.iloc[4],
            row.iloc[5],
            row.iloc[6],
            row.iloc[7],
            row.iloc[8],
        )
        # Normalize accelerometer data to units of g
        ag = math.sqrt(ax * ax + ay * ay + az * az)
        ax = ax / ag
        ay = ay / ag
        az = az / ag
        # Normalize magnetometer data from nT to µT
        mx = mx / 1000
        my = my / 1000
        mz = mz / 1000

        roll, pitch, yaw = calculate_angles(ax, ay, az, mx, my, mz)
        print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        azimuth, inclination, toolface, quaternion = attitude_to_wellbore_quaternion(
            roll, pitch, yaw
        )
        print(f"Azimuth: {azimuth}, Inclination: {inclination}, Toolface: {toolface}")
        print("=====")
