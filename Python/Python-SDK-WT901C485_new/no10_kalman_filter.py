import numpy as np
import pandas as pd

from no7_calc_from_acc_mag import calculate_angles, attitude_to_wellbore_quaternion


class KalmanFilter:
    def __init__(self, Q=1e-4, R=1e-2):
        """
        一维卡尔曼滤波器
        Q: 过程噪声协方差 (系统不确定性)
        R: 观测噪声协方差 (测量不确定性)
        """
        self.Q = Q  # 过程噪声协方差
        self.R = R  # 观测噪声协方差
        self.P = 1.0  # 估计误差协方差
        self.x = 0.0  # 初始状态估计
        self.K = 0.0  # 卡尔曼增益
        self.is_initialized = False

    def update(self, z):
        """
        卡尔曼滤波器更新
        z: 观测值 (测量值)
        """
        if not self.is_initialized:
            self.x = z
            self.is_initialized = True
            return self.x

        # 预测步骤
        # x_pred = x_prev (假设状态转移矩阵 F = 1)
        x_pred = self.x
        P_pred = self.P + self.Q

        # 更新步骤
        # 计算卡尔曼增益
        self.K = P_pred / (P_pred + self.R)

        # 更新状态估计
        self.x = x_pred + self.K * (z - x_pred)

        # 更新误差协方差
        self.P = (1 - self.K) * P_pred

        return self.x

    def reset(self):
        """重置滤波器"""
        self.P = 1.0
        self.x = 0.0
        self.K = 0.0
        self.is_initialized = False


class MultiAxisKalmanFilter:
    def __init__(self, Q=1e-4, R=1e-2):
        """
        多轴卡尔曼滤波器
        为每个轴创建独立的卡尔曼滤波器
        """
        self.Q = Q
        self.R = R
        self.filters = {}

    def filter_data(self, data):
        """
        对多维数据进行卡尔曼滤波
        data: 输入数据数组 [ax, ay, az, mx, my, mz]
        """
        filtered_data = np.zeros_like(data)
        axis_names = ["ax", "ay", "az", "mx", "my", "mz"]

        for i, axis in enumerate(axis_names):
            if axis not in self.filters:
                self.filters[axis] = KalmanFilter(self.Q, self.R)

            filtered_values = []
            for j in range(len(data)):
                filtered_value = self.filters[axis].update(data[j, i])
                filtered_values.append(filtered_value)

            filtered_data[:, i] = filtered_values

            # 重置滤波器以供下次使用
            self.filters[axis].reset()

        return filtered_data


def process_sensor_data_with_kalman_filter(data_list, real_angels, Q=1e-4, R=1e-2):
    """
    处理传感器数据序列并应用卡尔曼滤波

    data_list: 列表，每个元素为 [ax, ay, az, mx, my, mz]
    real_angels: 真实角度数据
    Q: 过程噪声协方差
    R: 观测噪声协方差
    """

    # 将数据转换为numpy数组便于处理
    data_array = np.array(data_list)

    # 创建多轴卡尔曼滤波器
    kalman_filter = MultiAxisKalmanFilter(Q=Q, R=R)

    # 对所有轴的数据进行卡尔曼滤波
    filtered_array = kalman_filter.filter_data(data_array)

    # 分离滤波后的数据
    ax_filtered = filtered_array[:, 0]
    ay_filtered = filtered_array[:, 1]
    az_filtered = filtered_array[:, 2]
    mx_filtered = filtered_array[:, 3]
    my_filtered = filtered_array[:, 4]
    mz_filtered = filtered_array[:, 5]

    # 计算每个滤波后数据点的井眼角度
    results = []
    for i in range(len(data_list)):
        roll, pitch, yaw = calculate_angles(
            ax_filtered[i],
            ay_filtered[i],
            az_filtered[i],
            mx_filtered[i],
            my_filtered[i],
            mz_filtered[i],
        )
        azimuth, inclination, toolface, quaternion = attitude_to_wellbore_quaternion(
            roll, pitch, yaw
        )

        real_roll = real_angels[i][0]
        real_pitch = real_angels[i][1]
        real_yaw = real_angels[i][2]
        real_azimuth, real_inclination, real_toolface, _ = (
            attitude_to_wellbore_quaternion(real_roll, real_pitch, real_yaw)
        )

        results.append(
            {
                "index": i,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "real_roll": real_roll,
                "real_pitch": real_pitch,
                "real_yaw": real_yaw,
                "diff_roll": roll - real_roll,
                "diff_pitch": pitch - real_pitch,
                "diff_yaw": yaw - real_yaw,
                "azimuth": azimuth,
                "inclination": inclination,
                "toolface": toolface,
                "real_azimuth": real_azimuth,
                "real_inclination": real_inclination,
                "real_toolface": real_toolface,
                "diff_azimuth": azimuth - real_azimuth,
                "diff_inclination": inclination - real_inclination,
                "diff_toolface": toolface - real_toolface,
                "raw_data": data_list[i],
                "filtered_data": [
                    ax_filtered[i],
                    ay_filtered[i],
                    az_filtered[i],
                    mx_filtered[i],
                    my_filtered[i],
                    mz_filtered[i],
                ],
            }
        )

    return results


if __name__ == "__main__":
    # L6964-6981 井段数据
    sample_data = [
        [0.144, 0.2842, 0.9419, -18.525, -31.694, -12.805],
        [0.1484, 0.2886, 0.9448, -18.538, -31.681, -12.805],
        [0.1455, 0.2886, 0.9438, -18.551, -31.668, -12.792],
        [0.1445, 0.29, 0.9473, -18.551, -31.668, -12.792],
        [0.1499, 0.2856, 0.9492, -18.551, -31.681, -12.792],
        [0.1479, 0.2808, 0.9482, -18.564, -31.668, -12.792],
        [0.1411, 0.2827, 0.9443, -18.564, -31.668, -12.792],
        [0.1411, 0.2891, 0.9463, -18.59, -31.668, -12.792],
        [0.1396, 0.2891, 0.9502, -18.59, -31.655, -12.792],
        [0.1465, 0.2891, 0.9521, -18.59, -31.655, -12.792],
        [0.147, 0.2881, 0.938, -18.603, -31.655, -12.792],
        [0.1523, 0.2817, 0.9409, -18.616, -31.642, -12.805],
        [0.1401, 0.2769, 0.9487, -18.629, -31.616, -12.831],
        [0.1377, 0.2803, 0.9536, -18.655, -31.603, -12.857],
        [0.1426, 0.2842, 0.9497, -18.668, -31.59, -12.883],
        [0.1392, 0.2822, 0.9517, -18.681, -31.577, -12.883],
        [0.1436, 0.2832, 0.9502, -18.694, -31.564, -12.909],
        [0.1431, 0.2896, 0.937, -18.681, -31.564, -12.909],
    ]

    # 对应的实际姿态角 [roll, pitch, yaw]
    real_angels = [
        [16.941, -8.525, -150.007],
        [16.963, -8.525, -149.996],
        [16.974, -8.52, -149.991],
        [16.974, -8.514, -149.985],
        [16.974, -8.503, -149.969],
        [16.985, -8.498, -149.925],
        [16.99, -8.503, -149.925],
        [16.985, -8.503, -149.936],
        [16.974, -8.492, -149.93],
        [16.963, -8.492, -149.903],
        [16.952, -8.498, -149.854],
        [16.913, -8.47, -149.799],
        [16.88, -8.438, -149.744],
        [16.859, -8.416, -149.711],
        [16.853, -8.416, -149.694],
        [16.859, -8.427, -149.694],
        [16.886, -8.438, -149.7],
        [16.908, -8.438, -149.7],
    ]

    print("=" * 100)
    print("卡尔曼滤波器测试")
    print("=" * 100)

    # 测试不同参数的卡尔曼滤波器
    test_params = [
        {"Q": 1e-5, "R": 1e-2, "name": "低过程噪声 (Q=1e-5, R=1e-2)"},
        {"Q": 1e-4, "R": 1e-2, "name": "中等过程噪声 (Q=1e-4, R=1e-2)"},
        {"Q": 1e-3, "R": 1e-2, "name": "高过程噪声 (Q=1e-3, R=1e-2)"},
        {"Q": 1e-4, "R": 1e-3, "name": "低观测噪声 (Q=1e-4, R=1e-3)"},
    ]

    for params in test_params:
        print(f"\n{params['name']}:")
        print("-" * 80)

        # 处理数据
        filtered_results = process_sensor_data_with_kalman_filter(
            sample_data, real_angels, Q=params["Q"], R=params["R"]
        )

        # 显示最后3个数据点的结果
        for result in filtered_results[-3:]:
            print(f"数据点 {result['index']}:")
            print(f"  原始数据: {[f'{x:.6f}' for x in result['raw_data']]}")
            print(f"  滤波数据: {[f'{x:.6f}' for x in result['filtered_data']]}")
            print(
                f"  姿态角: Roll={result['roll']:.2f}°, Pitch={result['pitch']:.2f}°, Yaw={result['yaw']:.2f}°"
            )
            print(
                f"  实际角: Roll={result['real_roll']:.2f}°, Pitch={result['real_pitch']:.2f}°, Yaw={result['real_yaw']:.2f}°"
            )
            print(
                f"  误差: ΔRoll={result['diff_roll']:.2f}°, ΔPitch={result['diff_pitch']:.2f}°, ΔYaw={result['diff_yaw']:.2f}°"
            )
            print(
                f"  井眼角度误差: ΔAzimuth={result['diff_azimuth']:.2f}°, ΔInclination={result['diff_inclination']:.2f}°, ΔToolface={result['diff_toolface']:.2f}°"
            )
            print()
