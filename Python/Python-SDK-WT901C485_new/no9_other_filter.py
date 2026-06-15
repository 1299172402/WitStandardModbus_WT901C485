import numpy as np
import pandas as pd
from scipy import signal

from no7_calc_from_acc_mag import calculate_angles, attitude_to_wellbore_quaternion


class LowPassFilter:
    def __init__(self, cutoff_freq=0.1, filter_order=2):
        """
        低通滤波器
        cutoff_freq: 截止频率 (0-1之间，相对于奈奎斯特频率)
        filter_order: 滤波器阶数
        """
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order

    def butterworth_filter(self, data):
        """
        巴特沃斯低通滤波
        data: 输入数据数组
        """
        if len(data) < 4:  # 需要足够的数据点
            return data

        # 设计巴特沃斯低通滤波器
        b, a = signal.butter(self.filter_order, self.cutoff_freq, btype="low")

        # 使用filtfilt进行零相位滤波
        filtered_data = signal.filtfilt(b, a, data)

        return filtered_data

    def exponential_filter(self, data, alpha=0.3):
        """
        指数移动平均滤波 (简单低通滤波)
        data: 输入数据数组
        alpha: 平滑因子 (0-1之间，越小越平滑)
        """
        filtered_data = np.zeros_like(data)
        filtered_data[0] = data[0]

        for i in range(1, len(data)):
            filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i - 1]

        return filtered_data


class SimpleFilter:
    def __init__(self, window_size=5):
        """
        简单移动平均滤波器
        window_size: 滑动窗口大小
        """
        self.window_size = window_size

    def moving_average_filter(self, data):
        """
        移动平均滤波
        data: 输入数据数组
        """
        if len(data) < self.window_size:
            return data

        filtered_data = []
        for i in range(len(data)):
            if i < self.window_size - 1:
                # 前面几个点，使用从开始到当前点的平均值
                window_data = data[0 : i + 1]
            else:
                # 使用滑动窗口
                window_data = data[i - self.window_size + 1 : i + 1]

            filtered_data.append(np.mean(window_data))

        return np.array(filtered_data)


def process_sensor_data_with_lowpass_filter(
    data_list, real_angels, filter_type="butterworth", **filter_params
):
    """
    处理传感器数据序列并应用低通滤波

    data_list: 列表，每个元素为 [ax, ay, az, mx, my, mz]
    real_angels: 真实角度数据
    filter_type: 滤波器类型 ('butterworth', 'exponential', 'moving_average')
    **filter_params: 滤波器参数
    """

    # 将数据转换为numpy数组便于处理
    data_array = np.array(data_list)

    # 分离各个传感器数据
    ax_data = data_array[:, 0]
    ay_data = data_array[:, 1]
    az_data = data_array[:, 2]
    mx_data = data_array[:, 3]
    my_data = data_array[:, 4]
    mz_data = data_array[:, 5]

    # 根据滤波器类型选择滤波器
    if filter_type == "butterworth":
        cutoff_freq = filter_params.get("cutoff_freq", 0.1)
        filter_order = filter_params.get("filter_order", 2)
        filter = LowPassFilter(cutoff_freq, filter_order)

        # 对每个轴的数据进行巴特沃斯滤波
        ax_filtered = filter.butterworth_filter(ax_data)
        ay_filtered = filter.butterworth_filter(ay_data)
        az_filtered = filter.butterworth_filter(az_data)
        mx_filtered = filter.butterworth_filter(mx_data)
        my_filtered = filter.butterworth_filter(my_data)
        mz_filtered = filter.butterworth_filter(mz_data)

    elif filter_type == "exponential":
        alpha = filter_params.get("alpha", 0.3)
        filter = LowPassFilter()

        # 对每个轴的数据进行指数滤波
        ax_filtered = filter.exponential_filter(ax_data, alpha)
        ay_filtered = filter.exponential_filter(ay_data, alpha)
        az_filtered = filter.exponential_filter(az_data, alpha)
        mx_filtered = filter.exponential_filter(mx_data, alpha)
        my_filtered = filter.exponential_filter(my_data, alpha)
        mz_filtered = filter.exponential_filter(mz_data, alpha)

    elif filter_type == "moving_average":
        window_size = filter_params.get("window_size", 5)
        filter = SimpleFilter(window_size)

        # 对每个轴的数据进行移动平均滤波
        ax_filtered = filter.moving_average_filter(ax_data)
        ay_filtered = filter.moving_average_filter(ay_data)
        az_filtered = filter.moving_average_filter(az_data)
        mx_filtered = filter.moving_average_filter(mx_data)
        my_filtered = filter.moving_average_filter(my_data)
        mz_filtered = filter.moving_average_filter(mz_data)
    else:
        raise ValueError(f"不支持的滤波器类型: {filter_type}")

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
    print("测试不同滤波器效果")
    print("=" * 100)

    # 测试巴特沃斯低通滤波
    print("\n1. 巴特沃斯低通滤波器 (cutoff_freq=0.15, order=2):")
    print("-" * 80)
    butterworth_results = process_sensor_data_with_lowpass_filter(
        sample_data,
        real_angels,
        filter_type="butterworth",
        cutoff_freq=0.15,
        filter_order=2,
    )

    for i, result in enumerate(butterworth_results[-3:]):  # 只显示最后3个结果
        print(f"数据点 {result['index']}:")
        print(
            f"  井眼角度误差: ΔAzimuth={result['diff_azimuth']:.2f}°, ΔInclination={result['diff_inclination']:.2f}°, ΔToolface={result['diff_toolface']:.2f}°"
        )

    # 测试指数移动平均滤波
    print("\n2. 指数移动平均滤波器 (alpha=0.3):")
    print("-" * 80)
    exponential_results = process_sensor_data_with_lowpass_filter(
        sample_data, real_angels, filter_type="exponential", alpha=0.3
    )

    for i, result in enumerate(exponential_results[-3:]):  # 只显示最后3个结果
        print(f"数据点 {result['index']}:")
        print(
            f"  井眼角度误差: ΔAzimuth={result['diff_azimuth']:.2f}°, ΔInclination={result['diff_inclination']:.2f}°, ΔToolface={result['diff_toolface']:.2f}°"
        )

    # 测试移动平均滤波 (对比)
    print("\n3. 移动平均滤波器 (window_size=3):")
    print("-" * 80)
    moving_avg_results = process_sensor_data_with_lowpass_filter(
        sample_data, real_angels, filter_type="moving_average", window_size=3
    )

    for i, result in enumerate(moving_avg_results[-3:]):  # 只显示最后3个结果
        print(f"数据点 {result['index']}:")
        print(
            f"  井眼角度误差: ΔAzimuth={result['diff_azimuth']:.2f}°, ΔInclination={result['diff_inclination']:.2f}°, ΔToolface={result['diff_toolface']:.2f}°"
        )
