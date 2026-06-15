import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from no14_ackf_filter import process_sensor_data_with_ackf
from no12_2_ukf_filter_modify import process_sensor_data_with_ukf
from no7_calc_from_acc_mag import calculate_angles

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


def generate_test_data(n_points=2000, noise_level=0.1):
    """
    生成测试数据，包含噪声的传感器数据
    """
    # 基础信号
    t = np.linspace(0, 20, n_points)

    # 模拟传感器数据的基础值（基于你的sample_data）
    # base_acc = [0.144, 0.284, 0.942]  # ax, ay, az
    # base_mag = [-18.5, -31.7, -12.8]  # mx, my, mz
    # base_acc = [0.031, 0.222, 0.982]
    # base_mag = [0.447, -41.61, -29.32]

    # base_acc = [-0.112793, -0.311523, 0.939453]
    # base_mag = [-3.452202, -20.613003, -48.219704]

    # base_acc = [0.319824, 0.307373, 0.891846]
    # base_mag = [11.697798, -29.463005, -38.769699]


    base_acc = [-0.104248, 0.296631, 0.944336]
    base_mag = [10.797798, -39.663002, -37.419701]


    # 添加周期性变化和噪声
    data_list = []
    real_angels = []

    for i in range(n_points):
        # 添加周期性变化
        acc_noise = np.random.normal(0, noise_level * 0.01, 3)
        mag_noise = np.random.normal(0, noise_level * 0.5, 3)

        # 添加一些周期性变化
        phase_shift = 3 * np.sin(2 * np.pi * t[i] / 10)

        ax = base_acc[0] + acc_noise[0] + phase_shift * 0.01
        ay = base_acc[1] + acc_noise[1] + phase_shift * 0.01
        az = base_acc[2] + acc_noise[2]

        mx = base_mag[0] + mag_noise[0] + phase_shift
        my = base_mag[1] + mag_noise[1] + phase_shift
        mz = base_mag[2] + mag_noise[2]

        data_list.append([ax, ay, az, mx, my, mz])

        # 计算真实角度（无噪声版本）
        ax_true = base_acc[0] + phase_shift * 0.01
        ay_true = base_acc[1] + phase_shift * 0.01
        az_true = base_acc[2]
        mx_true = base_mag[0] + phase_shift
        my_true = base_mag[1] + phase_shift
        mz_true = base_mag[2]

        roll_true, pitch_true, yaw_true = calculate_angles(ax_true, ay_true, az_true, mx_true, my_true, mz_true)
        real_angels.append([roll_true, pitch_true, yaw_true])

    return data_list, real_angels


def compare_filters(data_list, real_angels):
    """
    比较不同滤波器的效果
    """
    print("开始滤波处理...")

    # 计算原始观测数据
    raw_data = []
    for i, (ax, ay, az, mx, my, mz) in enumerate(data_list):
        roll_obs, pitch_obs, yaw_obs = calculate_angles(ax, ay, az, mx, my, mz)
        raw_data.append([roll_obs, pitch_obs, yaw_obs])

    # ACKF滤波
    print("ACKF滤波中...")
    ackf_results = process_sensor_data_with_ackf(
        data_list,
        real_angels,
        Q=np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]),
        R=np.diag([1e-2, 1e-2, 1e-2]),
        dt=1.0,
        window_size=5,
        forget_factor=0.95,
    )

    # UKF滤波
    print("UKF滤波中...")
    ukf_results = process_sensor_data_with_ukf(
        data_list,
        real_angels,
        Q=np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]),
        R=np.diag([1e-2, 1e-2, 1e-2]),
        dt=1.0,
        alpha=1e-3,
        beta=2.0,
        kappa=0,
    )

    return raw_data, ackf_results, ukf_results


# def plot_comparison(raw_data, ackf_results, ukf_results, real_angels):
#     """
#     绘制比较图
#     """
#     n_points = len(raw_data)
#     x_axis = np.arange(n_points)

#     # 提取数据
#     raw_angles = np.array(raw_data)
#     ackf_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in ackf_results])
#     ukf_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in ukf_results])
#     real_angles = np.array(real_angels)

#     angle_names = ["roll", "pitch", "yaw"]
#     angle_labels = ["翻滚角 (°)", "俯仰角 (°)", "航向角 (°)"]

#     # 创建图形
#     fig, axes = plt.subplots(3, 2, figsize=(15, 12))
#     fig.suptitle("滤波器效果比较", fontsize=16, fontweight="bold")

#     # 绘制角度比较
#     for i in range(3):
#         # 左列：原始数据vs滤波数据
#         ax1 = axes[i, 0]
#         ax1.plot(
#             x_axis, raw_angles[:, i], "b-", alpha=0.7, linewidth=0.8, label="观测数据"
#         )
#         ax1.plot(x_axis, ukf_angles[:, i], "g-", linewidth=1.2, label="UKF滤波数据")
#         ax1.plot(x_axis, ackf_angles[:, i], "r-", linewidth=1.2, label="ACKF滤波数据")
#         ax1.plot(x_axis, real_angles[:, i], "k-", linewidth=1.5, label="真实数据")

#         ax1.set_ylabel(angle_labels[i])
#         ax1.set_xlabel("测量点数据点")
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
#         ax1.set_title(f"{angle_names[i].upper()} 角度比较")

#         # 右列：误差比较
#         ax2 = axes[i, 1]
#         ackf_error = ackf_angles[:, i] - real_angles[:, i]
#         ukf_error = ukf_angles[:, i] - real_angles[:, i]
#         raw_error = raw_angles[:, i] - real_angles[:, i]

#         ax2.plot(x_axis, raw_error, "b-", alpha=0.7, linewidth=0.8, label="观测误差")
#         ax2.plot(x_axis, ukf_error, "g-", linewidth=1.2, label="UKF误差")
#         ax2.plot(x_axis, ackf_error, "r-", linewidth=1.2, label="ACKF误差")
#         ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)

#         ax2.set_ylabel(f"{angle_labels[i]} 误差")
#         ax2.set_xlabel("测量点数据点")
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
#         ax2.set_title(f"{angle_names[i].upper()} 误差比较")

#     plt.tight_layout()
#     plt.show()

#     # 创建类似论文图的格式
#     fig2, axes2 = plt.subplots(6, 1, figsize=(12, 16))
#     fig2.suptitle("改进ACKF滤波结果", fontsize=16, fontweight="bold")

#     # 绘制6个子图
#     subplot_data = [
#         (raw_angles[:, 0], "观测数据", "Roll (°)"),
#         (ackf_angles[:, 0], "ACKF滤波数据", "Roll (°)"),
#         (raw_angles[:, 1], "观测数据", "Pitch (°)"),
#         (ackf_angles[:, 1], "ACKF滤波数据", "Pitch (°)"),
#         (raw_angles[:, 2], "观测数据", "Yaw (°)"),
#         (ackf_angles[:, 2], "ACKF滤波数据", "Yaw (°)"),
#     ]

#     colors = ["blue", "red", "blue", "red", "blue", "red"]

#     for i, (data, label, ylabel) in enumerate(subplot_data):
#         axes2[i].plot(x_axis, data, color=colors[i], linewidth=1.0, label=label)
#         axes2[i].plot(
#             x_axis,
#             real_angles[:, i // 2],
#             "k--",
#             linewidth=1.5,
#             alpha=0.7,
#             label="真实数据",
#         )
#         axes2[i].set_ylabel(ylabel)
#         axes2[i].legend()
#         axes2[i].grid(True, alpha=0.3)
#         axes2[i].set_xlim(0, len(x_axis))

#     axes2[-1].set_xlabel("测量点数据点")
#     plt.tight_layout()
#     plt.show()


def plot_comparison(raw_data, ackf_results, ukf_results, real_angels):
    """
    绘制比较图
    """
    n_points = len(raw_data)
    x_axis = np.arange(n_points)

    # 提取数据
    raw_angles = np.array(raw_data)
    ackf_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in ackf_results])
    ukf_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in ukf_results])
    real_angles = np.array(real_angels)

    # 提取井眼角度数据
    ackf_wellbore = np.array(
        [
            [
                r["azimuth"] if r["azimuth"] <= 180 else r["azimuth"] - 360,
                r["inclination"],
                r["toolface"],
            ]
            for r in ackf_results
        ]
    )
    ukf_wellbore = np.array(
        [
            [
                r["azimuth"] if r["azimuth"] <= 180 else r["azimuth"] - 360,
                r["inclination"],
                r["toolface"],
            ]
            for r in ukf_results
        ]
    )
    real_wellbore = np.array([[r["real_azimuth"], r["real_inclination"], r["real_toolface"]] for r in ackf_results])

    # 计算观测数据的井眼角度
    raw_wellbore = []
    for i in range(len(raw_data)):
        from no7_calc_from_acc_mag import attitude_to_wellbore_quaternion

        azimuth_obs, inclination_obs, toolface_obs, _ = attitude_to_wellbore_quaternion(raw_angles[i, 0], raw_angles[i, 1], raw_angles[i, 2])

        if azimuth_obs > 180:  # 目前为0度附近，有些值会串到360度附近
            azimuth_obs -= 360

        raw_wellbore.append([azimuth_obs, inclination_obs, toolface_obs])
    raw_wellbore = np.array(raw_wellbore)

    angle_names = ["roll", "pitch", "yaw"]
    angle_labels = ["翻滚角 (°)", "俯仰角 (°)", "航向角 (°)"]

    wellbore_names = ["azimuth", "inclination", "toolface"]
    wellbore_labels = ["方位角 (°)", "井斜角 (°)", "工具面角 (°)"]

    # 创建姿态角对比图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("姿态角滤波器效果比较", fontsize=16, fontweight="bold")

    # 绘制姿态角比较
    for i in range(3):
        # 左列：原始数据vs滤波数据
        ax1 = axes[i, 0]
        ax1.plot(x_axis, raw_angles[:, i], "b-", alpha=0.7, linewidth=0.8, label="观测数据")
        ax1.plot(x_axis, ukf_angles[:, i], "g-", linewidth=1.2, label="UKF滤波数据")
        ax1.plot(x_axis, ackf_angles[:, i], "r-", linewidth=1.2, label="ACKF滤波数据")
        ax1.plot(x_axis, real_angles[:, i], "k-", linewidth=1.5, label="真实数据")

        ax1.set_ylabel(angle_labels[i])
        ax1.set_xlabel("测量点数据点")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{angle_names[i].upper()} 角度比较")

        # 右列：误差比较
        ax2 = axes[i, 1]
        ackf_error = ackf_angles[:, i] - real_angles[:, i]
        ukf_error = ukf_angles[:, i] - real_angles[:, i]
        raw_error = raw_angles[:, i] - real_angles[:, i]

        ax2.plot(x_axis, raw_error, "b-", alpha=0.7, linewidth=0.8, label="观测误差")
        ax2.plot(x_axis, ukf_error, "g-", linewidth=1.2, label="UKF误差")
        ax2.plot(x_axis, ackf_error, "r-", linewidth=1.2, label="ACKF误差")
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)

        ax2.set_ylabel(f"{angle_labels[i]} 误差")
        ax2.set_xlabel("测量点数据点")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"{angle_names[i].upper()} 误差比较")

    plt.tight_layout()
    plt.show()

    # 创建井眼角度对比图
    fig2, axes2 = plt.subplots(3, 2, figsize=(15, 12))
    fig2.suptitle("井眼角度滤波器效果比较", fontsize=16, fontweight="bold")

    # 绘制井眼角度比较
    for i in range(3):
        # 左列：观测数据vs滤波数据vs真实数据
        ax1 = axes2[i, 0]
        ax1.plot(x_axis, raw_wellbore[:, i], "b-", alpha=0.7, linewidth=0.8, label="观测数据")
        ax1.plot(x_axis, ukf_wellbore[:, i], "g-", linewidth=1.2, label="UKF滤波数据")
        ax1.plot(x_axis, ackf_wellbore[:, i], "r-", linewidth=1.2, label="ACKF滤波数据")
        ax1.plot(x_axis, real_wellbore[:, i], "k-", linewidth=1.5, label="真实数据")

        ax1.set_ylabel(wellbore_labels[i])
        ax1.set_xlabel("测量点数据点")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{wellbore_names[i].upper()} 角度比较")

        # 右列：误差比较
        ax2 = axes2[i, 1]
        raw_wellbore_error = raw_wellbore[:, i] - real_wellbore[:, i]
        ackf_wellbore_error = ackf_wellbore[:, i] - real_wellbore[:, i]
        ukf_wellbore_error = ukf_wellbore[:, i] - real_wellbore[:, i]

        ax2.plot(x_axis, raw_wellbore_error, "b-", alpha=0.7, linewidth=0.8, label="观测误差")
        ax2.plot(x_axis, ukf_wellbore_error, "g-", linewidth=1.2, label="UKF误差")
        ax2.plot(x_axis, ackf_wellbore_error, "r-", linewidth=1.2, label="ACKF误差")
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)

        ax2.set_ylabel(f"{wellbore_labels[i]} 误差")
        ax2.set_xlabel("测量点数据点")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"{wellbore_names[i].upper()} 误差比较")

    plt.tight_layout()
    plt.show()

    # 创建类似论文图的格式（保持原有的姿态角图）
    fig3, axes3 = plt.subplots(6, 1, figsize=(12, 16))
    fig3.suptitle("改进ACKF滤波结果", fontsize=16, fontweight="bold")

    # 绘制6个子图
    subplot_data = [
        (raw_angles[:, 0], "观测数据", "Roll (°)"),
        (ackf_angles[:, 0], "ACKF滤波数据", "Roll (°)"),
        (raw_angles[:, 1], "观测数据", "Pitch (°)"),
        (ackf_angles[:, 1], "ACKF滤波数据", "Pitch (°)"),
        (raw_angles[:, 2], "观测数据", "Yaw (°)"),
        (ackf_angles[:, 2], "ACKF滤波数据", "Yaw (°)"),
    ]

    colors = ["blue", "red", "blue", "red", "blue", "red"]

    for i, (data, label, ylabel) in enumerate(subplot_data):
        axes3[i].plot(x_axis, data, color=colors[i], linewidth=1.0, label=label)
        axes3[i].plot(
            x_axis,
            real_angles[:, i // 2],
            "k--",
            linewidth=1.5,
            alpha=0.7,
            label="真实数据",
        )
        axes3[i].set_ylabel(ylabel)
        axes3[i].legend()
        axes3[i].grid(True, alpha=0.3)
        axes3[i].set_xlim(0, len(x_axis))

    axes3[-1].set_xlabel("测量点数据点")
    plt.tight_layout()
    plt.show()


def calculate_metrics(raw_data, filtered_results, real_angels):
    """
    计算滤波器性能指标
    """
    raw_angles = np.array(raw_data)
    filtered_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in filtered_results])
    real_angles = np.array(real_angels)

    # 计算RMSE
    raw_rmse = np.sqrt(np.mean((raw_angles - real_angles) ** 2, axis=0))
    filtered_rmse = np.sqrt(np.mean((filtered_angles - real_angles) ** 2, axis=0))

    # 计算MAE
    raw_mae = np.mean(np.abs(raw_angles - real_angles), axis=0)
    filtered_mae = np.mean(np.abs(filtered_angles - real_angles), axis=0)

    # 计算标准差
    raw_std = np.std(raw_angles - real_angles, axis=0)
    filtered_std = np.std(filtered_angles - real_angles, axis=0)

    return {
        "rmse": (raw_rmse, filtered_rmse),
        "mae": (raw_mae, filtered_mae),
        "std": (raw_std, filtered_std),
    }


def print_metrics_comparison(raw_data, ackf_results, ukf_results, real_angels):
    """
    打印性能指标比较
    """
    ackf_metrics = calculate_metrics(raw_data, ackf_results, real_angels)
    ukf_metrics = calculate_metrics(raw_data, ukf_results, real_angels)

    # 计算井眼角度性能指标
    ackf_wellbore = np.array(
        [
            [
                r["azimuth"] if r["azimuth"] <= 180 else r["azimuth"] - 360,
                r["inclination"],
                r["toolface"],
            ]
            for r in ackf_results
        ]
    )
    ukf_wellbore = np.array(
        [
            [
                r["azimuth"] if r["azimuth"] <= 180 else r["azimuth"] - 360,
                r["inclination"],
                r["toolface"],
            ]
            for r in ukf_results
        ]
    )
    real_wellbore = np.array([[r["real_azimuth"], r["real_inclination"], r["real_toolface"]] for r in ackf_results])

    # 计算观测数据的井眼角度
    raw_wellbore = []
    raw_angles = np.array(raw_data)
    for i in range(len(raw_data)):
        from no7_calc_from_acc_mag import attitude_to_wellbore_quaternion

        azimuth_obs, inclination_obs, toolface_obs, _ = attitude_to_wellbore_quaternion(raw_angles[i, 0], raw_angles[i, 1], raw_angles[i, 2])
        if azimuth_obs > 180:
            azimuth_obs -= 360
        raw_wellbore.append([azimuth_obs, inclination_obs, toolface_obs])
    raw_wellbore = np.array(raw_wellbore)

    # 计算RMSE
    raw_wellbore_rmse = np.sqrt(np.mean((raw_wellbore - real_wellbore) ** 2, axis=0))
    ackf_wellbore_rmse = np.sqrt(np.mean((ackf_wellbore - real_wellbore) ** 2, axis=0))
    ukf_wellbore_rmse = np.sqrt(np.mean((ukf_wellbore - real_wellbore) ** 2, axis=0))

    angle_names = ["Roll", "Pitch", "Yaw"]
    wellbore_names = ["Azimuth", "Inclination", "Toolface"]

    print("\n" + "=" * 80)
    print("滤波器性能比较")
    print("=" * 80)

    print("\n姿态角性能指标:")
    for i, angle in enumerate(angle_names):
        print(f"\n{angle} 角度性能指标:")
        print(f"  原始数据 RMSE: {ackf_metrics['rmse'][0][i]:.4f}°")
        print(f"  ACKF RMSE:     {ackf_metrics['rmse'][1][i]:.4f}°")
        print(f"  UKF RMSE:      {ukf_metrics['rmse'][1][i]:.4f}°")
        print(f"  ACKF改善:      {((ackf_metrics['rmse'][0][i] - ackf_metrics['rmse'][1][i])/ackf_metrics['rmse'][0][i]*100):.2f}%")
        print(f"  UKF改善:       {((ukf_metrics['rmse'][0][i] - ukf_metrics['rmse'][1][i])/ukf_metrics['rmse'][0][i]*100):.2f}%")

    print("\n井眼角度性能指标:")
    for i, angle in enumerate(wellbore_names):
        print(f"\n{angle} 角度性能指标:")
        print(f"  观测数据 RMSE: {raw_wellbore_rmse[i]:.4f}°")
        print(f"  ACKF RMSE:     {ackf_wellbore_rmse[i]:.4f}°")
        print(f"  UKF RMSE:      {ukf_wellbore_rmse[i]:.4f}°")

        # 计算改善百分比
        ackf_improvement = ((raw_wellbore_rmse[i] - ackf_wellbore_rmse[i]) / raw_wellbore_rmse[i] * 100) if raw_wellbore_rmse[i] != 0 else 0
        ukf_improvement = ((raw_wellbore_rmse[i] - ukf_wellbore_rmse[i]) / raw_wellbore_rmse[i] * 100) if raw_wellbore_rmse[i] != 0 else 0
        ackf_vs_ukf = ((ukf_wellbore_rmse[i] - ackf_wellbore_rmse[i]) / ukf_wellbore_rmse[i] * 100) if ukf_wellbore_rmse[i] != 0 else 0

        print(f"  ACKF改善:      {ackf_improvement:.2f}%")
        print(f"  UKF改善:       {ukf_improvement:.2f}%")
        print(f"  ACKF相比UKF改善: {ackf_vs_ukf:.2f}%")


import pandas as pd
import os


def export_data_to_excel(
    raw_data,
    ackf_results,
    ukf_results,
    real_angels,
    filename="test_no20_260514_1152_filter_comparison_data.xlsx",
):
    """
    导出所有图表数据到Excel文件
    """
    print(f"正在导出数据到 {filename}...")

    n_points = len(raw_data)
    x_axis = np.arange(n_points)

    # 提取姿态角数据
    raw_angles = np.array(raw_data)
    ackf_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in ackf_results])
    ukf_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in ukf_results])
    real_angles = np.array(real_angels)

    # 提取井眼角度数据
    ackf_wellbore = np.array(
        [
            [
                r["azimuth"] if r["azimuth"] <= 180 else r["azimuth"] - 360,
                r["inclination"],
                r["toolface"],
            ]
            for r in ackf_results
        ]
    )
    ukf_wellbore = np.array(
        [
            [
                r["azimuth"] if r["azimuth"] <= 180 else r["azimuth"] - 360,
                r["inclination"],
                r["toolface"],
            ]
            for r in ukf_results
        ]
    )
    real_wellbore = np.array([[r["real_azimuth"], r["real_inclination"], r["real_toolface"]] for r in ackf_results])

    # 计算观测数据的井眼角度
    raw_wellbore = []
    for i in range(len(raw_data)):
        from no7_calc_from_acc_mag import attitude_to_wellbore_quaternion

        azimuth_obs, inclination_obs, toolface_obs, _ = attitude_to_wellbore_quaternion(raw_angles[i, 0], raw_angles[i, 1], raw_angles[i, 2])
        if azimuth_obs > 180:
            azimuth_obs -= 360
        raw_wellbore.append([azimuth_obs, inclination_obs, toolface_obs])
    raw_wellbore = np.array(raw_wellbore)

    # 计算误差
    raw_angle_errors = raw_angles - real_angles
    ackf_angle_errors = ackf_angles - real_angles
    ukf_angle_errors = ukf_angles - real_angles

    raw_wellbore_errors = raw_wellbore - real_wellbore
    ackf_wellbore_errors = ackf_wellbore - real_wellbore
    ukf_wellbore_errors = ukf_wellbore - real_wellbore

    # 创建Excel写入器
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:

        # 1. 姿态角原始数据表
        attitude_data = pd.DataFrame(
            {
                "数据点": x_axis,
                "观测Roll(°)": raw_angles[:, 0],
                "观测Pitch(°)": raw_angles[:, 1],
                "观测Yaw(°)": raw_angles[:, 2],
                "ACKF_Roll(°)": ackf_angles[:, 0],
                "ACKF_Pitch(°)": ackf_angles[:, 1],
                "ACKF_Yaw(°)": ackf_angles[:, 2],
                "UKF_Roll(°)": ukf_angles[:, 0],
                "UKF_Pitch(°)": ukf_angles[:, 1],
                "UKF_Yaw(°)": ukf_angles[:, 2],
                "真实Roll(°)": real_angles[:, 0],
                "真实Pitch(°)": real_angles[:, 1],
                "真实Yaw(°)": real_angles[:, 2],
            }
        )
        attitude_data.to_excel(writer, sheet_name="姿态角数据", index=False)

        # 2. 姿态角误差数据表
        attitude_error_data = pd.DataFrame(
            {
                "数据点": x_axis,
                "观测Roll误差(°)": raw_angle_errors[:, 0],
                "观测Pitch误差(°)": raw_angle_errors[:, 1],
                "观测Yaw误差(°)": raw_angle_errors[:, 2],
                "ACKF_Roll误差(°)": ackf_angle_errors[:, 0],
                "ACKF_Pitch误差(°)": ackf_angle_errors[:, 1],
                "ACKF_Yaw误差(°)": ackf_angle_errors[:, 2],
                "UKF_Roll误差(°)": ukf_angle_errors[:, 0],
                "UKF_Pitch误差(°)": ukf_angle_errors[:, 1],
                "UKF_Yaw误差(°)": ukf_angle_errors[:, 2],
            }
        )
        attitude_error_data.to_excel(writer, sheet_name="姿态角误差", index=False)

        # 3. 井眼角度原始数据表
        wellbore_data = pd.DataFrame(
            {
                "数据点": x_axis,
                "观测方位角(°)": raw_wellbore[:, 0],
                "观测井斜角(°)": raw_wellbore[:, 1],
                "观测工具面角(°)": raw_wellbore[:, 2],
                "ACKF方位角(°)": ackf_wellbore[:, 0],
                "ACKF井斜角(°)": ackf_wellbore[:, 1],
                "ACKF工具面角(°)": ackf_wellbore[:, 2],
                "UKF方位角(°)": ukf_wellbore[:, 0],
                "UKF井斜角(°)": ukf_wellbore[:, 1],
                "UKF工具面角(°)": ukf_wellbore[:, 2],
                "真实方位角(°)": real_wellbore[:, 0],
                "真实井斜角(°)": real_wellbore[:, 1],
                "真实工具面角(°)": real_wellbore[:, 2],
            }
        )
        wellbore_data.to_excel(writer, sheet_name="井眼角度数据", index=False)

        # 4. 井眼角度误差数据表
        wellbore_error_data = pd.DataFrame(
            {
                "数据点": x_axis,
                "观测方位角误差(°)": raw_wellbore_errors[:, 0],
                "观测井斜角误差(°)": raw_wellbore_errors[:, 1],
                "观测工具面角误差(°)": raw_wellbore_errors[:, 2],
                "ACKF方位角误差(°)": ackf_wellbore_errors[:, 0],
                "ACKF井斜角误差(°)": ackf_wellbore_errors[:, 1],
                "ACKF工具面角误差(°)": ackf_wellbore_errors[:, 2],
                "UKF方位角误差(°)": ukf_wellbore_errors[:, 0],
                "UKF井斜角误差(°)": ukf_wellbore_errors[:, 1],
                "UKF工具面角误差(°)": ukf_wellbore_errors[:, 2],
            }
        )
        wellbore_error_data.to_excel(writer, sheet_name="井眼角度误差", index=False)

        # 5. 性能指标汇总表
        ackf_metrics = calculate_metrics(raw_data, ackf_results, real_angels)
        ukf_metrics = calculate_metrics(raw_data, ukf_results, real_angels)

        # 计算井眼角度RMSE
        raw_wellbore_rmse = np.sqrt(np.mean((raw_wellbore - real_wellbore) ** 2, axis=0))
        ackf_wellbore_rmse = np.sqrt(np.mean((ackf_wellbore - real_wellbore) ** 2, axis=0))
        ukf_wellbore_rmse = np.sqrt(np.mean((ukf_wellbore - real_wellbore) ** 2, axis=0))

        metrics_summary = pd.DataFrame(
            {
                "角度类型": [
                    "Roll",
                    "Pitch",
                    "Yaw",
                    "Azimuth",
                    "Inclination",
                    "Toolface",
                ],
                "观测数据RMSE(°)": [
                    ackf_metrics["rmse"][0][0],
                    ackf_metrics["rmse"][0][1],
                    ackf_metrics["rmse"][0][2],
                    raw_wellbore_rmse[0],
                    raw_wellbore_rmse[1],
                    raw_wellbore_rmse[2],
                ],
                "ACKF_RMSE(°)": [
                    ackf_metrics["rmse"][1][0],
                    ackf_metrics["rmse"][1][1],
                    ackf_metrics["rmse"][1][2],
                    ackf_wellbore_rmse[0],
                    ackf_wellbore_rmse[1],
                    ackf_wellbore_rmse[2],
                ],
                "UKF_RMSE(°)": [
                    ukf_metrics["rmse"][1][0],
                    ukf_metrics["rmse"][1][1],
                    ukf_metrics["rmse"][1][2],
                    ukf_wellbore_rmse[0],
                    ukf_wellbore_rmse[1],
                    ukf_wellbore_rmse[2],
                ],
                "ACKF改善率(%)": [
                    ((ackf_metrics["rmse"][0][0] - ackf_metrics["rmse"][1][0]) / ackf_metrics["rmse"][0][0] * 100),
                    ((ackf_metrics["rmse"][0][1] - ackf_metrics["rmse"][1][1]) / ackf_metrics["rmse"][0][1] * 100),
                    ((ackf_metrics["rmse"][0][2] - ackf_metrics["rmse"][1][2]) / ackf_metrics["rmse"][0][2] * 100),
                    ((raw_wellbore_rmse[0] - ackf_wellbore_rmse[0]) / raw_wellbore_rmse[0] * 100 if raw_wellbore_rmse[0] != 0 else 0),
                    ((raw_wellbore_rmse[1] - ackf_wellbore_rmse[1]) / raw_wellbore_rmse[1] * 100 if raw_wellbore_rmse[1] != 0 else 0),
                    ((raw_wellbore_rmse[2] - ackf_wellbore_rmse[2]) / raw_wellbore_rmse[2] * 100 if raw_wellbore_rmse[2] != 0 else 0),
                ],
                "UKF改善率(%)": [
                    ((ukf_metrics["rmse"][0][0] - ukf_metrics["rmse"][1][0]) / ukf_metrics["rmse"][0][0] * 100),
                    ((ukf_metrics["rmse"][0][1] - ukf_metrics["rmse"][1][1]) / ukf_metrics["rmse"][0][1] * 100),
                    ((ukf_metrics["rmse"][0][2] - ukf_metrics["rmse"][1][2]) / ukf_metrics["rmse"][0][2] * 100),
                    ((raw_wellbore_rmse[0] - ukf_wellbore_rmse[0]) / raw_wellbore_rmse[0] * 100 if raw_wellbore_rmse[0] != 0 else 0),
                    ((raw_wellbore_rmse[1] - ukf_wellbore_rmse[1]) / raw_wellbore_rmse[1] * 100 if raw_wellbore_rmse[1] != 0 else 0),
                    ((raw_wellbore_rmse[2] - ukf_wellbore_rmse[2]) / raw_wellbore_rmse[2] * 100 if raw_wellbore_rmse[2] != 0 else 0),
                ],
            }
        )
        metrics_summary.to_excel(writer, sheet_name="性能指标汇总", index=False)

        # 6. 论文图数据 (6个子图的数据)
        paper_figure_data = pd.DataFrame(
            {
                "数据点": x_axis,
                "观测Roll(°)": raw_angles[:, 0],
                "ACKF_Roll(°)": ackf_angles[:, 0],
                "真实Roll(°)": real_angles[:, 0],
                "观测Pitch(°)": raw_angles[:, 1],
                "ACKF_Pitch(°)": ackf_angles[:, 1],
                "真实Pitch(°)": real_angles[:, 1],
                "观测Yaw(°)": raw_angles[:, 2],
                "ACKF_Yaw(°)": ackf_angles[:, 2],
                "真实Yaw(°)": real_angles[:, 2],
            }
        )
        paper_figure_data.to_excel(writer, sheet_name="论文图数据", index=False)

    print(f"数据已成功导出到 {os.path.abspath(filename)}")
    print(f"Excel文件包含以下工作表:")
    print("  - 姿态角数据: 原始观测、滤波结果和真实值")
    print("  - 姿态角误差: 各方法的误差数据")
    print("  - 井眼角度数据: 井眼角度的原始观测、滤波结果和真实值")
    print("  - 井眼角度误差: 井眼角度的误差数据")
    print("  - 性能指标汇总: RMSE和改善率统计")
    print("  - 论文图数据: 用于绘制论文图的数据")


if __name__ == "__main__":
    print("生成测试数据...")

    # 生成测试数据
    data_list, real_angels = generate_test_data(n_points=2000, noise_level=0.5)

    print(f"生成了 {len(data_list)} 个数据点")

    # 比较滤波器
    raw_data, ackf_results, ukf_results = compare_filters(data_list, real_angels)

    # print(ackf_results)
    for r in ackf_results:
        r["real_azimuth"] = r["real_azimuth"] if r["real_azimuth"] <= 180 else r["real_azimuth"] - 360

    # 绘制比较图
    print("绘制比较图...")
    plot_comparison(raw_data, ackf_results, ukf_results, real_angels)

    # 打印性能指标
    print_metrics_comparison(raw_data, ackf_results, ukf_results, real_angels)

    # 导出数据到Excel
    export_data_to_excel(raw_data, ackf_results, ukf_results, real_angels)

    print("\n处理完成！")
