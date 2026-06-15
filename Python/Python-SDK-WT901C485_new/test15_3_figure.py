import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from test15_1_ackf_sensor import process_sensor_data_with_ackf_sensor
from test15_2_ukf_sensor import process_sensor_data_with_ukf_sensor
from test15_2_2_ukf_sensor import process_sensor_data_with_ukf_sensor_fixed
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

    # 模拟传感器数据的基础值
    base_acc = [0.144, 0.284, 0.942]  # ax, ay, az
    base_mag = [-18.5, -31.7, -12.8]  # mx, my, mz

    # 添加周期性变化和噪声
    data_list = []
    real_angels = []
    true_sensor_data = []  # 真实传感器数据（无噪声）

    for i in range(n_points):
        # 添加周期性变化
        acc_noise = np.random.normal(0, noise_level * 0.01, 3)
        mag_noise = np.random.normal(0, noise_level * 0.5, 3)

        # 添加一些周期性变化
        phase_shift = 0.1 * np.sin(2 * np.pi * t[i] / 10)

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

        true_sensor_data.append([ax_true, ay_true, az_true, mx_true, my_true, mz_true])

        roll_true, pitch_true, yaw_true = calculate_angles(
            ax_true, ay_true, az_true, mx_true, my_true, mz_true
        )
        real_angels.append([roll_true, pitch_true, yaw_true])

    return data_list, real_angels, true_sensor_data


def compare_filters(data_list, real_angels):
    """
    比较不同滤波器的效果 - 方案一：直接滤波传感器数据
    """
    print("开始滤波处理...")

    # 计算原始观测数据（角度）
    raw_data = []
    for i, (ax, ay, az, mx, my, mz) in enumerate(data_list):
        roll_obs, pitch_obs, yaw_obs = calculate_angles(ax, ay, az, mx, my, mz)
        raw_data.append([roll_obs, pitch_obs, yaw_obs])

    # ACKF滤波传感器数据
    print("ACKF滤波传感器数据中...")
    ackf_results = process_sensor_data_with_ackf_sensor(
        data_list,
        real_angels,
        Q=np.diag(
            [
                1e-5,
                1e-5,
                1e-5,
                1e-4,
                1e-4,
                1e-4,  # 传感器值
                1e-4,
                1e-4,
                1e-4,
                1e-3,
                1e-3,
                1e-3,
            ]
        ),  # 变化率
        R=np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]),  # 观测噪声
        dt=1.0,
        window_size=5,
        forget_factor=0.95,
    )

    # UKF滤波传感器数据
    print("UKF滤波传感器数据中...")
    ukf_results = process_sensor_data_with_ukf_sensor(
        data_list,
        real_angels,
        Q=np.diag(
            [
                1e-5,
                1e-5,
                1e-5,
                1e-4,
                1e-4,
                1e-4,  # 传感器值
                1e-4,
                1e-4,
                1e-4,
                1e-3,
                1e-3,
                1e-3,
            ]
        ),  # 变化率
        R=np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]),  # 观测噪声
        dt=1.0,
        alpha=1e-3,
        beta=2.0,
        kappa=0,
    )

    return raw_data, ackf_results, ukf_results


# 在compare_filters函数中修改UKF调用
# def compare_filters(data_list, real_angels):
#     """
#     比较不同滤波器的效果 - 方案一：直接滤波传感器数据
#     """
#     print("开始滤波处理...")

#     # 计算原始观测数据（角度）
#     raw_data = []
#     for i, (ax, ay, az, mx, my, mz) in enumerate(data_list):
#         roll_obs, pitch_obs, yaw_obs = calculate_angles(ax, ay, az, mx, my, mz)
#         raw_data.append([roll_obs, pitch_obs, yaw_obs])

#     # ACKF滤波传感器数据
#     print("ACKF滤波传感器数据中...")
#     ackf_results = process_sensor_data_with_ackf_sensor(
#         data_list,
#         real_angels,
#         Q=np.diag(
#             [
#                 1e-5,
#                 1e-5,
#                 1e-5,
#                 1e-4,
#                 1e-4,
#                 1e-4,  # 传感器值
#                 1e-4,
#                 1e-4,
#                 1e-4,
#                 1e-3,
#                 1e-3,
#                 1e-3,
#             ]
#         ),  # 变化率
#         R=np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]),  # 观测噪声
#         dt=1.0,
#         window_size=5,
#         forget_factor=0.95,
#     )

#     # UKF滤波传感器数据 - 使用修复版本
#     print("UKF滤波传感器数据中...")

#     ukf_results = process_sensor_data_with_ukf_sensor_fixed(
#         data_list,
#         real_angels,
#         Q=np.diag(
#             [
#                 # 1e-6,
#                 # 1e-6,
#                 # 1e-6,
#                 # 1e-5,
#                 # 1e-5,
#                 # 1e-5,  # 传感器值（更小）
#                 # 1e-5,
#                 # 1e-5,
#                 # 1e-5,
#                 # 1e-4,
#                 # 1e-4,
#                 # 1e-4,
#                 1e-5,
#                 1e-5,
#                 1e-5,
#                 1e-4,
#                 1e-4,
#                 1e-4,  # 传感器值
#                 1e-4,
#                 1e-4,
#                 1e-4,
#                 1e-3,
#                 1e-3,
#                 1e-3,
#             ]
#         ),  # 变化率（更小）
#         R=np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]),  # 观测噪声
#         dt=1.0,
#         alpha=0.01,  # 增大alpha
#         beta=2.0,
#         kappa=3 - 12,  # 设置合适的kappa
#     )

#     return raw_data, ackf_results, ukf_results


def plot_sensor_data_comparison(data_list, ackf_results, ukf_results, true_sensor_data):
    """
    绘制传感器数据比较图 - 方案一
    """
    n_points = len(data_list)
    x_axis = np.arange(n_points)

    # 提取传感器数据
    raw_sensor = np.array(data_list)
    true_sensor = np.array(true_sensor_data)

    # 从滤波结果中提取滤波后的传感器数据
    ackf_sensor = np.array([r["filtered_data"] for r in ackf_results])
    ukf_sensor = np.array([r["filtered_data"] for r in ukf_results])

    sensor_names = ["ax", "ay", "az", "mx", "my", "mz"]
    sensor_labels = [
        "加速度X (g)",
        "加速度Y (g)",
        "加速度Z (g)",
        "磁场X (μT)",
        "磁场Y (μT)",
        "磁场Z (μT)",
    ]

    # 创建图形 - 6个传感器数据
    fig, axes = plt.subplots(6, 2, figsize=(15, 20))
    fig.suptitle(
        "传感器数据滤波效果比较 (方案一：直接滤波传感器数据)",
        fontsize=16,
        fontweight="bold",
    )

    for i in range(6):
        # 左列：原始数据vs滤波数据
        ax1 = axes[i, 0]
        ax1.plot(
            x_axis, raw_sensor[:, i], "b-", alpha=0.7, linewidth=0.8, label="观测数据"
        )
        ax1.plot(x_axis, ackf_sensor[:, i], "r-", linewidth=1.2, label="ACKF滤波数据")
        ax1.plot(x_axis, ukf_sensor[:, i], "g-", linewidth=1.2, label="UKF滤波数据")
        ax1.plot(x_axis, true_sensor[:, i], "k-", linewidth=1.5, label="真实数据")

        ax1.set_ylabel(sensor_labels[i])
        ax1.set_xlabel("测量点数据点")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{sensor_names[i].upper()} 传感器数据比较")

        # 右列：误差比较
        ax2 = axes[i, 1]
        ackf_error = ackf_sensor[:, i] - true_sensor[:, i]
        ukf_error = ukf_sensor[:, i] - true_sensor[:, i]
        raw_error = raw_sensor[:, i] - true_sensor[:, i]

        ax2.plot(x_axis, raw_error, "b-", alpha=0.7, linewidth=0.8, label="观测误差")
        ax2.plot(x_axis, ackf_error, "r-", linewidth=1.2, label="ACKF误差")
        ax2.plot(x_axis, ukf_error, "g-", linewidth=1.2, label="UKF误差")
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)

        ax2.set_ylabel(f"{sensor_labels[i]} 误差")
        ax2.set_xlabel("测量点数据点")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"{sensor_names[i].upper()} 误差比较")

    plt.tight_layout()
    plt.show()


def plot_angle_comparison(raw_data, ackf_results, ukf_results, real_angels):
    """
    绘制角度比较图 - 方案一
    """
    n_points = len(raw_data)
    x_axis = np.arange(n_points)

    # 提取数据
    raw_angles = np.array(raw_data)
    ackf_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in ackf_results])
    ukf_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in ukf_results])
    real_angles = np.array(real_angels)

    angle_names = ["roll", "pitch", "yaw"]
    angle_labels = ["翻滚角 (°)", "俯仰角 (°)", "航向角 (°)"]

    # 创建图形
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(
        "角度滤波效果比较 (方案一：从滤波后传感器数据计算角度)",
        fontsize=16,
        fontweight="bold",
    )

    # 绘制角度比较
    for i in range(3):
        # 左列：原始数据vs滤波数据
        ax1 = axes[i, 0]
        ax1.plot(
            x_axis, raw_angles[:, i], "b-", alpha=0.7, linewidth=0.8, label="观测数据"
        )
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


def calculate_sensor_metrics(raw_sensor, filtered_sensor, true_sensor):
    """
    计算传感器数据的性能指标
    """
    raw_sensor = np.array(raw_sensor)
    filtered_sensor = np.array(filtered_sensor)
    true_sensor = np.array(true_sensor)

    # 计算RMSE
    raw_rmse = np.sqrt(np.mean((raw_sensor - true_sensor) ** 2, axis=0))
    filtered_rmse = np.sqrt(np.mean((filtered_sensor - true_sensor) ** 2, axis=0))

    # 计算MAE
    raw_mae = np.mean(np.abs(raw_sensor - true_sensor), axis=0)
    filtered_mae = np.mean(np.abs(filtered_sensor - true_sensor), axis=0)

    # 计算标准差
    raw_std = np.std(raw_sensor - true_sensor, axis=0)
    filtered_std = np.std(filtered_sensor - true_sensor, axis=0)

    return {
        "rmse": (raw_rmse, filtered_rmse),
        "mae": (raw_mae, filtered_mae),
        "std": (raw_std, filtered_std),
    }


def calculate_angle_metrics(raw_data, filtered_results, real_angels):
    """
    计算角度的性能指标
    """
    raw_angles = np.array(raw_data)
    filtered_angles = np.array(
        [[r["roll"], r["pitch"], r["yaw"]] for r in filtered_results]
    )
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


def print_comprehensive_metrics(
    raw_data, ackf_results, ukf_results, real_angels, data_list, true_sensor_data
):
    """
    打印全面的性能指标比较 - 方案一
    """
    # 角度指标
    ackf_angle_metrics = calculate_angle_metrics(raw_data, ackf_results, real_angels)
    ukf_angle_metrics = calculate_angle_metrics(raw_data, ukf_results, real_angels)

    # 传感器数据指标
    ackf_sensor = [r["filtered_data"] for r in ackf_results]
    ukf_sensor = [r["filtered_data"] for r in ukf_results]

    ackf_sensor_metrics = calculate_sensor_metrics(
        data_list, ackf_sensor, true_sensor_data
    )
    ukf_sensor_metrics = calculate_sensor_metrics(
        data_list, ukf_sensor, true_sensor_data
    )

    print("\n" + "=" * 80)
    print("滤波器性能全面比较 (方案一：直接滤波传感器数据)")
    print("=" * 80)

    # 角度性能
    angle_names = ["Roll", "Pitch", "Yaw"]
    print("\n【角度滤波性能】")
    for i, angle in enumerate(angle_names):
        print(f"\n{angle} 角度性能指标:")
        print(f"  原始数据 RMSE: {ackf_angle_metrics['rmse'][0][i]:.4f}°")
        print(f"  ACKF RMSE:     {ackf_angle_metrics['rmse'][1][i]:.4f}°")
        print(f"  UKF RMSE:      {ukf_angle_metrics['rmse'][1][i]:.4f}°")

        if ackf_angle_metrics["rmse"][0][i] > 0:
            print(
                f"  ACKF改善:      {((ackf_angle_metrics['rmse'][0][i] - ackf_angle_metrics['rmse'][1][i])/ackf_angle_metrics['rmse'][0][i]*100):.2f}%"
            )
        if ukf_angle_metrics["rmse"][0][i] > 0:
            print(
                f"  UKF改善:       {((ukf_angle_metrics['rmse'][0][i] - ukf_angle_metrics['rmse'][1][i])/ukf_angle_metrics['rmse'][0][i]*100):.2f}%"
            )

    # 传感器数据性能
    sensor_names = ["ax", "ay", "az", "mx", "my", "mz"]
    print("\n【传感器数据滤波性能】")
    for i, sensor in enumerate(sensor_names):
        print(f"\n{sensor} 传感器性能指标:")
        print(f"  原始数据 RMSE: {ackf_sensor_metrics['rmse'][0][i]:.6f}")
        print(f"  ACKF RMSE:     {ackf_sensor_metrics['rmse'][1][i]:.6f}")
        print(f"  UKF RMSE:      {ukf_sensor_metrics['rmse'][1][i]:.6f}")

        if ackf_sensor_metrics["rmse"][0][i] > 0:
            print(
                f"  ACKF改善:      {((ackf_sensor_metrics['rmse'][0][i] - ackf_sensor_metrics['rmse'][1][i])/ackf_sensor_metrics['rmse'][0][i]*100):.2f}%"
            )
        if ukf_sensor_metrics["rmse"][0][i] > 0:
            print(
                f"  UKF改善:       {((ukf_sensor_metrics['rmse'][0][i] - ukf_sensor_metrics['rmse'][1][i])/ukf_sensor_metrics['rmse'][0][i]*100):.2f}%"
            )


def export_data_to_excel(
    raw_data, ackf_results, ukf_results, real_angels, data_list, true_sensor_data
):
    """
    导出数据到Excel
    """
    # 传感器数据
    raw_sensor = np.array(data_list)
    true_sensor = np.array(true_sensor_data)
    ackf_sensor = np.array([r["filtered_data"] for r in ackf_results])
    ukf_sensor = np.array([r["filtered_data"] for r in ukf_results])

    # 角度数据
    raw_angles = np.array(raw_data)
    ackf_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in ackf_results])
    ukf_angles = np.array([[r["roll"], r["pitch"], r["yaw"]] for r in ukf_results])
    real_angles = np.array(real_angels)

    filename = "test15_sensor_filter_results.xlsx"

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # 传感器数据表
        sensor_data = []
        sensor_names = ["ax", "ay", "az", "mx", "my", "mz"]

        for i in range(len(data_list)):
            row = {"Index": i}
            for j, name in enumerate(sensor_names):
                row[f"{name}_raw"] = raw_sensor[i, j]
                row[f"{name}_true"] = true_sensor[i, j]
                row[f"{name}_ackf"] = ackf_sensor[i, j]
                row[f"{name}_ukf"] = ukf_sensor[i, j]
                row[f"{name}_ackf_error"] = ackf_sensor[i, j] - true_sensor[i, j]
                row[f"{name}_ukf_error"] = ukf_sensor[i, j] - true_sensor[i, j]
                row[f"{name}_raw_error"] = raw_sensor[i, j] - true_sensor[i, j]
            sensor_data.append(row)

        df_sensors = pd.DataFrame(sensor_data)
        df_sensors.to_excel(writer, sheet_name="传感器数据", index=False)

        # 角度数据表
        angle_data = []
        angle_names = ["roll", "pitch", "yaw"]

        for i in range(len(raw_data)):
            row = {"Index": i}
            for j, name in enumerate(angle_names):
                row[f"{name}_raw"] = raw_angles[i, j]
                row[f"{name}_true"] = real_angles[i, j]
                row[f"{name}_ackf"] = ackf_angles[i, j]
                row[f"{name}_ukf"] = ukf_angles[i, j]
                row[f"{name}_ackf_error"] = ackf_angles[i, j] - real_angles[i, j]
                row[f"{name}_ukf_error"] = ukf_angles[i, j] - real_angles[i, j]
                row[f"{name}_raw_error"] = raw_angles[i, j] - real_angles[i, j]
            angle_data.append(row)

        df_angles = pd.DataFrame(angle_data)
        df_angles.to_excel(writer, sheet_name="角度数据", index=False)

    print(f"数据已导出到: {filename}")


if __name__ == "__main__":
    print("生成测试数据...")
    print("=" * 80)
    print("方案一测试：直接滤波传感器数据 → 计算角度")
    print(
        "流程：[ax,ay,az,mx,my,mz] → 卡尔曼滤波 → [ax_f,ay_f,az_f,mx_f,my_f,mz_f] → calculate_angles → [roll,pitch,yaw]"
    )
    print("=" * 80)

    # 生成测试数据
    data_list, real_angels, true_sensor_data = generate_test_data(
        n_points=2000, noise_level=1.0
    )

    print(f"生成了 {len(data_list)} 个数据点")

    # 比较滤波器
    raw_data, ackf_results, ukf_results = compare_filters(data_list, real_angels)

    # 绘制传感器数据比较图
    print("绘制传感器数据比较图...")
    plot_sensor_data_comparison(data_list, ackf_results, ukf_results, true_sensor_data)

    # 绘制角度比较图
    print("绘制角度比较图...")
    plot_angle_comparison(raw_data, ackf_results, ukf_results, real_angels)

    # 打印性能指标
    print_comprehensive_metrics(
        raw_data, ackf_results, ukf_results, real_angels, data_list, true_sensor_data
    )

    # 导出数据到Excel
    print("\n导出数据到Excel...")
    export_data_to_excel(
        raw_data, ackf_results, ukf_results, real_angels, data_list, true_sensor_data
    )

    print("\n处理完成！")
