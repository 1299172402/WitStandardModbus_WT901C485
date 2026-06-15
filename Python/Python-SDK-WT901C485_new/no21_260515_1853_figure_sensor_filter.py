import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
from no7_calc_from_acc_mag import calculate_angles

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


class SensorDataUKF:
    """专门用于传感器数据滤波的UKF"""

    def __init__(self, Q=None, R=None, dt=1.0, alpha=1e-3, beta=2.0, kappa=None):
        self.dt = dt
        self.n_states = 6  # [ax, ay, az, mx, my, mz]
        self.n_obs = 6  # 直接观测所有传感器数据

        # UKF参数
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa if kappa is not None else 3 - self.n_states
        self.lambda_ = self.alpha**2 * (self.n_states + self.kappa) - self.n_states
        self.n_sigma = 2 * self.n_states + 1

        # 权重计算
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)
        self.Wm[0] = self.lambda_ / (self.n_states + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n_states + self.lambda_) + (1 - self.alpha**2 + self.beta)
        for i in range(1, self.n_sigma):
            self.Wm[i] = 0.5 / (self.n_states + self.lambda_)
            self.Wc[i] = 0.5 / (self.n_states + self.lambda_)

        # 状态向量初始化
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 0.1

        # 噪声协方差矩阵
        if Q is None:
            # 加速度计和磁力计的过程噪声
            self.Q = np.diag([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])
        else:
            self.Q = Q

        if R is None:
            # 观测噪声
            self.R = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
            # self.R = np.diag([1e-3, 1e-3, 1e-3, 5e-2, 5e-2, 5e-2])
        else:
            self.R = R

        self.is_initialized = False

    def f(self, x, dt):
        """状态转移函数 - 传感器数据变化较慢，简单模型"""
        return x  # 假设传感器数据在短时间内变化缓慢

    def h(self, x):
        """观测函数 - 直接观测传感器数据"""
        return x

    def generate_sigma_points(self, x, P):
        """生成sigma点"""
        n = len(x)
        sigma_points = np.zeros((self.n_sigma, n))

        try:
            sqrt = np.linalg.cholesky((n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(P)
            sqrt = U @ np.diag(np.sqrt(np.maximum(s, 1e-12))) @ Vt
            sqrt = sqrt * np.sqrt(n + self.lambda_)

        sigma_points[0] = x
        for i in range(n):
            sigma_points[i + 1] = x + sqrt[i]
            sigma_points[i + 1 + n] = x - sqrt[i]

        return sigma_points

    def predict(self):
        """预测步骤"""
        sigma_points = self.generate_sigma_points(self.x, self.P)

        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            sigma_points_pred[i] = self.f(sigma_points[i], self.dt)

        self.x_pred = np.zeros(self.n_states)
        for i in range(self.n_sigma):
            self.x_pred += self.Wm[i] * sigma_points_pred[i]

        self.P_pred = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_sigma):
            diff = sigma_points_pred[i] - self.x_pred
            self.P_pred += self.Wc[i] * np.outer(diff, diff)

        self.P_pred += self.Q
        return sigma_points_pred

    def update(self, z):
        """更新步骤"""
        if not self.is_initialized:
            self.x = z.copy()
            self.is_initialized = True
            return self.x

        self.predict()

        sigma_points = self.generate_sigma_points(self.x_pred, self.P_pred)

        sigma_points_obs = np.zeros((self.n_sigma, self.n_obs))
        for i in range(self.n_sigma):
            sigma_points_obs[i] = self.h(sigma_points[i])

        z_pred = np.zeros(self.n_obs)
        for i in range(self.n_sigma):
            z_pred += self.Wm[i] * sigma_points_obs[i]

        Pzz = np.zeros((self.n_obs, self.n_obs))
        for i in range(self.n_sigma):
            diff = sigma_points_obs[i] - z_pred
            Pzz += self.Wc[i] * np.outer(diff, diff)
        Pzz += self.R

        Pxz = np.zeros((self.n_states, self.n_obs))
        for i in range(self.n_sigma):
            x_diff = sigma_points[i] - self.x_pred
            z_diff = sigma_points_obs[i] - z_pred
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)

        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        innovation = z - z_pred
        self.x = self.x_pred + K @ innovation
        self.P = self.P_pred - K @ Pzz @ K.T

        return self.x


class SensorDataACKF:
    """容积卡尔曼滤波（CKF/ACKF）
    使用12维状态：6个传感器通道 × [值, 速度] 的恒速模型
    x = [ax, v_ax, ay, v_ay, az, v_az, mx, v_mx, my, v_my, mz, v_mz]
    状态转移：x_pos[k+1] = x_pos[k] + x_vel[k]*dt
    观测：z = [ax, ay, az, mx, my, mz]（只观测位置分量）
    用容积点代替sigma点，Q/R 与 UKF 保持一致（乘以dt²缩放）。
    """

    def __init__(self, Q_pos=None, R=None, dt=0.02):
        self.dt = dt
        self.n_obs = 6
        self.n_states = 12

        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 0.1

        # 过程噪声：位置分量用 UKF 的 Q（但要缩放到 dt²），速度分量稍大
        if Q_pos is None:
            Q_pos = np.diag([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])
        # 构建12维Q：位置用 Q_pos，速度用 Q_pos/dt²（保证1步后位置噪声 ≈ Q_pos）
        q_vel = np.diag(Q_pos) / (dt**2 + 1e-12)  # 速度噪声
        Q_full = np.zeros((12, 12))
        for i in range(6):
            Q_full[2 * i, 2 * i] = Q_pos[i, i]  # 位置过程噪声
            Q_full[2 * i + 1, 2 * i + 1] = q_vel[i]  # 速度过程噪声
        self.Q = Q_full

        if R is None:
            self.R = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
        else:
            self.R = R

        self.is_initialized = False
        self.num_cubature_points = 2 * self.n_states  # = 24

    def f(self, x, dt):
        """恒速模型"""
        x_pred = x.copy()
        for i in range(self.n_obs):
            x_pred[2 * i] = x[2 * i] + x[2 * i + 1] * dt
        return x_pred

    def h(self, x):
        """观测位置分量"""
        return x[0::2]

    def generate_cubature_points(self, x, P):
        n = len(x)
        try:
            sqrt_P = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(P)
            sqrt_P = U @ np.diag(np.sqrt(np.maximum(s, 1e-12))) @ Vt

        sqrt_n = np.sqrt(n)
        points = np.zeros((2 * n, n))
        for i in range(n):
            points[i] = x + sqrt_n * sqrt_P[:, i]
            points[n + i] = x - sqrt_n * sqrt_P[:, i]
        return points

    def predict(self):
        cubature_points = self.generate_cubature_points(self.x, self.P)
        pred = np.zeros_like(cubature_points)
        for i in range(self.num_cubature_points):
            pred[i] = self.f(cubature_points[i], self.dt)

        self.x_pred = np.mean(pred, axis=0)
        self.P_pred = np.zeros((self.n_states, self.n_states))
        for i in range(self.num_cubature_points):
            diff = pred[i] - self.x_pred
            self.P_pred += np.outer(diff, diff)
        self.P_pred = self.P_pred / self.num_cubature_points + self.Q

    def update(self, z):
        if not self.is_initialized:
            self.x = np.zeros(self.n_states)
            self.x[0::2] = z
            self.is_initialized = True
            return z.copy()

        self.predict()

        cubature_points = self.generate_cubature_points(self.x_pred, self.P_pred)
        pred_obs = np.zeros((self.num_cubature_points, self.n_obs))
        for i in range(self.num_cubature_points):
            pred_obs[i] = self.h(cubature_points[i])
        z_pred = np.mean(pred_obs, axis=0)

        Pzz = np.zeros((self.n_obs, self.n_obs))
        for i in range(self.num_cubature_points):
            diff = pred_obs[i] - z_pred
            Pzz += np.outer(diff, diff)
        Pzz = Pzz / self.num_cubature_points + self.R

        Pxz = np.zeros((self.n_states, self.n_obs))
        for i in range(self.num_cubature_points):
            Pxz += np.outer(cubature_points[i] - self.x_pred, pred_obs[i] - z_pred)
        Pxz = Pxz / self.num_cubature_points

        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        innovation = z - z_pred
        self.x = self.x_pred + K @ innovation
        self.P = self.P_pred - K @ Pzz @ K.T
        self.P = (self.P + self.P.T) / 2

        return self.x[0::2].copy()


def generate_test_sensor_data(n_points=2000, noise_level=1.0):
    """
    生成带噪声的传感器测试数据
    """
    t = np.linspace(0, 20, n_points)

    # 基础传感器数据
    base_acc = [0.031, 0.222, 0.982]
    base_mag = [0.447, -41.61, -29.32]

    raw_data = []
    true_data = []

    for i in range(n_points):
        # 添加周期性变化（模拟真实的传感器信号变化）
        phase_shift = 1 * np.sin(2 * np.pi * t[i] / 10)

        # 真实信号（无噪声）
        ax_true = base_acc[0] + phase_shift * 0.01
        ay_true = base_acc[1] + phase_shift * 0.01
        az_true = base_acc[2]
        mx_true = base_mag[0] + phase_shift
        my_true = base_mag[1] + phase_shift
        mz_true = base_mag[2]

        true_data.append([ax_true, ay_true, az_true, mx_true, my_true, mz_true])

        # 添加噪声
        acc_noise = np.random.normal(0, noise_level * 0.01, 3)
        mag_noise = np.random.normal(0, noise_level * 0.5, 3)

        ax_noisy = ax_true + acc_noise[0]
        ay_noisy = ay_true + acc_noise[1]
        az_noisy = az_true + acc_noise[2]
        mx_noisy = mx_true + mag_noise[0]
        my_noisy = my_true + mag_noise[1]
        mz_noisy = mz_true + mag_noise[2]

        raw_data.append([ax_noisy, ay_noisy, az_noisy, mx_noisy, my_noisy, mz_noisy])

    return raw_data, true_data


def filter_sensor_data(raw_data, true_data):
    """
    对传感器数据进行UKF和ACKF滤波
    """
    print("开始传感器数据滤波...")

    # 初始化滤波器
    ukf = SensorDataUKF(
        Q=np.diag([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]),
        R=np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]),
        dt=1.0,
        alpha=1e-3,
        beta=2.0,
        kappa=0,
    )

    ackf = SensorDataACKF(
        Q_pos=np.diag([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]),
        R=np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]),
        dt=1.0,
    )

    # 滤波处理
    ukf_results = []
    ackf_results = []

    for i, sensor_data in enumerate(raw_data):
        sensor_array = np.array(sensor_data)

        # UKF滤波
        ukf_filtered = ukf.update(sensor_array)
        ukf_results.append(ukf_filtered.copy())

        # ACKF滤波
        ackf_filtered = ackf.update(sensor_array)
        ackf_results.append(ackf_filtered.copy())

        if (i + 1) % 200 == 0:
            print(f"已处理 {i + 1}/{len(raw_data)} 个数据点")

    return ukf_results, ackf_results


def plot_sensor_comparison(raw_data, ukf_results, ackf_results, true_data):
    """
    绘制传感器数据滤波对比图
    """
    n_points = len(raw_data)
    x_axis = np.arange(n_points)

    # 转换为numpy数组
    raw_array = np.array(raw_data)
    ukf_array = np.array(ukf_results)
    ackf_array = np.array(ackf_results)
    true_array = np.array(true_data)

    # 传感器名称和单位
    sensor_names = ["ax", "ay", "az", "mx", "my", "mz"]
    sensor_labels = ["ax (g)", "ay (g)", "az (g)", "mx (μT)", "my (μT)", "mz (μT)"]

    # 创建6个子图
    fig, axes = plt.subplots(6, 1, figsize=(12, 18))
    fig.suptitle("传感器数据滤波对比（UKF vs ACKF）", fontsize=16, fontweight="bold")

    for i in range(6):
        ax = axes[i]

        # 绘制三条线
        ax.plot(x_axis, raw_array[:, i], "b-", alpha=0.7, linewidth=0.8, label="原始数据")
        ax.plot(x_axis, ukf_array[:, i], "g-", linewidth=1.2, label="UKF滤波")
        ax.plot(x_axis, ackf_array[:, i], "r-", linewidth=1.2, label="ACKF滤波")
        ax.plot(x_axis, true_array[:, i], "k--", linewidth=1.5, alpha=0.8, label="真实数据")

        ax.set_ylabel(sensor_labels[i])
        ax.set_title(f"{sensor_names[i].upper()} 传感器数据滤波对比")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(x_axis))

    axes[-1].set_xlabel("数据点")
    plt.tight_layout()
    plt.show()


def calculate_sensor_metrics(raw_data, filtered_data, true_data, filter_name):
    """
    计算传感器数据滤波性能指标
    """
    raw_array = np.array(raw_data)
    filtered_array = np.array(filtered_data)
    true_array = np.array(true_data)

    # 计算RMSE
    raw_rmse = np.sqrt(np.mean((raw_array - true_array) ** 2, axis=0))
    filtered_rmse = np.sqrt(np.mean((filtered_array - true_array) ** 2, axis=0))

    # 计算MAE
    raw_mae = np.mean(np.abs(raw_array - true_array), axis=0)
    filtered_mae = np.mean(np.abs(filtered_array - true_array), axis=0)

    # 计算改善率
    improvement = (raw_rmse - filtered_rmse) / raw_rmse * 100

    return {
        "filter_name": filter_name,
        "raw_rmse": raw_rmse,
        "filtered_rmse": filtered_rmse,
        "raw_mae": raw_mae,
        "filtered_mae": filtered_mae,
        "improvement": improvement,
    }


def print_sensor_metrics_comparison(raw_data, ukf_results, ackf_results, true_data):
    """
    打印传感器数据滤波性能对比
    """
    ukf_metrics = calculate_sensor_metrics(raw_data, ukf_results, true_data, "UKF")
    ackf_metrics = calculate_sensor_metrics(raw_data, ackf_results, true_data, "ACKF")

    sensor_names = ["ax", "ay", "az", "mx", "my", "mz"]
    sensor_units = ["g", "g", "g", "μT", "μT", "μT"]

    print("\n" + "=" * 80)
    print("传感器数据滤波性能对比")
    print("=" * 80)

    for i, (sensor, unit) in enumerate(zip(sensor_names, sensor_units)):
        print(f"\n{sensor.upper()} 传感器性能指标:")
        print(f"  原始数据 RMSE: {ukf_metrics['raw_rmse'][i]:.6f} {unit}")
        print(f"  UKF RMSE:      {ukf_metrics['filtered_rmse'][i]:.6f} {unit}")
        print(f"  ACKF RMSE:     {ackf_metrics['filtered_rmse'][i]:.6f} {unit}")
        print(f"  UKF改善:       {ukf_metrics['improvement'][i]:.2f}%")
        print(f"  ACKF改善:      {ackf_metrics['improvement'][i]:.2f}%")

        # UKF vs ACKF对比
        if ackf_metrics["filtered_rmse"][i] != 0:
            ackf_vs_ukf = (ukf_metrics["filtered_rmse"][i] - ackf_metrics["filtered_rmse"][i]) / ukf_metrics["filtered_rmse"][i] * 100
            print(f"  ACKF相比UKF:   {ackf_vs_ukf:.2f}%")


def export_sensor_data_to_excel(
    raw_data,
    ukf_results,
    ackf_results,
    true_data,
    filename="no21_260515_1853_sensor_filter_comparison.xlsx",
):
    """
    导出传感器数据滤波结果到Excel
    """
    print(f"正在导出传感器数据到 {filename}...")

    n_points = len(raw_data)
    x_axis = np.arange(n_points)

    # 转换为numpy数组
    raw_array = np.array(raw_data)
    ukf_array = np.array(ukf_results)
    ackf_array = np.array(ackf_results)
    true_array = np.array(true_data)

    # 计算误差
    raw_errors = raw_array - true_array
    ukf_errors = ukf_array - true_array
    ackf_errors = ackf_array - true_array

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:

        # 1. 原始传感器数据
        sensor_data = pd.DataFrame(
            {
                "数据点": x_axis,
                "原始ax(g)": raw_array[:, 0],
                "原始ay(g)": raw_array[:, 1],
                "原始az(g)": raw_array[:, 2],
                "原始mx(μT)": raw_array[:, 3],
                "原始my(μT)": raw_array[:, 4],
                "原始mz(μT)": raw_array[:, 5],
                "UKF_ax(g)": ukf_array[:, 0],
                "UKF_ay(g)": ukf_array[:, 1],
                "UKF_az(g)": ukf_array[:, 2],
                "UKF_mx(μT)": ukf_array[:, 3],
                "UKF_my(μT)": ukf_array[:, 4],
                "UKF_mz(μT)": ukf_array[:, 5],
                "ACKF_ax(g)": ackf_array[:, 0],
                "ACKF_ay(g)": ackf_array[:, 1],
                "ACKF_az(g)": ackf_array[:, 2],
                "ACKF_mx(μT)": ackf_array[:, 3],
                "ACKF_my(μT)": ackf_array[:, 4],
                "ACKF_mz(μT)": ackf_array[:, 5],
                "真实ax(g)": true_array[:, 0],
                "真实ay(g)": true_array[:, 1],
                "真实az(g)": true_array[:, 2],
                "真实mx(μT)": true_array[:, 3],
                "真实my(μT)": true_array[:, 4],
                "真实mz(μT)": true_array[:, 5],
            }
        )
        sensor_data.to_excel(writer, sheet_name="传感器数据", index=False)

        # 2. 误差数据
        error_data = pd.DataFrame(
            {
                "数据点": x_axis,
                "原始ax误差": raw_errors[:, 0],
                "原始ay误差": raw_errors[:, 1],
                "原始az误差": raw_errors[:, 2],
                "原始mx误差": raw_errors[:, 3],
                "原始my误差": raw_errors[:, 4],
                "原始mz误差": raw_errors[:, 5],
                "UKF_ax误差": ukf_errors[:, 0],
                "UKF_ay误差": ukf_errors[:, 1],
                "UKF_az误差": ukf_errors[:, 2],
                "UKF_mx误差": ukf_errors[:, 3],
                "UKF_my误差": ukf_errors[:, 4],
                "UKF_mz误差": ukf_errors[:, 5],
                "ACKF_ax误差": ackf_errors[:, 0],
                "ACKF_ay误差": ackf_errors[:, 1],
                "ACKF_az误差": ackf_errors[:, 2],
                "ACKF_mx误差": ackf_errors[:, 3],
                "ACKF_my误差": ackf_errors[:, 4],
                "ACKF_mz误差": ackf_errors[:, 5],
            }
        )
        error_data.to_excel(writer, sheet_name="误差数据", index=False)

        # 3. 性能指标汇总
        ukf_metrics = calculate_sensor_metrics(raw_data, ukf_results, true_data, "UKF")
        ackf_metrics = calculate_sensor_metrics(raw_data, ackf_results, true_data, "ACKF")

        metrics_summary = pd.DataFrame(
            {
                "传感器": ["ax", "ay", "az", "mx", "my", "mz"],
                "单位": ["g", "g", "g", "μT", "μT", "μT"],
                "原始数据RMSE": ukf_metrics["raw_rmse"],
                "UKF_RMSE": ukf_metrics["filtered_rmse"],
                "ACKF_RMSE": ackf_metrics["filtered_rmse"],
                "UKF改善率(%)": ukf_metrics["improvement"],
                "ACKF改善率(%)": ackf_metrics["improvement"],
            }
        )
        metrics_summary.to_excel(writer, sheet_name="性能指标", index=False)

    print(f"数据已成功导出到 {os.path.abspath(filename)}")


def export_sensor_data_to_excel_no_true(
    raw_data,
    ukf_results,
    ackf_results,
    time_data=None,
    filename="no21_260515_1853_sensor_filter_comparison_no_true.xlsx",
):
    """
    导出传感器数据滤波结果到Excel（无真实数据版本）
    真实值列输出为0
    """
    print(f"正在导出传感器数据到 {filename}...")

    n_points = len(raw_data)
    if time_data is not None:
        x_axis = time_data
        x_label = "时间(s)"
    else:
        x_axis = np.arange(n_points)
        x_label = "数据点"

    # 转换为numpy数组
    raw_array = np.array(raw_data)
    ukf_array = np.array(ukf_results)
    ackf_array = np.array(ackf_results)

    # 创建全零的真实数据数组
    true_array = np.zeros_like(raw_array)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:

        # 1. 传感器数据（真实值列输出为0）
        sensor_data = pd.DataFrame(
            {
                x_label: x_axis,
                "原始ax(g)": raw_array[:, 0],
                "原始ay(g)": raw_array[:, 1],
                "原始az(g)": raw_array[:, 2],
                "原始mx(μT)": raw_array[:, 3],
                "原始my(μT)": raw_array[:, 4],
                "原始mz(μT)": raw_array[:, 5],
                "UKF_ax(g)": ukf_array[:, 0],
                "UKF_ay(g)": ukf_array[:, 1],
                "UKF_az(g)": ukf_array[:, 2],
                "UKF_mx(μT)": ukf_array[:, 3],
                "UKF_my(μT)": ukf_array[:, 4],
                "UKF_mz(μT)": ukf_array[:, 5],
                "ACKF_ax(g)": ackf_array[:, 0],
                "ACKF_ay(g)": ackf_array[:, 1],
                "ACKF_az(g)": ackf_array[:, 2],
                "ACKF_mx(μT)": ackf_array[:, 3],
                "ACKF_my(μT)": ackf_array[:, 4],
                "ACKF_mz(μT)": ackf_array[:, 5],
                "真实ax(g)": true_array[:, 0],
                "真实ay(g)": true_array[:, 1],
                "真实az(g)": true_array[:, 2],
                "真实mx(μT)": true_array[:, 3],
                "真实my(μT)": true_array[:, 4],
                "真实mz(μT)": true_array[:, 5],
            }
        )
        sensor_data.to_excel(writer, sheet_name="传感器数据", index=False)

        # 2. 平滑度指标汇总（无真实数据时使用标准差和变化率）
        sensor_names = ["ax", "ay", "az", "mx", "my", "mz"]
        sensor_units = ["g", "g", "g", "μT", "μT", "μT"]

        metrics_data = []
        for i, (sensor, unit) in enumerate(zip(sensor_names, sensor_units)):
            raw_std = np.std(raw_array[:, i])
            ukf_std = np.std(ukf_array[:, i])
            ackf_std = np.std(ackf_array[:, i])

            # 相邻点变化率
            raw_diff = np.mean(np.abs(np.diff(raw_array[:, i])))
            ukf_diff = np.mean(np.abs(np.diff(ukf_array[:, i])))
            ackf_diff = np.mean(np.abs(np.diff(ackf_array[:, i])))

            std_reduction_ukf = (1 - ukf_std / raw_std) * 100 if raw_std > 0 else 0
            std_reduction_ackf = (1 - ackf_std / raw_std) * 100 if raw_std > 0 else 0
            diff_reduction_ukf = (1 - ukf_diff / raw_diff) * 100 if raw_diff > 0 else 0
            diff_reduction_ackf = (1 - ackf_diff / raw_diff) * 100 if raw_diff > 0 else 0

            metrics_data.append(
                {
                    "传感器": sensor,
                    "单位": unit,
                    "原始标准差": raw_std,
                    "UKF标准差": ukf_std,
                    "ACKF标准差": ackf_std,
                    "UKF标准差降低(%)": std_reduction_ukf,
                    "ACKF标准差降低(%)": std_reduction_ackf,
                    "原始平均变化率": raw_diff,
                    "UKF平均变化率": ukf_diff,
                    "ACKF平均变化率": ackf_diff,
                    "UKF变化率降低(%)": diff_reduction_ukf,
                    "ACKF变化率降低(%)": diff_reduction_ackf,
                }
            )

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(writer, sheet_name="平滑度指标", index=False)

    print(f"数据已成功导出到 {os.path.abspath(filename)}")


def load_real_sensor_data(excel_path, sheet_name=1):
    """
    从Excel文件读取真实传感器数据
    sheet_name=1 表示 Sheet2（0-indexed）
    返回 raw_data (list of lists) 和 time 列
    """
    print(f"正在从 {excel_path} 读取传感器数据...")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"  Sheet 列名: {list(df.columns)}")
    print(f"  数据行数: {len(df)}")

    # 自动查找列名（兼容不同命名方式）
    col_map = {}
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if col_lower in ("time", "t", "时间", "timestamp"):
            col_map["time"] = col
        elif col_lower in ("ax", "accel_x", "acc_x"):
            col_map["ax"] = col
        elif col_lower in ("ay", "accel_y", "acc_y"):
            col_map["ay"] = col
        elif col_lower in ("az", "accel_z", "acc_z"):
            col_map["az"] = col
        elif col_lower in ("mx", "mag_x"):
            col_map["mx"] = col
        elif col_lower in ("my", "mag_y"):
            col_map["my"] = col
        elif col_lower in ("mz", "mag_z"):
            col_map["mz"] = col

    required = ["ax", "ay", "az", "mx", "my", "mz"]
    for key in required:
        if key not in col_map:
            raise ValueError(f"找不到列 '{key}'，当前列: {list(df.columns)}")

    # 提取时间列（如果有）
    time_data = None
    if "time" in col_map:
        time_data = df[col_map["time"]].values

    # 提取传感器数据
    raw_data = []
    for _, row in df.iterrows():
        raw_data.append(
            [
                float(row[col_map["ax"]]),
                float(row[col_map["ay"]]),
                float(row[col_map["az"]]),
                float(row[col_map["mx"]]),
                float(row[col_map["my"]]),
                float(row[col_map["mz"]]),
            ]
        )

    print(f"  成功读取 {len(raw_data)} 个数据点")
    if time_data is not None:
        print(f"  时间范围: {time_data[0]:.3f}s ~ {time_data[-1]:.3f}s")

    return raw_data, time_data


def filter_sensor_data_no_true(raw_data):
    """
    对真实传感器数据进行UKF和ACKF滤波（无真值参考）
    UKF: 恒等模型 + 适中参数 → 更贴合观测值，减少过度平滑
    ACKF: 恒速模型 + 优化参数 → 减少噪声影响，保持平滑
    """
    print("开始传感器数据滤波...")

    # UKF: 调整参数使其更贴合观测值，减少过度平滑
    ukf = SensorDataUKF(
        Q=np.diag([5e-3, 5e-3, 5e-3, 5e-2, 5e-2, 5e-2]),  # 增加Q，让模型更不相信预测
        R=np.diag([1e-2, 1e-2, 1e-2, 0.5, 0.5, 0.5]),  # 减小R，让滤波器更相信观测值
        dt=1.0,
        alpha=1e-1,
        beta=4.0,
        kappa=0,
    )

    # ACKF: 优化参数，减少噪声同时保持平滑
    ackf = SensorDataACKF(
        Q_pos=np.diag([1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5]),  # 进一步减小Q，更平滑
        R=np.diag([3e-3, 3e-3, 3e-3, 3e-2, 3e-2, 3e-2]),  # 进一步增加R，减少噪声影响
        dt=0.02,
    )

    ukf_results = []
    ackf_results = []

    for i, sensor_data in enumerate(raw_data):
        sensor_array = np.array(sensor_data)
        ukf_results.append(ukf.update(sensor_array).copy())
        ackf_results.append(ackf.update(sensor_array).copy())

        if (i + 1) % 500 == 0:
            print(f"已处理 {i + 1}/{len(raw_data)} 个数据点")

    return ukf_results, ackf_results


def plot_sensor_comparison_no_true(raw_data, ukf_results, ackf_results, time_data=None):
    """
    绘制真实传感器数据滤波对比图（无真值，仅对比原始 vs 滤波）
    """
    n_points = len(raw_data)
    if time_data is not None:
        x_axis = time_data
        x_label = "时间 (s)"
    else:
        x_axis = np.arange(n_points)
        x_label = "数据点"

    raw_array = np.array(raw_data)
    ukf_array = np.array(ukf_results)
    ackf_array = np.array(ackf_results)

    sensor_names = ["ax", "ay", "az", "mx", "my", "mz"]
    sensor_labels = ["ax (g)", "ay (g)", "az (g)", "mx (μT)", "my (μT)", "mz (μT)"]

    fig, axes = plt.subplots(6, 1, figsize=(14, 20))
    fig.suptitle("真实传感器数据滤波对比（UKF vs ACKF）", fontsize=16, fontweight="bold")

    for i in range(6):
        ax = axes[i]
        ax.plot(x_axis, raw_array[:, i], "b-", alpha=0.5, linewidth=0.6, label="原始数据")
        ax.plot(x_axis, ukf_array[:, i], "g-", linewidth=1.2, label="UKF滤波")
        ax.plot(x_axis, ackf_array[:, i], "r-", linewidth=1.2, label="ACKF滤波")

        ax.set_ylabel(sensor_labels[i])
        ax.set_title(f"{sensor_names[i].upper()} 传感器数据滤波对比")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)
    plt.tight_layout()
    plt.show()


def print_metrics_no_true(raw_data, ukf_results, ackf_results):
    """
    无真值时的性能对比：计算各滤波器输出的平滑度指标
    （用标准差、峰峰值、变化率来衡量平滑效果）
    """
    raw_array = np.array(raw_data)
    ukf_array = np.array(ukf_results)
    ackf_array = np.array(ackf_results)

    sensor_names = ["ax", "ay", "az", "mx", "my", "mz"]
    sensor_units = ["g", "g", "g", "μT", "μT", "μT"]

    print("\n" + "=" * 90)
    print("真实传感器数据 - 滤波平滑效果对比（无真值）")
    print("=" * 90)

    for i, (sensor, unit) in enumerate(zip(sensor_names, sensor_units)):
        raw_std = np.std(raw_array[:, i])
        ukf_std = np.std(ukf_array[:, i])
        ackf_std = np.std(ackf_array[:, i])

        # 相邻点变化率（越小越平滑）
        raw_diff = np.mean(np.abs(np.diff(raw_array[:, i])))
        ukf_diff = np.mean(np.abs(np.diff(ukf_array[:, i])))
        ackf_diff = np.mean(np.abs(np.diff(ackf_array[:, i])))

        std_reduction_ukf = (1 - ukf_std / raw_std) * 100 if raw_std > 0 else 0
        std_reduction_ackf = (1 - ackf_std / raw_std) * 100 if raw_std > 0 else 0
        diff_reduction_ukf = (1 - ukf_diff / raw_diff) * 100 if raw_diff > 0 else 0
        diff_reduction_ackf = (1 - ackf_diff / raw_diff) * 100 if raw_diff > 0 else 0

        print(f"\n{sensor.upper()} ({unit}):")
        print(f"  标准差  原始: {raw_std:.6f}  UKF: {ukf_std:.6f}  ACKF: {ackf_std:.6f}")
        print(f"  标准差降低  UKF: {std_reduction_ukf:.1f}%  ACKF: {std_reduction_ackf:.1f}%")
        print(f"  平均变化率  原始: {raw_diff:.6f}  UKF: {ukf_diff:.6f}  ACKF: {ackf_diff:.6f}")
        print(f"  变化率降低  UKF: {diff_reduction_ukf:.1f}%  ACKF: {diff_reduction_ackf:.1f}%")


if __name__ == "__main__":
    # ========== 数据源选择 ==========
    USE_REAL_DATA = True  # True=从Excel读取真实数据, False=生成仿真数据

    if USE_REAL_DATA:
        excel_path = r"D:\Documents\GitHub\RSS_board_UART\rss-20260310-1345.xlsx"
        raw_data, time_data = load_real_sensor_data(excel_path, sheet_name=1)
        true_data = None  # 真实数据没有真值
    else:
        print("生成传感器测试数据...")
        raw_data, true_data = generate_test_sensor_data(n_points=2000, noise_level=0.1)
        time_data = None
        print(f"生成了 {len(raw_data)} 个传感器数据点")

    # ========== 滤波处理 ==========
    if true_data is not None:
        # 有真值时，用传统滤波+真值对比
        ukf_results, ackf_results = filter_sensor_data(raw_data, true_data)
        plot_sensor_comparison(raw_data, ukf_results, ackf_results, true_data)
        print_sensor_metrics_comparison(raw_data, ukf_results, ackf_results, true_data)
        export_sensor_data_to_excel(raw_data, ukf_results, ackf_results, true_data)
    else:
        # 无真值时，对比原始 vs 滤波的平滑效果
        ukf_results, ackf_results = filter_sensor_data_no_true(raw_data)
        plot_sensor_comparison_no_true(raw_data, ukf_results, ackf_results, time_data)
        print_metrics_no_true(raw_data, ukf_results, ackf_results)
        export_sensor_data_to_excel_no_true(raw_data, ukf_results, ackf_results, time_data)

    print("\n传感器数据滤波对比完成！")
