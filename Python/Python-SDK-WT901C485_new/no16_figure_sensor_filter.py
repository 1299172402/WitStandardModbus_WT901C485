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
        self.Wc[0] = self.lambda_ / (self.n_states + self.lambda_) + (
            1 - self.alpha**2 + self.beta
        )
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
    """专门用于传感器数据滤波的ACKF"""

    def __init__(self, Q=None, R=None, dt=1.0, window_size=5, forget_factor=0.95):
        self.dt = dt
        self.n_states = 6  # [ax, ay, az, mx, my, mz]
        self.n_obs = 6  # 直接观测所有传感器数据
        self.window_size = window_size
        self.forget_factor = forget_factor

        # 状态向量初始化
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 0.1

        # 噪声协方差矩阵
        if Q is None:
            self.Q = np.diag([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])
        else:
            self.Q = Q

        if R is None:
            self.R = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
        else:
            self.R = R

        # 自适应参数
        self.innovation_history = []
        self.is_initialized = False
        self.num_cubature_points = 2 * self.n_states

    def f(self, x, dt):
        """状态转移函数"""
        return x  # 传感器数据变化缓慢

    def h(self, x):
        """观测函数"""
        return x

    def generate_cubature_points(self, x, P):
        """生成容积点"""
        n = len(x)

        try:
            sqrt_P = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(P)
            sqrt_P = U @ np.diag(np.sqrt(np.maximum(s, 1e-12))) @ Vt

        sqrt_n = np.sqrt(n)
        Xi = np.zeros((2 * n, n))

        for i in range(n):
            Xi[i, i] = sqrt_n
            Xi[n + i, i] = -sqrt_n

        cubature_points = np.zeros((2 * n, n))
        for i in range(2 * n):
            cubature_points[i] = x + sqrt_P @ Xi[i]

        return cubature_points

    def adapt_noise_covariance(self, innovation, S):
        """自适应调整噪声协方差矩阵"""
        self.innovation_history.append(innovation.copy())

        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)

        if len(self.innovation_history) < 3:
            return

        innovations = np.array(self.innovation_history)
        weights = np.array(
            [
                self.forget_factor ** (len(innovations) - 1 - i)
                for i in range(len(innovations))
            ]
        )
        weights = weights / np.sum(weights)

        mean_innovation = np.average(innovations, axis=0, weights=weights)
        actual_cov = np.zeros((self.n_obs, self.n_obs))

        for i, innov in enumerate(innovations):
            diff = innov - mean_innovation
            actual_cov += weights[i] * np.outer(diff, diff)

        try:
            ratio = np.trace(actual_cov) / np.trace(S)
            ratio = np.clip(ratio, 0.1, 10.0)

            alpha = 0.1
            self.R = (1 - alpha) * self.R + alpha * ratio * self.R

            eigenvals = np.linalg.eigvals(self.R)
            if np.any(eigenvals <= 0):
                self.R = self.R + np.eye(self.n_obs) * 1e-6
        except (np.linalg.LinAlgError, ZeroDivisionError):
            pass

    def predict(self):
        """预测步骤"""
        cubature_points = self.generate_cubature_points(self.x, self.P)

        predicted_points = np.zeros_like(cubature_points)
        for i in range(self.num_cubature_points):
            predicted_points[i] = self.f(cubature_points[i], self.dt)

        self.x_pred = np.mean(predicted_points, axis=0)

        self.P_pred = np.zeros((self.n_states, self.n_states))
        for i in range(self.num_cubature_points):
            diff = predicted_points[i] - self.x_pred
            self.P_pred += np.outer(diff, diff)

        self.P_pred = self.P_pred / self.num_cubature_points + self.Q

    def update(self, z):
        """更新步骤"""
        if not self.is_initialized:
            self.x = z.copy()
            self.is_initialized = True
            return self.x

        self.predict()

        cubature_points = self.generate_cubature_points(self.x_pred, self.P_pred)

        predicted_obs = np.zeros((self.num_cubature_points, self.n_obs))
        for i in range(self.num_cubature_points):
            predicted_obs[i] = self.h(cubature_points[i])

        z_pred = np.mean(predicted_obs, axis=0)

        Pzz = np.zeros((self.n_obs, self.n_obs))
        for i in range(self.num_cubature_points):
            diff = predicted_obs[i] - z_pred
            Pzz += np.outer(diff, diff)
        Pzz = Pzz / self.num_cubature_points + self.R

        Pxz = np.zeros((self.n_states, self.n_obs))
        for i in range(self.num_cubature_points):
            x_diff = cubature_points[i] - self.x_pred
            z_diff = predicted_obs[i] - z_pred
            Pxz += np.outer(x_diff, z_diff)
        Pxz = Pxz / self.num_cubature_points

        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        innovation = z - z_pred
        self.adapt_noise_covariance(innovation, Pzz)

        self.x = self.x_pred + K @ innovation
        self.P = self.P_pred - K @ Pzz @ K.T

        return self.x


def generate_test_sensor_data(n_points=1000, noise_level=1.0):
    """
    生成带噪声的传感器测试数据
    """
    t = np.linspace(0, 20, n_points)

    # 基础传感器数据
    base_acc = [0.144, 0.284, 0.942]
    base_mag = [-18.5, -31.7, -12.8]

    raw_data = []
    true_data = []

    for i in range(n_points):
        # 添加周期性变化（模拟真实的传感器信号变化）
        phase_shift = 0.1 * np.sin(2 * np.pi * t[i] / 10)

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
        Q=np.diag([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]),
        R=np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]),
        dt=1.0,
        window_size=5,
        forget_factor=0.95,
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
        ax.plot(
            x_axis, raw_array[:, i], "b-", alpha=0.7, linewidth=0.8, label="原始数据"
        )
        ax.plot(x_axis, ukf_array[:, i], "g-", linewidth=1.2, label="UKF滤波")
        ax.plot(x_axis, ackf_array[:, i], "r-", linewidth=1.2, label="ACKF滤波")
        ax.plot(
            x_axis, true_array[:, i], "k--", linewidth=1.5, alpha=0.8, label="真实数据"
        )

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
            ackf_vs_ukf = (
                (ukf_metrics["filtered_rmse"][i] - ackf_metrics["filtered_rmse"][i])
                / ukf_metrics["filtered_rmse"][i]
                * 100
            )
            print(f"  ACKF相比UKF:   {ackf_vs_ukf:.2f}%")


def export_sensor_data_to_excel(
    raw_data,
    ukf_results,
    ackf_results,
    true_data,
    filename="no16_sensor_filter_comparison.xlsx",
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
        ackf_metrics = calculate_sensor_metrics(
            raw_data, ackf_results, true_data, "ACKF"
        )

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


if __name__ == "__main__":
    print("生成传感器测试数据...")

    # 生成测试数据
    raw_data, true_data = generate_test_sensor_data(n_points=1000, noise_level=1.0)

    print(f"生成了 {len(raw_data)} 个传感器数据点")

    # 滤波处理
    ukf_results, ackf_results = filter_sensor_data(raw_data, true_data)

    # 绘制对比图
    print("绘制传感器数据滤波对比图...")
    plot_sensor_comparison(raw_data, ukf_results, ackf_results, true_data)

    # 打印性能指标
    print_sensor_metrics_comparison(raw_data, ukf_results, ackf_results, true_data)

    # 导出数据到Excel
    export_sensor_data_to_excel(raw_data, ukf_results, ackf_results, true_data)

    print("\n传感器数据滤波对比完成！")
