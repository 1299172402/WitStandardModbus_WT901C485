import numpy as np
import pandas as pd

from no7_calc_from_acc_mag import calculate_angles, attitude_to_wellbore_quaternion


class UnscentedKalmanFilterSensorFixed:
    def __init__(self, Q=None, R=None, dt=1.0, alpha=0.001, beta=2.0, kappa=None):
        """
        修复的无迹卡尔曼滤波器 (UKF) - 直接滤波传感器数据
        """
        self.dt = dt
        self.n_states = 12  # 6个传感器值 + 6个变化率
        self.n_obs = 6  # [ax, ay, az, mx, my, mz] 观测值

        # 优化UKF参数 - 关键修复点
        self.alpha = alpha  # 增大alpha，使sigma点分布更广
        self.beta = beta
        # 对于高维状态，kappa通常设为3-n
        self.kappa = kappa if kappa is not None else 3 - self.n_states

        # 计算lambda参数
        self.lambda_ = self.alpha**2 * (self.n_states + self.kappa) - self.n_states

        # 确保lambda不会导致负权重
        if self.lambda_ < 0:
            self.lambda_ = 0.1

        # sigma点数量
        self.n_sigma = 2 * self.n_states + 1

        # 权重计算 - 重新设计以提高稳定性
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)

        # 中心点权重
        self.Wm[0] = self.lambda_ / (self.n_states + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n_states + self.lambda_) + (
            1 - self.alpha**2 + self.beta
        )

        # 其他点权重
        weight_other = 0.5 / (self.n_states + self.lambda_)
        for i in range(1, self.n_sigma):
            self.Wm[i] = weight_other
            self.Wc[i] = weight_other

        # 检查权重总和
        print(f"UKF权重总和 - Wm: {np.sum(self.Wm):.6f}, Wc: {np.sum(self.Wc):.6f}")
        print(f"UKF lambda: {self.lambda_:.6f}")

        # 状态向量初始化
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 0.1  # 减小初始协方差

        # 噪声协方差矩阵 - 重新调整
        if Q is None:
            # 大幅减小过程噪声，特别是变化率的噪声
            self.Q = np.diag(
                [
                    1e-6,
                    1e-6,
                    1e-6,
                    1e-5,
                    1e-5,
                    1e-5,  # 传感器值（更小）
                    1e-5,
                    1e-5,
                    1e-5,
                    1e-4,
                    1e-4,
                    1e-4,  # 变化率（更小）
                ]
            )
        else:
            self.Q = Q

        if R is None:
            # 观测噪声保持合理
            self.R = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
        else:
            self.R = R

        self.is_initialized = False
        self.step_count = 0

    def f(self, x, dt):
        """
        状态转移函数 - 添加阻尼以提高稳定性
        """
        x_next = x.copy()

        # 传感器值更新
        x_next[0] += x[6] * dt  # ax
        x_next[1] += x[7] * dt  # ay
        x_next[2] += x[8] * dt  # az
        x_next[3] += x[9] * dt  # mx
        x_next[4] += x[10] * dt  # my
        x_next[5] += x[11] * dt  # mz

        # 变化率添加阻尼（防止发散）
        damping = 0.99
        x_next[6:] *= damping

        return x_next

    def h(self, x):
        """
        观测函数
        """
        return x[:6]  # 直接观测传感器值

    def generate_sigma_points(self, x, P):
        """
        改进的sigma点生成 - 提高数值稳定性
        """
        n = len(x)
        sigma_points = np.zeros((self.n_sigma, n))

        # 添加正则化以确保正定性
        P_regularized = P + np.eye(n) * 1e-8

        # 尝试Cholesky分解，失败则使用SVD
        try:
            # 使用更稳定的方法计算矩阵平方根
            eigenvals, eigenvecs = np.linalg.eigh(P_regularized)
            eigenvals = np.maximum(eigenvals, 1e-10)  # 确保正值
            sqrt_matrix = (
                eigenvecs
                @ np.diag(np.sqrt(eigenvals * (n + self.lambda_)))
                @ eigenvecs.T
            )
        except:
            # 备用方法
            U, s, Vt = np.linalg.svd(P_regularized)
            s = np.maximum(s, 1e-10)
            sqrt_matrix = U @ np.diag(np.sqrt(s * (n + self.lambda_))) @ Vt

        # 中心点
        sigma_points[0] = x

        # 生成其他sigma点
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_matrix[i]
            sigma_points[i + 1 + n] = x - sqrt_matrix[i]

        return sigma_points

    def predict(self):
        """
        预测步骤 - 增强稳定性
        """
        # 生成sigma点
        sigma_points = self.generate_sigma_points(self.x, self.P)

        # 传播sigma点
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            sigma_points_pred[i] = self.f(sigma_points[i], self.dt)

        # 计算预测状态均值
        self.x_pred = np.zeros(self.n_states)
        for i in range(self.n_sigma):
            self.x_pred += self.Wm[i] * sigma_points_pred[i]

        # 计算预测协方差
        self.P_pred = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_sigma):
            diff = sigma_points_pred[i] - self.x_pred
            self.P_pred += self.Wc[i] * np.outer(diff, diff)

        self.P_pred += self.Q

        # 确保协方差矩阵的正定性和数值稳定性
        eigenvals, eigenvecs = np.linalg.eigh(self.P_pred)
        eigenvals = np.maximum(eigenvals, 1e-10)
        self.P_pred = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return sigma_points_pred

    def update(self, z):
        """
        更新步骤 - 增加保护机制
        """
        self.step_count += 1

        if not self.is_initialized:
            self.x[:6] = z
            self.x[6:] = 0
            self.is_initialized = True
            return self.x[:6]

        # 预测步骤
        sigma_points_pred = self.predict()

        # 生成新的sigma点用于观测更新
        sigma_points = self.generate_sigma_points(self.x_pred, self.P_pred)

        # 传播sigma点通过观测函数
        sigma_points_obs = np.zeros((self.n_sigma, self.n_obs))
        for i in range(self.n_sigma):
            sigma_points_obs[i] = self.h(sigma_points[i])

        # 计算预测观测均值
        z_pred = np.zeros(self.n_obs)
        for i in range(self.n_sigma):
            z_pred += self.Wm[i] * sigma_points_obs[i]

        # 计算观测协方差矩阵
        Pzz = np.zeros((self.n_obs, self.n_obs))
        for i in range(self.n_sigma):
            diff = sigma_points_obs[i] - z_pred
            Pzz += self.Wc[i] * np.outer(diff, diff)
        Pzz += self.R

        # 计算交叉协方差矩阵
        Pxz = np.zeros((self.n_states, self.n_obs))
        for i in range(self.n_sigma):
            x_diff = sigma_points[i] - self.x_pred
            z_diff = sigma_points_obs[i] - z_pred
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)

        # 确保观测协方差矩阵的正定性
        eigenvals, eigenvecs = np.linalg.eigh(Pzz)
        eigenvals = np.maximum(eigenvals, 1e-10)
        Pzz = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # 计算卡尔曼增益 - 使用更稳定的方法
        try:
            K = np.linalg.solve(Pzz.T, Pxz.T).T
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        # 计算残差
        innovation = z - z_pred

        # 限制innovation的大小，防止突变
        innovation_norm = np.linalg.norm(innovation)
        max_innovation = 0.1  # 最大允许的innovation
        if innovation_norm > max_innovation:
            innovation = innovation * (max_innovation / innovation_norm)

        # 更新状态和协方差
        self.x = self.x_pred + K @ innovation

        # 使用Joseph形式的协方差更新以提高数值稳定性
        I = np.eye(self.n_states)
        I_KH = I - K @ np.eye(self.n_obs, self.n_states)
        self.P = I_KH @ self.P_pred @ I_KH.T + K @ self.R @ K.T

        # 确保更新后的协方差矩阵正定
        eigenvals, eigenvecs = np.linalg.eigh(self.P)
        eigenvals = np.maximum(eigenvals, 1e-10)
        self.P = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return self.x[:6]


def process_sensor_data_with_ukf_sensor_fixed(
    data_list, real_angels, Q=None, R=None, dt=1.0, alpha=0.01, beta=2.0, kappa=None
):
    """
    处理传感器数据序列并应用修复的无迹卡尔曼滤波 - 直接滤波传感器数据
    """
    # 创建修复的无迹卡尔曼滤波器
    ukf = UnscentedKalmanFilterSensorFixed(
        Q=Q, R=R, dt=dt, alpha=alpha, beta=beta, kappa=kappa
    )

    results = []
    for i in range(len(data_list)):
        # 原始传感器数据
        ax, ay, az, mx, my, mz = data_list[i]

        # UKF滤波传感器数据
        ax_filtered, ay_filtered, az_filtered, mx_filtered, my_filtered, mz_filtered = (
            ukf.update([ax, ay, az, mx, my, mz])
        )

        # 从原始传感器数据计算角度（作为观测角度）
        roll_obs, pitch_obs, yaw_obs = calculate_angles(ax, ay, az, mx, my, mz)

        # 从滤波后的传感器数据计算角度
        roll_filtered, pitch_filtered, yaw_filtered = calculate_angles(
            ax_filtered, ay_filtered, az_filtered, mx_filtered, my_filtered, mz_filtered
        )

        # 计算井眼角度
        azimuth, inclination, toolface, quaternion = attitude_to_wellbore_quaternion(
            roll_filtered, pitch_filtered, yaw_filtered
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
                "roll": roll_filtered,
                "pitch": pitch_filtered,
                "yaw": yaw_filtered,
                "roll_obs": roll_obs,
                "pitch_obs": pitch_obs,
                "yaw_obs": yaw_obs,
                "real_roll": real_roll,
                "real_pitch": real_pitch,
                "real_yaw": real_yaw,
                "diff_roll": roll_filtered - real_roll,
                "diff_pitch": pitch_filtered - real_pitch,
                "diff_yaw": yaw_filtered - real_yaw,
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
                    ax_filtered,
                    ay_filtered,
                    az_filtered,
                    mx_filtered,
                    my_filtered,
                    mz_filtered,
                ],
            }
        )

    return results
