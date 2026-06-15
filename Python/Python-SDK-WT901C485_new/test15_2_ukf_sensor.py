import numpy as np
import pandas as pd

from no7_calc_from_acc_mag import calculate_angles, attitude_to_wellbore_quaternion


class UnscentedKalmanFilterSensor:
    def __init__(self, Q=None, R=None, dt=1.0, alpha=1e-3, beta=2.0, kappa=None):
        """
        无迹卡尔曼滤波器 (UKF) - 直接滤波传感器数据
        """
        self.dt = dt
        self.n_states = 12  # 6个传感器值 + 6个变化率
        self.n_obs = 6  # [ax, ay, az, mx, my, mz] 观测值

        # 调整UKF参数
        self.alpha = max(alpha, 1e-4)
        self.beta = beta
        self.kappa = kappa if kappa is not None else max(3 - self.n_states, 0)

        # 计算lambda参数
        self.lambda_ = self.alpha**2 * (self.n_states + self.kappa) - self.n_states

        # sigma点数量
        self.n_sigma = 2 * self.n_states + 1

        # 权重计算
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)

        self.Wm[0] = self.lambda_ / (self.n_states + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n_states + self.lambda_) + (
            1 - self.alpha**2 + self.beta
        )

        weight_other = 0.5 / (self.n_states + self.lambda_)
        for i in range(1, self.n_sigma):
            self.Wm[i] = weight_other
            self.Wc[i] = weight_other

        # 状态向量初始化
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 1.0

        # 噪声协方差矩阵
        if Q is None:
            self.Q = np.diag(
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
            )  # 变化率
        else:
            self.Q = Q

        if R is None:
            self.R = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
        else:
            self.R = R

        self.is_initialized = False

    def f(self, x, dt):
        """
        状态转移函数
        """
        x_next = x.copy()
        # 传感器值 = 传感器值 + 变化率 * 时间
        x_next[0] += x[6] * dt  # ax
        x_next[1] += x[7] * dt  # ay
        x_next[2] += x[8] * dt  # az
        x_next[3] += x[9] * dt  # mx
        x_next[4] += x[10] * dt  # my
        x_next[5] += x[11] * dt  # mz
        return x_next

    def h(self, x):
        """
        观测函数
        """
        return x[:6]  # 直接观测传感器值

    def generate_sigma_points(self, x, P):
        """
        生成sigma点
        """
        n = len(x)
        sigma_points = np.zeros((self.n_sigma, n))

        # 确保协方差矩阵的正定性
        P_regularized = P + np.eye(n) * 1e-9

        # 计算矩阵平方根
        try:
            sqrt = np.linalg.cholesky((n + self.lambda_) * P_regularized)
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(P_regularized)
            s = np.maximum(s, 1e-12)
            sqrt = U @ np.diag(np.sqrt(s * (n + self.lambda_))) @ Vt

        # 中心点
        sigma_points[0] = x

        # 生成其他sigma点
        for i in range(n):
            sigma_points[i + 1] = x + sqrt[i]
            sigma_points[i + 1 + n] = x - sqrt[i]

        return sigma_points

    def predict(self):
        """
        预测步骤
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

        # 确保协方差矩阵的正定性
        eigenvals, eigenvecs = np.linalg.eigh(self.P_pred)
        eigenvals = np.maximum(eigenvals, 1e-12)
        self.P_pred = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return sigma_points_pred

    def update(self, z):
        """
        更新步骤
        z: 观测值 [ax, ay, az, mx, my, mz]
        """
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
        eigenvals = np.maximum(eigenvals, 1e-12)
        Pzz = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # 计算卡尔曼增益
        try:
            K = np.linalg.solve(Pzz.T, Pxz.T).T
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        # 计算残差
        innovation = z - z_pred

        # 更新状态和协方差
        self.x = self.x_pred + K @ innovation

        # Joseph形式的协方差更新
        I_KH = np.eye(self.n_states) - K @ np.eye(self.n_obs, self.n_states)
        self.P = I_KH @ self.P_pred @ I_KH.T + K @ self.R @ K.T

        # 确保更新后的协方差矩阵正定
        eigenvals, eigenvecs = np.linalg.eigh(self.P)
        eigenvals = np.maximum(eigenvals, 1e-12)
        self.P = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return self.x[:6]


def process_sensor_data_with_ukf_sensor(
    data_list, real_angels, Q=None, R=None, dt=1.0, alpha=1e-3, beta=2.0, kappa=None
):
    """
    处理传感器数据序列并应用无迹卡尔曼滤波 - 直接滤波传感器数据
    """
    # 创建无迹卡尔曼滤波器
    ukf = UnscentedKalmanFilterSensor(
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
