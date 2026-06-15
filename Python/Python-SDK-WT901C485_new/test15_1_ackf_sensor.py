import numpy as np
import pandas as pd

from no7_calc_from_acc_mag import calculate_angles, attitude_to_wellbore_quaternion


class AdaptiveCubatureKalmanFilterSensor:
    def __init__(self, Q=None, R=None, dt=1.0, window_size=5, forget_factor=0.95):
        """
        自适应容积卡尔曼滤波器 (ACKF) - 直接滤波传感器数据
        状态向量: [ax, ay, az, mx, my, mz, ax_rate, ay_rate, az_rate, mx_rate, my_rate, mz_rate]
        """
        self.dt = dt
        self.n_states = 12  # 6个传感器值 + 6个变化率
        self.n_obs = 6  # [ax, ay, az, mx, my, mz] 观测值
        self.window_size = window_size
        self.forget_factor = forget_factor

        # 状态向量初始化
        self.x = np.zeros(self.n_states)  # 状态估计

        # 协方差矩阵初始化
        self.P = np.eye(self.n_states) * 1.0  # 状态协方差矩阵

        # 初始噪声协方差矩阵
        if Q is None:
            # 传感器值的过程噪声较小，变化率的过程噪声较大
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
            # 观测噪声：加速度计噪声小，磁力计噪声大
            self.R = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
        else:
            self.R = R

        # 自适应参数
        self.innovation_history = []  # 新息序列历史
        self.is_initialized = False

        # 容积点数 = 2 * n_states
        self.num_cubature_points = 2 * self.n_states

    def f(self, x, dt):
        """
        状态转移函数 (非线性)
        x: 当前状态 [ax, ay, az, mx, my, mz, ax_rate, ay_rate, az_rate, mx_rate, my_rate, mz_rate]
        dt: 时间步长
        """
        x_next = x.copy()
        # 传感器值 = 传感器值 + 变化率 * 时间
        x_next[0] += x[6] * dt  # ax = ax + ax_rate * dt
        x_next[1] += x[7] * dt  # ay = ay + ay_rate * dt
        x_next[2] += x[8] * dt  # az = az + az_rate * dt
        x_next[3] += x[9] * dt  # mx = mx + mx_rate * dt
        x_next[4] += x[10] * dt  # my = my + my_rate * dt
        x_next[5] += x[11] * dt  # mz = mz + mz_rate * dt
        # 变化率保持不变（可以根据需要添加阻尼等）
        return x_next

    def h(self, x):
        """
        观测函数
        从状态向量中提取观测值 [ax, ay, az, mx, my, mz]
        """
        return x[:6]  # 直接观测传感器值

    def generate_cubature_points(self, x, P):
        """
        生成容积点
        """
        n = len(x)

        # 计算矩阵平方根
        try:
            sqrt_P = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # 如果矩阵不是正定的，使用SVD分解
            U, s, Vt = np.linalg.svd(P)
            sqrt_P = U @ np.diag(np.sqrt(np.maximum(s, 1e-12))) @ Vt

        # 标准容积点
        sqrt_n = np.sqrt(n)
        Xi = np.zeros((2 * n, n))

        # 正方向容积点
        for i in range(n):
            Xi[i, i] = sqrt_n

        # 负方向容积点
        for i in range(n):
            Xi[n + i, i] = -sqrt_n

        # 变换容积点到实际状态空间
        cubature_points = np.zeros((2 * n, n))
        for i in range(2 * n):
            cubature_points[i] = x + sqrt_P @ Xi[i]

        return cubature_points

    def adapt_noise_covariance(self, innovation, S):
        """
        自适应调整噪声协方差矩阵
        """
        # 存储新息历史
        self.innovation_history.append(innovation.copy())

        # 保持窗口大小
        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)

        if len(self.innovation_history) < 3:
            return

        # 计算新息的统计特性
        innovations = np.array(self.innovation_history)

        # 实际新息协方差（使用遗忘因子加权）
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

        # 自适应调整观测噪声协方差
        try:
            ratio = np.trace(actual_cov) / np.trace(S)
            ratio = np.clip(ratio, 0.1, 10.0)

            # 平滑调整
            alpha = 0.1
            self.R = (1 - alpha) * self.R + alpha * ratio * self.R

            # 确保R矩阵的正定性
            eigenvals = np.linalg.eigvals(self.R)
            if np.any(eigenvals <= 0):
                self.R = self.R + np.eye(self.n_obs) * 1e-6

        except (np.linalg.LinAlgError, ZeroDivisionError):
            pass

    def predict(self):
        """
        预测步骤
        """
        # 生成容积点
        cubature_points = self.generate_cubature_points(self.x, self.P)

        # 传播容积点通过状态转移函数
        predicted_points = np.zeros_like(cubature_points)
        for i in range(self.num_cubature_points):
            predicted_points[i] = self.f(cubature_points[i], self.dt)

        # 计算预测状态均值
        self.x_pred = np.mean(predicted_points, axis=0)

        # 计算预测协方差
        self.P_pred = np.zeros((self.n_states, self.n_states))
        for i in range(self.num_cubature_points):
            diff = predicted_points[i] - self.x_pred
            self.P_pred += np.outer(diff, diff)

        self.P_pred = self.P_pred / self.num_cubature_points + self.Q

    def update(self, z):
        """
        更新步骤
        z: 观测值 [ax, ay, az, mx, my, mz]
        """
        if not self.is_initialized:
            # 第一次初始化
            self.x[:6] = z  # 设置初始传感器值
            self.x[6:] = 0  # 初始变化率为0
            self.is_initialized = True
            return self.x[:6]

        # 预测步骤
        self.predict()

        # 生成预测状态的容积点
        cubature_points = self.generate_cubature_points(self.x_pred, self.P_pred)

        # 传播容积点通过观测函数
        predicted_obs = np.zeros((self.num_cubature_points, self.n_obs))
        for i in range(self.num_cubature_points):
            predicted_obs[i] = self.h(cubature_points[i])

        # 计算预测观测均值
        z_pred = np.mean(predicted_obs, axis=0)

        # 计算观测协方差矩阵
        Pzz = np.zeros((self.n_obs, self.n_obs))
        for i in range(self.num_cubature_points):
            diff = predicted_obs[i] - z_pred
            Pzz += np.outer(diff, diff)
        Pzz = Pzz / self.num_cubature_points + self.R

        # 计算交叉协方差矩阵
        Pxz = np.zeros((self.n_states, self.n_obs))
        for i in range(self.num_cubature_points):
            x_diff = cubature_points[i] - self.x_pred
            z_diff = predicted_obs[i] - z_pred
            Pxz += np.outer(x_diff, z_diff)
        Pxz = Pxz / self.num_cubature_points

        # 计算卡尔曼增益
        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        # 计算残差
        innovation = z - z_pred

        # 自适应调整噪声协方差
        self.adapt_noise_covariance(innovation, Pzz)

        # 更新状态和协方差
        self.x = self.x_pred + K @ innovation
        self.P = self.P_pred - K @ Pzz @ K.T

        return self.x[:6]  # 返回估计的传感器值

    def reset(self):
        """重置滤波器"""
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 1.0
        self.innovation_history = []
        self.is_initialized = False


def process_sensor_data_with_ackf_sensor(
    data_list, real_angels, Q=None, R=None, dt=1.0, window_size=5, forget_factor=0.95
):
    """
    处理传感器数据序列并应用自适应容积卡尔曼滤波 - 直接滤波传感器数据
    """
    # 创建自适应容积卡尔曼滤波器
    ackf = AdaptiveCubatureKalmanFilterSensor(
        Q=Q, R=R, dt=dt, window_size=window_size, forget_factor=forget_factor
    )

    results = []
    for i in range(len(data_list)):
        # 原始传感器数据
        ax, ay, az, mx, my, mz = data_list[i]

        # ACKF滤波传感器数据
        ax_filtered, ay_filtered, az_filtered, mx_filtered, my_filtered, mz_filtered = (
            ackf.update([ax, ay, az, mx, my, mz])
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
