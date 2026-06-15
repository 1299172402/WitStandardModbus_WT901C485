import numpy as np
import pandas as pd

from no7_calc_from_acc_mag import calculate_angles, attitude_to_wellbore_quaternion

###
### 修复了无迹卡尔曼滤波器中的数值不稳定问题，之前的版本数据会发散爆炸
###


class UnscentedKalmanFilter:
    def __init__(self, Q=None, R=None, dt=1.0, alpha=1e-3, beta=2.0, kappa=None):
        """
        无迹卡尔曼滤波器 (UKF)
        """
        self.dt = dt
        self.n_states = 6
        self.n_obs = 3

        # 调整UKF参数，提高数值稳定性
        self.alpha = max(alpha, 1e-4)  # 防止alpha过小
        self.beta = beta
        self.kappa = kappa if kappa is not None else max(3 - self.n_states, 0)

        # 计算lambda参数
        self.lambda_ = self.alpha**2 * (self.n_states + self.kappa) - self.n_states

        # sigma点数量
        self.n_sigma = 2 * self.n_states + 1

        # 权重计算 - 确保数值稳定性
        self.Wm = np.zeros(self.n_sigma)
        self.Wc = np.zeros(self.n_sigma)

        self.Wm[0] = self.lambda_ / (self.n_states + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n_states + self.lambda_) + (
            1 - self.alpha**2 + self.beta
        )

        # 确保权重不会过小
        weight_other = 0.5 / (self.n_states + self.lambda_)
        for i in range(1, self.n_sigma):
            self.Wm[i] = weight_other
            self.Wc[i] = weight_other

        # 状态向量初始化
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 1.0

        # 噪声协方差矩阵 - 使用保守的初始值
        if Q is None:
            self.Q = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])  # 增大过程噪声
        else:
            self.Q = Q

        if R is None:
            self.R = np.diag([1e-1, 1e-1, 1e-1])  # 增大观测噪声
        else:
            self.R = R

        self.is_initialized = False

    def f(self, x, dt):
        """
        状态转移函数 (非线性)
        x: 当前状态 [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        dt: 时间步长
        """
        # 简化的状态转移模型：位置 = 位置 + 速度 * 时间
        x_next = x.copy()
        x_next[0] += x[3] * dt  # roll = roll + roll_rate * dt
        x_next[1] += x[4] * dt  # pitch = pitch + pitch_rate * dt
        x_next[2] += x[5] * dt  # yaw = yaw + yaw_rate * dt
        # 角速度保持不变（可以根据需要添加阻尼等）
        return x_next

    def h(self, x):
        """
        观测函数
        从状态向量中提取观测值 [roll, pitch, yaw]
        """
        return x[:3]  # 直接观测姿态角

    def generate_sigma_points(self, x, P):
        """
        生成sigma点 - 增强数值稳定性
        """
        n = len(x)
        sigma_points = np.zeros((self.n_sigma, n))

        # 确保协方差矩阵的正定性
        P_regularized = P + np.eye(n) * 1e-9

        # 计算矩阵平方根
        try:
            sqrt = np.linalg.cholesky((n + self.lambda_) * P_regularized)
        except np.linalg.LinAlgError:
            # 使用更稳定的SVD分解
            U, s, Vt = np.linalg.svd(P_regularized)
            s = np.maximum(s, 1e-12)  # 防止奇异值过小
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
        eigenvals = np.maximum(eigenvals, 1e-12)
        self.P_pred = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return sigma_points_pred

    def update(self, z):
        """
        更新步骤 - 修复数值不稳定问题
        """
        if not self.is_initialized:
            self.x[:3] = z
            self.x[3:] = 0
            self.is_initialized = True
            return self.x[:3]

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

        # 计算卡尔曼增益 - 使用更稳定的方法
        try:
            K = np.linalg.solve(Pzz.T, Pxz.T).T
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        # 计算残差，处理角度跳跃
        innovation = z - z_pred
        for i in range(len(innovation)):
            # 优化的角度归一化
            innovation[i] = ((innovation[i] + 180) % 360) - 180

        # 限制innovation的大小，防止异常值
        innovation = np.clip(innovation, -90, 90)

        # 更新状态和协方差
        self.x = self.x_pred + K @ innovation

        # Joseph形式的协方差更新，保证数值稳定性
        I_KH = np.eye(self.n_states) - K @ np.eye(self.n_obs, self.n_states)
        self.P = I_KH @ self.P_pred @ I_KH.T + K @ self.R @ K.T

        # 确保更新后的协方差矩阵正定
        eigenvals, eigenvecs = np.linalg.eigh(self.P)
        eigenvals = np.maximum(eigenvals, 1e-12)
        self.P = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return self.x[:3]

    def reset(self):
        """重置滤波器"""
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 1.0
        self.is_initialized = False


def process_sensor_data_with_ukf(
    data_list, real_angels, Q=None, R=None, dt=1.0, alpha=1e-3, beta=2.0, kappa=None
):
    """
    处理传感器数据序列并应用无迹卡尔曼滤波

    data_list: 列表，每个元素为 [ax, ay, az, mx, my, mz]
    real_angels: 真实角度数据
    Q: 过程噪声协方差矩阵
    R: 观测噪声协方差矩阵
    dt: 时间步长
    alpha: UKF分布参数
    beta: UKF先验知识参数
    kappa: UKF缩放参数
    """

    # 创建无迹卡尔曼滤波器
    ukf = UnscentedKalmanFilter(Q=Q, R=R, dt=dt, alpha=alpha, beta=beta, kappa=kappa)

    # 计算每个数据点的井眼角度
    results = []
    for i in range(len(data_list)):
        # 从传感器数据计算姿态角作为观测值
        ax, ay, az, mx, my, mz = data_list[i]
        roll_obs, pitch_obs, yaw_obs = calculate_angles(ax, ay, az, mx, my, mz)

        # UKF滤波
        roll_filtered, pitch_filtered, yaw_filtered = ukf.update(
            [roll_obs, pitch_obs, yaw_obs]
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
                "filtered_data": [ax, ay, az, mx, my, mz],
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
    print("无迹卡尔曼滤波器测试")
    print("=" * 100)

    # 测试不同参数的UKF
    test_params = [
        {
            "Q": np.diag([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]),
            "R": np.diag([1e-2, 1e-2, 1e-2]),
            "dt": 1.0,
            "alpha": 1e-3,
            "beta": 2.0,
            "kappa": 0,
            "name": "标准参数 UKF (alpha=1e-3)",
        },
        {
            "Q": np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]),
            "R": np.diag([1e-2, 1e-2, 1e-2]),
            "dt": 1.0,
            "alpha": 1e-2,
            "beta": 2.0,
            "kappa": 0,
            "name": "高alpha参数 UKF (alpha=1e-2)",
        },
        {
            "Q": np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]),
            "R": np.diag([1e-3, 1e-3, 1e-3]),
            "dt": 1.0,
            "alpha": 1e-3,
            "beta": 2.0,
            "kappa": 3,
            "name": "低观测噪声 UKF (R=1e-3)",
        },
    ]

    for params in test_params:
        print(f"\n{params['name']}:")
        print("-" * 80)

        # 处理数据
        ukf_results = process_sensor_data_with_ukf(
            sample_data,
            real_angels,
            Q=params["Q"],
            R=params["R"],
            dt=params["dt"],
            alpha=params["alpha"],
            beta=params["beta"],
            kappa=params["kappa"],
        )

        # 显示最后3个数据点的结果
        for result in ukf_results[-3:]:
            print(f"数据点 {result['index']}:")
            print(
                f"  观测角度: Roll={result['roll_obs']:.2f}°, Pitch={result['pitch_obs']:.2f}°, Yaw={result['yaw_obs']:.2f}°"
            )
            print(
                f"  滤波角度: Roll={result['roll']:.2f}°, Pitch={result['pitch']:.2f}°, Yaw={result['yaw']:.2f}°"
            )
            print(
                f"  实际角度: Roll={result['real_roll']:.2f}°, Pitch={result['real_pitch']:.2f}°, Yaw={result['real_yaw']:.2f}°"
            )
            print(
                f"  误差: ΔRoll={result['diff_roll']:.2f}°, ΔPitch={result['diff_pitch']:.2f}°, ΔYaw={result['diff_yaw']:.2f}°"
            )
            print(
                f"  井眼角度误差: ΔAzimuth={result['diff_azimuth']:.2f}°, ΔInclination={result['diff_inclination']:.2f}°, ΔToolface={result['diff_toolface']:.2f}°"
            )
            print()
