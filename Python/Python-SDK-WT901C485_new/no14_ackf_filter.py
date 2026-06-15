import numpy as np
import pandas as pd

from no7_calc_from_acc_mag import calculate_angles, attitude_to_wellbore_quaternion


class AdaptiveCubatureKalmanFilter:
    def __init__(self, Q=None, R=None, dt=1.0, window_size=5, forget_factor=0.95):
        """
        自适应容积卡尔曼滤波器 (ACKF)
        状态向量: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]

        Q: 初始过程噪声协方差矩阵 (6x6)
        R: 初始观测噪声协方差矩阵 (3x3)
        dt: 时间步长
        window_size: 自适应窗口大小
        forget_factor: 遗忘因子 (0 < forget_factor <= 1)
        """
        self.dt = dt
        self.n_states = 6  # [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.n_obs = 3  # [roll, pitch, yaw] 观测值
        self.window_size = window_size
        self.forget_factor = forget_factor

        # 状态向量初始化
        self.x = np.zeros(self.n_states)  # 状态估计

        # 协方差矩阵初始化
        self.P = np.eye(self.n_states) * 1.0  # 状态协方差矩阵

        # 初始噪声协方差矩阵
        if Q is None:
            self.Q = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])
        else:
            self.Q = Q

        if R is None:
            self.R = np.diag([1e-2, 1e-2, 1e-2])
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
        x: 当前状态 [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        dt: 时间步长
        """
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

    def generate_cubature_points(self, x, P):
        """
        生成容积点
        x: 状态均值
        P: 状态协方差矩阵
        """
        n = len(x)

        # 计算矩阵平方根
        try:
            sqrt_P = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # 如果矩阵不是正定的，使用SVD分解
            U, s, Vt = np.linalg.svd(P)
            sqrt_P = U @ np.diag(np.sqrt(np.maximum(s, 1e-12))) @ Vt

        # 标准容积点（单位球面上的点）
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
        innovation: 当前新息
        S: 新息协方差矩阵
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

        # 理论新息协方差
        theoretical_cov = S

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
        # 使用比值来调整R矩阵
        try:
            ratio = np.trace(actual_cov) / np.trace(theoretical_cov)

            # 限制调整幅度，避免数值不稳定
            ratio = np.clip(ratio, 0.1, 10.0)

            # 平滑调整
            alpha = 0.1  # 调整速度
            self.R = (1 - alpha) * self.R + alpha * ratio * self.R

            # 确保R矩阵的正定性
            eigenvals = np.linalg.eigvals(self.R)
            if np.any(eigenvals <= 0):
                self.R = self.R + np.eye(self.n_obs) * 1e-6

        except (np.linalg.LinAlgError, ZeroDivisionError):
            pass  # 保持当前R不变

    def adapt_process_noise(self):
        """
        自适应调整过程噪声协方差矩阵
        基于状态估计的不确定性
        """
        # 根据状态协方差的迹来调整过程噪声
        trace_P = np.trace(self.P)

        # 如果状态不确定性过大，增加过程噪声
        if trace_P > 10.0:
            self.Q = self.Q * 1.01
        # 如果状态不确定性过小，减少过程噪声
        elif trace_P < 0.1:
            self.Q = self.Q * 0.99

        # 限制Q的范围
        Q_min = np.diag([1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5])
        Q_max = np.diag([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1])

        self.Q = np.maximum(self.Q, Q_min)
        self.Q = np.minimum(self.Q, Q_max)

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

        # 自适应调整过程噪声
        self.adapt_process_noise()

    def update(self, z):
        """
        更新步骤
        z: 观测值 [roll, pitch, yaw]
        """
        if not self.is_initialized:
            # 第一次初始化
            self.x[:3] = z  # 设置初始姿态角
            self.x[3:] = 0  # 初始角速度为0
            self.is_initialized = True
            return self.x[:3]

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

        # 计算残差，处理角度跳跃
        innovation = z - z_pred
        for i in range(len(innovation)):
            # 优化的角度归一化
            innovation[i] = ((innovation[i] + 180) % 360) - 180

        # 自适应调整噪声协方差
        self.adapt_noise_covariance(innovation, Pzz)

        # 更新状态和协方差
        self.x = self.x_pred + K @ innovation
        self.P = self.P_pred - K @ Pzz @ K.T

        return self.x[:3]  # 返回估计的姿态角

    def reset(self):
        """重置滤波器"""
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 1.0
        self.innovation_history = []
        self.is_initialized = False


def process_sensor_data_with_ackf(
    data_list, real_angels, Q=None, R=None, dt=1.0, window_size=5, forget_factor=0.95
):
    """
    处理传感器数据序列并应用自适应容积卡尔曼滤波

    data_list: 列表，每个元素为 [ax, ay, az, mx, my, mz]
    real_angels: 真实角度数据
    Q: 过程噪声协方差矩阵
    R: 观测噪声协方差矩阵
    dt: 时间步长
    window_size: 自适应窗口大小
    forget_factor: 遗忘因子
    """

    # 创建自适应容积卡尔曼滤波器
    ackf = AdaptiveCubatureKalmanFilter(
        Q=Q, R=R, dt=dt, window_size=window_size, forget_factor=forget_factor
    )

    # 计算每个数据点的井眼角度
    results = []
    for i in range(len(data_list)):
        # 从传感器数据计算姿态角作为观测值
        ax, ay, az, mx, my, mz = data_list[i]
        roll_obs, pitch_obs, yaw_obs = calculate_angles(ax, ay, az, mx, my, mz)

        # ACKF滤波
        roll_filtered, pitch_filtered, yaw_filtered = ackf.update(
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
    print("自适应容积卡尔曼滤波器测试")
    print("=" * 100)

    # 测试不同参数的ACKF
    test_params = [
        {
            "Q": np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]),
            "R": np.diag([1e-2, 1e-2, 1e-2]),
            "dt": 1.0,
            "window_size": 5,
            "forget_factor": 0.95,
            "name": "标准参数 ACKF",
        },
        {
            "Q": np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]),
            "R": np.diag([1e-2, 1e-2, 1e-2]),
            "dt": 1.0,
            "window_size": 3,
            "forget_factor": 0.9,
            "name": "快速自适应 ACKF (小窗口)",
        },
        {
            "Q": np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]),
            "R": np.diag([1e-2, 1e-2, 1e-2]),
            "dt": 1.0,
            "window_size": 8,
            "forget_factor": 0.98,
            "name": "平滑自适应 ACKF (大窗口)",
        },
    ]

    for params in test_params:
        print(f"\n{params['name']}:")
        print("-" * 80)

        # 处理数据
        ackf_results = process_sensor_data_with_ackf(
            sample_data,
            real_angels,
            Q=params["Q"],
            R=params["R"],
            dt=params["dt"],
            window_size=params["window_size"],
            forget_factor=params["forget_factor"],
        )

        # 显示结果
        for result in ackf_results:
            print(f"数据点 {result['index']}:")
            print(f"  原始数据: {[f'{x:.6f}' for x in result['raw_data']]}")
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
                f"  井眼角度: Azimuth={result['azimuth']:.2f}°, Inclination={result['inclination']:.2f}°, Toolface={result['toolface']:.2f}°"
            )
            print(
                f"  实际井眼角度: Azimuth={result['real_azimuth']:.2f}°, Inclination={result['real_inclination']:.2f}°, Toolface={result['real_toolface']:.2f}°"
            )
            print(
                f"  井眼角度误差: ΔAzimuth={result['diff_azimuth']:.2f}°, ΔInclination={result['diff_inclination']:.2f}°, ΔToolface={result['diff_toolface']:.2f}°"
            )
            print()
