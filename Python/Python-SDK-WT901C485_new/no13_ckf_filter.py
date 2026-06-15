import numpy as np
import pandas as pd

from no7_calc_from_acc_mag import calculate_angles, attitude_to_wellbore_quaternion


class CubatureKalmanFilter:
    def __init__(self, Q=None, R=None, dt=1.0):
        """
        容积卡尔曼滤波器 (CKF)
        状态向量: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]

        Q: 过程噪声协方差矩阵 (6x6)
        R: 观测噪声协方差矩阵 (3x3)
        dt: 时间步长
        """
        self.dt = dt
        self.n_states = 6  # [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.n_obs = 3  # [roll, pitch, yaw] 观测值

        # 状态向量初始化
        self.x = np.zeros(self.n_states)  # 状态估计

        # 协方差矩阵初始化
        self.P = np.eye(self.n_states) * 1.0  # 状态协方差矩阵

        # 过程噪声协方差矩阵
        if Q is None:
            self.Q = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])
        else:
            self.Q = Q

        # 观测噪声协方差矩阵
        if R is None:
            self.R = np.diag([1e-2, 1e-2, 1e-2])
        else:
            self.R = R

        self.is_initialized = False

        # 容积点数 = 2 * n_states
        self.num_cubature_points = 2 * self.n_states

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
            # 如果矩阵奇异，使用伪逆
            K = Pxz @ np.linalg.pinv(Pzz)

        # 计算残差，处理角度跳跃
        innovation = z - z_pred
        for i in range(len(innovation)):
            while innovation[i] > 180:
                innovation[i] -= 360
            while innovation[i] < -180:
                innovation[i] += 360

        # 更新状态和协方差
        self.x = self.x_pred + K @ innovation
        self.P = self.P_pred - K @ Pzz @ K.T

        return self.x[:3]  # 返回估计的姿态角

    def reset(self):
        """重置滤波器"""
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 1.0
        self.is_initialized = False


def process_sensor_data_with_ckf(data_list, real_angels, Q=None, R=None, dt=1.0):
    """
    处理传感器数据序列并应用容积卡尔曼滤波

    data_list: 列表，每个元素为 [ax, ay, az, mx, my, mz]
    real_angels: 真实角度数据
    Q: 过程噪声协方差矩阵
    R: 观测噪声协方差矩阵
    dt: 时间步长
    """

    # 创建容积卡尔曼滤波器
    ckf = CubatureKalmanFilter(Q=Q, R=R, dt=dt)

    # 计算每个数据点的井眼角度
    results = []
    for i in range(len(data_list)):
        # 从传感器数据计算姿态角作为观测值
        ax, ay, az, mx, my, mz = data_list[i]
        roll_obs, pitch_obs, yaw_obs = calculate_angles(ax, ay, az, mx, my, mz)

        # CKF滤波
        roll_filtered, pitch_filtered, yaw_filtered = ckf.update(
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
                "filtered_data": [
                    ax,
                    ay,
                    az,
                    mx,
                    my,
                    mz,
                ],  # CKF直接处理角度，原始传感器数据保持不变
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
    print("容积卡尔曼滤波器测试")
    print("=" * 100)

    # 测试不同参数的CKF
    test_params = [
        {
            "Q": np.diag([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]),
            "R": np.diag([1e-2, 1e-2, 1e-2]),
            "dt": 1.0,
            "name": "低过程噪声 CKF",
        },
        {
            "Q": np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]),
            "R": np.diag([1e-2, 1e-2, 1e-2]),
            "dt": 1.0,
            "name": "中等过程噪声 CKF",
        },
        {
            "Q": np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2]),
            "R": np.diag([1e-2, 1e-2, 1e-2]),
            "dt": 1.0,
            "name": "高过程噪声 CKF",
        },
        {
            "Q": np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]),
            "R": np.diag([1e-3, 1e-3, 1e-3]),
            "dt": 1.0,
            "name": "低观测噪声 CKF",
        },
    ]

    for params in test_params:
        print(f"\n{params['name']}:")
        print("-" * 80)

        # 处理数据
        ckf_results = process_sensor_data_with_ckf(
            sample_data, real_angels, Q=params["Q"], R=params["R"], dt=params["dt"]
        )

        # 显示最后3个数据点的结果
        for result in ckf_results[-3:]:
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
