import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import random

# 设置matplotlib支持中文
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class DrillAccelerationSimulator:
    def __init__(self, sampling_rate=1000, duration=10):
        """
        旋转导向设备加速度噪声模拟器

        参数:
        sampling_rate: 采样率 (Hz)
        duration: 模拟时长 (秒)
        """
        self.fs = sampling_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        self.n_samples = len(self.t)

    def generate_drilling_vibration(self, rpm=120, wob=50000):
        """
        生成钻进振动信号

        参数:
        rpm: 转速 (转/分钟)
        wob: 钻压 (牛顿)
        """
        # 转速相关频率
        rotation_freq = rpm / 60  # Hz

        # 钻进主频率成分
        drilling_freqs = [
            rotation_freq,  # 基频
            2 * rotation_freq,  # 二次谐波
            3 * rotation_freq,  # 三次谐波
            rotation_freq / 2,  # 亚谐波
        ]

        # 钻压相关的低频振动
        wob_factor = wob / 100000  # 归一化钻压影响

        vibration = np.zeros_like(self.t)

        for i, freq in enumerate(drilling_freqs):
            amplitude = (0.5 + 0.3 * wob_factor) / (i + 1)  # 递减幅值
            phase = random.uniform(0, 2 * np.pi)
            vibration += amplitude * np.sin(2 * np.pi * freq * self.t + phase)

        return vibration

    def generate_formation_noise(self):
        """生成地层相关的随机噪声"""
        # 地层硬度变化引起的低频噪声 (0.1-5 Hz)
        formation_noise = 0.3 * signal.butter_lowpass_filter(
            np.random.normal(0, 1, self.n_samples), 5, self.fs
        )

        # 岩石破碎产生的冲击噪声 (高频脉冲)
        impact_noise = np.zeros_like(self.t)
        n_impacts = int(self.duration * 10)  # 平均每秒10次冲击
        impact_times = np.random.uniform(0, self.duration, n_impacts)

        for impact_time in impact_times:
            impact_idx = int(impact_time * self.fs)
            if impact_idx < self.n_samples:
                # 指数衰减脉冲
                decay_samples = min(int(0.01 * self.fs), self.n_samples - impact_idx)
                decay = np.exp(-np.arange(decay_samples) / (0.002 * self.fs))
                amplitude = np.random.uniform(0.5, 2.0)
                impact_noise[impact_idx : impact_idx + decay_samples] += (
                    amplitude * decay
                )

        return formation_noise + impact_noise

    def generate_mechanical_noise(self):
        """生成机械系统噪声"""
        # 轴承噪声 (高频)
        bearing_freq = np.random.uniform(200, 500)
        bearing_noise = 0.1 * np.sin(
            2 * np.pi * bearing_freq * self.t + np.random.uniform(0, 2 * np.pi)
        )

        # 齿轮噪声
        gear_freq = np.random.uniform(50, 150)
        gear_noise = 0.15 * np.sin(
            2 * np.pi * gear_freq * self.t + np.random.uniform(0, 2 * np.pi)
        )

        # 液压系统噪声
        hydraulic_noise = 0.05 * signal.butter_bandpass_filter(
            np.random.normal(0, 1, self.n_samples), 10, 100, self.fs
        )

        return bearing_noise + gear_noise + hydraulic_noise

    def generate_three_axis_acceleration(self, rpm=120, wob=50000, inclination=30):
        """
        生成三轴加速度信号

        参数:
        rpm: 转速
        wob: 钻压
        inclination: 井斜角度 (度)
        """
        # 基础振动信号
        base_vibration = self.generate_drilling_vibration(rpm, wob)
        formation_noise = self.generate_formation_noise()
        mechanical_noise = self.generate_mechanical_noise()

        # 白噪声
        white_noise = 0.1 * np.random.normal(0, 1, self.n_samples)

        # 重力影响
        inclination_rad = np.radians(inclination)
        gravity_x = 9.8 * np.sin(inclination_rad)
        gravity_z = 9.8 * np.cos(inclination_rad)

        # X轴 (水平横向)
        acc_x = (
            base_vibration * 0.8
            + formation_noise * 0.6
            + mechanical_noise * 1.0
            + white_noise
            + gravity_x
        )

        # Y轴 (水平纵向)
        acc_y = (
            base_vibration * 0.9
            + formation_noise * 0.7
            + mechanical_noise * 0.8
            + white_noise * 0.9
        )

        # Z轴 (垂直) - 主要振动方向
        acc_z = (
            base_vibration * 1.2
            + formation_noise * 1.0
            + mechanical_noise * 0.6
            + white_noise * 0.8
            + gravity_z
        )

        return acc_x, acc_y, acc_z

    def calculate_spectrum(self, signal_data):
        """计算频谱"""
        spectrum = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1 / self.fs)

        # 只取正频率部分
        positive_freqs = freqs[: len(freqs) // 2]
        positive_spectrum = np.abs(spectrum[: len(spectrum) // 2])

        return positive_freqs, positive_spectrum

    def plot_results(self, acc_x, acc_y, acc_z):
        """绘制结果"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("旋转导向设备三轴加速度噪声模拟", fontsize=16)

        # 时域信号
        axes[0, 0].plot(
            self.t[: int(2 * self.fs)], acc_x[: int(2 * self.fs)], "r-", linewidth=0.8
        )
        axes[0, 0].set_title("X轴加速度时域信号 (前2秒)")
        axes[0, 0].set_xlabel("时间 (s)")
        axes[0, 0].set_ylabel("加速度 (m/s2)")
        axes[0, 0].grid(True)

        axes[1, 0].plot(
            self.t[: int(2 * self.fs)], acc_y[: int(2 * self.fs)], "g-", linewidth=0.8
        )
        axes[1, 0].set_title("Y轴加速度时域信号 (前2秒)")
        axes[1, 0].set_xlabel("时间 (s)")
        axes[1, 0].set_ylabel("加速度 (m/s2)")
        axes[1, 0].grid(True)

        axes[2, 0].plot(
            self.t[: int(2 * self.fs)], acc_z[: int(2 * self.fs)], "b-", linewidth=0.8
        )
        axes[2, 0].set_title("Z轴加速度时域信号 (前2秒)")
        axes[2, 0].set_xlabel("时间 (s)")
        axes[2, 0].set_ylabel("加速度 (m/s2)")
        axes[2, 0].grid(True)

        # 频域信号
        freqs_x, spectrum_x = self.calculate_spectrum(acc_x)
        freqs_y, spectrum_y = self.calculate_spectrum(acc_y)
        freqs_z, spectrum_z = self.calculate_spectrum(acc_z)

        axes[0, 1].semilogy(freqs_x, spectrum_x, "r-", linewidth=1)
        axes[0, 1].set_title("X轴加速度频谱")
        axes[0, 1].set_xlabel("频率 (Hz)")
        axes[0, 1].set_ylabel("幅值")
        axes[0, 1].set_xlim(0, 200)
        axes[0, 1].grid(True)

        axes[1, 1].semilogy(freqs_y, spectrum_y, "g-", linewidth=1)
        axes[1, 1].set_title("Y轴加速度频谱")
        axes[1, 1].set_xlabel("频率 (Hz)")
        axes[1, 1].set_ylabel("幅值")
        axes[1, 1].set_xlim(0, 200)
        axes[1, 1].grid(True)

        axes[2, 1].semilogy(freqs_z, spectrum_z, "b-", linewidth=1)
        axes[2, 1].set_title("Z轴加速度频谱")
        axes[2, 1].set_xlabel("频率 (Hz)")
        axes[2, 1].set_ylabel("幅值")
        axes[2, 1].set_xlim(0, 200)
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.show()


# 辅助滤波函数
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)


# 为signal模块添加滤波函数
signal.butter_lowpass_filter = butter_lowpass_filter
signal.butter_bandpass_filter = butter_bandpass_filter

# 主程序
if __name__ == "__main__":
    # 创建模拟器
    simulator = DrillAccelerationSimulator(sampling_rate=100, duration=2)

    # 生成三轴加速度信号
    # 参数: 转速120rpm, 钻压50kN, 井斜45.390096度
    acc_x, acc_y, acc_z = simulator.generate_three_axis_acceleration(
        rpm=120, wob=50000, inclination=45.390096
    )

    # 绘制结果
    simulator.plot_results(acc_x, acc_y, acc_z)

    # 保存数据
    np.savetxt(
        "drill_acceleration_data.csv",
        np.column_stack([simulator.t, acc_x, acc_y, acc_z]),
        header="Time,Acc_X,Acc_Y,Acc_Z",
        delimiter=",",
        comments="",
    )

    print("模拟完成！数据已保存到 drill_acceleration_data.csv")
    print(f"采样率: {simulator.fs} Hz")
    print(f"数据长度: {simulator.duration} 秒")
    print(f"X轴RMS: {np.sqrt(np.mean(acc_x**2)):.3f} m/s²")
    print(f"Y轴RMS: {np.sqrt(np.mean(acc_y**2)):.3f} m/s²")
    print(f"Z轴RMS: {np.sqrt(np.mean(acc_z**2)):.3f} m/s²")
