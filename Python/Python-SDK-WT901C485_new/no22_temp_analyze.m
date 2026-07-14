%% 临时分析脚本 — 计算 UKF 和 ACKF 与原始数据的偏差
% 假设 ukf_results, ackf_results, raw_data 在 workspace 中

sensor_names = {'ax', 'ay', 'az', 'mx', 'my', 'mz'};

fprintf('\n%s\n', repmat('=', 1, 90));
fprintf('  误差详细分析\n');
fprintf('%s\n', repmat('=', 1, 90));

for i = 1:6
    raw_i  = raw_data(:, i);
    ukf_i  = ukf_results(:, i);
    ackf_i = ackf_results(:, i);
    
    % UKF 与原始数据的误差
    ukf_err = ukf_i - raw_i;
    ackf_err = ackf_i - raw_i;
    
    % UKF 和 ACKF 之间的差
    diff_ukf_ackf = ukf_i - ackf_i;
    
    % ===== 统计指标 =====
    % 均方根误差 (RMSE)
    ukf_rmse = sqrt(mean(ukf_err.^2));
    ackf_rmse = sqrt(mean(ackf_err.^2));
    
    % 平均绝对误差 (MAE)
    ukf_mae = mean(abs(ukf_err));
    ackf_mae = mean(abs(ackf_err));
    
    % 最大绝对偏差
    ukf_max = max(abs(ukf_err));
    ackf_max = max(abs(ackf_err));
    
    % UKF与ACKF之间的差异
    diff_mean = mean(abs(diff_ukf_ackf));
    diff_max = max(abs(diff_ukf_ackf));
    diff_std = std(diff_ukf_ackf);
    
    % ===== 信号本身的变化幅度 =====
    raw_range = max(raw_i) - min(raw_i);
    raw_std = std(raw_i);
    
    fprintf('\n%s 通道:\n', upper(sensor_names{i}));
    fprintf('  原始数据范围: %.4f (std=%.4f)\n', raw_range, raw_std);
    fprintf('  UKF vs 原始:  RMSE=%.6f, MAE=%.6f, Max=%.6f\n', ukf_rmse, ukf_mae, ukf_max);
    fprintf('  ACKF vs 原始: RMSE=%.6f, MAE=%.6f, Max=%.6f\n', ackf_rmse, ackf_mae, ackf_max);
    fprintf('  UKF vs ACKF:  平均偏差=%.6f, 最大偏差=%.6f, std=%.6f\n', diff_mean, diff_max, diff_std);
    
    % 用RMSE与信号范围的比例来判断偏差是否严重
    ukf_err_ratio = ukf_rmse / raw_range * 100;
    ackf_err_ratio = ackf_rmse / raw_range * 100;
    fprintf('  UKF误差/信号幅度 = %.1f%%\n', ukf_err_ratio);
    fprintf('  ACKF误差/信号幅度 = %.1f%%\n', ackf_err_ratio);
    
    % 检查ACKF是否过于平滑（变化率对比）
    ukf_chg = mean(abs(diff(ukf_i)));
    ackf_chg = mean(abs(diff(ackf_i)));
    raw_chg = mean(abs(diff(raw_i)));
    fprintf('  变化率: 原始=%.6f, UKF=%.6f (%.0f%%), ACKF=%.6f (%.0f%%)\n', ...
        raw_chg, ukf_chg, ukf_chg/raw_chg*100, ackf_chg, ackf_chg/raw_chg*100);
end

fprintf('\n%s\n', repmat('=', 1, 90));
fprintf('  结论：UKF 和 ACKF 之间的平均偏差越大，肉眼可见差别越大\n');
fprintf('%s\n', repmat('=', 1, 90));
