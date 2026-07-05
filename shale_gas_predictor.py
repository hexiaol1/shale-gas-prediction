# -*- coding: utf-8 -*-
"""
常压区页岩气产能预测程序 — 核心模块
=====================================
包含数据加载、物性计算、递减模型、物质平衡、可视化等全部功能。

作者: Shale Gas Analytics
版本: 1.0.0
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.special import gamma as gamma_func
from scipy.special import gammainc, gammaincc

# NumPy 兼容性: trapz -> trapezoid, nan_to_num 已移除 (NumPy 2.x)
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
_nan_to_num = getattr(np, 'nan_to_num', None)
if _nan_to_num is None:
    def _nan_to_num(x, nan=0.0, posinf=None, neginf=None):
        """简易替代: 替换 NaN/inf (兼容 NumPy 2.x)"""
        x = np.where(np.isnan(x), nan, x)
        if posinf is not None:
            x = np.where(x == np.inf, posinf, x)
        if neginf is not None:
            x = np.where(x == -np.inf, neginf, x)
        return x
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# =====================================================================
# 全局绘图设置 — 中文字体
# =====================================================================
# 尝试查找系统中已有的中文字体；app.py 中的下载式设置会在此之后覆盖
_TRY_ZH = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
           'Noto Sans SC', 'Noto Sans CJK SC', 'Source Han Sans SC']
for _f in _TRY_ZH:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        # 已安装，设为默认
        matplotlib.rcParams['font.sans-serif'] = [_f] + matplotlib.rcParams['font.sans-serif']
        break
    except Exception:
        continue
matplotlib.rcParams['axes.unicode_minus'] = False


# =====================================================================
# 1. 生产数据类
# =====================================================================
class ProductionData:
    """加载、存储和预处理生产数据"""

    REQUIRED_COLS = {'time', 'rate'}  # 必须列
    RECOMMENDED_COLS = {'cum_prod', 'pressure', 'pressure_bhp'}  # 推荐列

    def __init__(self, filepath=None, dataframe=None):
        """
        从 CSV/Excel 文件或已有的 DataFrame 加载数据。

        参数
        ----------
        filepath : str or Path
            .csv 或 .xlsx 文件路径。
            必须包含列: 'time' (生产时间, 天), 'rate' (产气量, 万方/天 或 m³/d)。
            可选: 'cum_prod' (累计产气量, 万方 或 m³),
                  'pressure' (井口压力/套压, MPa),
                  'pressure_bhp' (井底流压, MPa)。
        dataframe : pd.DataFrame
            已有 DataFrame，列要求同上。
        """
        if filepath is not None:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"文件未找到: {filepath}")
            ext = filepath.suffix.lower()
            if ext == '.csv':
                self.data = pd.read_csv(filepath, encoding='utf-8')
            elif ext in ('.xlsx', '.xls'):
                self.data = pd.read_excel(filepath)
            else:
                raise ValueError(f"不支持的文件格式: {ext}，请使用 CSV 或 Excel。")
        elif dataframe is not None:
            self.data = dataframe.copy()
        else:
            self.data = pd.DataFrame()
            return

        # 统一列名（小写、去空格、下划线）
        self.data.columns = [c.strip().lower().replace(' ', '_') for c in self.data.columns]

        # 列名映射 —— 允许常用的别名
        col_map = {
            't': 'time', '时间': 'time', '生产时间': 'time', 'days': 'time',
            'q': 'rate', '产量': 'rate', '产气量': 'rate', '日产量': 'rate',
            'gas_rate': 'rate', '日产气量': 'rate',
            'gp': 'cum_prod', '累积产量': 'cum_prod', '累计产量': 'cum_prod',
            'cum_production': 'cum_prod', 'cumulative_gas': 'cum_prod',
            'p': 'pressure', '套压': 'pressure', '井口压力': 'pressure',
            'pwf': 'pressure_bhp', '流压': 'pressure_bhp', '井底流压': 'pressure_bhp',
        }
        self.data.rename(columns=col_map, inplace=True)

        # 检查必需列
        missing = self.REQUIRED_COLS - set(self.data.columns)
        if missing:
            raise ValueError(f"缺少必需列: {missing}。请确保数据包含 time 和 rate 列。")
        if 'cum_prod' not in self.data.columns:
            # 自动计算累计产量
            self.data['cum_prod'] = self.data['rate'].cumsum()

        # 排序并重置索引
        self.data.sort_values('time', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        # 基本统计
        self._compute_stats()

    def _compute_stats(self):
        """计算基本统计信息"""
        self.n_points = len(self.data)
        self.time_min = float(self.data['time'].min())
        self.time_max = float(self.data['time'].max())
        self.rate_max = float(self.data['rate'].max())
        self.rate_min = float(self.data['rate'].min())
        self.cum_total = float(self.data['cum_prod'].max())

    def clean_outliers(self, z_thresh=3):
        """
        基于 Z-score 剔除异常产气数据点。

        参数
        ----------
        z_thresh : float
            Z-score 阈值，默认 3。越大越宽松。
        """
        rates = self.data['rate'].values
        mean, std = np.nanmean(rates), np.nanstd(rates)
        if std == 0:
            return
        z = np.abs((rates - mean) / std)
        mask = z < z_thresh
        n_removed = (~mask).sum()
        if n_removed > 0:
            self.data = self.data[mask].copy()
            self.data.reset_index(drop=True, inplace=True)
            self._compute_stats()
        return n_removed

    def filter_by_time(self, start_time=None, end_time=None):
        """按时间范围筛选数据"""
        mask = pd.Series(True, index=self.data.index)
        if start_time is not None:
            mask &= (self.data['time'] >= start_time)
        if end_time is not None:
            mask &= (self.data['time'] <= end_time)
        self.data = self.data[mask].copy()
        self.data.reset_index(drop=True, inplace=True)
        self._compute_stats()

    def get_time(self):
        """获取时间数组（天）"""
        return self.data['time'].values

    def get_rate(self):
        """获取产气量数组"""
        return self.data['rate'].values

    def get_cumprod(self):
        """获取累计产气量数组"""
        return self.data['cum_prod'].values

    def get_pressure(self):
        """获取压力数组（如果存在）"""
        if 'pressure' in self.data.columns:
            return self.data['pressure'].values
        return None

    def __repr__(self):
        return (f"ProductionData({self.n_points} 个数据点, "
                f"时间范围: {self.time_min:.0f} ~ {self.time_max:.0f} 天, "
                f"累计产气: {self.cum_total:.2f} 万方)")


# =====================================================================
# 2. 天然气物性计算
# =====================================================================
class GasProperties:
    """天然气高压物性参数计算 (适用于常压区页岩气)"""

    def __init__(self, gamma_g=0.6, co2_pct=0.0, h2s_pct=0.0, n2_pct=0.0):
        """
        参数
        ----------
        gamma_g : float
            天然气相对密度 (空气=1.0), 默认 0.6
        co2_pct : float
            CO2 摩尔分数 (0~1), 默认 0
        h2s_pct : float
            H2S 摩尔分数 (0~1), 默认 0
        n2_pct : float
            N2 摩尔分数 (0~1), 默认 0
        """
        self.gamma_g = gamma_g
        self.co2_pct = co2_pct
        self.h2s_pct = h2s_pct
        self.n2_pct = n2_pct
        # 考虑非烃组分校正
        self._corr_factor = 1.0 - 0.5 * (co2_pct + h2s_pct) + 0.3 * n2_pct

    def pseudo_critical_properties(self):
        """
        计算拟临界温度和压力 (Sutton 方法，适用于常压页岩气藏)。

        返回
        -------
        dict : {'pc': 拟临界压力 (MPa), 'tc': 拟临界温度 (K)}
        """
        gamma = self.gamma_g
        # Sutton (1985) 关联式
        if gamma < 0.57:
            pc = 4.881 - 0.3861 * gamma
            tc = 88.282 + 168.243 * gamma
        else:
            pc = 4.8688 - 0.3564 * gamma + 0.01623 * gamma ** 2
            tc = 87.359 + 159.673 * gamma - 8.196 * gamma ** 2

        # 非烃校正 (Wichert-Aziz)
        eps = 120 * ((self.co2_pct + self.h2s_pct) ** 0.9
                     - (self.co2_pct + self.h2s_pct) ** 1.6) \
              + 15 * (self.h2s_pct ** 0.5 - self.h2s_pct ** 4)
        tc_corr = tc - eps
        pc_corr = pc * tc_corr / (tc + self.h2s_pct * (1 - self.h2s_pct) * eps)
        return {'pc': pc_corr, 'tc': tc_corr}

    def z_factor(self, p, T):
        """
        计算天然气压缩因子 Z (DAK 方法)。

        参数
        ----------
        p : float or array
            压力 (MPa)
        T : float
            温度 (K)

        返回
        -------
        z : float or array
            压缩因子 (无因次)
        """
        pc_tc = self.pseudo_critical_properties()
        p_pr = p / pc_tc['pc']  # 拟对比压力
        T_pr = T / pc_tc['tc']  # 拟对比温度

        # Dranchuk-Abu-Kassem 关联式
        A1, A2, A3, A4 = 0.3265, -1.0700, -0.5339, 0.01569
        A5, A6, A7, A8 = -0.05165, 0.5475, -0.7361, 0.1844
        A9, A10, A11 = 0.1056, 0.6134, 0.7210

        # 迭代求解 Z
        z = np.ones_like(p_pr, dtype=float)
        for i in range(50):
            rho_pr = 0.27 * p_pr / (z * T_pr)
            # DAK 方程
            f = (z - 1
                 + (A1 + A2 / T_pr + A3 / T_pr ** 3 + A4 / T_pr ** 4 + A5 / T_pr ** 5) * rho_pr
                 + (A6 + A7 / T_pr + A8 / T_pr ** 2) * rho_pr ** 2
                 - A9 * (A7 / T_pr + A8 / T_pr ** 2) * rho_pr ** 5
                 + A10 * (1 + A11 * rho_pr ** 2) * (rho_pr ** 2 / T_pr ** 3)
                 * np.exp(-A11 * rho_pr ** 2))
            # 导数
            drho_dz = -0.27 * p_pr / (z ** 2 * T_pr)
            df_dz = (1
                     + (A1 + A2 / T_pr + A3 / T_pr ** 3 + A4 / T_pr ** 4 + A5 / T_pr ** 5) * drho_dz
                     + 2 * (A6 + A7 / T_pr + A8 / T_pr ** 2) * rho_pr * drho_dz
                     - 5 * A9 * (A7 / T_pr + A8 / T_pr ** 2) * rho_pr ** 4 * drho_dz
                     + A10 / T_pr ** 3 * (
                             drho_dz * (1 + A11 * rho_pr ** 2) * np.exp(-A11 * rho_pr ** 2)
                             * rho_pr ** 2
                             + rho_pr ** 2 * (2 * A11 * rho_pr * drho_dz)
                             * np.exp(-A11 * rho_pr ** 2)
                             + rho_pr ** 2 * (1 + A11 * rho_pr ** 2)
                             * np.exp(-A11 * rho_pr ** 2) * (-2 * A11 * rho_pr * drho_dz))
                     )
            z_new = z - f / df_dz
            if np.max(np.abs(z_new - z)) < 1e-8:
                break
            z = z_new
        return z

    def gas_viscosity(self, p, T):
        """
        计算天然气粘度 (Lee-Gonzalez-Eakin 方法)。

        参数
        ----------
        p : float or array
            压力 (MPa)
        T : float
            温度 (K)

        返回
        -------
        mu_g : float or array
            气体粘度 (mPa·s)
        """
        p = np.asarray(p)
        T = np.asarray(T)
        M = self.gamma_g * 28.97  # 摩尔质量
        rho_g = p * M / (self.z_factor(p, T) * 8.314 * T)  # kg/m³

        K = (9.4 + 0.02 * M) * T ** 1.5 / (209 + 19 * M + T)
        X = 3.5 + 986 / T + 0.01 * M
        Y = 2.4 - 0.2 * X

        return K * np.exp(X * (rho_g / 1000) ** Y) * 1e-4

    def gas_formation_volume_factor(self, p, T):
        """
        计算天然气体积系数 Bg。

        参数
        ----------
        p : float or array
            压力 (MPa)
        T : float
            温度 (K)

        返回
        -------
        Bg : float or array
            天然气体积系数 (m³/m³, 地下体积/地面体积)
        """
        z = self.z_factor(p, T)
        return 3.458e-4 * z * T / p

    def gas_compressibility(self, p, T):
        """
        计算天然气等温压缩系数 Cg。

        参数
        ----------
        p : float or array
            压力 (MPa)
        T : float
            温度 (K)

        返回
        -------
        Cg : float or array
            等温压缩系数 (MPa⁻¹)
        """
        pc_tc = self.pseudo_critical_properties()
        p_pr = p / pc_tc['pc']
        z = self.z_factor(p, T)

        # 使用 DAK 关联式求导数 dz/dp_pr
        A1, A2, A3, A4 = 0.3265, -1.0700, -0.5339, 0.01569
        A5, A6, A7, A8 = -0.05165, 0.5475, -0.7361, 0.1844
        A9, A10, A11 = 0.1056, 0.6134, 0.7210

        T_pr = T / pc_tc['tc']
        rho_pr = 0.27 * p_pr / (z * T_pr)

        # dz/drho_pr (显式微分，省略，使用差分近似)
        h = 1e-6
        rho_pr_p = rho_pr + h
        z_p = z
        f_val = (rho_pr_p - 0.27 * p_pr / (z_p * T_pr))
        # 直接差分
        z1 = z * 1.001
        rho_pr1 = 0.27 * p_pr / (z1 * T_pr)
        dz_drho = (z1 - z) / (rho_pr1 - rho_pr + 1e-12)

        c_pr = (1 / p_pr - 1 / z * dz_drho * rho_pr / p_pr)
        return c_pr / pc_tc['pc']

    @staticmethod
    def real_gas_potential(p, p_base, T, z_func):
        """
        计算真实气体拟压力 (Real Gas Pseudo-Pressure)。

        参数
        ----------
        p : float
            当前压力 (MPa)
        p_base : float
            基准压力 (MPa)
        T : float
            温度 (K)
        z_func : callable
            计算 Z 因子的函数

        返回
        -------
        m_p : float
            拟压力积分值 (MPa²/mPa·s)
        """
        import scipy.integrate as integrate

        def integrand(p_prime):
            z = z_func(p_prime)
            mu = GasProperties().gas_viscosity(p_prime, T)
            return p_prime / (z * mu)

        result, _ = integrate.quad(integrand, p_base, p, limit=200)
        return result


# =====================================================================
# 3. 递减模型定义
# =====================================================================
class DeclineModels:
    """
    页岩气产能递减分析模型集合。

    包含: Arps (指数/双曲/调和), Duong, SEPD 模型。
    每个模型提供:
      - func: 原始函数 (time -> rate)
      - cum_func: 累计产量函数 (time -> Gp)
      - fit: 从历史数据拟合参数
      - predict: 预测未来产量
    """

    # ============== 模型函数 (供 curve_fit 使用) ==============

    @staticmethod
    def _arps_exp(t, qi, Di):
        """Arps 指数递减 (b=0)"""
        return qi * np.exp(-Di * t)

    @staticmethod
    def _arps_hyperbolic(t, qi, Di, b):
        """Arps 双曲递减 (0 < b < 1)"""
        # 避免除零
        b = np.clip(b, 0.001, 0.999)
        return qi / (1 + b * Di * t) ** (1.0 / b)

    @staticmethod
    def _arps_harmonic(t, qi, Di):
        """Arps 调和递减 (b=1)"""
        return qi / (1.0 + Di * t)

    @staticmethod
    def _duong(t, qi, m, a):
        """Duong (2011) 页岩气递减模型"""
        m = np.clip(m, 0.001, 4.999)
        a = np.clip(a, 0.001, 10.0)
        # q(t) = qi * t^(-m) * exp[a/(1-m) * (t^(1-m) - 1)]
        if m == 1:
            return qi * t ** (-1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = qi * t ** (-m) * np.exp(a / (1.0 - m) * (t ** (1.0 - m) - 1.0))
        result = _nan_to_num(result, nan=0.0, posinf=0.0)
        return result

    @staticmethod
    def _sepd(t, qi, tau, n):
        """Stretched Exponential Production Decline (SEPD)"""
        n = np.clip(n, 0.001, 1.0)
        tau = np.clip(tau, 1.0, None)
        return qi * np.exp(-(t / tau) ** n)

    # ============== 累计产量函数 ==============

    @staticmethod
    def cum_arps_exp(t, qi, Di):
        """Arps 指数递减累计产量"""
        return qi / Di * (1.0 - np.exp(-Di * t))

    @staticmethod
    def cum_arps_hyperbolic(t, qi, Di, b):
        """Arps 双曲递减累计产量"""
        b = np.clip(b, 0.001, 0.999)
        if b == 0:
            return DeclineModels.cum_arps_exp(t, qi, Di)
        qi_t = qi / (1 + b * Di * t) ** (1.0 / b)
        return (qi ** b) / ((1 - b) * Di) * (qi ** (1 - b) - qi_t ** (1 - b))

    @staticmethod
    def cum_arps_harmonic(t, qi, Di):
        """Arps 调和递减累计产量"""
        return qi / Di * np.log(1.0 + Di * t)

    @staticmethod
    def cum_duong(t, qi, m, a):
        """Duong 累计产量"""
        m = np.clip(m, 0.001, 4.999)
        a = np.clip(a, 0.001, 10.0)
        if m < 1:
            # 简化 Gp = qi/a * exp(a/(1-m) * (t^(1-m) - 1))
            with np.errstate(over='ignore'):
                exp_term = a / (1.0 - m) * (t ** (1.0 - m) - 1.0)
                # 防溢出
                exp_term = np.clip(exp_term, -700, 700)
                result = qi / a * np.exp(exp_term)
            result = _nan_to_num(result, nan=0.0, posinf=1e10)
            return result
        else:
            # m >= 1 时数值积分
            rates = DeclineModels._duong(np.arange(1, t + 1), qi, m, a)
            return np.cumsum(rates)

    @staticmethod
    def cum_sepd(t, qi, tau, n):
        """SEPD 累计产量（使用不完全 Gamma 函数）"""
        n = np.clip(n, 0.001, 1.0)
        tau = np.clip(tau, 1.0, None)
        # Gp = (qi * tau / n) * gamma_inc(1/n, (t/tau)^n)
        x = (t / tau) ** n
        a = 1.0 / n
        # 下不完全 Gamma 函数: gamma_inc(a, x) = integral_0^x y^(a-1) e^(-y) dy
        # scipy: gammainc(a, x) = gamma_inc(a, x) / Gamma(a)
        gamma_term = gammainc(a, x) * gamma_func(a)
        return qi * tau / n * gamma_term

    # ============== 拟合 ==============

    @staticmethod
    def fit_arps(time, rate, model_type='hyperbolic'):
        """
        拟合 Arps 递减模型。

        参数
        ----------
        time : np.ndarray
            生产时间 (天)
        rate : np.ndarray
            产气量
        model_type : str
            'exponential', 'hyperbolic', 或 'harmonic'

        返回
        -------
        dict : 拟合结果 {params, pcov, label, func, cum_func, rmse, aic}
        """
        t, q = time.astype(float), rate.astype(float)

        # 初始值猜测
        qi_guess = q[0]
        # 用最后一段估计递减率
        n_late = max(3, len(q) // 4)
        if len(q) > n_late:
            Di_guess = -(np.log(q[-1] / qi_guess) / t[-1]) if q[-1] > 0 else 0.01
        else:
            Di_guess = 0.01
        Di_guess = max(Di_guess, 1e-6)

        try:
            if model_type == 'exponential':
                popt, pcov = curve_fit(
                    DeclineModels._arps_exp, t, q,
                    p0=[qi_guess, Di_guess],
                    bounds=([0, 0], [np.inf, 0.5]),
                    maxfev=5000
                )
                params = {'qi': popt[0], 'Di': popt[1], 'b': 0}
                label = 'Arps 指数递减 (b=0)'
                func = lambda tt: DeclineModels._arps_exp(tt, *popt)
                cum_func = lambda tt: DeclineModels.cum_arps_exp(tt, *popt)

            elif model_type == 'harmonic':
                popt, pcov = curve_fit(
                    DeclineModels._arps_harmonic, t, q,
                    p0=[qi_guess, Di_guess],
                    bounds=([0, 0], [np.inf, 0.5]),
                    maxfev=5000
                )
                params = {'qi': popt[0], 'Di': popt[1], 'b': 1}
                label = 'Arps 调和递减 (b=1)'
                func = lambda tt: DeclineModels._arps_harmonic(tt, *popt)
                cum_func = lambda tt: DeclineModels.cum_arps_harmonic(tt, *popt)

            else:  # hyperbolic
                b_guess = 0.5
                popt, pcov = curve_fit(
                    DeclineModels._arps_hyperbolic, t, q,
                    p0=[qi_guess, Di_guess, b_guess],
                    bounds=([0, 0, 0.01], [np.inf, 0.5, 2.0]),
                    maxfev=5000
                )
                # 约束 b 到 [0.01, 0.99] 以避免无物理意义的 EUR 发散
                if popt[2] > 0.99:
                    # b > 0.99 时重新用调和递减拟合
                    try:
                        popt_h, _ = curve_fit(
                            DeclineModels._arps_harmonic, t, q,
                            p0=[qi_guess, Di_guess],
                            bounds=([0, 0], [np.inf, 0.5]),
                            maxfev=5000
                        )
                        popt = [popt_h[0], popt_h[1], 1.0]
                    except Exception:
                        pass
                params = {'qi': popt[0], 'Di': popt[1], 'b': popt[2]}
                label = f'Arps 双曲递减 (b={popt[2]:.3f})'
                func = lambda tt: DeclineModels._arps_hyperbolic(tt, *popt)
                cum_func = lambda tt: DeclineModels.cum_arps_hyperbolic(tt, *popt)

            # 评估拟合质量
            pred = func(t)
            rmse = np.sqrt(np.mean((q - pred) ** 2))
            # AIC: AIC = n * ln(RSS/n) + 2k
            rss = np.sum((q - pred) ** 2)
            n_obs = len(q)
            n_params = len(popt)
            aic = n_obs * np.log(rss / n_obs + 1e-10) + 2 * n_params

            return {'params': params, 'pcov': pcov, 'label': label,
                    'func': func, 'cum_func': cum_func,
                    'rmse': rmse, 'aic': aic, 'type': model_type}

        except Exception as e:
            return {'params': None, 'label': f'Arps {model_type} (拟合失败)',
                    'func': None, 'cum_func': None,
                    'rmse': np.inf, 'aic': np.inf, 'type': model_type}

    @staticmethod
    def fit_duong(time, rate):
        """
        拟合 Duong (2011) 模型 —— 全参数拟合 + 诊断初始化。

        参数
        ----------
        time : np.ndarray
        rate : np.ndarray

        返回
        -------
        dict : 拟合结果
        """
        t, q = time.astype(float), rate.astype(float)
        # 过滤 t=0
        valid = t > 0
        t, q = t[valid], q[valid]
        if len(t) < 3:
            return {'params': None, 'label': 'Duong 模型 (数据不足)',
                    'func': None, 'cum_func': None, 'rmse': np.inf, 'aic': np.inf}

        # ---- 第一阶段: 诊断图获取初始猜测 ----
        cum_q = np.cumsum(q)
        ratio = q / (cum_q + 1e-10)
        with np.errstate(all='ignore'):
            log_t = np.log(t)
            log_ratio = np.log(ratio + 1e-10)
        valid2 = np.isfinite(log_t) & np.isfinite(log_ratio)
        if valid2.sum() < 3:
            return {'params': None, 'label': 'Duong 模型 (拟合失败)',
                    'func': None, 'cum_func': None, 'rmse': np.inf, 'aic': np.inf}

        # q/Gp = a * t^(-m)  →  ln(q/Gp) = ln(a) - m * ln(t)
        A = np.vstack([np.ones(valid2.sum()), -log_t[valid2]]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, log_ratio[valid2], rcond=None)
        ln_a_diag = coeffs[0]
        m_diag = np.clip(coeffs[1], 0.01, 4.99)
        a_diag = np.exp(ln_a_diag)

        # 对于常压页岩，a 通常很小 (0.01~0.5)，限制防止中间量爆炸
        a_diag = np.clip(a_diag, 0.001, 1.0)

        # ---- 第二阶段: 全三参数拟合 ----
        # 初始值
        qi_guess = q[0]
        p0 = [qi_guess, m_diag, a_diag]

        def _duong_wrapper(t, qi, m, a):
            """包装器，确保稳定性"""
            m = np.clip(m, 0.001, 4.99)
            a = np.clip(a, 0.001, 5.0)
            if m == 1:
                return qi * t ** (-1.0)
            exp_arg = a / (1.0 - m) * (t ** (1.0 - m) - 1.0)
            exp_arg = np.clip(exp_arg, -50, 50)  # 限制指数防止溢出
            return qi * t ** (-m) * np.exp(exp_arg)

        try:
            # 有界拟合
            popt, pcov = curve_fit(
                _duong_wrapper, t, q,
                p0=p0,
                bounds=([0, 0.001, 0.001], [np.inf, 4.99, 5.0]),
                maxfev=10000
            )
            qi_val, m_val, a_val = popt
            params = {'qi': qi_val, 'm': m_val, 'a': a_val}

            # 检查结果合理性: 若 a 太大或 m 接近 1，标记警告但不失败
            warning = ''
            if m_val > 0.95 and abs(m_val - 1.0) < 0.05:
                warning += '(m接近1)'

            label = f'Duong 模型 (m={m_val:.3f}, a={a_val:.4f}) {warning}'

            def func(tt):
                tt = np.asarray(tt, float)
                exp_arg = a_val / (1.0 - m_val) * (tt ** (1.0 - m_val) - 1.0)
                exp_arg = np.clip(exp_arg, -50, 50)
                return qi_val * tt ** (-m_val) * np.exp(exp_arg)

            def cum_func(tt):
                tt = np.asarray(tt, float)
                if m_val < 1:
                    exp_term = a_val / (1.0 - m_val) * (tt ** (1.0 - m_val) - 1.0)
                    exp_term = np.clip(exp_term, -50, 50)
                    result = qi_val / a_val * np.exp(exp_term)
                    result = _nan_to_num(result, nan=0.0, posinf=1e10)
                    return result
                else:
                    return np.array([_trapz(
                        _duong_wrapper(np.arange(1, max(2, int(ti) + 1)),
                                       qi_val, m_val, a_val)
                    ) for ti in tt])

            pred = func(t)
            # 检查是否有效
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)) or np.any(pred < 0):
                raise ValueError("预测包含无效值")

            rmse = np.sqrt(np.mean((q - pred) ** 2))
            rss = np.sum((q - pred) ** 2)
            aic = len(t) * np.log(rss / len(t) + 1e-10) + 2 * 3

            return {'params': params, 'pcov': pcov, 'label': label,
                    'func': func, 'cum_func': cum_func,
                    'rmse': rmse, 'aic': aic, 'type': 'duong'}

        except Exception as e:
            # 如果全参数拟合失败，尝试限定 m 范围后再次拟合
            try:
                def _duong_fixed(t, qi, m):
                    a = a_diag  # 固定 a 为诊断值
                    m = np.clip(m, 0.001, 4.99)
                    exp_arg = a / (1.0 - m) * (t ** (1.0 - m) - 1.0)
                    exp_arg = np.clip(exp_arg, -50, 50)
                    return qi * t ** (-m) * np.exp(exp_arg)

                popt, _ = curve_fit(_duong_fixed, t, q, p0=[qi_guess, m_diag],
                                    bounds=([0, 0.001], [np.inf, 4.99]),
                                    maxfev=5000)
                params = {'qi': popt[0], 'm': popt[1], 'a': a_diag}
                label = f'Duong 变体 (m={popt[1]:.3f}, a={a_diag:.4f})'
                qi_v, m_v = popt[0], popt[1]
                a_v = a_diag

                def func(tt):
                    tt = np.asarray(tt, float)
                    exp_arg = a_v / (1.0 - m_v) * (tt ** (1.0 - m_v) - 1.0)
                    exp_arg = np.clip(exp_arg, -50, 50)
                    return qi_v * tt ** (-m_v) * np.exp(exp_arg)

                def cum_func(tt):
                    tt = np.asarray(tt, float)
                    if m_v < 1:
                        exp_term = a_v / (1.0 - m_v) * (tt ** (1.0 - m_v) - 1.0)
                        exp_term = np.clip(exp_term, -50, 50)
                        return qi_v / a_v * np.exp(exp_term)
                    else:
                        return np.array([_trapz(
                            func(np.arange(1, max(2, int(ti) + 1)))
                        ) for ti in tt])

                pred = func(t)
                rmse = np.sqrt(np.mean((q - pred) ** 2))
                rss = np.sum((q - pred) ** 2)
                aic = len(t) * np.log(rss / len(t) + 1e-10) + 2 * 2

                return {'params': params, 'pcov': None, 'label': label,
                        'func': func, 'cum_func': cum_func,
                        'rmse': rmse, 'aic': aic, 'type': 'duong'}

            except Exception:
                return {'params': None, 'label': 'Duong 模型 (拟合失败)',
                        'func': None, 'cum_func': None,
                        'rmse': np.inf, 'aic': np.inf, 'type': 'duong'}

    @staticmethod
    def fit_sepd(time, rate):
        """
        拟合 SEPD (Stretched Exponential) 模型。

        参数
        ----------
        time : np.ndarray
        rate : np.ndarray

        返回
        -------
        dict : 拟合结果
        """
        t, q = time.astype(float), rate.astype(float)
        t_min = t.min()
        if t_min <= 0:
            t = t - t_min + 1  # 平移，使 t>0

        # 初始值: n ~ 0.5, tau ~ t_mid
        qi_guess = q[0]
        tau_guess = t[-1] * 0.5
        n_guess = 0.5

        try:
            popt, pcov = curve_fit(
                DeclineModels._sepd, t, q,
                p0=[qi_guess, tau_guess, n_guess],
                bounds=([0, 10, 0.01], [np.inf, 1e6, 1.0]),
                maxfev=10000
            )
            params = {'qi': popt[0], 'tau': popt[1], 'n': popt[2]}
            label = f'SEPD (n={popt[2]:.3f}, τ={popt[1]:.0f}d)'

            def func(tt):
                tt = np.asarray(tt, float)
                if t_min <= 0:
                    tt = tt - t_min + 1
                return DeclineModels._sepd(tt, *popt)

            def cum_func(tt):
                tt = np.asarray(tt, float)
                if t_min <= 0:
                    tt = tt - t_min + 1
                return DeclineModels.cum_sepd(tt, *popt)

            pred = func(t)
            rmse = np.sqrt(np.mean((q - pred) ** 2))
            rss = np.sum((q - pred) ** 2)
            n_obs = len(t)
            n_params = 3
            aic = n_obs * np.log(rss / n_obs + 1e-10) + 2 * n_params

            return {'params': params, 'pcov': pcov, 'label': label,
                    'func': func, 'cum_func': cum_func,
                    'rmse': rmse, 'aic': aic, 'type': 'sepd'}
        except Exception as e:
            return {'params': None, 'label': 'SEPD 模型 (拟合失败)',
                    'func': None, 'cum_func': None,
                    'rmse': np.inf, 'aic': np.inf, 'type': 'sepd'}


# =====================================================================
# 4. 物质平衡法
# =====================================================================
class MaterialBalance:
    """
    页岩气物质平衡分析 —— 动态法估算原始地质储量 (OGIP)。

    适用于常压区页岩气藏，考虑:
      - 自由气膨胀
      - 吸附气解吸 (Langmuir 等温线)
      - 孔隙压缩性
    """

    def __init__(self, reservoir_params, gas_props, T):
        """
        参数
        ----------
        reservoir_params : dict
            储层参数:
              - phi: 孔隙度 (小数)
              - Sw: 含水饱和度 (小数)
              - h: 有效厚度 (m)
              - A: 含气面积 (km²)
              - rho_b: 岩石密度 (g/cm³), 用于吸附气计算
              - cf: 孔隙压缩系数 (MPa⁻¹), 常压区 ~ (3-6)e-4
        gas_props : GasProperties
            天然气物性计算对象
        T : float
            储层温度 (K)
        """
        self.phi = reservoir_params.get('phi', 0.05)
        self.Sw = reservoir_params.get('Sw', 0.3)
        self.h = reservoir_params.get('h', 30)
        self.A = reservoir_params.get('A', 10)
        self.rho_b = reservoir_params.get('rho_b', 2.5)
        self.cf = reservoir_params.get('cf', 4e-4)
        self.gas = gas_props
        self.T = T

        # Langmuir 参数 (常压区页岩典型值)
        self.VL = reservoir_params.get('VL', 2.0)  # Langmuir 体积 (m³/ton)
        self.pL = reservoir_params.get('pL', 5.0)  # Langmuir 压力 (MPa)

    def ogip_free_gas(self, p):
        """
        计算自由气地质储量 (万方)。

        V = A * h * phi * (1-Sw) / Bgi
        """
        Bg = self.gas.gas_formation_volume_factor(p, self.T)
        # A (km²) -> m², h (m)
        V_bulk = self.A * 1e6 * self.h  # m³
        V_pore = V_bulk * self.phi * (1 - self.Sw)  # m³
        V_sc = V_pore / Bg  # 标准状态下体积 m³
        return V_sc / 1e4  # 转换为万方

    def ogip_adsorbed_gas(self, p):
        """
        计算吸附气地质储量 (万方)。

        使用 Langmuir 等温线: V = VL * p / (pL + p)
        """
        # 岩石质量
        V_bulk = self.A * 1e6 * self.h  # m³
        mass_rock = V_bulk * self.rho_b * 1e3  # kg = g/cm³ * m³ = ton (近似)
        # 实际上: 1 g/cm³ = 1 ton/m³
        mass_rock_ton = V_bulk * self.rho_b  # tonnes
        V_ads = self.VL * p / (self.pL + p)  # m³/ton
        return V_ads * mass_rock_ton / 1e4  # 万方

    def ogip_total(self, p):
        """计算总地质储量 (万方)"""
        return self.ogip_free_gas(p) + self.ogip_adsorbed_gas(p)

    def pz_plot(self, pressures, cum_prod):
        """
        计算 p/Z vs Gp 关系，用于动态法求 OGIP。

        参数
        ----------
        pressures : array
            各时间点的储层平均压力 (MPa)
        cum_prod : array
            累计产气量 (万方)

        返回
        -------
        dict : {'p_z': p/Z 数组, 'ogip_estimate': OGIP 估算值}
        """
        p = np.asarray(pressures, float)
        z = self.gas.z_factor(p, self.T)
        p_z = p / z

        # p/Z vs Gp 线性回归 -> 外推至 p/Z=0 得 OGIP
        gp = np.asarray(cum_prod, float)
        A = np.vstack([gp, np.ones_like(gp)]).T
        slope, intercept = np.linalg.lstsq(A, p_z, rcond=None)[0]

        ogip_estimate = -intercept / slope if slope != 0 else np.nan
        return {
            'p_z': p_z,
            'slope': slope,
            'intercept': intercept,
            'ogip_estimate': ogip_estimate,
            'func': lambda gp: slope * gp + intercept
        }

    def forecast_by_material_balance(self, p_init, cum_prod_total, steps=100):
        """
        基于物质平衡预测产量递减。

        参数
        ----------
        p_init : float
            原始地层压力 (MPa)
        cum_prod_total : float
            最终累计产气量目标 (万方)
        steps : int
            计算步数

        返回
        -------
        pd.DataFrame
        """
        gp_range = np.linspace(0, cum_prod_total, steps)
        p_init_z_init = p_init / self.gas.z_factor(p_init, self.T)

        pres = []
        for gp in gp_range:
            p_z_target = p_init_z_init * (1 - gp / cum_prod_total)
            # 求 p
            def f(p):
                return p / self.gas.z_factor(p, self.T) - p_z_target
            try:
                from scipy.optimize import fsolve
                p_sol = fsolve(f, p_init * 0.5)[0]
                pres.append(p_sol)
            except Exception:
                pres.append(np.nan)

        return pd.DataFrame({
            'cum_prod': gp_range,
            'pressure': pres,
            'p_per_z': p_init_z_init * (1 - gp_range / cum_prod_total)
        })


# =====================================================================
# 5. 综合预测引擎
# =====================================================================
class ShaleGasPredictor:
    """
    常压区页岩气产能预测综合引擎。

    整合:
      - 数据加载与预处理 (ProductionData)
      - 天然气物性计算 (GasProperties)
      - 多种递减模型拟合 (Arps / Duong / SEPD)
      - 物质平衡分析 (MaterialBalance)
      - 产能预测与 EUR 估算
      - 可视化 (Visualizer)
    """

    def __init__(self, data_file=None, df=None):
        """
        初始化预测器。

        参数
        ----------
        data_file : str or Path
            生产数据文件路径 (CSV 或 Excel)
        df : pd.DataFrame
            或直接传入 DataFrame
        """
        self.data = ProductionData(filepath=data_file, dataframe=df)
        self.results = {}  # 存储各模型拟合结果
        self.predictions = {}  # 存储预测结果
        self.reservoir_params = {}  # 储层参数
        self.gas_props = None
        self.T = None

    def set_reservoir_params(self, **kwargs):
        """
        设置储层参数。

        常用参数:
          phi      : 孔隙度 (小数), 常压区页岩 ~0.03-0.08
          Sw       : 含水饱和度 (小数), ~0.2-0.4
          h        : 有效厚度 (m), ~10-60
          A        : 含气面积 (km²)
          rho_b    : 岩石密度 (g/cm³), ~2.5-2.6
          cf       : 孔隙压缩系数 (MPa⁻¹), ~(3-6)e-4
          VL       : Langmuir 体积 (m³/ton), ~1-3
          pL       : Langmuir 压力 (MPa), ~3-8
          gamma_g  : 天然气相对密度, ~0.55-0.7
          T        : 储层温度 (K), ~340-370
          p_i      : 原始地层压力 (MPa), 常压区 ~20-35
        """
        self.reservoir_params = kwargs
        # 自动创建物性计算对象
        if 'gamma_g' in kwargs:
            co2 = kwargs.get('co2_pct', 0)
            h2s = kwargs.get('h2s_pct', 0)
            n2 = kwargs.get('n2_pct', 0)
            self.gas_props = GasProperties(kwargs['gamma_g'], co2, h2s, n2)
        if 'T' in kwargs:
            self.T = kwargs['T']

    # ---------- 递减模型拟合 ----------

    def fit_arps(self, model_type='hyperbolic'):
        """拟合 Arps 递减模型"""
        t = self.data.get_time()
        q = self.data.get_rate()
        result = DeclineModels.fit_arps(t, q, model_type)
        self.results[f'arps_{model_type}'] = result
        return result

    def fit_duong(self):
        """拟合 Duong 模型"""
        t = self.data.get_time()
        q = self.data.get_rate()
        result = DeclineModels.fit_duong(t, q)
        self.results['duong'] = result
        return result

    def fit_sepd(self):
        """拟合 SEPD 模型"""
        t = self.data.get_time()
        q = self.data.get_rate()
        result = DeclineModels.fit_sepd(t, q)
        self.results['sepd'] = result
        return result

    def fit_all_models(self, arps_types=None):
        """
        拟合所有模型并自动对比。

        参数
        ----------
        arps_types : list
            要拟合的 Arps 子类型, 默认 ['exponential', 'hyperbolic', 'harmonic']

        返回
        -------
        dict : 所有拟合结果
        """
        if arps_types is None:
            arps_types = ['exponential', 'hyperbolic', 'harmonic']
        for at in arps_types:
            self.fit_arps(at)
        self.fit_duong()
        self.fit_sepd()
        return self.results

    # ---------- 智能拟合（自动选择最优模型）----------

    def fit_best_model(self, criteria='aic'):
        """
        自动拟合所有模型并选择最优。

        参数
        ----------
        criteria : str
            'aic' 或 'rmse', 选择最优模型的标准。

        返回
        -------
        best_result : dict
            最优模型的结果
        """
        # 只拟合尚未拟合的模型
        for at in ['exponential', 'hyperbolic', 'harmonic']:
            key = f'arps_{at}'
            if key not in self.results or self.results[key]['func'] is None:
                self.fit_arps(at)
        if 'duong' not in self.results or self.results['duong']['func'] is None:
            self.fit_duong()
        if 'sepd' not in self.results or self.results['sepd']['func'] is None:
            self.fit_sepd()
        valid = {k: v for k, v in self.results.items()
                 if v['func'] is not None}
        if not valid:
            return None
        # 按 AIC 或 RMSE 排序
        key_fn = lambda x: x[1][criteria]
        sorted_results = sorted(valid.items(), key=key_fn)
        best_name, best_result = sorted_results[0]
        best_result['selected'] = best_name
        self.results['best'] = best_result
        return best_result

    # ---------- 产能预测 ----------

    def predict(self, model_key, forecast_days=3650):
        """
        基于指定模型对未来产能进行预测。

        参数
        ----------
        model_key : str
            使用的模型标识符:
              'arps_exponential', 'arps_hyperbolic', 'arps_harmonic',
              'duong', 'sepd', 'best'
        forecast_days : int
            预测天数 (默认 3650 天 ≈ 10 年)

        返回
        -------
        pd.DataFrame : 预测结果表
        """
        if model_key == 'best':
            if 'best' not in self.results:
                self.fit_best_model()
            result = self.results.get('best')
        else:
            result = self.results.get(model_key)

        if result is None or result['func'] is None:
            raise ValueError(f"模型 '{model_key}' 尚未拟合或拟合失败。"
                             f"可用模型: {list(self.results.keys())}")

        # 预测时间范围: 从数据结束到 forecast_days
        t_last = self.data.time_max
        t_future = np.arange(1, forecast_days + 1, dtype=float)
        t_hist = self.data.get_time()

        # 历史拟合值
        q_hist_fit = result['func'](t_hist)
        cum_hist_fit = result['cum_func'](t_hist)

        # 未来预测值
        q_future = result['func'](t_future)
        cum_future = result['cum_func'](t_future)

        # 总 EUR
        eur = float(cum_future[-1])

        df_future = pd.DataFrame({
            'time': t_future,
            'rate_pred': q_future,
            'cum_pred': cum_future,
        })
        # 标记阶段
        df_future['phase'] = '预测'
        df_future.loc[t_future <= t_last, 'phase'] = '历史拟合'

        result['forecast'] = df_future
        result['eur'] = eur

        self.predictions[model_key] = result
        return df_future

    def predict_all(self, forecast_days=3650):
        """对所有已拟合的模型进行预测（跳过 'best' 别名避免重复）"""
        for key in list(self.results.keys()):
            if key == 'best':
                continue
            if self.results[key]['func'] is not None:
                try:
                    self.predict(key, forecast_days)
                except Exception:
                    continue

    # ---------- EUR 估算汇总 ----------

    def eur_summary(self):
        """
        生成各模型 EUR 估算汇总表。

        返回
        -------
        pd.DataFrame
        """
        rows = []
        for key, result in self.results.items():
            if key == 'best':
                continue
            if result.get('eur') is not None:
                rows.append({
                    '模型': result['label'],
                    'EUR (万方)': result['eur'],
                    'EUR (亿方)': result['eur'] / 10000,
                    'RMSE': result.get('rmse', np.nan),
                    'AIC': result.get('aic', np.nan),
                })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.sort_values('AIC', inplace=True)
        return df

    def eur_by_arps_limits(self, terminal_rate=None):
        """
        基于 Arps 递减极限估算 EUR（考虑最小经济极限产量）。

        参数
        ----------
        terminal_rate : float
            经济极限产量 (万方/天)。默认取峰值产量的 5%。

        返回
        -------
        dict : 各模型的 EUR 及经济可采年限
        """
        if terminal_rate is None:
            terminal_rate = self.data.rate_max * 0.05

        results = {}
        for key in ['arps_exponential', 'arps_hyperbolic', 'arps_harmonic']:
            result = self.results.get(key)
            if result is None or result['func'] is None:
                continue
            func = result['func']
            cum_func = result['cum_func']

            # 求达到极限产量的时间
            def f(t):
                return func(t) - terminal_rate
            try:
                from scipy.optimize import brentq
                t_limit = brentq(f, 1, 50000)
                eur_limit = cum_func(t_limit)
                results[key] = {
                    'label': result['label'],
                    'terminal_rate': terminal_rate,
                    't_limit_days': t_limit,
                    't_limit_years': t_limit / 365,
                    'eur_limit': eur_limit,
                    'eur_limit_亿方': eur_limit / 10000,
                }
            except Exception:
                continue
        return results

    # ---------- 物质平衡分析 ----------

    def analyze_material_balance(self, p_i=None):
        """
        物质平衡法分析。

        参数
        ----------
        p_i : float
            原始地层压力 (MPa)。如未提供则从储层参数中取。

        返回
        -------
        dict : 物质平衡分析结果
        """
        if not self.gas_props:
            raise ValueError("请先通过 set_reservoir_params() 设置储层参数。")
        if p_i is None:
            p_i = self.reservoir_params.get('p_i', 30)
        if self.T is None:
            raise ValueError("请设置储层温度 T。")

        mb = MaterialBalance(self.reservoir_params, self.gas_props, self.T)

        # 静态 OGIP
        ogip_free = mb.ogip_free_gas(p_i)
        ogip_ads = mb.ogip_adsorbed_gas(p_i)
        ogip_total = ogip_free + ogip_ads

        # 动态 p/Z 分析
        pressure_data = self.data.get_pressure()
        cum_prod = self.data.get_cumprod()
        pz_result = None
        if pressure_data is not None and len(pressure_data) > 2:
            pz_result = mb.pz_plot(pressure_data, cum_prod)

        return {
            'ogip_free': ogip_free,
            'ogip_ads': ogip_ads,
            'ogip_total': ogip_total,
            'ogip_free_亿方': ogip_free / 10000,
            'ogip_ads_亿方': ogip_ads / 10000,
            'ogip_total_亿方': ogip_total / 10000,
            'pz_analysis': pz_result,
            'mb_model': mb
        }

    # =================================================================
    # 6. 可视化
    # =================================================================
    def plot_production_history(self, figsize=(12, 5)):
        """绘制生产历史曲线"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        t = self.data.get_time()
        q = self.data.get_rate()
        cum = self.data.get_cumprod()

        # 日产量
        axes[0].plot(t, q, 'o-', color='#2c6b9e', markersize=3, linewidth=1.5,
                     label='日产气量')
        axes[0].set_xlabel('生产时间 (天)')
        axes[0].set_ylabel('日产气量 (万方/天)')
        axes[0].set_title('日产气量历史')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 累计产量
        axes[1].plot(t, cum / 10000, 's-', color='#d35400', markersize=3,
                     linewidth=1.5, label='累计产气量')
        axes[1].set_xlabel('生产时间 (天)')
        axes[1].set_ylabel('累计产气量 (亿方)')
        axes[1].set_title('累计产气量历史')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        return fig

    def plot_decline_fit(self, model_keys=None, figsize=(14, 6)):
        """
        绘制递减模型拟合对比图。

        参数
        ----------
        model_keys : list
            要展示的模型列表。默认展示所有已拟合模型。
        """
        t = self.data.get_time()
        q = self.data.get_rate()
        cum = self.data.get_cumprod()

        if model_keys is None:
            model_keys = [k for k, v in self.results.items()
                          if v['func'] is not None]

        colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6',
                  '#1abc9c', '#e67e22']

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 左图: 产量拟合
        axes[0].plot(t, q, 'o', color='#2c3e50', markersize=3, alpha=0.6,
                     label='实际产量')
        for i, key in enumerate(model_keys):
            result = self.results.get(key)
            if result is None or result['func'] is None:
                continue
            t_fine = np.linspace(t.min(), t.max(), 500)
            q_fit = result['func'](t_fine)
            color = colors[i % len(colors)]
            axes[0].plot(t_fine, q_fit, '-', color=color, linewidth=2,
                         label=f"{result['label']} (RMSE={result.get('rmse', np.nan):.3f})")

        axes[0].set_xlabel('生产时间 (天)')
        axes[0].set_ylabel('日产气量 (万方/天)')
        axes[0].set_title('产量递减模型拟合对比')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[0].set_yscale('log')

        # 右图: 累计产量拟合
        axes[1].plot(t, cum / 10000, 'o', color='#2c3e50', markersize=3,
                     alpha=0.6, label='实际累计产量')
        for i, key in enumerate(model_keys):
            result = self.results.get(key)
            if result is None or result['cum_func'] is None:
                continue
            t_fine = np.linspace(t.min(), t.max(), 500)
            cum_fit = result['cum_func'](t_fine)
            color = colors[i % len(colors)]
            axes[1].plot(t_fine, cum_fit / 10000, '--', color=color, linewidth=1.5,
                         label=f"{result['label']}")

        axes[1].set_xlabel('生产时间 (天)')
        axes[1].set_ylabel('累计产气量 (亿方)')
        axes[1].set_title('累计产量拟合对比')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        return fig

    def plot_forecast(self, model_key='best', history_years=5, figsize=(14, 10)):
        """
        绘制产能预测综合图。

        参数
        ----------
        model_key : str
            使用的模型。
        history_years : float
            图中展示的历史年数。
        """
        if model_key == 'best' and 'best' not in self.results:
            self.fit_best_model()

        if model_key not in self.predictions:
            try:
                self.predict(model_key)
            except Exception:
                if model_key != 'best':
                    raise
                # 回退到第一个可用模型
                for k in self.results:
                    if self.results[k]['func'] is not None:
                        self.predict(k)
                        model_key = k
                        break

        result = self.predictions.get(model_key, self.results.get(model_key))
        if result is None or 'forecast' not in result:
            raise ValueError(f"模型 '{model_key}' 无预测数据。")

        df_fc = result['forecast']
        t = self.data.get_time()
        q = self.data.get_rate()
        cum = self.data.get_cumprod()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # (1) 产量历史 + 预测 (线性坐标)
        t_hist = df_fc[df_fc['phase'] == '历史拟合']['time'].values
        q_hist = df_fc[df_fc['phase'] == '历史拟合']['rate_pred'].values
        t_pred = df_fc[df_fc['phase'] == '预测']['time'].values
        q_pred = df_fc[df_fc['phase'] == '预测']['rate_pred'].values

        axes[0, 0].plot(t, q, 'o', color='#2c3e50', markersize=3, alpha=0.5,
                        label='实际产量')
        if len(t_hist) > 0:
            axes[0, 0].plot(t_hist, q_hist, '-', color='#3498db', linewidth=2,
                            label='历史拟合')
        axes[0, 0].plot(t_pred, q_pred, '--', color='#e74c3c', linewidth=2,
                        label='产量预测')
        axes[0, 0].axvline(x=self.data.time_max, color='gray', linestyle=':',
                           alpha=0.7, label='预测起点')
        axes[0, 0].set_xlabel('生产时间 (天)')
        axes[0, 0].set_ylabel('日产气量 (万方/天)')
        axes[0, 0].set_title(f"产量预测 ({result['label']})")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=8)

        # (2) 产量历史 + 预测 (对数坐标)
        axes[0, 1].plot(t, q, 'o', color='#2c3e50', markersize=3, alpha=0.5,
                        label='实际产量')
        if len(t_hist) > 0:
            axes[0, 1].plot(t_hist, q_hist, '-', color='#3498db', linewidth=2)
        axes[0, 1].plot(t_pred, q_pred, '--', color='#e74c3c', linewidth=2,
                        label=f"EUR={result.get('eur', 0)/10000:.2f} 亿方")
        axes[0, 1].axvline(x=self.data.time_max, color='gray', linestyle=':',
                           alpha=0.7)
        axes[0, 1].set_xlabel('生产时间 (天)')
        axes[0, 1].set_ylabel('日产气量 (万方/天)')
        axes[0, 1].set_title('产量预测 (对数坐标)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=9)

        # (3) 累计产量
        cum_hist = df_fc[df_fc['phase'] == '历史拟合']['cum_pred'].values
        cum_pred = df_fc[df_fc['phase'] == '预测']['cum_pred'].values
        axes[1, 0].plot(t, cum / 10000, 'o', color='#2c3e50', markersize=3,
                        alpha=0.5, label='实际累计产量')
        if len(cum_hist) > 0:
            axes[1, 0].plot(t_hist, cum_hist / 10000, '-', color='#3498db',
                            linewidth=2, label='历史拟合累计')
        axes[1, 0].plot(t_pred, cum_pred / 10000, '--', color='#e74c3c',
                        linewidth=2, label=f"预测累计 (EUR={result.get('eur', 0)/10000:.2f} 亿方)")
        axes[1, 0].set_xlabel('生产时间 (天)')
        axes[1, 0].set_ylabel('累计产气量 (亿方)')
        axes[1, 0].set_title('累计产量预测')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=9)

        # (4) 递减率 / 年产量
        pred_mask = df_fc['phase'] == '预测'
        yearly = df_fc[pred_mask].copy()
        yearly['year'] = np.ceil(yearly['time'] / 365)
        yearly_agg = yearly.groupby('year').agg(
            年产气量=('rate_pred', 'sum'),
            年末累计=('cum_pred', 'last')
        ).reset_index()
        yearly_agg['年产气量_亿方'] = yearly_agg['年产气量'] / 10000
        axes[1, 1].bar(yearly_agg['year'], yearly_agg['年产气量_亿方'],
                       color='#2ecc71', alpha=0.7, width=0.7)
        axes[1, 1].set_xlabel('生产年份')
        axes[1, 1].set_ylabel('年产气量 (亿方)')
        axes[1, 1].set_title('年度产量预测')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_pz_analysis(self, figsize=(8, 6)):
        """绘制 p/Z 分析图"""
        pressure_data = self.data.get_pressure()
        cum_prod = self.data.get_cumprod()
        if pressure_data is None:
            raise ValueError("数据中无压力信息，无法进行 p/Z 分析。")

        mb = MaterialBalance(self.reservoir_params, self.gas_props, self.T)
        result = mb.pz_plot(pressure_data, cum_prod)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(cum_prod / 10000, result['p_z'], 'o', color='#2c3e50',
                markersize=5, label='p/Z 数据点')
        gp_fit = np.linspace(0, cum_prod.max() * 1.5, 100)
        ax.plot(gp_fit / 10000, result['func'](gp_fit), '--', color='#e74c3c',
                linewidth=2, label='线性回归')
        # OGIP 点
        ogip = result['ogip_estimate']
        if not np.isnan(ogip):
            ax.axvline(x=ogip / 10000, color='#27ae60', linestyle=':',
                       linewidth=2, label=f"OGIP ≈ {ogip/10000:.2f} 亿方")
            ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_xlabel('累计产气量 (亿方)')
        ax.set_ylabel('p/Z (MPa)')
        ax.set_title('p/Z 物质平衡分析')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_duong_diagnostic(self, figsize=(8, 6)):
        """Duong 模型诊断图: ln(q/Gp) vs ln(t)"""
        t = self.data.get_time()
        q = self.data.get_rate()
        cum = self.data.get_cumprod()
        valid = (t > 0) & (cum > 0)
        t, q, cum = t[valid], q[valid], cum[valid]

        ratio = q / cum
        log_t = np.log(t)
        log_ratio = np.log(ratio + 1e-10)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(log_t, log_ratio, 'o', color='#2c3e50', markersize=4,
                label='q/Gp 数据')
        # 线性拟合
        A = np.vstack([np.ones_like(log_t), -log_t]).T
        slope, intercept = np.linalg.lstsq(A, log_ratio, rcond=None)[0]
        # 注意 A 的第二列是 -ln(t)，所以回归: ln(q/Gp) = ln(a) - m*ln(t)
        # y = c0 + c1 * (-ln(t))  -> c0 = ln(a), c1 = m
        ln_a = A[:, 0].dot(log_ratio) / A[:, 0].sum()  # simplified
        m_est = intercept
        a_est = np.exp(slope)

        t_fit = np.logspace(np.log10(t.min()), np.log10(t.max()), 100)
        ax.plot(np.log(t_fit), np.log(a_est) - m_est * np.log(t_fit),
                '-', color='#e74c3c', linewidth=2,
                label=f"拟合: ln(q/Gp) = ln({a_est:.4f}) - {m_est:.3f}·ln(t)")

        ax.set_xlabel('ln(t)')
        ax.set_ylabel('ln(q / Gp)')
        ax.set_title('Duong 模型诊断图')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_model_comparison(self, figsize=(14, 6)):
        """多模型 EUR 对比柱状图"""
        summary = self.eur_summary()
        if summary.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, '尚无预测结果', ha='center', va='center', fontsize=14)
            return fig

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # EUR 对比
        models = summary['模型'].values
        eur_values = summary['EUR (万方)'].values
        rmse_values = summary['RMSE'].values
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

        axes[0].barh(range(len(models)), eur_values / 10000, color=colors, alpha=0.8)
        axes[0].set_yticks(range(len(models)))
        axes[0].set_yticklabels([m[:25] + '...' if len(m) > 25 else m for m in models],
                                fontsize=9)
        axes[0].set_xlabel('EUR (亿方)')
        axes[0].set_title('各模型 EUR 估算对比')
        axes[0].grid(True, alpha=0.3, axis='x')
        # 标注数值
        for i, v in enumerate(eur_values / 10000):
            axes[0].text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=9)

        # RMSE 对比
        axes[1].barh(range(len(models)), rmse_values, color=colors, alpha=0.8)
        axes[1].set_yticks(range(len(models)))
        axes[1].set_yticklabels([m[:25] + '...' if len(m) > 25 else m for m in models],
                                fontsize=9)
        axes[1].set_xlabel('RMSE (万方/天)')
        axes[1].set_title('拟合精度 (RMSE) 对比 (越小越好)')
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    # ---------- 报告生成 ----------

    def generate_report(self, output_path=None, model_key='best'):
        """
        生成产能预测报告 (PDF/PNG 导出 + 文本摘要)。

        参数
        ----------
        output_path : str or Path
            输出路径 (支持 .png, .pdf 扩展名)。None 则仅显示。
        model_key : str
            主要展示的模型。
        """
        # 确保已拟合和预测
        if not self.results:
            self.fit_all_models()
        if model_key not in self.predictions:
            try:
                self.predict(model_key)
            except Exception:
                model_key = 'arps_hyperbolic'
                if model_key not in self.results:
                    self.fit_arps('hyperbolic')
                self.predict(model_key)

        # 生成综合图
        fig = self.plot_production_history()
        fig2 = self.plot_decline_fit()
        fig3 = self.plot_forecast(model_key)
        fig4 = self.plot_model_comparison()

        if output_path:
            output_path = Path(output_path)
            # 合并大图
            from matplotlib.backends.backend_pdf import PdfPages
            if output_path.suffix.lower() == '.pdf':
                with PdfPages(output_path) as pdf:
                    pdf.savefig(fig)
                    pdf.savefig(fig2)
                    pdf.savefig(fig3)
                    pdf.savefig(fig4)
            else:
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"报告图片已保存至: {output_path}")

        plt.show()
        return fig, fig2, fig3, fig4

    def print_summary(self):
        """在控制台打印预测结果摘要"""
        print("=" * 70)
        print("      常压区页岩气产能预测结果摘要")
        print("=" * 70)
        print(f"\n📊 生产数据概况:")
        print(f"   • 数据点数: {self.data.n_points}")
        print(f"   • 生产时间: {self.data.time_min:.0f} ~ {self.data.time_max:.0f} 天"
              f" ({self.data.time_max / 365:.1f} 年)")
        print(f"   • 最高日产: {self.data.rate_max:.2f} 万方/天")
        print(f"   • 当前日产: {self.data.rate_min:.2f} 万方/天")
        print(f"   • 累计产气: {self.data.cum_total:.2f} 万方"
              f" ({self.data.cum_total / 10000:.4f} 亿方)")

        if self.results:
            print(f"\n📈 递减模型拟合结果:")
            for key, result in self.results.items():
                if key == 'best':
                    continue
                if result['params'] is not None:
                    params_str = ', '.join(
                        [f"{k}={v:.4f}" for k, v in result['params'].items()]
                    )
                    print(f"   • {result['label']}")
                    print(f"     参数: {params_str}")
                    print(f"     RMSE: {result.get('rmse', np.nan):.4f}, "
                          f"AIC: {result.get('aic', np.nan):.2f}")

        if self.predictions:
            print(f"\n🔮 产能预测 (EUR):")
            for key, result in self.predictions.items():
                if result.get('eur') is not None:
                    print(f"   • {result['label']}: "
                          f"{result['eur']:.2f} 万方 ({result['eur']/10000:.2f} 亿方)")

        if 'ogip_total_亿方' in self.reservoir_params:
            print(f"\n⛰️  物质平衡分析:")
            print(f"   • OGIP (自由气): {self.reservoir_params.get('ogip_free_亿方', 'N/A')} 亿方")
            print(f"   • OGIP (吸附气): {self.reservoir_params.get('ogip_ads_亿方', 'N/A')} 亿方")

        print(f"\n{'=' * 70}")
