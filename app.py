# -*- coding: utf-8 -*-
"""
常压区页岩气产能预测系统 — Streamlit 应用
============================================
单文件网页应用，基于 Streamlit 框架。

启动方式:
    streamlit run app.py

部署方式:
    1. 上传到 GitHub 仓库
    2. 在 Streamlit Cloud (share.streamlit.io) 选择该仓库
    3. 入口文件设为 app.py

依赖:
    pip install streamlit numpy pandas scipy matplotlib openpyxl
"""

import os, sys, io, datetime, urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# =====================================================================
# 中文字体设置（自动下载 + 注册，兼容 Streamlit Cloud Linux 环境）
# =====================================================================
def _setup_chinese_font():
    """查找或下载中文字体，返回可用字体名称"""
    import matplotlib.font_manager as fm

    # 1) 检查系统中是否已有中文字体
    known_zh = ['SimHei', 'Microsoft YaHei', 'Microsoft JhengHei',
                'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
                'Noto Sans SC', 'Noto Sans CJK SC', 'Noto Serif CJK SC',
                'Source Han Sans SC', 'Source Han Serif SC',
                'AR PL UMing CN', 'AR PL UKai CN',
                'Droid Sans Fallback', 'FangSong', 'KaiTi', 'SimSun']
    found = set(f.name for f in fm.fontManager.ttflist)
    for name in known_zh:
        if name in found:
            return name

    # 2) 在 Linux (Streamlit Cloud) 上下载 Noto Sans SC
    font_dir = Path(__file__).parent / 'fonts'
    font_dir.mkdir(exist_ok=True)
    font_path = font_dir / 'NotoSansSC-Regular.otf'

    if not font_path.exists():
        urls = [
            'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf',
            'https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf',
        ]
        for url in urls:
            try:
                urllib.request.urlretrieve(url, font_path)
                if font_path.stat().st_size > 100000:
                    break
            except Exception:
                continue

    if font_path.exists() and font_path.stat().st_size > 100000:
        try:
            fm.fontManager.addfont(str(font_path))
            font_name = fm.FontProperties(fname=str(font_path)).get_name()
            return font_name
        except Exception:
            pass

    return 'DejaVu Sans'  # fallback

# 添加当前路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shale_gas_predictor import ShaleGasPredictor, ProductionData, DeclineModels, GasProperties
from shale_gas_predictor import MaterialBalance

# 下载并应用中文字体（必须在 import shale_gas_predictor 之后，确保覆盖其模块级设置）
_ZH_FONT = _setup_chinese_font()
plt.rcParams['font.sans-serif'] = [_ZH_FONT, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 页面配置
# =====================================================================
st.set_page_config(
    page_title="常压区页岩气产能预测系统",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 工具函数
# =====================================================================

@st.cache_data
def create_example_data():
    """生成示例数据（带缓存）"""
    np.random.seed(42)
    qi, Di, b = 8.0, 0.008, 0.65
    n_days = 1800
    p_i = 28.0
    time = np.arange(1, n_days + 1, dtype=float)
    q_true = qi / (1 + b * Di * time) ** (1.0 / b)
    noise = np.random.normal(0, 0.15 * q_true ** 0.7, len(time))
    noise = np.clip(noise, -q_true * 0.3, q_true * 0.3)
    rate = np.maximum(q_true + noise, 0.05)
    cum_prod = np.cumsum(rate)
    pressure = p_i * (1 - 0.015 * np.log(time + 1))
    pressure += np.random.normal(0, 0.3, len(time))
    pressure = np.maximum(pressure, 5.0)
    monthly = np.arange(0, n_days, 30)
    return pd.DataFrame({
        'time': time[monthly],
        'rate': rate[monthly],
        'cum_prod': cum_prod[monthly],
        'pressure': pressure[monthly],
    })


@st.cache_resource(hash_funcs={pd.DataFrame: lambda df: hash(df.to_json() + str(df.shape))})
def run_prediction(data_df, params_dict, forecast_years):
    """运行完整分析流程（带缓存，cache_resource 避免 pickle 序列化问题）"""
    predictor = ShaleGasPredictor(df=data_df)
    predictor.set_reservoir_params(**params_dict)
    predictor.fit_all_models()
    predictor.fit_best_model()
    forecast_days = forecast_years * 365
    predictor.predict_all(forecast_days=forecast_days)
    return predictor


# =====================================================================
# 侧边栏 — 数据加载 + 参数设置
# =====================================================================

st.sidebar.markdown("## ⛰️ 页岩气产能预测")
st.sidebar.markdown("---")

# -------- 数据来源 --------
st.sidebar.markdown("### 📂 数据来源")
data_source = st.sidebar.radio(
    "选择数据",
    ["示例数据（内置）", "上传文件（CSV/Excel）"],
    index=0,
    help="选择内置的常压页岩气示例数据，或上传自己的生产数据",
)

data_df = None
uploaded_file = None

if data_source == "上传文件（CSV/Excel）":
    uploaded_file = st.sidebar.file_uploader(
        "上传生产数据文件",
        type=["csv", "xlsx", "xls"],
        help="必需列: time (天), rate (万方/天)。可选: cum_prod, pressure (MPa)",
    )
    if uploaded_file is not None:
        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == '.csv':
                data_df = pd.read_csv(io.StringIO(uploaded_file.read().decode('utf-8-sig')))
            else:
                data_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"文件读取失败: {e}")

if data_source == "示例数据（内置）" or (uploaded_file is None and data_source == "上传文件（CSV/Excel）"):
    data_df = create_example_data()
    if data_source == "示例数据（内置）":
        st.sidebar.info("✅ 已加载示例数据（常压页岩气井，60 个月）")
    else:
        st.sidebar.info("📤 请上传数据文件")

st.sidebar.markdown("---")

# -------- 预测年限 --------
st.sidebar.markdown("### 🔮 预测参数")
forecast_years = st.sidebar.slider(
    "预测年限", min_value=1, max_value=50, value=20, step=1,
    help="基于历史数据向前预测的年数",
    disabled=(data_df is None),
)

# -------- 储层参数 --------
st.sidebar.markdown("### ⚙️ 储层参数（可选）")
st.sidebar.caption("仅物质平衡分析需要，递减预测可不填")

with st.sidebar.expander("展开/折叠参数", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        phi = st.number_input("孔隙度 φ", 0.01, 0.15, 0.055, 0.001, help="常压区 0.03~0.08")
        Sw = st.number_input("含水饱和度 Sw", 0.1, 0.6, 0.30, 0.01, help="0.20~0.45")
        h = st.number_input("有效厚度 h (m)", 5.0, 100.0, 35.0, 1.0, help="10~60m")
        A = st.number_input("含气面积 A (km²)", 0.5, 50.0, 8.0, 0.5, help="井控面积")
        rho_b = st.number_input("岩石密度 (g/cm³)", 2.0, 3.0, 2.55, 0.01)
        cf = st.number_input("孔隙压缩 cf (MPa⁻¹)", 0.0001, 0.001, 0.00045, 0.00005,
                              format="%.5f", help="(3~6)e-4")
    with col2:
        VL = st.number_input("Langmuir体积 (m³/t)", 0.5, 5.0, 2.5, 0.1, help="1~3.5")
        pL = st.number_input("Langmuir压力 (MPa)", 1.0, 15.0, 5.0, 0.1, help="3~8")
        gamma_g = st.number_input("气体相对密度", 0.5, 0.8, 0.62, 0.01, help="0.55~0.72")
        T = st.number_input("储层温度 (K)", 320, 400, 355, 1, help="340~380K (≈80°C)")
        p_i = st.number_input("原始地层压力 (MPa)", 10.0, 60.0, 28.0, 0.5, help="常压 20~35")

params_dict = {
    'phi': phi, 'Sw': Sw, 'h': h, 'A': A,
    'rho_b': rho_b, 'cf': cf, 'VL': VL, 'pL': pL,
    'gamma_g': gamma_g, 'T': T, 'p_i': p_i,
}

st.sidebar.markdown("---")

# -------- 运行按钮 --------
run_disabled = (data_df is None)
run_btn = st.sidebar.button(
    "🚀 开始分析",
    type="primary",
    use_container_width=True,
    disabled=run_disabled,
    help="点击运行递减模型拟合 + 产能预测",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 使用说明:\n"
    "1. 选择数据来源（示例/上传）\n"
    "2. 调整预测年限\n"
    "3. 储层参数可选填\n"
    "4. 点击「开始分析」\n"
    "5. 浏览各选项卡结果"
)


# =====================================================================
# 主页面
# =====================================================================

st.title("⛰️ 常压区页岩气产能预测系统")
st.markdown(
    '<p style="color: #666; font-size: 14px;">'
    '基于递减分析 (Arps / Duong / SEPD) + 物质平衡法的页岩气井产能预测工具'
    '</p>',
    unsafe_allow_html=True,
)

# 如果还没运行，显示欢迎信息
if not run_btn:
    st.info("👈 请在左侧边栏选择数据来源并点击「开始分析」")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 递减模型", "Arps / Duong / SEPD", "5种模型自动对比")
    with col2:
        st.metric("🔮 产能预测", "1~50年", "EUR + 年产气量")
    with col3:
        st.metric("⛰️ 物质平衡", "OGIP估算", "自由气 + 吸附气")

    st.markdown("""
    #### 支持的模型

    | 模型 | 说明 | 参数 |
    |------|------|------|
    | **Arps 指数** | 经典递减，b=0 | qi, Di |
    | **Arps 双曲** | 页岩最常用，0<b<1 | qi, Di, b |
    | **Arps 调和** | b=1，上限参考 | qi, Di |
    | **Duong** | 专为页岩设计，线性流 | qi, m, a |
    | **SEPD** | 扩展指数，多尺度流动 | qi, τ, n |

    #### 数据格式要求

    - **必需列**: `time` (生产天数), `rate` (日产气量, 万方/天)
    - **可选列**: `cum_prod` (累计产量), `pressure` (压力, MPa)
    - 支持 CSV 和 Excel 格式
    """)
    st.stop()

# =====================================================================
# 运行分析
# =====================================================================

with st.spinner("🔄 正在拟合递减模型并进行产能预测..."):
    try:
        predictor = run_prediction(data_df, params_dict, forecast_years)
        best = predictor.results.get('best')
        summary = predictor.eur_summary()
        st.success(f"✅ 分析完成！最优模型: {best['label'] if best else 'N/A'}")
    except Exception as e:
        st.error(f"分析失败: {e}")
        st.stop()

# =====================================================================
# 选项卡
# =====================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 数据概览",
    "📈 递减模型拟合",
    "🔮 产能预测",
    "📋 模型对比",
    "⛰️ 物质平衡",
    "📄 使用说明",
])

# =====================================================================
# Tab 1: 数据概览
# =====================================================================
with tab1:
    st.subheader("📊 生产数据概览")

    # 统计卡片
    stats = predictor.data
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("数据点数", stats.n_points)
    col2.metric("生产时间", f"{stats.time_max:.0f} 天 ({stats.time_max/365:.1f}年)")
    col3.metric("最高日产", f"{stats.rate_max:.2f} 万方/天")
    col4.metric("当前日产", f"{stats.rate_min:.2f} 万方/天")
    col5.metric("累计产气", f"{stats.cum_total:.0f} 万方 ({stats.cum_total/10000:.2f}亿方)")

    # 生产历史曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    t = stats.get_time()
    q = stats.get_rate()
    cum = stats.get_cumprod()

    axes[0].plot(t, q, 'o-', color='#1a73e8', markersize=3, linewidth=1.5, label='日产气量')
    axes[0].set_xlabel('生产时间 (天)')
    axes[0].set_ylabel('日产气量 (万方/天)')
    axes[0].set_title('日产气量历史')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, cum / 10000, 's-', color='#f9ab00', markersize=3, linewidth=1.5, label='累计产气量')
    axes[1].set_xlabel('生产时间 (天)')
    axes[1].set_ylabel('累计产气量 (亿方)')
    axes[1].set_title('累计产气量历史')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # 数据表格
    with st.expander("查看原始数据"):
        show_df = stats.data.copy()
        show_df.columns = [{'time': '时间(天)', 'rate': '日产气(万方/天)',
                            'cum_prod': '累计产气(万方)', 'pressure': '压力(MPa)'}
                           .get(c, c) for c in show_df.columns]
        st.dataframe(show_df, use_container_width=True, height=300)

# =====================================================================
# Tab 2: 递减模型拟合
# =====================================================================
with tab2:
    st.subheader("📈 递减模型拟合结果")

    # 模型参数表
    model_rows = []
    for key in ['arps_exponential', 'arps_hyperbolic', 'arps_harmonic', 'duong', 'sepd']:
        result = predictor.results.get(key)
        if result and result['params']:
            p = result['params']
            row = {'模型': result['label']}
            if 'qi' in p: row['qi (万方/天)'] = round(p['qi'], 4)
            if 'Di' in p: row['Di (1/天)'] = round(p['Di'], 5)
            if 'b' in p: row['b'] = round(p['b'], 4)
            if 'm' in p: row['m'] = round(p['m'], 4)
            if 'a' in p and p['a'] < 100: row['a'] = round(p['a'], 4)
            if 'n' in p: row['n'] = round(p['n'], 4)
            if 'tau' in p: row['τ (天)'] = round(p['tau'], 1)
            row['RMSE'] = round(result.get('rmse', 0), 4)
            row['AIC'] = round(result.get('aic', 0), 2)
            model_rows.append(row)

    if model_rows:
        st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

    # 最优模型
    if best:
        st.info(
            f"🏆 **最优模型**: {best['label']}  "
            f"(AIC={best.get('aic', 0):.2f}, RMSE={best.get('rmse', 0):.4f})"
        )

    # 拟合对比图
    fig = predictor.plot_decline_fit()
    st.pyplot(fig)

    with st.expander("📖 模型选择解读"):
        st.markdown("""
        ### 如何判断哪个模型更好？

        | 指标 | 说明 |
        |------|------|
        | **AIC** (越小越好) | 平衡拟合精度和复杂度，是最主要依据 |
        | **RMSE** (越小越好) | 仅反映拟合误差，忽略模型复杂度 |

        **经验规则：**
        - ΔAIC < 2 → 两模型无显著差异
        - ΔAIC > 10 → 低分模型明显较差

        **常压页岩气参数范围：**
        - Arps b 值: 0.5~0.85（超出此范围需谨慎）
        - Duong m 值: 0.8~1.2
        - SEPD n 值: 0.3~0.7
        """)

# =====================================================================
# Tab 3: 产能预测
# =====================================================================
with tab3:
    st.subheader("🔮 产能预测")

    # 选择模型
    available_models = [k for k, v in predictor.predictions.items() if v.get('forecast') is not None]
    if not available_models:
        st.warning("暂无预测数据")
        st.stop()

    model_keys_display = {
        'arps_exponential': 'Arps 指数递减',
        'arps_hyperbolic': 'Arps 双曲递减',
        'arps_harmonic': 'Arps 调和递减',
        'duong': 'Duong 模型',
        'sepd': 'SEPD 模型',
    }
    display_options = []
    display_map = {}
    for k in available_models:
        label = model_keys_display.get(k, k)
        if k in predictor.results and predictor.results[k]:
            label = predictor.results[k].get('label', label)
            rmse = predictor.results[k].get('rmse')
            if rmse:
                label += f" (RMSE={rmse:.3f})"
        display_options.append(label)
        display_map[label] = k

    selected_display = st.selectbox(
        "选择预测模型",
        options=display_options,
        index=0,
        help="选择用于展示预测结果的模型",
    )
    selected_key = display_map[selected_display]

    # 预测结果
    result = predictor.predictions[selected_key]
    eur_val = result.get('eur', 0)

    # EUR 展示
    col1, col2, col3 = st.columns(3)
    col1.metric("EUR (最终可采储量)", f"{eur_val:.0f} 万方")
    col2.metric("EUR (亿方)", f"{eur_val / 10000:.4f} 亿方")
    col3.metric("模型", result.get('label', ''))

    # 预测图
    fig = predictor.plot_forecast(selected_key)
    st.pyplot(fig)

    # 年度产量表
    fc = result['forecast']
    pred_only = fc[fc['phase'] == '预测'].copy()
    if len(pred_only) > 0:
        pred_only['year'] = np.ceil(pred_only['time'] / 365).astype(int)
        yearly = pred_only.groupby('year').agg(
            年产气量_万方=('rate_pred', 'sum'),
            年末累计_万方=('cum_pred', 'last')
        ).reset_index()
        yearly.columns = ['年份', '年产气量(万方)', '年末累计(万方)']
        yearly['年产气量(亿方)'] = (yearly['年产气量(万方)'] / 10000).round(4)
        yearly['年末累计(亿方)'] = (yearly['年末累计(万方)'] / 10000).round(4)

        with st.expander("📅 年度产量明细"):
            st.dataframe(yearly, use_container_width=True, hide_index=True)

        # 总 EUR 汇总对比
        with st.expander("📊 各模型 EUR 汇总"):
            if not summary.empty:
                st.dataframe(summary, use_container_width=True, hide_index=True)
                # 推荐区间
                eurs = summary['EUR (亿方)'].values
                if len(eurs) >= 3:
                    sorted_eurs = sorted(eurs)
                    mid = sorted_eurs[1:-1]
                    st.info(
                        f"**推荐 EUR 区间**: {min(mid):.4f} ~ {max(mid):.4f} 亿方 "
                        f"(去掉最优和最悲观的中间模型)"
                    )

# =====================================================================
# Tab 4: 模型对比
# =====================================================================
with tab4:
    st.subheader("📋 模型 EUR 对比")

    if not summary.empty:
        fig = predictor.plot_model_comparison()
        st.pyplot(fig)

        # 排序表
        st.subheader("按 AIC 排序（越小越优）")
        sorted_summary = summary.sort_values('AIC')
        st.dataframe(sorted_summary, use_container_width=True, hide_index=True)

    # Duong 诊断图
    st.subheader("Duong 模型诊断")
    st.caption("若 ln(q/Gp) vs ln(t) 呈线性，则适合用 Duong 模型")
    fig_d = predictor.plot_duong_diagnostic()
    st.pyplot(fig_d)

# =====================================================================
# Tab 5: 物质平衡
# =====================================================================
with tab5:
    st.subheader("⛰️ 物质平衡分析 — OGIP 估算")

    # 检查是否有压力数据
    pressure_data = predictor.data.get_pressure()
    if pressure_data is None:
        st.warning("数据中无压力列 (pressure)，无法进行 p/Z 物质平衡分析。")
        st.info("如需使用此功能，请在数据文件中包含 pressure (MPa) 列。")

    # 检查是否已设置完整储层参数
    has_reservoir = all(k in params_dict for k in ['phi', 'Sw', 'h', 'A', 'gamma_g', 'T'])

    if has_reservoir:
        try:
            mb_result = predictor.analyze_material_balance(p_i=params_dict.get('p_i', 28.0))

            col1, col2, col3 = st.columns(3)
            col1.metric("自由气 OGIP", f"{mb_result['ogip_free']:.0f} 万方 "
                        f"({mb_result['ogip_free_亿方']:.2f} 亿方)")
            col2.metric("吸附气 OGIP", f"{mb_result['ogip_ads']:.0f} 万方 "
                        f"({mb_result['ogip_ads_亿方']:.2f} 亿方)")
            col3.metric("总 OGIP", f"{mb_result['ogip_total']:.0f} 万方 "
                        f"({mb_result['ogip_total_亿方']:.2f} 亿方)")

            # p/Z 分析
            pz = mb_result.get('pz_analysis')
            if pz and pz.get('ogip_estimate') and not np.isnan(pz['ogip_estimate']):
                st.subheader("p/Z 分析")
                col1, col2 = st.columns(2)
                col1.metric("p/Z 法 OGIP",
                            f"{pz['ogip_estimate']:.0f} 万方 ({pz['ogip_estimate']/10000:.2f} 亿方)")
                col2.metric("采收率 (EUR/OGIP)",
                            f"{eur_val / pz['ogip_estimate'] * 100:.1f}%" if eur_val > 0 else "N/A")

                # p/Z 图
                fig_pz = predictor.plot_pz_analysis()
                st.pyplot(fig_pz)
            else:
                st.info("压力数据不足，无法进行 p/Z 分析（至少需要 3 个压力点）")

            # 参数汇总
            with st.expander("储层参数汇总"):
                st.json(params_dict)

        except Exception as e:
            st.error(f"物质平衡分析失败: {e}")
    else:
        st.warning("请先在侧边栏展开储层参数并填入完整信息")

# =====================================================================
# Tab 6: 使用说明
# =====================================================================
with tab6:
    st.subheader("📄 使用说明")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        ### 快速开始
        1. 左侧边栏选择 **数据来源**（示例数据或上传文件）
        2. 调整 **预测年限**（默认 20 年）
        3. 可选填 **储层参数**（仅物质平衡需要）
        4. 点击 **「开始分析」**
        """)

        st.markdown("""
        ### 数据格式要求

        **必需列：**

        | 列名 | 说明 | 单位 |
        |------|------|------|
        | time | 生产时间 | 天 |
        | rate | 日产气量 | 万方/天 |

        **可选列：**
        - cum_prod — 累计产气量（万方）
        - pressure — 压力（MPa）

        **支持格式：** CSV (.csv)、Excel (.xlsx / .xls)
        """)

    with col2:
        st.markdown("""
        ### 模型说明

        | 模型 | 适用场景 |
        |------|---------|
        | **Arps 指数** | 裂缝性气藏、边界控制流 |
        | **Arps 双曲** | 常规页岩气（最推荐） |
        | **Arps 调和** | 强水驱气藏（偏乐观） |
        | **Duong** | 页岩线性流特征明显时 |
        | **SEPD** | 多尺度流动、复杂裂缝 |

        ### 常压区页岩气特征

        - 压力系数: 0.9~1.2
        - 初始产量: 3~10 万方/天
        - 首年递减: 40%~65%
        - 吸附气占比: 30%~60%
        - b 值范围: 0.5~0.85
        """)

    st.markdown("---")
    st.subheader("🔬 技术原理")

    tab_p1, tab_p2, tab_p3, tab_p4 = st.tabs(["Arps 递减", "Duong 模型", "SEPD 模型", "物质平衡法"])

    with tab_p1:
        st.markdown("""
        **Arps 递减 (1945)**

        通用形式:
        $$q(t) = \\frac{q_i}{(1 + b D_i t)^{1/b}}$$

        | b 值 | 类型 | 公式 |
        |------|------|------|
        | b=0 | 指数递减 | $q = q_i e^{-D_i t}$ |
        | 0<b<1 | 双曲递减（推荐） | $q = q_i / (1 + b D_i t)^{1/b}$ |
        | b=1 | 调和递减 | $q = q_i / (1 + D_i t)$ |

        EUR (双曲):
        $$EUR = \\frac{q_i^b}{(1-b)D_i} [q_i^{1-b} - q_{ab}^{1-b}]$$
        """)

    with tab_p2:
        st.markdown("""
        **Duong 模型 (2011)**

        $$q(t) = q_1 \\cdot t^{-m} \\cdot \\exp\\left[\\frac{a}{1-m}(t^{1-m} - 1)\\right]$$

        - $m$: 双对数图 $\\ln(q/G_p)$ vs $\\ln(t)$ 的斜率
        - $a$: 截距参数
        - **诊断**: 若 $\\ln(q/G_p)$ vs $\\ln(t)$ 呈线性，适用 Duong 模型
        """)

    with tab_p3:
        st.markdown("""
        **SEPD (Stretched Exponential)**

        $$q(t) = q_i \\cdot \\exp\\left[-\\left(\\frac{t}{\\tau}\\right)^n\\right]$$

        $$EUR = \\frac{q_i \\tau}{n} \\cdot \\Gamma\\left(\\frac{1}{n}\\right)$$

        - $\\tau$: 特征时间常数
        - $n$: 拉伸指数 (0 < n ≤ 1)
        - $n \\to 1$ 趋近指数递减
        """)

    with tab_p4:
        st.markdown("""
        **物质平衡法 — OGIP 估算**

        常规气藏 p/Z 分析法:
        $$\\frac{p}{Z} = \\frac{p_i}{Z_i}\\left(1 - \\frac{G_p}{G}\\right)$$

        外推 $p/Z = 0$ 得原始地质储量 OGIP。

        页岩气修正（吸附气）:
        - 自由气膨胀
        - 吸附气解吸 (Langmuir 等温线)
        - 孔隙压缩性
        """)

    st.markdown("---")
    st.markdown("""
    **⚠️ 免责声明**: 本程序提供的预测结果仅供参考，不构成工程决策的唯一依据。
    实际产能评价应结合地质研究、数值模拟和开发动态综合判断。
    """)

# =====================================================================
# 页脚
# =====================================================================
st.markdown("---")
st.caption(
    "⛰️ 常压区页岩气产能预测系统 v1.0 | "
    "基于递减分析 + 物质平衡法 | "
    "仅供工程参考"
)
