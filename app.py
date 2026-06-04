import html
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


# DEFAULT_TRAVEL_DAYS：用户没有写旅行天数时，默认按 3 天处理。
DEFAULT_TRAVEL_DAYS = 3

# DEFAULT_TRAVEL_NIGHTS：用户没有写住宿晚数时，默认按 2 晚处理。
DEFAULT_TRAVEL_NIGHTS = 2

# DEFAULT_BUDGET_LEVEL：用户没有写预算时，默认使用普通预算。
DEFAULT_BUDGET_LEVEL = "普通预算"

# DEFAULT_BUDGET_CURRENCY：用户输入预算数字但没有写货币单位时，默认按人民币处理。
DEFAULT_BUDGET_CURRENCY = "CNY"

# DEFAULT_DESTINATION：用户没有写明确目的地时，用于演示的默认目的地。
DEFAULT_DESTINATION = "东京"

# DEEPSEEK_BASE_URL：DeepSeek API 的基础地址，OpenAI SDK 会通过这个地址请求 DeepSeek。
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DEFAULT_DEEPSEEK_MODEL：用户没有在 .env 配置模型时，默认使用的 DeepSeek 模型。
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

# DEFAULT_SEARCH_MAX_RESULTS：每个搜索查询最多保留的结果数量。
DEFAULT_SEARCH_MAX_RESULTS = 3

# DEFAULT_TAVILY_SEARCH_DEPTH：Tavily 默认使用 basic 搜索，控制搜索额度消耗。
DEFAULT_TAVILY_SEARCH_DEPTH = "basic"

# DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE：每份攻略默认最多调用 Tavily 的次数。
DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE = 1

# TAVILY_CACHE_FILE：Tavily 搜索结果本地缓存文件，避免 24 小时内重复消耗额度。
TAVILY_CACHE_FILE = "tavily_cache.json"

# TAVILY_CACHE_TTL_SECONDS：Tavily 缓存有效期，默认 24 小时。
TAVILY_CACHE_TTL_SECONDS = 24 * 60 * 60

# TAVILY_CACHE_PATH：Tavily 缓存文件的绝对路径。
TAVILY_CACHE_PATH = Path(__file__).with_name(TAVILY_CACHE_FILE)

# MAX_GENERATIONS_PER_SESSION：Beta 测试版每个浏览器会话最多生成攻略次数，避免 API 被滥用。
MAX_GENERATIONS_PER_SESSION = 3

# OPEN_METEO_GEOCODING_URL：Open-Meteo 免费地理编码接口，不需要 API Key。
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# OPEN_METEO_FORECAST_URL：Open-Meteo 免费天气预报接口，不需要 API Key。
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WEATHER_FORECAST_DAYS：天气模块默认展示最近几天。
WEATHER_FORECAST_DAYS = 3

# BETA_NOTICE_TEXT：上线前页面底部展示的 Beta 和隐私安全提醒。
BETA_NOTICE_TEXT = "当前为 Beta 测试版。AI 生成内容仅供参考，门票、预约、开放时间、交通政策等信息请以官方渠道为准。请勿输入身份证号、手机号、住址、护照号等敏感个人信息。"

# SAMPLE_PROMPTS：Hero 输入区的示例旅行需求，点击后会自动填入对话框。
SAMPLE_PROMPTS = [
    {"label": "东京3天动漫美食游", "prompt": "我想去东京旅游，喜欢动漫、美食和夜景，预算5000，3 天 2 晚"},
    {"label": "杭州7天舒适游", "prompt": "杭州7日游，想去西湖、灵隐寺、龙井村，也想吃杭州美食和看夜景，预算一万，要求舒适一点"},
    {"label": "南京3天 + 江西4天", "prompt": "南京3天，然后去江西4天，喜欢历史文化、美食和夜景，预算8000"},
    {"label": "大阪京都5日自由行", "prompt": "我想去大阪京都自由行，喜欢历史、美食、购物和拍照，普通预算，5 天 4 晚"},
]


if load_dotenv:
    # load_dotenv：读取本地 .env 文件，方便初学者不用每次手动设置环境变量。
    load_dotenv()


def get_config_value(config_name: str, default_value: str = "") -> str:
    """get_config_value：优先从 .env/环境变量读取配置，其次兼容 Streamlit secrets。"""

    # env_value：load_dotenv 后从系统环境变量中读取到的配置值。
    env_value = os.getenv(config_name)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()

    try:
        # secret_value：Streamlit Cloud 部署时可从 st.secrets 读取的配置值。
        secret_value = st.secrets.get(config_name)
        if secret_value is not None and str(secret_value).strip():
            return str(secret_value).strip()
    except Exception:
        return default_value

    return default_value


def get_bool_config(config_name: str, default_value: bool = False) -> bool:
    """get_bool_config：把环境变量或 secrets 中的开关配置转换成布尔值。"""

    # raw_value：配置原始字符串。
    raw_value = get_config_value(config_name, str(default_value)).strip().lower()
    return raw_value in {"1", "true", "yes", "y", "on", "启用", "是"}


def is_debug_enabled() -> bool:
    """is_debug_enabled：判断是否显示开发者调试信息，默认关闭。"""

    # secret_value：Streamlit Cloud secrets 中的调试开关，优先级最高。
    secret_value = None
    try:
        secret_value = st.secrets.get("SHOW_DEBUG")
    except Exception:
        secret_value = None

    if secret_value is not None and str(secret_value).strip():
        return str(secret_value).strip().lower() in {"true", "1", "yes"}

    # env_value：本地环境变量中的调试开关。
    env_value = os.getenv("SHOW_DEBUG", "")
    return str(env_value).strip().lower() in {"true", "1", "yes"}


def get_int_config(config_name: str, default_value: int) -> int:
    """get_int_config：读取整数配置，非法值自动使用默认值。"""

    # raw_value：配置原始字符串。
    raw_value = get_config_value(config_name, str(default_value)).strip()
    try:
        return int(raw_value)
    except ValueError:
        return default_value


def setup_page() -> None:
    """setup_page：设置 Streamlit 页面基础信息和自定义样式。"""

    st.set_page_config(
        page_title="AI 旅游攻略 Agent",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # custom_css：控制页面视觉风格，让 Streamlit 默认界面更接近高级旅行杂志和 AI 工具。
    custom_css = """
    <style>
    :root {
        --bg-deep: #05070f;
        --panel: rgba(15, 23, 42, 0.58);
        --panel-strong: rgba(15, 23, 42, 0.82);
        --panel-warm: rgba(120, 75, 32, 0.18);
        --line: rgba(255, 255, 255, 0.14);
        --line-soft: rgba(255, 255, 255, 0.08);
        --text-soft: #cbd5e1;
        --text-muted: #94a3b8;
        --cyan: #38bdf8;
        --mint: #34d399;
        --rose: #fb7185;
        --gold: #f6c76f;
        --champagne: #fdecc8;
        --orange: #fb923c;
        --shadow: 0 28px 90px rgba(0, 0, 0, 0.34);
        --glass-blur: blur(20px);
    }

    *,
    *::before,
    *::after {
        box-sizing: border-box;
    }

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        max-width: 100%;
        overflow-x: hidden;
    }

    [data-testid="stMain"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="column"],
    [data-testid="stForm"],
    [data-testid="stTextArea"],
    [data-testid="stMarkdownContainer"] {
        max-width: 100%;
        min-width: 0;
    }

    img,
    iframe,
    table,
    svg {
        max-width: 100%;
    }

    .stApp {
        color: #f8fafc;
        background:
            linear-gradient(118deg, rgba(246, 199, 111, 0.16) 0%, transparent 24%),
            linear-gradient(242deg, rgba(251, 146, 60, 0.12) 0%, transparent 31%),
            linear-gradient(180deg, rgba(255, 255, 255, 0.045), transparent 24%),
            linear-gradient(145deg, #04060d 0%, #0b1020 36%, #111827 68%, #05070f 100%);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.028) 1px, transparent 1px);
        background-size: 72px 72px;
        mask-image: linear-gradient(180deg, rgba(0,0,0,0.72), transparent 72%);
        opacity: 0.38;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    .block-container {
        width: 100%;
        max-width: 1180px;
        padding-top: 1.35rem;
        padding-bottom: 5rem;
    }

    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2.8rem;
        padding: 0.82rem 1.05rem;
        border: 1px solid rgba(246, 199, 111, 0.16);
        border-radius: 999px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02)),
            rgba(8, 12, 24, 0.62);
        box-shadow: 0 16px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.02rem;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: inline-grid;
        place-items: center;
        color: #111827;
        background: linear-gradient(135deg, var(--gold), #fff7d6 48%, var(--orange));
        box-shadow: 0 0 0 1px rgba(255,255,255,0.32), 0 12px 28px rgba(251, 146, 60, 0.24);
    }

    .nav-links {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: var(--text-soft);
        font-size: 0.92rem;
    }

    .nav-links span {
        padding: 0.45rem 0.72rem;
        border-radius: 999px;
        color: #dbeafe;
    }

    .hero {
        position: relative;
        margin-bottom: 1.75rem;
        padding: 0.25rem 0 0.35rem;
    }

    .hero::after {
        content: "";
        display: block;
        width: min(420px, 68vw);
        height: 1px;
        margin-top: 1.5rem;
        background: linear-gradient(90deg, rgba(246, 199, 111, 0.72), transparent);
    }

    .hero-layout {
        display: grid;
        grid-template-columns: minmax(0, 1.16fr) minmax(280px, 0.84fr);
        gap: 1.35rem;
        align-items: stretch;
        max-width: 100%;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 0.8rem;
        border: 1px solid rgba(246, 199, 111, 0.34);
        border-radius: 999px;
        color: #fde68a;
        background: rgba(120, 75, 32, 0.22);
        font-size: 0.84rem;
        margin-bottom: 1.15rem;
    }

    .hero-proof {
        display: flex;
        flex-wrap: wrap;
        gap: 0.58rem;
        margin-top: 1.15rem;
    }

    .hero-proof span {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        padding: 0.38rem 0.68rem;
        color: #e5e7eb;
        background: rgba(15, 23, 42, 0.42);
        font-size: 0.84rem;
        backdrop-filter: blur(12px);
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2.65rem, 5.8vw, 5.85rem);
        line-height: 0.98;
        letter-spacing: 0;
        max-width: 930px;
        color: #fff7ed;
        text-wrap: balance;
    }

    .hero p {
        margin: 1.15rem 0 0;
        max-width: 720px;
        color: #d1d5db;
        font-size: 1.08rem;
        line-height: 1.85;
    }

    .hero-panel {
        min-width: 0;
        min-height: 100%;
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 1.25rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.035)),
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(251, 146, 60, 0.06));
        box-shadow: var(--shadow);
        backdrop-filter: var(--glass-blur);
        position: relative;
        overflow: hidden;
    }

    .hero-panel::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(90deg, transparent 0, rgba(246, 199, 111, 0.08) 1px, transparent 1px),
            linear-gradient(180deg, transparent 0, rgba(255, 255, 255, 0.045) 1px, transparent 1px);
        background-size: 34px 34px;
        opacity: 0.5;
    }

    .mini-card {
        position: relative;
        z-index: 1;
        border: 1px solid var(--line-soft);
        border-radius: 20px;
        padding: 1rem;
        background: rgba(3, 7, 18, 0.42);
        margin-bottom: 0.85rem;
    }

    .mini-card span {
        display: block;
        color: var(--gold);
        font-size: 0.78rem;
        margin-bottom: 0.45rem;
    }

    .mini-card strong {
        display: block;
        font-size: 1.2rem;
        margin-bottom: 0.35rem;
    }

    .mini-card p {
        margin: 0;
        color: var(--text-muted);
        line-height: 1.55;
        font-size: 0.92rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line) !important;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.105), rgba(255, 255, 255, 0.035)),
            rgba(10, 15, 30, 0.58) !important;
        border-radius: 26px !important;
        box-shadow: var(--shadow);
        backdrop-filter: var(--glass-blur);
    }

    [data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #fff7ed;
        letter-spacing: 0;
    }

    .input-title {
        font-size: 1.12rem;
        color: #fef3c7;
        margin: 0 0 0.35rem;
        font-weight: 700;
    }

    .input-kicker {
        color: var(--gold);
        font-size: 0.78rem;
        text-transform: uppercase;
        margin: 0 0 0.32rem;
    }

    .sample-title {
        color: var(--text-muted);
        font-size: 0.88rem;
        margin: 0.85rem 0 0.45rem;
    }

    .stTextArea textarea {
        min-height: 150px !important;
        border-radius: 22px !important;
        border: 1px solid rgba(246, 199, 111, 0.34) !important;
        background:
            linear-gradient(145deg, rgba(2, 6, 23, 0.78), rgba(15, 23, 42, 0.62)) !important;
        color: #f8fafc !important;
        font-size: 1.03rem !important;
        line-height: 1.65 !important;
        box-shadow:
            inset 0 0 0 1px rgba(255, 255, 255, 0.04),
            0 18px 50px rgba(0, 0, 0, 0.16);
    }

    .stTextArea textarea:focus {
        border-color: rgba(246, 199, 111, 0.88) !important;
        box-shadow: 0 0 0 4px rgba(246, 199, 111, 0.14) !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        width: 100%;
        border: 1px solid rgba(246, 199, 111, 0.26);
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(246, 199, 111, 0.95), rgba(251, 146, 60, 0.92));
        color: #17120a;
        font-weight: 800;
        padding: 0.78rem 1rem;
        box-shadow: 0 14px 34px rgba(251, 146, 60, 0.18);
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        color: #17120a;
        filter: brightness(1.05);
        border-color: rgba(255, 247, 237, 0.55);
    }

    .cover-card {
        width: 100%;
        max-width: 100%;
        aspect-ratio: 16 / 9;
        min-height: 420px;
        border-radius: 30px;
        border: 1px solid rgba(246, 199, 111, 0.24);
        background-size: cover;
        background-position: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 34px 110px rgba(0, 0, 0, 0.45);
        margin: 2.15rem 0 1.45rem;
    }

    .cover-card::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: 1;
        pointer-events: none;
        background:
            linear-gradient(90deg, rgba(246, 199, 111, 0.16) 1px, transparent 1px),
            linear-gradient(180deg, rgba(255, 255, 255, 0.07) 1px, transparent 1px),
            linear-gradient(135deg, transparent 0 62%, rgba(246, 199, 111, 0.12) 62% 63%, transparent 63%);
        background-size: 88px 88px, 88px 88px, 100% 100%;
        opacity: 0.42;
    }

    .cover-card::after {
        content: "";
        position: absolute;
        inset: 0;
        z-index: 0;
        background:
            linear-gradient(90deg, rgba(2, 6, 23, 0.82), rgba(2, 6, 23, 0.28) 58%, rgba(2, 6, 23, 0.68)),
            linear-gradient(180deg, rgba(2, 6, 23, 0.02) 20%, rgba(2, 6, 23, 0.88));
    }

    .cover-content {
        position: absolute;
        inset: auto clamp(1.25rem, 4vw, 3.2rem) clamp(1.25rem, 4vw, 3.2rem) clamp(1.25rem, 4vw, 3.2rem);
        z-index: 2;
    }

    .cover-content .label {
        color: #fde68a;
        font-size: 0.88rem;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .cover-content h2 {
        margin: 0;
        font-size: clamp(2.55rem, 6.2vw, 5.4rem);
        line-height: 0.98;
        letter-spacing: 0;
    }

    .cover-dayline {
        color: var(--champagne);
        font-size: clamp(1rem, 2.2vw, 1.45rem);
        font-weight: 800;
        margin-top: 0.62rem;
    }

    .cover-content p {
        margin: 0.9rem 0 0;
        max-width: 760px;
        color: #f8fafc;
        font-size: 1rem;
        line-height: 1.75;
    }

    .cover-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .cover-badge {
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        color: #fff7ed;
        background: rgba(15, 23, 42, 0.46);
        backdrop-filter: blur(10px);
        font-size: 0.86rem;
    }

    .bento-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        grid-auto-rows: minmax(112px, auto);
        gap: 0.9rem;
        margin: 1rem 0 2rem;
    }

    .bento-card {
        min-width: 0;
        max-width: 100%;
        border: 1px solid var(--line-soft);
        border-radius: 24px;
        padding: 1.05rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.52);
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
        min-height: 112px;
    }

    .bento-card.large {
        grid-column: span 2;
    }

    .bento-card.warm {
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.19), rgba(251, 146, 60, 0.065)),
            rgba(15, 23, 42, 0.52);
    }

    .bento-card span {
        display: block;
        color: #fcd34d;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }

    .bento-card strong {
        display: block;
        color: #f8fafc;
        font-size: clamp(1.05rem, 2.2vw, 1.45rem);
        line-height: 1.18;
        letter-spacing: 0;
    }

    .bento-card p {
        color: var(--text-muted);
        line-height: 1.55;
        margin: 0.55rem 0 0;
        font-size: 0.92rem;
    }

    .segment-overview-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.85rem;
        max-width: 100%;
        margin: -0.45rem 0 1.9rem;
    }

    .segment-card {
        width: 100%;
        min-width: 0;
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 22px;
        padding: 1rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(56, 189, 248, 0.05)),
            rgba(15, 23, 42, 0.48);
        box-shadow: 0 16px 50px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
    }

    .segment-card span {
        display: block;
        color: #fcd34d;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }

    .segment-card strong {
        display: block;
        color: #fff7ed;
        font-size: 1.18rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .segment-card p {
        color: var(--text-muted);
        line-height: 1.58;
        margin: 0.55rem 0 0;
        font-size: 0.9rem;
        overflow-wrap: anywhere;
    }

    .section-heading {
        margin: 2.25rem 0 0.35rem;
        color: #fff7ed;
        font-size: clamp(1.55rem, 3vw, 2.1rem);
        letter-spacing: 0;
    }

    .section-subtitle {
        margin: 0 0 1.1rem;
        color: var(--text-muted);
        line-height: 1.65;
    }

    .timeline-grid,
    .food-grid,
    .info-grid,
    .warning-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.6rem;
        max-width: 100%;
    }

    .timeline-day,
    .food-card,
    .info-card,
    .warning-card {
        width: 100%;
        max-width: 100%;
        min-width: 0;
        border: 1px solid var(--line-soft);
        border-radius: 26px;
        padding: 1.15rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.25);
        backdrop-filter: var(--glass-blur);
    }

    .timeline-day h3,
    .food-card h3,
    .info-card h3,
    .warning-card h3 {
        margin: 0 0 0.85rem;
        color: #fff7ed;
        letter-spacing: 0;
        overflow-wrap: anywhere;
    }

    .card-title-row {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin-bottom: 0.82rem;
    }

    .card-title-row h3 {
        margin: 0;
    }

    .info-icon,
    .warning-icon {
        width: 36px;
        height: 36px;
        border-radius: 13px;
        display: grid;
        place-items: center;
        font-weight: 900;
        flex: 0 0 auto;
    }

    .info-icon {
        color: #082f49;
        background: linear-gradient(135deg, #7dd3fc, #38bdf8);
    }

    .warning-icon {
        color: #17120a;
        background: linear-gradient(135deg, #f6c76f, #fb923c);
    }

    .timeline-slot {
        display: grid;
        grid-template-columns: 44px minmax(0, 1fr);
        gap: 0.85rem;
        padding: 0.82rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .timeline-slot:first-of-type {
        border-top: 0;
        padding-top: 0;
    }

    .slot-icon {
        width: 40px;
        height: 40px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        color: #17120a;
        background: linear-gradient(135deg, #f6c76f, #fb923c);
        font-weight: 900;
        box-shadow: 0 12px 26px rgba(251, 146, 60, 0.18);
    }

    .slot-time {
        color: #fcd34d;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .slot-place {
        color: #f8fafc;
        font-weight: 800;
        margin-bottom: 0.25rem;
        overflow-wrap: anywhere;
    }

    .slot-original {
        color: #fde68a;
        font-size: 0.78rem;
        margin-bottom: 0.38rem;
        opacity: 0.92;
        overflow-wrap: anywhere;
    }

    .slot-desc,
    .food-card p,
    .info-card p,
    .warning-card p {
        color: #cbd5e1;
        line-height: 1.62;
        margin: 0;
        font-size: 0.93rem;
        overflow-wrap: anywhere;
    }

    .slot-meta-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.45rem;
        margin-top: 0.62rem;
    }

    .slot-meta-item {
        border: 1px solid rgba(255, 255, 255, 0.075);
        border-radius: 14px;
        padding: 0.48rem 0.56rem;
        color: #cbd5e1;
        background: rgba(2, 6, 23, 0.26);
        font-size: 0.8rem;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    .slot-meta-item strong {
        color: #fcd34d;
        font-weight: 800;
    }

    .food-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.9rem;
    }

    .food-meta span {
        border: 1px solid rgba(246, 199, 111, 0.24);
        border-radius: 999px;
        padding: 0.32rem 0.58rem;
        color: #fde68a;
        background: rgba(120, 75, 32, 0.18);
        font-size: 0.8rem;
    }

    .food-location {
        color: #9ca3af;
        font-size: 0.82rem;
        line-height: 1.55;
        margin: -0.42rem 0 0.76rem;
    }

    .food-map-keyword {
        color: #fcd34d;
        font-size: 0.78rem;
        line-height: 1.45;
        margin-top: 0.65rem;
        opacity: 0.9;
    }

    .info-card {
        background:
            linear-gradient(145deg, rgba(56, 189, 248, 0.13), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
    }

    .warning-card {
        border-color: rgba(251, 146, 60, 0.22);
        background:
            linear-gradient(145deg, rgba(251, 146, 60, 0.18), rgba(127, 29, 29, 0.10)),
            rgba(15, 23, 42, 0.54);
    }

    .weather-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        max-width: 100%;
        margin-bottom: 1.5rem;
    }

    .weather-card {
        width: 100%;
        max-width: 100%;
        min-width: 0;
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 26px;
        padding: 1.1rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(56, 189, 248, 0.055)),
            rgba(15, 23, 42, 0.54);
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.24);
        backdrop-filter: var(--glass-blur);
    }

    .weather-card h3 {
        margin: 0 0 0.8rem;
        color: #fff7ed;
        letter-spacing: 0;
        overflow-wrap: anywhere;
    }

    .weather-day {
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr);
        gap: 0.82rem;
        padding: 0.86rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .weather-day:first-of-type {
        border-top: 0;
        padding-top: 0;
    }

    .weather-icon {
        width: 40px;
        height: 40px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: rgba(246, 199, 111, 0.15);
        border: 1px solid rgba(246, 199, 111, 0.22);
        font-size: 1.2rem;
    }

    .weather-date {
        color: #fcd34d;
        font-weight: 800;
        font-size: 0.88rem;
        margin-bottom: 0.22rem;
    }

    .weather-main {
        color: #f8fafc;
        font-weight: 800;
        margin-bottom: 0.36rem;
        overflow-wrap: anywhere;
    }

    .weather-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        margin: 0.45rem 0;
    }

    .weather-meta span {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 999px;
        padding: 0.28rem 0.48rem;
        color: #fde68a;
        background: rgba(15, 23, 42, 0.42);
        font-size: 0.78rem;
        overflow-wrap: anywhere;
    }

    .weather-advice {
        color: #cbd5e1;
        line-height: 1.6;
        font-size: 0.9rem;
        margin: 0.55rem 0 0;
        overflow-wrap: anywhere;
    }

    .weather-fallback {
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 22px;
        padding: 1rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.12), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
        color: #cbd5e1;
        line-height: 1.65;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.20);
        backdrop-filter: var(--glass-blur);
    }

    .blessing-card {
        margin: 1.2rem 0 1.6rem;
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 24px;
        padding: 1rem 1.05rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.14), rgba(251, 146, 60, 0.045)),
            rgba(15, 23, 42, 0.56);
        color: #f8fafc;
        line-height: 1.72;
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
    }

    .markdown-actions {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 220px;
        gap: 0.8rem;
        align-items: center;
        margin-bottom: 0.7rem;
    }

    .hint {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.65;
        overflow-wrap: anywhere;
    }

    .search-status-pill {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        max-width: 100%;
        gap: 0.48rem;
        margin: 0.3rem 0 1.1rem;
        padding: 0.5rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(246, 199, 111, 0.22);
        background: rgba(15, 23, 42, 0.48);
        color: #fde68a;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
        backdrop-filter: var(--glass-blur);
        font-size: 0.86rem;
        line-height: 1.4;
    }

    .trust-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.45rem 0 1.15rem;
        max-width: 100%;
    }

    .trust-card {
        min-width: 0;
        border: 1px solid rgba(246, 199, 111, 0.16);
        border-radius: 18px;
        padding: 0.82rem 0.9rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.10), rgba(255, 255, 255, 0.025)),
            rgba(15, 23, 42, 0.48);
        box-shadow: 0 14px 42px rgba(0, 0, 0, 0.18);
        backdrop-filter: var(--glass-blur);
    }

    .trust-card span {
        display: block;
        color: #fcd34d;
        font-size: 0.76rem;
        margin-bottom: 0.32rem;
    }

    .trust-card strong {
        display: block;
        color: #fff7ed;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .trust-card p {
        color: var(--text-muted);
        font-size: 0.84rem;
        line-height: 1.5;
        margin: 0.42rem 0 0;
        overflow-wrap: anywhere;
    }

    .search-status-dot {
        width: 0.46rem;
        height: 0.46rem;
        flex: 0 0 auto;
        border-radius: 999px;
        background: var(--gold);
        box-shadow: 0 0 18px rgba(246, 199, 111, 0.52);
    }

    .generation-quota {
        color: #fde68a;
        font-size: 0.86rem;
        margin-top: 0.55rem;
        opacity: 0.92;
    }

    .result-action-panel {
        margin: 2rem 0 1rem;
        padding: 1.05rem;
        border: 1px solid rgba(246, 199, 111, 0.20);
        border-radius: 24px;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.56);
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.25);
        backdrop-filter: var(--glass-blur);
    }

    .result-action-panel h2 {
        margin: 0 0 0.25rem;
        color: #fff7ed;
        font-size: 1.25rem;
        letter-spacing: 0;
    }

    .result-action-panel p {
        margin: 0;
        color: var(--text-muted);
        line-height: 1.55;
        font-size: 0.9rem;
    }

    .result-action-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.8fr) minmax(0, 0.8fr);
        gap: 0.78rem;
        align-items: center;
        margin-top: 0.9rem;
        max-width: 100%;
    }

    .beta-notice {
        margin-top: 2.2rem;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(246, 199, 111, 0.22);
        border-radius: 20px;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.12), rgba(255, 255, 255, 0.035)),
            rgba(15, 23, 42, 0.58);
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.9rem;
        backdrop-filter: var(--glass-blur);
    }

    div[data-testid="stExpander"] {
        max-width: 100%;
        border: 1px solid var(--line-soft);
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.50);
        backdrop-filter: var(--glass-blur);
    }

    pre,
    code,
    .stCodeBlock {
        max-width: 100%;
        overflow-wrap: anywhere;
    }

    pre {
        white-space: pre-wrap;
    }

    @media (max-width: 768px) {
        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"] {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }

        .block-container {
            max-width: 100% !important;
            padding: 0.85rem 0.78rem 3rem !important;
        }

        .top-nav {
            width: 100%;
            border-radius: 22px;
            align-items: flex-start;
            margin-bottom: 1.45rem;
            padding: 0.72rem 0.78rem;
        }

        .nav-links {
            display: none;
        }

        .hero {
            margin-bottom: 1.2rem;
        }

        .hero-layout,
        .segment-overview-grid,
        .trust-strip,
        .slot-meta-grid,
        .result-action-grid,
        .weather-grid,
        .timeline-grid,
        .food-grid,
        .info-grid,
        .warning-grid,
        .markdown-actions,
        .bento-grid {
            grid-template-columns: minmax(0, 1fr) !important;
            width: 100%;
        }

        .hero h1 {
            font-size: clamp(2rem, 12vw, 3.25rem);
            line-height: 1.04;
        }

        .hero p {
            font-size: 0.96rem;
            line-height: 1.68;
        }

        .hero-panel {
            padding: 0.85rem;
            border-radius: 22px;
        }

        .bento-card.large {
            grid-column: span 1;
        }

        .bento-card,
        .segment-card,
        .trust-card,
        .weather-card,
        .timeline-day,
        .food-card,
        .info-card,
        .warning-card {
            width: 100%;
            padding: 0.92rem;
            border-radius: 20px;
            box-shadow: 0 14px 44px rgba(0, 0, 0, 0.22);
        }

        .cover-card {
            min-height: 240px;
            aspect-ratio: 4 / 3;
            border-radius: 22px;
            margin: 1.25rem 0 1rem;
            background-position: center;
        }

        .cover-content {
            inset: auto 1rem 1rem 1rem;
        }

        .cover-content h2 {
            font-size: clamp(2rem, 12vw, 3.2rem);
        }

        .cover-content p {
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .cover-badge {
            font-size: 0.76rem;
            padding: 0.34rem 0.55rem;
        }

        .section-heading {
            font-size: 1.35rem;
            margin-top: 1.55rem;
        }

        .section-subtitle {
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .timeline-slot {
            grid-template-columns: 36px minmax(0, 1fr);
            gap: 0.68rem;
        }

        .slot-icon {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            font-size: 0.75rem;
        }

        .weather-day {
            grid-template-columns: 36px minmax(0, 1fr);
            gap: 0.68rem;
        }

        .weather-icon {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            font-size: 1rem;
        }

        .food-meta span {
            max-width: 100%;
            white-space: normal;
        }

        .search-status-pill {
            width: 100%;
            align-items: flex-start;
            border-radius: 18px;
        }

        .result-action-panel {
            padding: 0.92rem;
            border-radius: 20px;
        }

        .hero-proof {
            gap: 0.42rem;
        }

        .hero-proof span {
            width: 100%;
            text-align: center;
        }

        [data-testid="column"] {
            width: 100% !important;
            min-width: 0 !important;
            flex: 1 1 100% !important;
        }

        .stTextArea textarea {
            width: 100% !important;
            min-height: 118px !important;
            font-size: 0.95rem !important;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {
            width: 100% !important;
            white-space: normal;
            line-height: 1.35;
        }

        .stApp::before {
            opacity: 0.18;
            background-size: 96px 96px;
        }

        table {
            display: block;
            overflow-x: auto;
        }
    }

    @media (max-width: 520px) {
        .block-container {
            padding-left: 0.62rem !important;
            padding-right: 0.62rem !important;
        }

        .bento-grid {
            grid-template-columns: 1fr;
        }

        .bento-card.large {
            grid-column: span 1;
        }

        .top-nav {
            border-radius: 18px;
        }

        .hero h1 {
            font-size: clamp(1.85rem, 13vw, 2.75rem);
        }

        .cover-card {
            min-height: 218px;
        }

        .cover-content .label {
            font-size: 0.72rem;
            letter-spacing: 0.08rem;
        }

        .cover-dayline {
            font-size: 0.92rem;
        }
    }
    /* =========================
       TripAgent Product UI v2
       ========================= */
    :root {
        --v2-bg-a: #060711;
        --v2-bg-b: #111827;
        --v2-ink: #fff7ed;
        --v2-muted: #a7b0c0;
        --v2-gold: #f7d58a;
        --v2-gold-2: #f59e0b;
        --v2-ice: #9bdcff;
        --v2-card: rgba(12, 18, 34, 0.66);
        --v2-card-strong: rgba(8, 13, 26, 0.82);
        --v2-line: rgba(255, 244, 214, 0.16);
        --v2-line-bright: rgba(247, 213, 138, 0.34);
        --v2-shadow: 0 28px 100px rgba(0, 0, 0, 0.42);
    }

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        width: 100%;
        max-width: 100%;
        overflow-x: hidden !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 16% 8%, rgba(247, 213, 138, 0.22), transparent 28%),
            radial-gradient(circle at 82% 12%, rgba(155, 220, 255, 0.15), transparent 26%),
            radial-gradient(circle at 70% 86%, rgba(245, 158, 11, 0.15), transparent 34%),
            linear-gradient(145deg, #050611 0%, #0b1020 42%, #171923 72%, #060711 100%) !important;
        color: var(--v2-ink);
    }

    .stApp::before {
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(247, 213, 138, 0.035) 1px, transparent 1px) !important;
        background-size: 92px 92px;
        opacity: 0.42;
    }

    .block-container {
        max-width: 1240px !important;
        padding-top: 1rem !important;
    }

    .top-nav {
        position: sticky;
        top: 0.7rem;
        z-index: 5;
        margin-bottom: 1.6rem !important;
        padding: 0.74rem 0.92rem !important;
        border: 1px solid var(--v2-line-bright) !important;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.13), rgba(255, 255, 255, 0.035)),
            rgba(8, 12, 24, 0.78) !important;
        box-shadow: 0 18px 58px rgba(0, 0, 0, 0.34);
    }

    .brand-mark {
        background: linear-gradient(135deg, #fff2bf, #f7d58a 45%, #f59e0b) !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.34), 0 0 34px rgba(247, 213, 138, 0.22) !important;
    }

    .nav-links span {
        color: #f8fafc !important;
        border: 1px solid transparent;
    }

    .nav-links span:hover {
        border-color: rgba(247, 213, 138, 0.2);
        background: rgba(247, 213, 138, 0.08);
    }

    .hero.product-hero {
        position: relative;
        padding: clamp(1.1rem, 3vw, 2rem);
        border: 1px solid rgba(247, 213, 138, 0.20);
        border-radius: 34px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.115), rgba(255, 255, 255, 0.035)),
            radial-gradient(circle at 10% 0%, rgba(247, 213, 138, 0.15), transparent 38%),
            radial-gradient(circle at 90% 12%, rgba(155, 220, 255, 0.10), transparent 32%),
            rgba(9, 14, 28, 0.62);
        box-shadow: var(--v2-shadow);
        backdrop-filter: blur(26px);
        overflow: hidden;
        margin-bottom: 1.2rem;
    }

    .hero.product-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(90deg, transparent 0 8%, rgba(247, 213, 138, 0.12) 8% 8.12%, transparent 8.12%),
            linear-gradient(180deg, transparent 0 18%, rgba(255, 255, 255, 0.08) 18% 18.12%, transparent 18.12%);
        opacity: 0.55;
    }

    .hero.product-hero::after {
        display: none;
    }

    .hero-layout {
        position: relative;
        z-index: 1;
        grid-template-columns: minmax(0, 1.06fr) minmax(330px, 0.94fr) !important;
        gap: clamp(1rem, 2.4vw, 2rem) !important;
        align-items: center !important;
    }

    .eyebrow {
        border-color: rgba(247, 213, 138, 0.36) !important;
        color: #fff2bf !important;
        background: rgba(247, 213, 138, 0.10) !important;
        box-shadow: 0 12px 38px rgba(245, 158, 11, 0.13);
    }

    .hero h1,
    .hero-title {
        max-width: 900px;
        margin: 0.35rem 0 0.85rem !important;
        padding: 0.08rem 0;
        font-size: clamp(3rem, 5.8vw, 4.5rem) !important;
        line-height: 1.1 !important;
        color: transparent !important;
        background: linear-gradient(102deg, #fff7ed 0%, #f7d58a 58%, #9bdcff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        text-wrap: balance;
        overflow-wrap: break-word;
    }

    .hero-title-line {
        display: inline;
    }

    .hero p {
        max-width: 720px;
        color: #d7dbe5 !important;
        font-size: clamp(1rem, 1.8vw, 1.22rem) !important;
        line-height: 1.85 !important;
    }

    .hero-proof span {
        border-color: rgba(247, 213, 138, 0.20) !important;
        background: rgba(255, 255, 255, 0.07) !important;
    }

    .hero-panel.product-preview {
        min-height: 430px;
        border: 1px solid rgba(247, 213, 138, 0.24) !important;
        border-radius: 30px !important;
        padding: 1.1rem !important;
        background:
            radial-gradient(circle at 20% 10%, rgba(247, 213, 138, 0.18), transparent 34%),
            linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.035)),
            rgba(6, 10, 22, 0.72) !important;
        overflow: hidden;
    }

    .preview-cover {
        position: relative;
        min-height: 176px;
        border-radius: 24px;
        border: 1px solid rgba(247, 213, 138, 0.22);
        background:
            linear-gradient(120deg, rgba(2, 6, 23, 0.20), rgba(2, 6, 23, 0.82)),
            radial-gradient(circle at 20% 20%, rgba(247, 213, 138, 0.72), transparent 20%),
            linear-gradient(135deg, #1e293b, #7c2d12 58%, #020617);
        box-shadow: 0 20px 65px rgba(0, 0, 0, 0.32);
        overflow: hidden;
    }

    .preview-cover::after {
        content: "";
        position: absolute;
        inset: 18px;
        border: 1px solid rgba(255, 247, 237, 0.18);
        border-radius: 18px;
    }

    .preview-cover-label {
        position: absolute;
        left: 1rem;
        bottom: 1rem;
        z-index: 1;
    }

    .preview-cover-label span {
        display: block;
        color: #f7d58a;
        font-size: 0.76rem;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .preview-cover-label strong {
        display: block;
        font-size: 1.55rem;
        color: #fff7ed;
        line-height: 1.08;
    }

    .preview-steps {
        display: grid;
        gap: 0.72rem;
        margin-top: 0.92rem;
    }

    .preview-step {
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr);
        gap: 0.72rem;
        align-items: center;
        padding: 0.72rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.055);
    }

    .preview-step b {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 14px;
        color: #17120a;
        background: linear-gradient(135deg, #fff2bf, #f59e0b);
    }

    .preview-step span {
        display: block;
        color: #f7d58a;
        font-size: 0.78rem;
        margin-bottom: 0.18rem;
    }

    .preview-step p {
        margin: 0 !important;
        color: #dbe4f0 !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-kicker) {
        position: relative;
        border: 1px solid rgba(247, 213, 138, 0.25) !important;
        border-radius: 30px !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.034)),
            rgba(8, 13, 26, 0.72) !important;
        box-shadow: 0 24px 86px rgba(0, 0, 0, 0.34), 0 0 0 1px rgba(255,255,255,0.035) inset !important;
        backdrop-filter: blur(26px);
        overflow: hidden;
        padding: 0.4rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-kicker)::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: radial-gradient(circle at 6% 0%, rgba(247, 213, 138, 0.16), transparent 34%);
    }

    .input-kicker {
        color: #f7d58a !important;
        letter-spacing: 0.12rem;
    }

    .input-title {
        color: #fff7ed !important;
        font-size: 1.26rem !important;
    }

    .sample-title,
    .hint {
        color: #aeb8ca !important;
    }

    .stTextArea textarea {
        min-height: 170px !important;
        border-radius: 24px !important;
        border: 1px solid rgba(247, 213, 138, 0.36) !important;
        background:
            linear-gradient(145deg, rgba(2,6,23,0.88), rgba(15,23,42,0.70)) !important;
        box-shadow: 0 18px 58px rgba(0,0,0,0.26), 0 0 0 1px rgba(255,255,255,0.045) inset !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        min-height: 44px;
        border: 1px solid rgba(255, 247, 237, 0.24) !important;
        border-radius: 999px !important;
        background:
            linear-gradient(135deg, #fff2bf 0%, #f7d58a 36%, #f59e0b 100%) !important;
        color: #16120b !important;
        box-shadow: 0 15px 38px rgba(245, 158, 11, 0.22) !important;
        font-weight: 900 !important;
    }

    .cover-card {
        min-height: 520px !important;
        border-radius: 36px !important;
        border: 1px solid rgba(247, 213, 138, 0.30) !important;
        box-shadow: 0 42px 130px rgba(0, 0, 0, 0.52), 0 0 80px rgba(247, 213, 138, 0.08) inset !important;
        isolation: isolate;
    }

    .cover-card::before {
        z-index: 2 !important;
        background:
            linear-gradient(90deg, rgba(247, 213, 138, 0.20) 1px, transparent 1px),
            linear-gradient(180deg, rgba(255,255,255,0.09) 1px, transparent 1px),
            radial-gradient(circle at 88% 18%, rgba(155, 220, 255, 0.18), transparent 28%) !important;
        background-size: 94px 94px, 94px 94px, 100% 100% !important;
        opacity: 0.5 !important;
    }

    .cover-card::after {
        background:
            linear-gradient(90deg, rgba(2, 6, 23, 0.88), rgba(2, 6, 23, 0.34) 56%, rgba(2, 6, 23, 0.78)),
            linear-gradient(180deg, rgba(2, 6, 23, 0.02), rgba(2, 6, 23, 0.88)) !important;
    }

    .cover-content {
        z-index: 3 !important;
    }

    .cover-content .label {
        color: #f7d58a !important;
        letter-spacing: 0.18rem !important;
    }

    .cover-content h2 {
        color: #fff7ed !important;
        font-size: clamp(3rem, 7vw, 6.4rem) !important;
        text-shadow: 0 22px 80px rgba(0,0,0,0.5);
    }

    .cover-dayline {
        color: #fff2bf !important;
        font-size: clamp(1.1rem, 2.4vw, 1.6rem) !important;
    }

    .cover-badge,
    .weather-meta span,
    .food-meta span {
        border-color: rgba(247, 213, 138, 0.24) !important;
        background: rgba(247, 213, 138, 0.10) !important;
        color: #fff2bf !important;
    }

    .section-heading {
        margin-top: 2.6rem !important;
        font-size: clamp(1.7rem, 3vw, 2.35rem) !important;
        color: #fff7ed !important;
    }

    .section-heading::after {
        content: "";
        display: block;
        width: 86px;
        height: 2px;
        margin-top: 0.45rem;
        background: linear-gradient(90deg, #f7d58a, transparent);
    }

    .section-subtitle {
        color: #aeb8ca !important;
        max-width: 820px;
    }

    .bento-grid {
        grid-template-columns: repeat(6, minmax(0, 1fr)) !important;
        gap: 1rem !important;
    }

    .bento-card {
        grid-column: span 2;
        min-height: 148px !important;
        border-radius: 28px !important;
        border: 1px solid rgba(247, 213, 138, 0.18) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.105), rgba(255,255,255,0.026)),
            rgba(10, 16, 31, 0.64) !important;
        box-shadow: 0 22px 76px rgba(0, 0, 0, 0.28) !important;
    }

    .bento-card.large {
        grid-column: span 3 !important;
    }

    .bento-card.warm {
        background:
            radial-gradient(circle at 12% 10%, rgba(247, 213, 138, 0.18), transparent 38%),
            linear-gradient(145deg, rgba(247, 213, 138, 0.14), rgba(255,255,255,0.03)),
            rgba(10, 16, 31, 0.64) !important;
    }

    .bento-card span,
    .segment-card span {
        color: #f7d58a !important;
        letter-spacing: 0.04rem;
        text-transform: uppercase;
    }

    .bento-card strong {
        color: #fff7ed !important;
        font-size: clamp(1.25rem, 2.2vw, 1.68rem) !important;
    }

    .timeline-grid {
        grid-template-columns: minmax(0, 1fr) !important;
        gap: 1.25rem !important;
    }

    .timeline-day {
        position: relative;
        border-radius: 30px !important;
        border: 1px solid rgba(247, 213, 138, 0.18) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.105), rgba(255,255,255,0.026)),
            rgba(10, 16, 31, 0.66) !important;
        box-shadow: 0 24px 86px rgba(0,0,0,0.30) !important;
        padding: 1.25rem 1.25rem 1.25rem 1.45rem !important;
        overflow: hidden;
    }

    .timeline-day::before {
        content: "";
        position: absolute;
        left: 2.05rem;
        top: 4.6rem;
        bottom: 1.4rem;
        width: 1px;
        background: linear-gradient(180deg, #f7d58a, rgba(247, 213, 138, 0.05));
    }

    .timeline-day h3 {
        font-size: clamp(1.2rem, 2vw, 1.55rem) !important;
        color: #fff7ed !important;
        padding-left: 0.2rem;
    }

    .timeline-slot {
        position: relative;
        grid-template-columns: 54px minmax(0, 1fr) !important;
        gap: 1rem !important;
        border-top: 0 !important;
        padding: 0.7rem 0 !important;
    }

    .slot-icon {
        position: relative;
        z-index: 1;
        width: 46px !important;
        height: 46px !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #fff2bf, #f59e0b) !important;
        box-shadow: 0 14px 32px rgba(245, 158, 11, 0.23) !important;
    }

    .timeline-slot > div:nth-child(2) {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        background: rgba(255,255,255,0.045);
        padding: 0.86rem 0.92rem;
    }

    .slot-time,
    .slot-original,
    .weather-date {
        color: #f7d58a !important;
    }

    .slot-place,
    .weather-main {
        color: #fff7ed !important;
    }

    .slot-meta-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    }

    .slot-meta-item {
        border-color: rgba(247, 213, 138, 0.12) !important;
        background: rgba(7, 11, 22, 0.46) !important;
    }

    .food-grid,
    .info-grid,
    .warning-grid,
    .weather-grid,
    .budget-grid {
        gap: 1rem !important;
    }

    .food-card,
    .info-card,
    .warning-card,
    .weather-card,
    .budget-card,
    .segment-card {
        border-radius: 28px !important;
        border: 1px solid rgba(247, 213, 138, 0.17) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.025)),
            rgba(10, 16, 31, 0.64) !important;
        box-shadow: 0 22px 76px rgba(0,0,0,0.28) !important;
    }

    .budget-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        max-width: 100%;
        margin-bottom: 1.6rem;
    }

    .budget-card {
        min-width: 0;
        padding: 1rem;
    }

    .budget-card span {
        display: block;
        color: #f7d58a;
        font-size: 0.78rem;
        letter-spacing: 0.04rem;
        margin-bottom: 0.38rem;
        text-transform: uppercase;
    }

    .budget-card p {
        color: #cbd5e1;
        line-height: 1.62;
        margin: 0;
        overflow-wrap: anywhere;
    }

    .food-card {
        position: relative;
        overflow: hidden;
    }

    .food-card::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: radial-gradient(circle at 88% 10%, rgba(247, 213, 138, 0.13), transparent 28%);
    }

    .food-card h3 {
        position: relative;
        color: #fff7ed !important;
        font-size: 1.22rem;
    }

    .food-location,
    .food-map-keyword {
        position: relative;
        color: #aeb8ca !important;
    }

    .weather-card h3,
    .info-card h3,
    .warning-card h3 {
        color: #fff7ed !important;
    }

    .weather-day {
        border-top-color: rgba(247, 213, 138, 0.10) !important;
    }

    .weather-icon,
    .info-icon,
    .warning-icon {
        background: linear-gradient(135deg, #fff2bf, #f59e0b) !important;
        color: #17120a !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-actions-title),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.source-card-title) {
        border-radius: 28px !important;
        border: 1px solid rgba(247, 213, 138, 0.20) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03)),
            rgba(10, 16, 31, 0.64) !important;
        box-shadow: 0 22px 76px rgba(0,0,0,0.28) !important;
    }

    .result-actions-title,
    .source-card-title {
        color: #fff7ed;
        font-size: 1.25rem;
        font-weight: 900;
        margin: 0 0 0.35rem;
    }

    .source-card-text {
        color: #cbd5e1;
        line-height: 1.65;
        margin: 0.25rem 0;
    }

    .search-status-pill,
    .trust-card,
    .weather-fallback,
    .blessing-card {
        border-color: rgba(247, 213, 138, 0.18) !important;
        background:
            linear-gradient(145deg, rgba(247, 213, 138, 0.10), rgba(255,255,255,0.025)),
            rgba(10, 16, 31, 0.60) !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 0.72rem 0.68rem 3rem !important;
        }

        .top-nav {
            position: static;
            border-radius: 22px !important;
            margin-bottom: 0.9rem !important;
        }

        .hero.product-hero {
            border-radius: 26px;
            padding: 1rem;
        }

        .hero-layout,
        .bento-grid,
        .timeline-grid,
        .food-grid,
        .info-grid,
        .warning-grid,
        .weather-grid,
        .budget-grid,
        .segment-overview-grid,
        .trust-strip,
        .result-action-grid,
        .slot-meta-grid {
            grid-template-columns: minmax(0, 1fr) !important;
            width: 100% !important;
        }

        .hero h1,
        .hero-title {
            max-width: min(100%, 900px);
            margin: 0.55rem 0 1rem !important;
            font-size: clamp(2.375rem, 10.8vw, 2.875rem) !important;
            line-height: 1.12 !important;
            letter-spacing: 0 !important;
        }

        .hero-title-line {
            display: block;
        }

        .hero p {
            font-size: 0.96rem !important;
            line-height: 1.65 !important;
        }

        .hero-panel.product-preview {
            min-height: auto;
        }

        .preview-cover {
            min-height: 150px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-kicker) {
            border-radius: 24px !important;
        }

        .stTextArea textarea {
            min-height: 128px !important;
        }

        .cover-card {
            min-height: 300px !important;
            aspect-ratio: 4 / 3 !important;
            border-radius: 26px !important;
        }

        .cover-content h2 {
            font-size: clamp(2rem, 12vw, 3.3rem) !important;
        }

        .bento-card,
        .bento-card.large {
            grid-column: span 1 !important;
            min-height: auto !important;
        }

        .timeline-day::before {
            left: 1.55rem;
            top: 4.5rem;
        }

        .timeline-slot {
            grid-template-columns: 42px minmax(0, 1fr) !important;
            gap: 0.72rem !important;
        }

        .slot-icon {
            width: 36px !important;
            height: 36px !important;
            border-radius: 13px !important;
            font-size: 0.76rem !important;
        }

        .timeline-slot > div:nth-child(2) {
            padding: 0.72rem;
            border-radius: 17px;
        }

        .weather-day {
            grid-template-columns: 38px minmax(0, 1fr) !important;
        }

        .weather-meta span,
        .food-meta span {
            width: 100%;
        }

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 0 !important;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {
            width: 100% !important;
            white-space: normal !important;
        }
    }

    @media (max-width: 520px) {
        .hero-proof span {
            width: 100%;
        }

        .preview-step {
            grid-template-columns: 36px minmax(0, 1fr);
        }

        .preview-step b {
            width: 34px;
            height: 34px;
        }

        .section-heading {
            font-size: 1.45rem !important;
        }

        .cover-content {
            inset: auto 0.9rem 0.95rem 0.9rem !important;
        }
    }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)


def parse_chinese_number(number_text: str) -> int:
    """parse_chinese_number：把常见中文数字转换成整数。"""

    # chinese_number_map：保存中文数字到阿拉伯数字的对应关系。
    chinese_number_map = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    if number_text.isdigit():
        return int(number_text)

    if number_text == "十":
        return 10

    if number_text.startswith("十"):
        return 10 + chinese_number_map.get(number_text[-1], 0)

    if "十" in number_text:
        # parts：中文数字按“十”拆分后的十位和个位。
        parts = number_text.split("十")
        tens = chinese_number_map.get(parts[0], 1)
        ones = chinese_number_map.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones

    return chinese_number_map.get(number_text, DEFAULT_TRAVEL_DAYS)


def parse_chinese_amount(amount_text: str) -> float | None:
    """parse_chinese_amount：把中文预算金额转换成数字，例如“一万”转成 10000。"""

    # cleaned_amount：清理空格后的中文金额文本。
    cleaned_amount = amount_text.strip()
    if not cleaned_amount:
        return None

    if re.fullmatch(r"[0-9][0-9,]*(?:\.\d+)?", cleaned_amount):
        return float(cleaned_amount.replace(",", ""))

    # chinese_digit_map：中文数字字符和数值的对应关系。
    chinese_digit_map = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    if cleaned_amount.endswith("万"):
        # base_text：中文金额中“万”前面的数字部分。
        base_text = cleaned_amount[:-1]
        if not base_text:
            return 10000.0
        base_value = parse_chinese_number(base_text)
        return float(base_value * 10000)

    if cleaned_amount in chinese_digit_map:
        return float(chinese_digit_map[cleaned_amount])

    if "千" in cleaned_amount:
        # thousand_parts：中文金额按“千”拆分后的千位和余数。
        thousand_parts = cleaned_amount.split("千", 1)
        thousands = parse_chinese_number(thousand_parts[0] or "一") * 1000
        rest = parse_chinese_amount(thousand_parts[1]) if thousand_parts[1] else 0
        return float(thousands + (rest or 0))

    return None


def normalize_currency_unit(currency_text: str | None) -> str:
    """normalize_currency_unit：把用户输入的货币单位统一成标准货币代码。"""

    if not currency_text:
        return DEFAULT_BUDGET_CURRENCY

    # normalized_unit：统一大小写并去除空格后的货币单位。
    normalized_unit = currency_text.strip().upper()

    # currency_alias_map：常见货币表达和标准货币代码的对应关系。
    currency_alias_map = {
        "人民币": "CNY",
        "RMB": "CNY",
        "CNY": "CNY",
        "元": "CNY",
        "块": "CNY",
        "日元": "JPY",
        "日币": "JPY",
        "日圓": "JPY",
        "JPY": "JPY",
        "美元": "USD",
        "美金": "USD",
        "USD": "USD",
        "欧元": "EUR",
        "欧": "EUR",
        "EUR": "EUR",
        "韩元": "KRW",
        "韩币": "KRW",
        "KRW": "KRW",
    }

    return currency_alias_map.get(normalized_unit, DEFAULT_BUDGET_CURRENCY)


def get_currency_name(currency_code: str) -> str:
    """get_currency_name：把标准货币代码转换成中文显示名称。"""

    # currency_name_map：标准货币代码和中文名称的对应关系。
    currency_name_map = {
        "CNY": "人民币",
        "JPY": "日元",
        "USD": "美元",
        "EUR": "欧元",
        "KRW": "韩元",
    }

    return currency_name_map.get(currency_code, currency_code)


def format_budget_amount(amount: float) -> str:
    """format_budget_amount：把预算金额格式化为适合页面展示的文本。"""

    # numeric_amount：兼容 int 和 float 的预算金额数值。
    numeric_amount = float(amount)

    if numeric_amount.is_integer():
        return f"{int(numeric_amount):,}"

    return f"{numeric_amount:,.2f}".rstrip("0").rstrip(".")


def parse_budget_info(cleaned_input: str, budget_level: str) -> dict:
    """parse_budget_info：识别用户输入中的预算金额和货币单位。"""

    # budget_pattern：匹配“预算5000”“预算一万”“预算10万日元”“预算800 USD”等表达。
    budget_pattern = re.compile(
        r"(?:预算|总预算|花费|费用)\s*"
        r"(?:约|大概|大约|控制在|不超过|不超|以内|左右|是|为|:|：)?\s*"
        r"([0-9][0-9,]*(?:\.\d+)?|[零一二两三四五六七八九十百千万]+)\s*"
        r"(万)?\s*"
        r"(人民币|日元|日币|日圓|美元|美金|欧元|欧|韩元|韩币|USD|EUR|JPY|KRW|RMB|CNY|元|块)?",
        re.IGNORECASE,
    )

    # budget_match：预算金额匹配结果。
    budget_match = budget_pattern.search(cleaned_input)
    if not budget_match:
        return {
            "amount": None,
            "currency": None,
            "currency_name": None,
            "display": budget_level,
            "level": budget_level,
            "has_explicit_amount": False,
        }

    # amount_text：预算金额文本。
    amount_text = budget_match.group(1)

    # amount_value：预算金额数值。
    amount_value = parse_chinese_amount(amount_text)
    if amount_value is None:
        amount_value = float(amount_text.replace(",", ""))

    if budget_match.group(2) and "万" not in amount_text:
        amount_value *= 10000

    # currency_code：标准货币代码；用户没写单位时默认 CNY。
    currency_code = normalize_currency_unit(budget_match.group(3))

    # currency_name：中文货币名称。
    currency_name = get_currency_name(currency_code)

    # budget_display：页面和提示词展示的预算文本。
    budget_display = f"{format_budget_amount(amount_value)} {currency_name} ({currency_code})"

    # normalized_amount：整数金额保存为 int，便于 Debug 区显示 10000 而不是 10000.0。
    normalized_amount = int(amount_value) if float(amount_value).is_integer() else amount_value

    return {
        "amount": normalized_amount,
        "currency": currency_code,
        "currency_name": currency_name,
        "display": budget_display,
        "level": budget_level,
        "has_explicit_amount": True,
    }


def infer_destination_currency(destination: str) -> str | None:
    """infer_destination_currency：根据目的地粗略推断当地常用货币。"""

    # destination_currency_keywords：国外目的地关键词和当地货币代码。
    destination_currency_keywords = {
        "JPY": ["日本", "东京", "大阪", "京都", "北海道", "冲绳", "奈良", "福冈", "名古屋", "札幌", "箱根"],
        "KRW": ["韩国", "首尔", "釜山", "济州"],
        "EUR": [
            "欧洲",
            "法国",
            "巴黎",
            "意大利",
            "罗马",
            "米兰",
            "德国",
            "柏林",
            "西班牙",
            "巴塞罗那",
            "荷兰",
            "阿姆斯特丹",
            "葡萄牙",
            "希腊",
            "瑞士",
        ],
        "USD": ["美国", "纽约", "洛杉矶", "旧金山", "西雅图", "夏威夷"],
    }

    for currency_code, keyword_list in destination_currency_keywords.items():
        if any(keyword in destination for keyword in keyword_list):
            return currency_code

    return None


def build_exchange_hint(parsed_request: dict) -> str | None:
    """build_exchange_hint：为国外目的地生成粗略换算提示。"""

    # budget_amount：用户输入的预算金额。
    budget_amount = parsed_request.get("budget_amount")
    if not budget_amount:
        return None

    # destination_currency：根据目的地推断出的当地货币。
    destination_currency = infer_destination_currency(parsed_request["destination"])
    if not destination_currency:
        return None

    # source_currency：用户输入预算的货币代码。
    source_currency = parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY

    # cny_to_currency_rate：人民币到其他货币的粗略换算比例。
    cny_to_currency_rate = {
        "JPY": 20.0,
        "KRW": 190.0,
        "EUR": 0.13,
        "USD": 0.14,
    }

    # currency_to_cny_rate：其他货币到人民币的粗略换算比例。
    currency_to_cny_rate = {
        "JPY": 0.05,
        "KRW": 0.0053,
        "EUR": 7.8,
        "USD": 7.2,
        "CNY": 1.0,
    }

    if source_currency == DEFAULT_BUDGET_CURRENCY and destination_currency in cny_to_currency_rate:
        # converted_amount：人民币预算换算成目的地当地货币的粗略金额。
        converted_amount = budget_amount * cny_to_currency_rate[destination_currency]
        return (
            f"粗略换算：{format_budget_amount(budget_amount)} 人民币约 "
            f"{format_budget_amount(converted_amount)} {get_currency_name(destination_currency)}"
            "，汇率仅供参考，请以出行前实际汇率为准。"
        )

    if source_currency != DEFAULT_BUDGET_CURRENCY and source_currency in currency_to_cny_rate:
        # converted_amount：外币预算换算成人民币的粗略金额。
        converted_amount = budget_amount * currency_to_cny_rate[source_currency]
        return (
            f"粗略换算：{format_budget_amount(budget_amount)} {get_currency_name(source_currency)}约 "
            f"{format_budget_amount(converted_amount)} 人民币"
            "，汇率仅供参考，请以出行前实际汇率为准。"
        )

    return None


def extract_destination(cleaned_input: str) -> str:
    """extract_destination：从用户输入中优先提取明确目的地。"""

    # destination_patterns：从强到弱排列的目的地匹配规则。
    destination_patterns = [
        r"(?:^|[，。,\s])([一-龥A-Za-z]{2,20})\s*(?:[0-9一二两三四五六七八九十]+)\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行)",
        r"想去(?!看看|看一看|看|尝|尝一尝|吃|逛)([一-龥A-Za-z]{2,20}?)(?:旅游|旅行|自由行|游|度假|玩|看|赏|吃|逛|，|。|,|\s|$)",
        r"去(?!看看|看一看|看|尝|尝一尝|吃|逛)([一-龥A-Za-z]{2,20}?)(?:旅游|旅行|自由行|游|度假|玩|看|赏|吃|逛|，|。|,|\s|$)",
        r"(?:^|[，。,\s])([一-龥A-Za-z]{2,20}?)\s*(?:旅游|旅行|自由行|度假|游)",
        r"目的地[:：]\s*([一-龥A-Za-z]{2,20})",
    ]

    # invalid_destination_words：不应被当成目的地的动作词或泛词。
    invalid_destination_words = {
        "看看",
        "看一看",
        "看",
        "尝",
        "尝一尝",
        "美食",
        "夜景",
        "其他景点",
        "景点",
    }

    for pattern in destination_patterns:
        match = re.search(pattern, cleaned_input)
        if not match:
            continue

        # destination：当前规则识别出的目的地。
        destination = match.group(1).strip()
        destination = re.sub(r"^(?:我|我们|本人)?(?:想去|要去|计划去|打算去|去)", "", destination).strip()
        if destination and destination not in invalid_destination_words:
            return destination

    return DEFAULT_DESTINATION


def extract_trip_days(cleaned_input: str) -> tuple[int, int]:
    """extract_trip_days：识别“7日游”“7天”“七日”等明确天数。"""

    # days_patterns：可识别的天数表达。
    days_patterns = [
        r"([0-9一二两三四五六七八九十]+)\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行)",
        r"([0-9一二两三四五六七八九十]+)\s*(?:天|日)(?!元|币)",
    ]

    for pattern in days_patterns:
        match = re.search(pattern, cleaned_input)
        if match:
            days = max(1, parse_chinese_number(match.group(1)))
            return days, max(0, days - 1)

    return DEFAULT_TRAVEL_DAYS, DEFAULT_TRAVEL_NIGHTS


def get_known_destination_names() -> list[str]:
    """get_known_destination_names：返回用于多目的地识别的常见目的地名称。"""

    return [
        "内蒙古",
        "黑龙江",
        "张家界",
        "九寨沟",
        "西双版纳",
        "香格里拉",
        "南京",
        "江西",
        "南昌",
        "景德镇",
        "婺源",
        "庐山",
        "上饶",
        "赣州",
        "上海",
        "苏州",
        "杭州",
        "香港",
        "澳门",
        "东京",
        "京都",
        "大阪",
        "广州",
        "深圳",
        "北京",
        "成都",
        "重庆",
        "西安",
        "云南",
        "大理",
        "丽江",
        "昆明",
        "福建",
        "厦门",
        "泉州",
        "福州",
        "广东",
        "海南",
        "三亚",
        "青岛",
        "长沙",
        "武汉",
        "天津",
        "首尔",
        "釜山",
        "济州",
        "日本",
        "韩国",
        "欧洲",
        "法国",
        "巴黎",
        "意大利",
        "罗马",
        "美国",
        "纽约",
        "洛杉矶",
    ]


def get_province_route_note(destination: str) -> str:
    """get_province_route_note：为省份或大区域目的地生成具体城市路线提示。"""

    # province_route_map：省份或大区域到经典城市组合的映射。
    province_route_map = {
        "江西": "你输入的是省份，系统为你选择较经典的江西路线：南昌、景德镇、婺源、庐山、上饶，可根据偏好调整。",
        "云南": "你输入的是省份，系统会优先按昆明、大理、丽江、香格里拉等经典路线规划，可根据偏好调整。",
        "福建": "你输入的是省份，系统会优先按厦门、泉州、福州或武夷山等经典路线规划，可根据偏好调整。",
        "广东": "你输入的是省份，系统会优先按广州、深圳、珠海或潮汕等经典路线规划，可根据偏好调整。",
        "海南": "你输入的是省份，系统会优先按海口、三亚、万宁等经典路线规划，可根据偏好调整。",
        "日本": "你输入的是国家，系统会优先按东京、京都、大阪等经典路线规划，可根据偏好调整。",
        "韩国": "你输入的是国家，系统会优先按首尔、釜山、济州等经典路线规划，可根据偏好调整。",
        "欧洲": "你输入的是大区域，系统会选择适合天数的城市组合，并明确说明推断依据。",
    }

    return province_route_map.get(destination, "")


def clean_destination_candidate(destination_text: str) -> str:
    """clean_destination_candidate：清理多目的地正则中捕获的目的地候选词。"""

    # cleaned_destination：去掉连接词、动词和标点后的目的地。
    cleaned_destination = destination_text.strip()
    cleaned_destination = re.sub(r"^[，。；、,\s+]+", "", cleaned_destination)
    cleaned_destination = re.sub(
        r"^(?:然后再去|然后去|接着去|随后去|之后去|先去|再去|然后再|然后|接着|随后|之后|先|再)",
        "",
        cleaned_destination,
    )
    cleaned_destination = re.sub(r"^(?:去|到|前往|游玩|玩|旅游|旅行|自由行)", "", cleaned_destination)
    cleaned_destination = re.sub(r"(?:游玩|玩|旅游|旅行|自由行|游)$", "", cleaned_destination)
    return cleaned_destination.strip(" ，。,.;；、+")


def is_valid_destination_candidate(destination_text: str) -> bool:
    """is_valid_destination_candidate：过滤不应当作为目的地的词。"""

    # invalid_destination_words：动作词、偏好词和泛词，不能当作目的地。
    invalid_destination_words = {
        "然后",
        "然后再",
        "再去",
        "喜欢",
        "历史",
        "文化",
        "历史文化",
        "美食",
        "夜景",
        "预算",
        "交通",
        "住宿",
        "餐饮",
        "景点",
        "其他景点",
    }

    if not destination_text or destination_text in invalid_destination_words:
        return False

    if len(destination_text) < 2 or len(destination_text) > 12:
        return False

    return bool(re.search(r"[一-龥A-Za-z]", destination_text))


def find_known_destination_matches(cleaned_input: str) -> list[dict]:
    """find_known_destination_matches：在用户输入中查找已知目的地并按位置去重。"""

    # known_destinations：常见目的地词表，按长度降序避免长地名被短词截断。
    known_destinations = sorted(get_known_destination_names(), key=len, reverse=True)

    # destination_matches：用户输入中出现的目的地及位置。
    destination_matches = []
    for destination in known_destinations:
        for match in re.finditer(re.escape(destination), cleaned_input):
            destination_matches.append({"destination": destination, "start": match.start(), "end": match.end()})

    destination_matches.sort(key=lambda item: item["start"])

    # deduped_matches：按文本位置去重，避免同一位置重复匹配。
    deduped_matches = []
    occupied_ranges = []
    for item in destination_matches:
        if any(item["start"] >= start and item["end"] <= end for start, end in occupied_ranges):
            continue
        deduped_matches.append(item)
        occupied_ranges.append((item["start"], item["end"]))

    # unique_matches：同一个目的地多次出现时只保留第一次。
    unique_matches = []
    seen_destinations = set()
    for item in deduped_matches:
        destination = item["destination"]
        if destination in seen_destinations:
            continue
        unique_matches.append(item)
        seen_destinations.add(destination)

    return unique_matches


def get_weather_reference_city(destination: str, travel_json: dict | None = None) -> tuple[str, str]:
    """get_weather_reference_city：把省份或大区域目的地转换为适合查询天气的主要城市。"""

    # destination_city_map：省份、国家或大区域到天气参考城市的映射。
    destination_city_map = {
        "江西": "南昌",
        "云南": "昆明",
        "福建": "厦门",
        "广东": "广州",
        "海南": "海口",
        "日本": "东京",
        "韩国": "首尔",
        "欧洲": "巴黎",
    }

    # city_from_map：从固定映射中得到的参考城市。
    city_from_map = destination_city_map.get(destination)
    if city_from_map:
        return city_from_map, f"{destination}主要城市天气参考：{city_from_map}"

    return destination, destination


def geocode_destination(destination: str) -> dict | None:
    """geocode_destination：使用 Open-Meteo Geocoding API 把目的地转换为经纬度。"""

    if not destination:
        return None

    try:
        # response：Open-Meteo 地理编码接口响应。
        response = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params={
                "name": destination,
                "count": 1,
                "language": "zh",
                "format": "json",
            },
            timeout=8,
        )
        response.raise_for_status()
        # geocode_data：地理编码 JSON 数据。
        geocode_data = response.json()
    except Exception:
        return None

    # result_list：Open-Meteo 返回的候选地点列表。
    result_list = geocode_data.get("results", [])
    if not result_list:
        return None

    # first_result：最匹配的地点。
    first_result = result_list[0]
    latitude = first_result.get("latitude")
    longitude = first_result.get("longitude")
    if latitude is None or longitude is None:
        return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "name": first_result.get("name", destination),
        "country": first_result.get("country", ""),
        "timezone": first_result.get("timezone", "auto"),
    }


def fetch_weather_forecast(latitude: float, longitude: float) -> dict | None:
    """fetch_weather_forecast：使用 Open-Meteo Forecast API 查询未来天气。"""

    try:
        # response：Open-Meteo 天气预报接口响应。
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": ",".join(
                    [
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                        "wind_speed_10m_max",
                    ]
                ),
                "hourly": "relative_humidity_2m",
                "timezone": "auto",
                "forecast_days": WEATHER_FORECAST_DAYS,
            },
            timeout=8,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def map_weather_code(weather_code: int | None) -> tuple[str, str]:
    """map_weather_code：把 Open-Meteo 天气代码转换成中文天气状态和图标。"""

    if weather_code is None:
        return "天气待确认", "🌦️"

    # weather_code_map：WMO 天气代码到中文状态的映射。
    weather_code_map = {
        0: ("晴", "☀️"),
        1: ("大致晴朗", "🌤️"),
        2: ("多云", "🌤️"),
        3: ("阴", "☁️"),
        45: ("有雾", "☁️"),
        48: ("雾凇", "☁️"),
        51: ("小毛毛雨", "🌧️"),
        53: ("毛毛雨", "🌧️"),
        55: ("较强毛毛雨", "🌧️"),
        56: ("冻毛毛雨", "🌧️"),
        57: ("强冻毛毛雨", "🌧️"),
        61: ("小雨", "🌧️"),
        63: ("中雨", "🌧️"),
        65: ("大雨", "🌧️"),
        66: ("冻雨", "🌧️"),
        67: ("强冻雨", "🌧️"),
        71: ("小雪", "🌨️"),
        73: ("中雪", "🌨️"),
        75: ("大雪", "🌨️"),
        77: ("雪粒", "🌨️"),
        80: ("阵雨", "🌧️"),
        81: ("较强阵雨", "🌧️"),
        82: ("强阵雨", "🌧️"),
        85: ("阵雪", "🌨️"),
        86: ("强阵雪", "🌨️"),
        95: ("雷雨", "⛈️"),
        96: ("雷雨伴冰雹", "⛈️"),
        99: ("强雷雨伴冰雹", "⛈️"),
    }

    return weather_code_map.get(weather_code, ("天气待确认", "🌦️"))


def format_weather_date(date_text: str) -> str:
    """format_weather_date：把 YYYY-MM-DD 日期转换成中文短日期。"""

    try:
        # date_value：解析后的日期对象。
        date_value = datetime.fromisoformat(date_text)
        return f"{date_value.month}月{date_value.day}日"
    except ValueError:
        return date_text


def format_weather_value(value: float | int | None, suffix: str) -> str:
    """format_weather_value：格式化天气数值，缺失时不编造。"""

    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{round(value)}{suffix}"
    return f"{value}{suffix}"


def calculate_daily_humidity(weather_data: dict) -> dict[str, int | None]:
    """calculate_daily_humidity：从小时级湿度数据计算每天平均湿度。"""

    # hourly_data：Open-Meteo 小时级天气数据。
    hourly_data = weather_data.get("hourly", {})

    # time_list/humidity_list：小时和相对湿度列表。
    time_list = hourly_data.get("time", [])
    humidity_list = hourly_data.get("relative_humidity_2m", [])

    # humidity_bucket：按日期收集的湿度值。
    humidity_bucket: dict[str, list[float]] = {}
    for time_text, humidity_value in zip(time_list, humidity_list):
        if humidity_value is None or not time_text:
            continue
        date_key = str(time_text).split("T", 1)[0]
        humidity_bucket.setdefault(date_key, []).append(float(humidity_value))

    # humidity_by_date：每天平均湿度。
    humidity_by_date: dict[str, int | None] = {}
    for date_key, values in humidity_bucket.items():
        humidity_by_date[date_key] = round(sum(values) / len(values)) if values else None

    return humidity_by_date


def build_single_weather_advice(weather_item: dict) -> str:
    """build_single_weather_advice：根据单日天气生成携带和出行提醒。"""

    # weather_text：中文天气状态。
    weather_text = str(weather_item.get("weather_text", ""))

    # precipitation_probability：降水概率。
    precipitation_probability = weather_item.get("precipitation_probability")

    # temperature_max/temperature_min/wind_speed：温度和风速。
    temperature_max = weather_item.get("temperature_max")
    temperature_min = weather_item.get("temperature_min")
    wind_speed = weather_item.get("wind_speed")

    # advice_parts：多条提醒合并后的建议。
    advice_parts = []

    # rainy_words：用于判断雨天的关键词。
    rainy_words = ["雨", "小雨", "中雨", "大雨", "阵雨", "雷雨"]
    is_rainy = precipitation_probability is not None and precipitation_probability >= 50
    if is_rainy or any(word in weather_text for word in rainy_words):
        advice_parts.append("可能下雨，建议携带雨伞、防水袋和防滑鞋，户外景点注意地面湿滑。☔")

    if temperature_max is not None and temperature_max >= 30:
        advice_parts.append("气温偏高，注意防晒、补水，尽量避开中午长时间暴晒。")

    if temperature_min is not None and temperature_min <= 10:
        advice_parts.append("早晚偏冷，建议带外套，注意昼夜温差。")

    if "晴" in weather_text:
        advice_parts.append("晴天适合户外游玩，记得准备防晒、墨镜和水。☀️")

    if "多云" in weather_text or "阴" in weather_text:
        advice_parts.append("适合步行游玩，但天气变化仍建议出门前查看实时天气。")

    if wind_speed is not None and wind_speed >= 38:
        advice_parts.append("风力偏大，注意保暖，高处观景或乘船行程要留意安全。🌬️")

    if not advice_parts:
        advice_parts.append("天气信息仅供参考，请出行前查看天气 App。")

    return " ".join(advice_parts)


def build_weather_advice(weather_data: dict) -> list[dict]:
    """build_weather_advice：把 Open-Meteo 天气数据整理成每日天气卡片。"""

    if not isinstance(weather_data, dict):
        return []

    # daily_data：Open-Meteo 每日天气数据。
    daily_data = weather_data.get("daily", {})
    date_list = daily_data.get("time", [])
    if not date_list:
        return []

    # humidity_by_date：按日期计算出的平均湿度。
    humidity_by_date = calculate_daily_humidity(weather_data)

    # weather_items：每日天气卡片数据。
    weather_items = []
    for index, date_text in enumerate(date_list[:WEATHER_FORECAST_DAYS]):
        # weather_code：Open-Meteo WMO 天气代码。
        weather_code_list = daily_data.get("weather_code", [])
        weather_code = weather_code_list[index] if index < len(weather_code_list) else None
        weather_text, weather_icon = map_weather_code(weather_code)

        # temperature_max/min：最高和最低温度。
        temperature_max_list = daily_data.get("temperature_2m_max", [])
        temperature_min_list = daily_data.get("temperature_2m_min", [])
        temperature_max = temperature_max_list[index] if index < len(temperature_max_list) else None
        temperature_min = temperature_min_list[index] if index < len(temperature_min_list) else None

        # precipitation_probability：最大降水概率。
        precipitation_list = daily_data.get("precipitation_probability_max", [])
        precipitation_probability = precipitation_list[index] if index < len(precipitation_list) else None

        # wind_speed：最大风速。
        wind_speed_list = daily_data.get("wind_speed_10m_max", [])
        wind_speed = wind_speed_list[index] if index < len(wind_speed_list) else None
        if wind_speed is not None and wind_speed >= 38:
            weather_icon = "🌬️"
            weather_text = f"{weather_text}，风力偏大"

        # weather_item：单日天气展示数据。
        weather_item = {
            "date": format_weather_date(str(date_text)),
            "raw_date": str(date_text),
            "weather_text": weather_text,
            "weather_icon": weather_icon,
            "temperature_max": temperature_max,
            "temperature_min": temperature_min,
            "humidity": humidity_by_date.get(str(date_text)),
            "precipitation_probability": precipitation_probability,
            "wind_speed": wind_speed,
            "will_rain": bool(
                (precipitation_probability is not None and precipitation_probability >= 50)
                or any(word in weather_text for word in ["雨", "阵雨", "雷雨"])
            ),
        }
        weather_item["advice"] = build_single_weather_advice(weather_item)
        weather_items.append(weather_item)

    return weather_items


def build_weather_cards(parsed_request: dict, travel_json: dict | None = None) -> list[dict]:
    """build_weather_cards：为单目的地或多目的地构建天气模块卡片数据。"""

    # trip_segments：目的地分段列表。
    trip_segments = parsed_request.get("trip_segments") or [{"destination": parsed_request["destination"]}]

    # weather_cards：所有目的地天气卡片。
    weather_cards = []
    seen_weather_destinations = set()
    for segment in trip_segments:
        # destination：当前天气卡片对应的目的地。
        destination = str(segment.get("destination", "")).strip()
        if not destination or destination in seen_weather_destinations:
            continue
        seen_weather_destinations.add(destination)

        # query_city/display_destination：用于查询天气的城市和页面展示标题。
        query_city, display_destination = get_weather_reference_city(destination, travel_json)

        # geocode_result：目的地经纬度。
        geocode_result = geocode_destination(query_city)
        if not geocode_result:
            weather_cards.append(
                {
                    "destination": display_destination,
                    "query_city": query_city,
                    "error": "天气信息暂时无法获取，请出行前查看天气 App。",
                    "days": [],
                }
            )
            continue

        # forecast_data：Open-Meteo 天气预报数据。
        forecast_data = fetch_weather_forecast(geocode_result["latitude"], geocode_result["longitude"])
        if not forecast_data:
            weather_cards.append(
                {
                    "destination": display_destination,
                    "query_city": query_city,
                    "error": "天气信息暂时无法获取，请出行前查看天气 App。",
                    "days": [],
                }
            )
            continue

        # day_weather_items：每日天气卡片数据。
        day_weather_items = build_weather_advice(forecast_data)
        if not day_weather_items:
            weather_cards.append(
                {
                    "destination": display_destination,
                    "query_city": query_city,
                    "error": "天气信息暂时无法获取，请出行前查看天气 App。",
                    "days": [],
                }
            )
            continue

        weather_cards.append(
            {
                "destination": display_destination,
                "query_city": query_city,
                "error": "",
                "days": day_weather_items,
            }
        )

    return weather_cards


def build_weather_markdown(weather_cards: list[dict] | None) -> str:
    """build_weather_markdown：把天气卡片转换成可复制 Markdown。"""

    if not weather_cards:
        return "## 天气与出行提醒 🌦️\n天气信息暂时无法获取，请出行前查看天气 App。"

    # markdown_lines：天气 Markdown 行。
    markdown_lines = ["## 天气与出行提醒 🌦️"]
    for weather_card in weather_cards:
        destination = str(weather_card.get("destination", "目的地"))
        markdown_lines.append(f"### {destination}｜未来 {WEATHER_FORECAST_DAYS} 天天气")
        if weather_card.get("error"):
            markdown_lines.append(str(weather_card["error"]))
            continue

        for day_weather in weather_card.get("days", []):
            will_rain_text = "可能下雨" if day_weather.get("will_rain") else "降雨风险较低"
            markdown_lines.extend(
                [
                    f"#### {day_weather.get('date', '')}",
                    f"- 天气：{day_weather.get('weather_text', '天气待确认')}",
                    (
                        f"- 温度：{format_weather_value(day_weather.get('temperature_min'), '°C')} - "
                        f"{format_weather_value(day_weather.get('temperature_max'), '°C')}"
                    ),
                    f"- 湿度：{format_weather_value(day_weather.get('humidity'), '%')}",
                    f"- 降水概率：{format_weather_value(day_weather.get('precipitation_probability'), '%')}",
                    f"- 是否可能下雨：{will_rain_text}",
                    f"- 建议：{day_weather.get('advice', '天气信息仅供参考，请出行前查看天气 App。')}",
                ]
            )

    return "\n".join(markdown_lines)


def remove_markdown_sections(markdown_text: str, heading_names: set[str]) -> str:
    """remove_markdown_sections：移除指定二级标题章节，避免 Markdown 原文重复。"""

    # markdown_lines：原始 Markdown 行。
    markdown_lines = markdown_text.splitlines()

    # kept_lines：保留下来的 Markdown 行。
    kept_lines = []
    skipping = False
    for line in markdown_lines:
        # heading_match：匹配二级标题。
        heading_match = re.match(r"^##\s+(.+?)\s*$", line)
        if heading_match:
            heading_text = heading_match.group(1).strip()
            skipping = heading_text in heading_names
            if skipping:
                continue
        if not skipping:
            kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def append_weather_and_blessing_to_markdown(markdown_text: str, weather_cards: list[dict] | None, generated_at: str) -> str:
    """append_weather_and_blessing_to_markdown：把天气、更新时间和祝福语追加到 Markdown 末尾。"""

    # cleaned_markdown：移除模型可能生成的旧信息区，避免重复和技术词外露。
    cleaned_markdown = remove_markdown_sections(
        markdown_text,
        {
            "天气与出行提醒",
            "天气与出行提醒 🌦️",
            "信息来源与更新时间",
            "信息与更新时间",
            "旅行祝福语",
        },
    )

    # source_markdown：用户友好的信息与更新时间。
    source_markdown = "\n".join(
        [
            "## 信息与更新时间",
            "本攻略由 AI 根据你的输入和当前可用信息整理生成。",
            f"更新时间：{generated_at}",
            "门票、预约、开放时间、交通政策和天气情况可能变化，请出行前以官方渠道和天气 App 为准。",
        ]
    )

    # blessing_markdown：旅行祝福语。
    blessing_markdown = (
        "## 旅行祝福语\n"
        "祝你这次旅行顺利又开心。记得提前确认天气、门票和交通安排，慢慢走、好好看，"
        "把喜欢的风景都装进记忆里。祝你旅途愉快呀～ 🌿✨🧳"
    )

    return "\n\n".join([cleaned_markdown, build_weather_markdown(weather_cards), source_markdown, blessing_markdown]).strip()


def extract_days_near_destination(text_after_destination: str) -> int | None:
    """extract_days_near_destination：在目的地后方的小片段中提取天数。"""

    # day_match：匹配“玩2天”“游玩3天”“4日”等表达。
    day_match = re.search(
        r"(?:游玩|玩|旅游|旅行|自由行|游)?\s*([0-9一二两三四五六七八九十]+)\s*(?:天|日)(?!元|币)",
        text_after_destination,
    )
    if not day_match:
        return None

    return max(1, parse_chinese_number(day_match.group(1)))


def build_trip_segment(destination: str, days: int, days_inferred: bool, preferences: list[str]) -> dict:
    """build_trip_segment：构造单个目的地分段对象。"""

    # note：省份、大区域或默认天数提示。
    note = get_province_route_note(destination)
    if days_inferred:
        inferred_note = f"用户未说明{destination}游玩天数，系统默认按{DEFAULT_TRAVEL_DAYS}天规划。"
        note = f"{inferred_note} {note}".strip()

    return {
        "destination": destination,
        "days": days,
        "nights": max(0, days - 1),
        "days_inferred": days_inferred,
        "preferences": preferences,
        "note": note,
    }


def parse_multi_destination_input(cleaned_input: str, preferences: list[str]) -> list[dict]:
    """parse_multi_destination_input：确定性识别多目的地和每段天数。"""

    # known_matches：先用已知目的地词表扫描，支持后续目的地没有写天数的情况。
    known_matches = find_known_destination_matches(cleaned_input)
    if len(known_matches) >= 2:
        # segments：按用户输入顺序构造的多目的地分段。
        segments = []
        for index, item in enumerate(known_matches):
            next_start = known_matches[index + 1]["start"] if index + 1 < len(known_matches) else len(cleaned_input)
            segment_text = cleaned_input[item["end"] : next_start]
            days = extract_days_near_destination(segment_text)
            days_inferred = days is None
            if days_inferred:
                days = DEFAULT_TRAVEL_DAYS
            segments.append(build_trip_segment(item["destination"], days, days_inferred, preferences))
        return segments

    # destination_day_pattern：兜底识别“目的地 + 天数”表达，支持连续书写如“上海2天苏州1天杭州3天”。
    destination_day_pattern = re.compile(
        r"([一-龥A-Za-z]{2,20}?)(?:游玩|玩|旅游|旅行|自由行|游)?\s*"
        r"([0-9一二两三四五六七八九十]+)\s*(?:天|日)(?!元|币)"
    )

    # regex_segments：从“目的地+天数”表达中识别出的分段。
    regex_segments = []
    seen_destinations = set()
    for match in destination_day_pattern.finditer(cleaned_input):
        destination = clean_destination_candidate(match.group(1))
        if not is_valid_destination_candidate(destination) or destination in seen_destinations:
            continue

        days = max(1, parse_chinese_number(match.group(2)))
        regex_segments.append(build_trip_segment(destination, days, False, preferences))
        seen_destinations.add(destination)

    if len(regex_segments) >= 2:
        return regex_segments

    return []


def extract_trip_segments(cleaned_input: str, preferences: list[str]) -> list[dict]:
    """extract_trip_segments：识别单目的地或多目的地分段行程。"""

    # multi_destination_segments：确定性多目的地解析结果，优先级最高。
    multi_destination_segments = parse_multi_destination_input(cleaned_input, preferences)
    if multi_destination_segments:
        return multi_destination_segments

    # deduped_matches：已知目的地匹配结果。
    deduped_matches = find_known_destination_matches(cleaned_input)

    if not deduped_matches:
        destination = extract_destination(cleaned_input)
        days, nights = extract_trip_days(cleaned_input)
        return [build_trip_segment(destination, days, False, preferences)]

    # segments：最终分段行程列表。
    segments = []
    for index, item in enumerate(deduped_matches):
        next_start = deduped_matches[index + 1]["start"] if index + 1 < len(deduped_matches) else len(cleaned_input)
        segment_text = cleaned_input[item["end"] : next_start]
        days = extract_days_near_destination(segment_text)
        days_inferred = days is None
        if days_inferred:
            days = DEFAULT_TRAVEL_DAYS

        destination = item["destination"]
        segments.append(build_trip_segment(destination, days, days_inferred, preferences))

    # 如果只识别出一个目的地，沿用全局天数解析，避免“杭州7日游”被默认覆盖。
    if len(segments) == 1:
        days, nights = extract_trip_days(cleaned_input)
        explicit_days_found = bool(re.search(r"[0-9一二两三四五六七八九十]+\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行|天|日)", cleaned_input))
        segments[0]["days"] = days
        segments[0]["nights"] = nights
        segments[0]["days_inferred"] = not explicit_days_found
        if segments[0]["days_inferred"]:
            inferred_note = f"用户未说明{segments[0]['destination']}游玩天数，系统默认按{DEFAULT_TRAVEL_DAYS}天规划。"
            segments[0]["note"] = f"{inferred_note} {get_province_route_note(segments[0]['destination'])}".strip()

    return segments


def infer_budget_level(cleaned_input: str) -> str:
    """infer_budget_level：识别用户输入中的预算风格档位。"""

    if re.search(r"穷游|省钱|低预算|便宜|学生党", cleaned_input):
        return "经济预算"

    if re.search(r"舒适一点|舒适|舒服|品质|高端|豪华|不差钱|预算充足", cleaned_input):
        return "舒适预算"

    return DEFAULT_BUDGET_LEVEL


def extract_preferences(cleaned_input: str) -> list[str]:
    """extract_preferences：从用户输入中提取旅行偏好、景点和体验主题。"""

    # preference_keywords：可识别的旅行偏好关键词。
    preference_keywords = [
        "历史文化",
        "西湖",
        "灵隐寺",
        "美食",
        "夜景",
        "自然",
        "动漫",
        "购物",
        "历史",
        "拍照",
        "文化",
        "博物馆",
        "亲子",
        "海边",
        "徒步",
        "温泉",
        "咖啡",
        "艺术",
    ]

    # preferences：从用户输入中识别出的偏好列表。
    preferences = []
    for keyword in preference_keywords:
        if keyword in {"历史", "文化"} and "历史文化" in preferences:
            continue
        if keyword in cleaned_input and keyword not in preferences:
            preferences.append(keyword)

    if not preferences:
        preferences = ["美食", "拍照"]

    return preferences


def extract_fact_check_spots(parsed_request: dict) -> list[str]:
    """extract_fact_check_spots：提取需要联网校验门票、预约和开放时间的景点。"""

    # generic_preferences：不适合作为具体景点搜索的泛偏好。
    generic_preferences = {
        "美食",
        "夜景",
        "自然",
        "动漫",
        "购物",
        "历史",
        "拍照",
        "文化",
        "博物馆",
        "亲子",
        "海边",
        "徒步",
        "温泉",
        "咖啡",
        "艺术",
    }

    # spot_list：从偏好里筛出的具体景点。
    spot_list = [item for item in parsed_request["preferences"] if item not in generic_preferences]

    if not spot_list:
        spot_list = ["主要景点"]

    return spot_list[:4]


def build_fact_search_queries(parsed_request: dict) -> list[str]:
    """build_fact_search_queries：生成省额度的合并搜索查询，避免每个景点单独搜索。"""

    # destination：目的地名称，多目的地时合并为一个省额度查询。
    destination = " ".join(parsed_request.get("destinations") or [parsed_request["destination"]])

    # spot_list：需要查询的景点列表。
    spot_list = extract_fact_check_spots(parsed_request)

    # important_spots：用户明确提到的重点景点，最多放入 4 个，避免 query 过长。
    important_spots = [spot for spot in spot_list if spot != "主要景点"][:4]

    # spot_text：重点景点文本，没有明确景点时使用热门景点兜底。
    spot_text = " ".join(important_spots) if important_spots else "热门景点"

    # query_list：省额度模式下的合并查询列表；默认只会执行第一条。
    query_list = [
        f"{destination} {spot_text} 旅游 景点 门票 预约 开放时间 最新 交通 政策",
        f"{destination} 官方 旅游 景区规则 门票 预约 开放时间 最新政策",
    ]

    return query_list


def get_tavily_api_key() -> str | None:
    """get_tavily_api_key：读取 Tavily API Key，并忽略示例占位值。"""

    # tavily_api_key：从 .env、环境变量或 Streamlit secrets 读取的 Tavily API Key。
    tavily_api_key = get_config_value("TAVILY_API_KEY", "").strip()
    if not tavily_api_key or tavily_api_key.startswith("tvly-your"):
        return None

    return tavily_api_key


def get_deepseek_api_key() -> str | None:
    """get_deepseek_api_key：读取 DeepSeek API Key，并忽略示例占位值。"""

    # deepseek_api_key：从 .env、环境变量或 Streamlit secrets 读取的 DeepSeek API Key。
    deepseek_api_key = get_config_value("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_api_key or deepseek_api_key.startswith("sk-your"):
        return None

    return deepseek_api_key


def get_deepseek_model_name() -> str:
    """get_deepseek_model_name：读取 DeepSeek 模型名，未配置时使用默认模型。"""

    return get_config_value("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def get_tavily_search_depth() -> str:
    """get_tavily_search_depth：读取 Tavily 搜索深度，并强制使用 basic 省额度模式。"""

    # configured_depth：用户配置的搜索深度；为省额度，任何非 basic 配置都会被降级为 basic。
    configured_depth = get_config_value("TAVILY_SEARCH_DEPTH", DEFAULT_TAVILY_SEARCH_DEPTH).strip().lower()
    return DEFAULT_TAVILY_SEARCH_DEPTH if configured_depth != DEFAULT_TAVILY_SEARCH_DEPTH else configured_depth


def get_tavily_max_searches_per_guide() -> int:
    """get_tavily_max_searches_per_guide：读取每份攻略最多搜索次数，并默认限制为 1 次。"""

    # configured_limit：用户配置的每份攻略最大 Tavily 调用次数。
    configured_limit = get_int_config("TAVILY_MAX_SEARCHES_PER_GUIDE", DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE)
    return max(0, min(configured_limit, DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE))


def normalize_tavily_query(destination: str, query: str) -> str:
    """normalize_tavily_query：把目的地和 query 标准化，用于判断相似搜索并命中缓存。"""

    # token_list：从 query 中提取的中英文关键词。
    token_list = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", query.lower())

    # normalized_tokens：去重排序后的关键词，使词序轻微变化时仍可复用缓存。
    normalized_tokens = sorted(set(token_list))

    return f"{destination.strip().lower()}|{'|'.join(normalized_tokens)}"


def build_tavily_cache_key(destination: str, query: str) -> str:
    """build_tavily_cache_key：为目的地和相似 query 生成稳定缓存 key。"""

    # normalized_query：标准化后的 query 文本。
    normalized_query = normalize_tavily_query(destination, query)
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


def load_tavily_cache() -> dict:
    """load_tavily_cache：读取本地 Tavily 缓存文件。"""

    if not TAVILY_CACHE_PATH.exists():
        return {}

    try:
        with TAVILY_CACHE_PATH.open("r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except Exception:
        return {}


def save_tavily_cache(cache_data: dict) -> None:
    """save_tavily_cache：把 Tavily 搜索结果写入本地缓存文件。"""

    with TAVILY_CACHE_PATH.open("w", encoding="utf-8") as cache_file:
        json.dump(cache_data, cache_file, ensure_ascii=False, indent=2)


def get_cached_tavily_results(destination: str, query: str) -> list[dict] | None:
    """get_cached_tavily_results：读取 24 小时内的 Tavily 缓存结果。"""

    # cache_data：本地缓存文件中的全部数据。
    cache_data = load_tavily_cache()

    # cache_key：当前目的地和 query 对应的缓存 key。
    cache_key = build_tavily_cache_key(destination, query)

    # cached_item：缓存中的单条搜索记录。
    cached_item = cache_data.get(cache_key)
    if not cached_item:
        return None

    # cached_at：缓存写入时间戳。
    cached_at = float(cached_item.get("cached_at", 0))
    if time.time() - cached_at > TAVILY_CACHE_TTL_SECONDS:
        return None

    return cached_item.get("results", [])


def set_cached_tavily_results(destination: str, query: str, results: list[dict]) -> None:
    """set_cached_tavily_results：缓存 Tavily 搜索结果，减少重复搜索消耗。"""

    # cache_data：本地缓存文件中的全部数据。
    cache_data = load_tavily_cache()

    # cache_key：当前目的地和 query 对应的缓存 key。
    cache_key = build_tavily_cache_key(destination, query)
    cache_data[cache_key] = {
        "destination": destination,
        "query": query,
        "cached_at": time.time(),
        "cached_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "results": results,
    }
    save_tavily_cache(cache_data)


def is_tavily_limit_error(error: Exception) -> bool:
    """is_tavily_limit_error：判断 Tavily 错误是否属于额度不足或请求受限。"""

    # error_text：错误文本，兼容 SDK 返回的不同异常格式。
    error_text = str(error).lower()
    limit_keywords = ["429", "quota", "rate limit", "ratelimit", "credits", "credit", "insufficient"]
    return any(keyword in error_text for keyword in limit_keywords)


def call_tavily_search(query: str, parsed_request: dict) -> tuple[list[dict], bool]:
    """call_tavily_search：调用 Tavily SDK 搜索，返回结果和是否命中缓存。"""

    # destination：目的地名称，用于缓存 key。
    destination = parsed_request["destination"]

    # cached_results：24 小时内缓存命中的搜索结果。
    cached_results = get_cached_tavily_results(destination, query)
    if cached_results is not None:
        return cached_results, True

    # tavily_api_key：Tavily API Key，从 .env、环境变量或 Streamlit secrets 读取。
    tavily_api_key = get_tavily_api_key()
    if not tavily_api_key or TavilyClient is None:
        return [], False

    # tavily_client：Tavily Python SDK 客户端。
    tavily_client = TavilyClient(api_key=tavily_api_key)

    # response_data：Tavily SDK 搜索返回结果；不启用 answer/raw/images/auto_parameters，控制额度消耗。
    response_data = tavily_client.search(
        query=query,
        search_depth=get_tavily_search_depth(),
        max_results=DEFAULT_SEARCH_MAX_RESULTS,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        auto_parameters=False,
        timeout=12,
    )

    # results：Tavily 搜索结果列表。
    results = response_data.get("results", [])
    set_cached_tavily_results(destination, query, results)
    return results, False


def build_facts_context(parsed_request: dict) -> tuple[str, list[dict], str | None]:
    """build_facts_context：联网搜索并整理 facts_context，供 DeepSeek 生成攻略时引用。"""

    # searched_at：事实校验执行时间。
    searched_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not get_bool_config("USE_TAVILY", True):
        facts_context = f"""
联网事实校验状态：未启用
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：USE_TAVILY=false
更新时间：{searched_at}
页面提示：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
""".strip()
        return facts_context, [], "未启用联网搜索"

    if TavilyClient is None:
        facts_context = f"""
联网事实校验状态：执行失败
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：未安装 tavily-python
更新时间：{searched_at}
页面提示：联网搜索失败，已切换普通模式。
""".strip()
        return facts_context, [], "联网搜索失败，已切换普通模式"

    # tavily_api_key：Tavily API Key，用于判断是否启用搜索。
    tavily_api_key = get_tavily_api_key()
    if not tavily_api_key:
        facts_context = f"""
联网事实校验状态：未配置
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：未配置 TAVILY_API_KEY
更新时间：{searched_at}
页面提示：未配置 Tavily，当前为普通生成模式。
""".strip()
        return facts_context, [], "未配置 Tavily，当前为普通生成模式"

    # max_searches：每份攻略最多 Tavily 调用次数，默认限制为 1 次。
    max_searches = get_tavily_max_searches_per_guide()
    if max_searches <= 0:
        facts_context = f"""
联网事实校验状态：未启用
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：TAVILY_MAX_SEARCHES_PER_GUIDE=0
更新时间：{searched_at}
页面提示：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
""".strip()
        return facts_context, [], "未启用联网搜索"

    # query_list：本次事实校验需要执行的搜索查询。
    query_list = build_fact_search_queries(parsed_request)[:max_searches]

    # source_records：用于展示和传给模型的搜索结果。
    source_records = []

    # used_cache：本次搜索是否命中过本地缓存。
    used_cache = False

    # context_blocks：facts_context 中的文本块。
    context_blocks = [
        "联网事实校验状态：已执行",
        f"更新时间：{searched_at}",
        f"搜索模式：Tavily basic，省额度模式，每份攻略最多 {max_searches} 次搜索。",
        "使用范围：门票、预约规则、开放时间、交通政策等易变化信息只能基于以下搜索结果整理。",
        "注意：根据搜索结果整理，仍需出行前二次确认。",
    ]

    try:
        for query in query_list:
            # results：单个 query 的网页搜索结果。
            results, cache_hit = call_tavily_search(query, parsed_request)
            used_cache = used_cache or cache_hit
            context_blocks.append(f"\n### 搜索查询：{query}")
            context_blocks.append(f"- 结果来源：{'本地 24 小时缓存' if cache_hit else 'Tavily basic 搜索'}")

            if not results:
                context_blocks.append("- 未查到可用结果。")
                continue

            for result in results:
                # title：搜索结果标题。
                title = result.get("title", "未命名来源")

                # url：搜索结果链接。
                url = result.get("url", "")

                # content：搜索结果摘要内容。
                content = result.get("content", "") or result.get("snippet", "")
                content = re.sub(r"\s+", " ", content).strip()

                source_records.append({"query": query, "title": title, "url": url, "content": content})
                context_blocks.append(f"- 标题：{title}\n  链接：{url}\n  摘要：{content[:360]}")
    except Exception as error:
        if is_tavily_limit_error(error):
            facts_context = f"""
联网事实校验状态：额度不足
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
更新时间：{searched_at}
错误：{error}
要求：门票、预约、开放时间、交通政策等易变化信息不可编造；请写“建议出行前二次确认”。
""".strip()
            return facts_context, source_records, "Tavily 额度不足，已切换普通模式"

        facts_context = f"""
联网事实校验状态：执行失败
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
更新时间：{searched_at}
错误：{error}
要求：门票、预约、开放时间、交通政策等易变化信息不可编造；如果 facts_context 没有查到，请写“建议出行前再次核对”。
""".strip()
        return facts_context, source_records, "联网搜索失败，已切换普通模式"

    if not source_records:
        context_blocks.append("\n结论：未查到足够搜索结果。不要编造门票、预约、开放时间，请提示建议出行前再次核对。")

    if used_cache:
        context_blocks[0] = "联网事实校验状态：缓存命中"
        return "\n".join(context_blocks), source_records, "使用缓存搜索结果"

    return "\n".join(context_blocks), source_records, "已启用联网搜索"


def parse_travel_request(user_input: str) -> dict:
    """parse_travel_request：从用户的一句话里提取目的地、天数、预算和偏好。"""

    # cleaned_input：去掉多余空格后的用户输入。
    cleaned_input = user_input.strip()

    # budget_level：识别到的预算档位。
    budget_level = infer_budget_level(cleaned_input)

    # budget_info：识别到的预算金额、货币单位和展示文本。
    budget_info = parse_budget_info(cleaned_input, budget_level)

    # preferences：从用户输入中识别出的偏好列表。
    preferences = extract_preferences(cleaned_input)

    # trip_segments：单目的地或多目的地分段行程。
    trip_segments = extract_trip_segments(cleaned_input, preferences)

    # destination_list：所有识别到的目的地。
    destination_list = [segment["destination"] for segment in trip_segments]

    # destination：用于页面总览展示的目的地文本。
    destination = " × ".join(destination_list) if len(destination_list) > 1 else destination_list[0]

    # days：总旅行天数，多目的地时为各段天数之和。
    days = sum(segment["days"] for segment in trip_segments)

    # nights：总住宿晚数，按连续旅行粗略估算。
    nights = max(0, days - 1)

    # nights_match：如果用户明确写了总住宿晚数，则尊重用户输入。
    nights_match = re.search(r"([0-9一二两三四五六七八九十]+)\s*晚", cleaned_input)
    if nights_match:
        nights = max(0, parse_chinese_number(nights_match.group(1)))

    # trip_type：单目的地或多目的地。
    trip_type = "multi_destination" if len(trip_segments) > 1 else "single_destination"

    # trip_notes：多目的地或省份默认推断提示。
    trip_notes = [segment["note"] for segment in trip_segments if segment.get("note")]

    # parsed_request：最终返回给页面和大模型的结构化旅行需求。
    parsed_request = {
        "trip_type": trip_type,
        "destination": destination,
        "destinations": destination_list,
        "trip_segments": trip_segments,
        "trip_notes": trip_notes,
        "days": days,
        "nights": nights,
        "total_days": days,
        "total_nights": nights,
        "budget": budget_info["display"],
        "budget_level": budget_info["level"],
        "style": budget_info["level"].replace("预算", ""),
        "budget_amount": budget_info["amount"],
        "budget_currency": budget_info["currency"],
        "currency": budget_info["currency"],
        "budget_currency_name": budget_info["currency_name"],
        "budget_has_explicit_amount": budget_info["has_explicit_amount"],
        "preferences": preferences,
    }

    # budget_exchange_hint：国外目的地的粗略换算提示。
    parsed_request["budget_exchange_hint"] = build_exchange_hint(parsed_request)

    return parsed_request


def build_ai_prompt(user_input: str, parsed_request: dict, facts_context: str) -> str:
    """build_ai_prompt：把用户输入、解析结果和联网事实上下文整理成给大模型的提示词。"""

    # preferences_text：把偏好列表合并成适合模型阅读的字符串。
    preferences_text = "、".join(parsed_request["preferences"])

    # exchange_hint_text：国外目的地预算粗略换算提示。
    exchange_hint_text = parsed_request.get("budget_exchange_hint") or "无"

    # budget_currency_text：预算货币单位说明。
    budget_currency_text = (
        f"{parsed_request['budget_currency_name']} ({parsed_request['budget_currency']})"
        if parsed_request.get("budget_currency")
        else "未指定具体金额"
    )

    # search_enabled：是否拿到了可用于事实校验的联网或缓存结果。
    search_enabled = "联网事实校验状态：已执行" in facts_context or "联网事实校验状态：缓存命中" in facts_context

    if search_enabled:
        # fact_rules：启用联网搜索时，对易变化信息使用 facts_context 的强约束规则。
        fact_rules = """
16. 门票、预约规则、开放时间、景区政策、交通政策等易变化信息，必须优先依据 facts_context 写。
17. 如果 facts_context 没有明确说明对应信息，不能编造，请写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。
18. 对门票、预约、开放时间这类信息，必须标注“根据搜索结果整理，仍需出行前二次确认”。
19. 必须增加“信息来源与更新时间”区域，列出来源标题、链接和更新时间；如果搜索结果没有覆盖某项信息，也要说明未查到。
20. 避坑提醒要明确、实用。
21. 必须包含以下二级标题，并保持标题文字完全一致：
""".strip()
    else:
        # fact_rules：普通生成模式下不使用联网结果，但提醒用户二次确认易变化信息。
        fact_rules = """
16. 当前未启用联网搜索，请按普通生成模式输出攻略。
17. DeepSeek 不能编造最新门票、预约规则、开放时间、景区政策或交通政策。
18. 对所有可能变化的信息，必须写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。
19. 必须增加“信息来源与更新时间”区域，并写明：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
20. 避坑提醒要明确、实用。
21. 必须包含以下二级标题，并保持标题文字完全一致：
""".strip()

    return f"""
用户原始需求：
{user_input}

联网事实校验 facts_context：
{facts_context}

系统已识别：
- 目的地：{parsed_request["destination"]}
- 目的地分段：{json.dumps(parsed_request.get("trip_segments", []), ensure_ascii=False)}
- 旅行天数：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 预算单位：{budget_currency_text}
- 预算换算提示：{exchange_hint_text}
- 偏好：{preferences_text}

请生成一份中文旅行攻略，要求：
1. 使用 Markdown。
2. 内容具体、可执行，不要泛泛而谈。
3. 每日行程必须严格生成 {parsed_request["days"]} 天，从 Day 1 到 Day {parsed_request["days"]}，不能少生成，也不能只生成 3 天。
4. 每一天必须有不同主题，标题格式必须是“### Day 1：主题名”，例如“### Day 1：西湖经典路线”。
5. 用户明确提到的景点和偏好必须优先安排：{preferences_text}。
6. 如果用户提到的景点不足以填满全部天数，请根据目的地、偏好、预算和 facts_context 补充适合景点；搜索结果不足时请写“不确定，请出行前核对”，不要编造实时事实。
7. 每天必须包含“上午”“中午”“下午”“晚上”四个时间段。
8. 每个时间段必须写成：- 上午：具体地点｜推荐理由｜预计耗时｜交通或预约提醒。
9. 不要反复使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
10. 同一景点、同一餐厅、同一区域不要重复出现，除非用户明确要求。
11. 美食推荐建议写成：- 名称：推荐理由｜人均预算｜适合场景。
12. 预算估算要分交通、住宿、餐饮、门票体验、机动费用，并明确预算单位。
13. 如果用户输入了预算数字但没有货币单位，必须按人民币 CNY 理解，不要按目的地当地货币理解。
14. 如果用户明确写了美元、日元、欧元、韩元或 USD/EUR/JPY/KRW，必须尊重用户输入的货币单位。
15. 如果存在“预算换算提示”，请在预算估算中补充这条提示，并说明汇率仅供参考。
{fact_rules}
## 旅行封面文案
## 详细旅游攻略
## 每日行程
## 美食推荐
## 交通建议
## 预算估算
## 避坑提醒
## 信息来源与更新时间
""".strip()


def call_deepseek_chat(prompt: str, instructions: str) -> tuple[str | None, str | None]:
    """call_deepseek_chat：执行一次 DeepSeek Chat Completions 调用并返回文本。"""

    if OpenAI is None:
        return None, "没有安装 openai 依赖，已使用本地演示攻略。"

    # api_key：DeepSeek API Key，从 .env、环境变量或 Streamlit secrets 读取。
    api_key = get_deepseek_api_key()
    if not api_key:
        return None, "未配置 DEEPSEEK_API_KEY，已使用本地演示攻略。"

    # model_name：当前使用的 DeepSeek 模型名称，可通过 DEEPSEEK_MODEL 修改。
    model_name = get_deepseek_model_name()

    try:
        # client：OpenAI SDK 客户端，通过 base_url 指向 DeepSeek 服务。
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

        # response：DeepSeek Chat Completions API 返回的大模型结果。
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        )

        # response_text：从模型回复中取出的文本。
        response_text = response.choices[0].message.content
        if not response_text:
            return None, "DeepSeek 返回内容为空，已使用本地演示攻略。"

        return response_text.strip(), None
    except Exception as error:
        return None, f"DeepSeek API 调用失败，已使用本地演示攻略。错误：{error}"


def build_structured_json_prompt(user_input: str, parsed_request: dict, facts_context: str) -> str:
    """build_structured_json_prompt：生成结构化旅行 JSON 的 DeepSeek 提示词。"""

    # preferences_json：用户偏好 JSON 文本，确保用户输入优先。
    preferences_json = json.dumps(parsed_request["preferences"], ensure_ascii=False)

    # trip_segments_json：分段目的地 JSON 文本，确保多目的地不被忽略。
    trip_segments_json = json.dumps(parsed_request.get("trip_segments", []), ensure_ascii=False, indent=2)

    # search_enabled：是否有可用于事实校验的 Tavily 搜索结果。
    search_enabled = "联网事实校验状态：已执行" in facts_context or "联网事实校验状态：缓存命中" in facts_context

    # fact_rule_text：联网事实约束文本。
    fact_rule_text = (
        "门票、预约、开放时间、景区政策、交通政策必须优先依据 facts_context；如果 facts_context 没有明确说明，不要编造，写“建议出行前二次确认”。"
        if search_enabled
        else "当前未启用联网搜索，不能编造最新门票、预约、开放时间、景区政策或交通政策；必须写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。"
    )

    return f"""
用户原始需求：
{user_input}

联网事实校验 facts_context：
{facts_context}

系统已识别参数，必须严格使用，不能被模型猜测或默认值覆盖：
- destination: {parsed_request["destination"]}
- trip_type: {parsed_request.get("trip_type", "single_destination")}
- destinations: {json.dumps(parsed_request.get("destinations", [parsed_request["destination"]]), ensure_ascii=False)}
- trip_segments: {trip_segments_json}
- days: {parsed_request["days"]}
- nights: {parsed_request["nights"]}
- budget_amount: {parsed_request.get("budget_amount")}
- currency: {parsed_request.get("budget_currency") or "CNY"}
- budget_level: {parsed_request.get("style") or parsed_request.get("budget_level")}
- preferences: {preferences_json}

请只返回合法 JSON，不要输出 Markdown，不要解释，不要使用代码块。
JSON 顶层必须包含：
trip_type, destination, destinations, total_days, days, nights, budget, preferences, trip_segments, daily_itinerary, food_recommendations

budget 必须包含：
amount, currency, level

trip_segments 必须和系统识别参数一致。
每个 trip_segments 对象必须包含：
destination, days, nights, days_inferred, theme, note

daily_itinerary 必须 exactly {parsed_request["days"]} 天，从 day 1 到 day {parsed_request["days"]}。
每天必须包含：
day, segment_destination, theme, morning, noon, afternoon, evening

morning/noon/afternoon/evening 每个对象必须包含以下非空字段：
time, place, original_name, reason, duration, transport, booking_note

food_recommendations 必须包含 4-6 家店或小吃点。
每个美食推荐对象必须包含以下非空字段：
name_cn, name_original, location, nearby_spot, reason, budget, scene, booking_note, map_keyword

写作规则：
1. 用户明确提到的景点和偏好必须优先安排：{preferences_json}。
2. 用户明确提到的所有目的地必须出现在 trip_segments 和 daily_itinerary 中，不允许只生成第一个目的地。
3. 用户明确输入的所有目的地必须完整出现在攻略中。不得只生成第一个目的地。对于多目的地输入，必须按 trip_segments 分段规划，并保证每日行程覆盖全部目的地。
4. 如果 days_inferred=true，必须在 note 中说明“用户未说明该目的地天数，系统默认按3天规划”。
5. 如果目的地是省份、国家或大区域，不要泛泛写省名/国家名；必须推荐具体城市路线，并在 note 中说明推断依据。
6. 多目的地 daily_itinerary 的 day 必须全程连续编号，不能每段都从 Day 1 重新开始。
7. 每一天主题必须不同，不能重复。
8. 同一景点、同一餐厅、同一区域不要重复安排。
9. 不要使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
10. place 必须是具体地点，original_name 必须包含中文名和英文/原名；没有英文名时写中文原名。
11. reason、transport、booking_note 必须具体，不能空泛。
12. 美食推荐如果是国外目的地，name_original 必须尽量保留英文名、当地语言原名或常用地图搜索名。
13. 如果无法确认具体地址，不要编造门牌号；location 可以写“市中心区域”“靠近某某景点”“建议以 Google Maps 搜索原名确认”。
14. map_keyword 必须适合复制到 Google Maps / Apple Maps / 百度地图 / 高德地图搜索。
15. {fact_rule_text}
""".strip()


def extract_json_text(model_output: str) -> str:
    """extract_json_text：从模型输出中提取 JSON 文本，兼容代码块和前后解释。"""

    # fenced_match：匹配 ```json 代码块中的 JSON。
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", model_output, flags=re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    # start_index/end_index：提取第一个对象到最后一个对象之间的内容。
    start_index = model_output.find("{")
    end_index = model_output.rfind("}")
    if start_index >= 0 and end_index > start_index:
        return model_output[start_index : end_index + 1].strip()

    return model_output.strip()


def parse_structured_json_output(model_output: str | None) -> tuple[dict | None, list[str]]:
    """parse_structured_json_output：把模型原始输出解析为 JSON 对象。"""

    if not model_output:
        return None, ["模型没有返回 JSON 内容。"]

    # json_text：提取后的 JSON 文本。
    json_text = extract_json_text(model_output)
    try:
        # parsed_json：解析后的 JSON 对象。
        parsed_json = json.loads(json_text)
    except json.JSONDecodeError as error:
        return None, [f"JSON 解析失败：{error}"]

    if not isinstance(parsed_json, dict):
        return None, ["JSON 顶层必须是对象。"]

    return parsed_json, []


def validate_structured_travel_json(travel_json: dict | None, parsed_request: dict) -> list[str]:
    """validate_structured_travel_json：校验每日行程 JSON 是否完整、准确、不重复。"""

    if not isinstance(travel_json, dict):
        return ["结构化结果不是 JSON 对象。"]

    # validation_errors：JSON 校验错误列表。
    validation_errors = []

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    if travel_json.get("destination") != parsed_request["destination"]:
        validation_errors.append(f"destination 不一致，应为 {parsed_request['destination']}。")

    if travel_json.get("days") != expected_days:
        validation_errors.append(f"days 不一致，应为 {expected_days}。")

    if travel_json.get("total_days") is not None and travel_json.get("total_days") != expected_days:
        validation_errors.append(f"total_days 不一致，应为 {expected_days}。")

    if travel_json.get("nights") != parsed_request["nights"]:
        validation_errors.append(f"nights 不一致，应为 {parsed_request['nights']}。")

    # expected_destinations：系统识别出的所有目的地。
    expected_destinations = parsed_request.get("destinations") or [parsed_request["destination"]]

    # destination_json：模型返回的目的地数组。
    destination_json = travel_json.get("destinations")
    if isinstance(destination_json, list):
        missing_destinations = [destination for destination in expected_destinations if destination not in destination_json]
        if missing_destinations:
            validation_errors.append(f"destinations 缺少目的地：{'、'.join(missing_destinations)}。")

    # trip_segments_json：模型返回的分段行程。
    trip_segments_json = travel_json.get("trip_segments")
    if not isinstance(trip_segments_json, list):
        validation_errors.append("trip_segments 必须是数组。")
    else:
        segment_map = {segment.get("destination"): segment for segment in trip_segments_json if isinstance(segment, dict)}
        for expected_segment in parsed_request.get("trip_segments", []):
            destination = expected_segment["destination"]
            segment = segment_map.get(destination)
            if not segment:
                validation_errors.append(f"trip_segments 缺少目的地：{destination}。")
                continue
            if segment.get("days") != expected_segment["days"]:
                validation_errors.append(f"{destination} days 不一致，应为 {expected_segment['days']}。")
            if segment.get("nights") != expected_segment["nights"]:
                validation_errors.append(f"{destination} nights 不一致，应为 {expected_segment['nights']}。")
            if "days_inferred" not in segment:
                validation_errors.append(f"{destination} 缺少 days_inferred 字段。")
            if not str(segment.get("theme", "")).strip():
                validation_errors.append(f"{destination} theme 不能为空。")
            if expected_segment.get("days_inferred") and not str(segment.get("note", "")).strip():
                validation_errors.append(f"{destination} 是默认推断天数，note 不能为空。")

    # budget_json：模型返回的预算对象。
    budget_json = travel_json.get("budget")
    if not isinstance(budget_json, dict):
        validation_errors.append("budget 必须是对象。")
    else:
        if parsed_request.get("budget_has_explicit_amount") and budget_json.get("amount") != parsed_request.get("budget_amount"):
            validation_errors.append(f"budget.amount 不一致，应为 {parsed_request.get('budget_amount')}。")
        if parsed_request.get("budget_currency") and budget_json.get("currency") != parsed_request.get("budget_currency"):
            validation_errors.append(f"budget.currency 不一致，应为 {parsed_request.get('budget_currency')}。")
        if not str(budget_json.get("level", "")).strip():
            validation_errors.append("budget.level 不能为空。")

    # preferences_json：模型返回的偏好列表。
    preferences_json = travel_json.get("preferences")
    if not isinstance(preferences_json, list):
        validation_errors.append("preferences 必须是数组。")
    else:
        missing_preferences = [preference for preference in parsed_request["preferences"] if preference not in preferences_json]
        if missing_preferences:
            validation_errors.append(f"preferences 缺少用户明确偏好：{'、'.join(missing_preferences)}。")

    # daily_itinerary：模型返回的每日行程数组。
    daily_itinerary = travel_json.get("daily_itinerary")
    if not isinstance(daily_itinerary, list):
        return validation_errors + ["daily_itinerary 必须是数组。"]

    if len(daily_itinerary) != expected_days:
        validation_errors.append(f"daily_itinerary 必须 exactly {expected_days} 天，当前为 {len(daily_itinerary)} 天。")

    # required_slots：每天必须包含的四个时间段。
    required_slots = ["morning", "noon", "afternoon", "evening"]

    # required_slot_fields：每个时间段对象必须包含的字段。
    required_slot_fields = ["time", "place", "original_name", "reason", "duration", "transport", "booking_note"]

    # generic_terms：不允许出现的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # seen_days/themes/places/segment_destinations：用于检查编号、主题、地点和目的地覆盖。
    seen_days = set()
    seen_themes = set()
    seen_places = set()
    seen_segment_destinations = set()

    for day_index, day_item in enumerate(daily_itinerary, start=1):
        if not isinstance(day_item, dict):
            validation_errors.append(f"Day {day_index} 必须是对象。")
            continue

        day_number = day_item.get("day")
        if day_number != day_index:
            validation_errors.append(f"Day {day_index} 的 day 字段应为 {day_index}，当前为 {day_number}。")
        if day_number in seen_days:
            validation_errors.append(f"Day 编号重复：{day_number}。")
        seen_days.add(day_number)

        segment_destination = str(day_item.get("segment_destination", "")).strip()
        if not segment_destination:
            validation_errors.append(f"Day {day_index} segment_destination 不能为空。")
        elif segment_destination not in expected_destinations:
            validation_errors.append(f"Day {day_index} segment_destination 不在用户目的地中：{segment_destination}。")
        seen_segment_destinations.add(segment_destination)

        theme = str(day_item.get("theme", "")).strip()
        if not theme:
            validation_errors.append(f"Day {day_index} theme 不能为空。")
        elif theme in seen_themes:
            validation_errors.append(f"Day {day_index} theme 重复：{theme}。")
        seen_themes.add(theme)

        for slot_name in required_slots:
            # slot_data：单个时间段对象。
            slot_data = day_item.get(slot_name)
            if not isinstance(slot_data, dict):
                validation_errors.append(f"Day {day_index} 缺少 {slot_name} 对象。")
                continue

            for field_name in required_slot_fields:
                field_value = slot_data.get(field_name)
                if field_value is None or not str(field_value).strip():
                    validation_errors.append(f"Day {day_index} {slot_name}.{field_name} 不能为空。")

            slot_text = " ".join(str(slot_data.get(field_name, "")) for field_name in required_slot_fields)
            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_index} {slot_name} 包含空泛模板词。")

            place = str(slot_data.get("place", "")).strip()
            if place:
                if place in seen_places:
                    validation_errors.append(f"重复安排地点：{place}。")
                seen_places.add(place)

    missing_days = [day for day in range(1, expected_days + 1) if day not in seen_days]
    if missing_days:
        validation_errors.append(f"daily_itinerary 缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    missing_segment_destinations = [
        destination for destination in expected_destinations if destination not in seen_segment_destinations
    ]
    if missing_segment_destinations:
        validation_errors.append(f"daily_itinerary 未覆盖目的地：{'、'.join(missing_segment_destinations)}。")

    # food_recommendations：模型返回的美食推荐数组。
    food_recommendations = travel_json.get("food_recommendations")
    if not isinstance(food_recommendations, list):
        validation_errors.append("food_recommendations 必须是数组。")
    elif not food_recommendations:
        validation_errors.append("food_recommendations 不能为空。")
    else:
        # required_food_fields：每个美食推荐对象必须包含的字段。
        required_food_fields = [
            "name_cn",
            "name_original",
            "location",
            "nearby_spot",
            "reason",
            "budget",
            "scene",
            "booking_note",
            "map_keyword",
        ]

        # seen_food_names：用于检查店铺名称是否重复。
        seen_food_names = set()
        for food_index, food_item in enumerate(food_recommendations, start=1):
            if not isinstance(food_item, dict):
                validation_errors.append(f"food_recommendations 第 {food_index} 项必须是对象。")
                continue

            for field_name in required_food_fields:
                field_value = food_item.get(field_name)
                if field_value is None or not str(field_value).strip():
                    validation_errors.append(f"food_recommendations 第 {food_index} 项 {field_name} 不能为空。")

            food_name_key = f"{food_item.get('name_cn', '')}|{food_item.get('name_original', '')}".strip()
            if food_name_key in seen_food_names:
                validation_errors.append(f"重复推荐店铺：{food_item.get('name_cn', '')}。")
            seen_food_names.add(food_name_key)

    return validation_errors


def normalize_structured_travel_json(travel_json: dict, parsed_request: dict) -> dict:
    """normalize_structured_travel_json：用系统识别参数覆盖 JSON 顶层关键字段，保证用户输入优先。"""

    # normalized_json：复制后的结构化结果。
    normalized_json = dict(travel_json)
    normalized_json["trip_type"] = parsed_request.get("trip_type", "single_destination")
    normalized_json["destination"] = parsed_request["destination"]
    normalized_json["destinations"] = parsed_request.get("destinations", [parsed_request["destination"]])
    normalized_json["total_days"] = parsed_request["days"]
    normalized_json["days"] = parsed_request["days"]
    normalized_json["nights"] = parsed_request["nights"]
    normalized_json["preferences"] = parsed_request["preferences"]
    normalized_json["trip_segments"] = [
        {
            "destination": segment["destination"],
            "days": segment["days"],
            "nights": segment["nights"],
            "days_inferred": segment.get("days_inferred", False),
            "theme": segment.get("theme") or get_province_route_note(segment["destination"]) or f"{segment['destination']}分段行程",
            "note": segment.get("note", ""),
        }
        for segment in parsed_request.get("trip_segments", [])
    ]
    normalized_json["budget"] = {
        "amount": parsed_request.get("budget_amount"),
        "currency": parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY,
        "level": parsed_request.get("style") or parsed_request.get("budget_level"),
    }
    return normalized_json


def build_json_repair_prompt(
    original_output: str,
    validation_errors: list[str],
    parsed_request: dict,
    facts_context: str,
) -> str:
    """build_json_repair_prompt：根据校验错误要求 DeepSeek 只修复 JSON。"""

    # error_text：校验错误说明。
    error_text = "\n".join(f"- {error}" for error in validation_errors)

    return f"""
你刚才返回的每日行程 JSON 没有通过校验，错误如下：
{error_text}

请基于原始内容修复 JSON。
必须返回合法 JSON。
必须包含 exactly {parsed_request["days"]} 天。
必须包含 trip_type/destination/destinations/total_days/trip_segments/daily_itinerary。
trip_segments 必须等于系统识别分段：
{json.dumps(parsed_request.get("trip_segments", []), ensure_ascii=False, indent=2)}
daily_itinerary 每天必须包含 segment_destination，且必须覆盖所有目的地。
每天必须包含 morning/noon/afternoon/evening。
每个时间段必须包含 time/place/original_name/reason/duration/transport/booking_note。
food_recommendations 必须包含 4-6 项。
每个美食推荐必须包含 name_cn/name_original/location/nearby_spot/reason/budget/scene/booking_note/map_keyword。
必须保留系统识别参数：
- destination: {parsed_request["destination"]}
- destinations: {json.dumps(parsed_request.get("destinations", [parsed_request["destination"]]), ensure_ascii=False)}
- total_days: {parsed_request["days"]}
- days: {parsed_request["days"]}
- nights: {parsed_request["nights"]}
- budget_amount: {parsed_request.get("budget_amount")}
- currency: {parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY}
- budget_level: {parsed_request.get("style") or parsed_request.get("budget_level")}
- preferences: {json.dumps(parsed_request["preferences"], ensure_ascii=False)}

联网事实校验 facts_context：
{facts_context}

原始输出：
{original_output}

不要输出 Markdown，不要解释，只返回 JSON。
""".strip()


def call_deepseek_structured_json_api(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
) -> tuple[dict | None, str | None, list[str], str | None]:
    """call_deepseek_structured_json_api：生成并校验结构化旅行 JSON，失败时自动修复一次。"""

    # instructions：要求模型只返回 JSON 的系统提示。
    instructions = """
你是旅行规划结构化数据生成器。只返回合法 JSON，不输出 Markdown，不解释。
所有用户明确输入的目的地、天数、预算、货币和偏好优先级最高。
不要编造门票、预约、开放时间、景区政策等实时信息。
""".strip()

    # json_prompt：第一次生成结构化 JSON 的提示词。
    json_prompt = build_structured_json_prompt(user_input, parsed_request, facts_context)

    # raw_output：第一次模型原始输出。
    raw_output, api_message = call_deepseek_chat(json_prompt, instructions)
    if not raw_output:
        return None, raw_output, [api_message or "DeepSeek 没有返回结构化 JSON。"], api_message

    # travel_json：第一次解析出的 JSON。
    travel_json, parse_errors = parse_structured_json_output(raw_output)
    validation_errors = parse_errors or validate_structured_travel_json(travel_json, parsed_request)
    if not validation_errors and travel_json:
        return normalize_structured_travel_json(travel_json, parsed_request), raw_output, [], None

    # repair_prompt：JSON 校验失败后的修复提示。
    repair_prompt = build_json_repair_prompt(raw_output, validation_errors, parsed_request, facts_context)

    # repaired_output：修复后的模型原始输出。
    repaired_output, repair_message = call_deepseek_chat(repair_prompt, instructions)
    if not repaired_output:
        return None, raw_output, validation_errors + [repair_message or "DeepSeek JSON 修复没有返回内容。"], repair_message

    # repaired_json：修复后解析出的 JSON。
    repaired_json, repair_parse_errors = parse_structured_json_output(repaired_output)
    repair_errors = repair_parse_errors or validate_structured_travel_json(repaired_json, parsed_request)
    if repair_errors:
        return None, repaired_output, repair_errors, "每日行程 JSON 修复后仍未通过校验。"

    return normalize_structured_travel_json(repaired_json, parsed_request), repaired_output, [], "每日行程 JSON 第一次未通过校验，已自动修复。"


def build_markdown_from_json_prompt(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
    travel_json: dict,
) -> str:
    """build_markdown_from_json_prompt：基于结构化 JSON 生成 Markdown 攻略提示词。"""

    # structured_json_text：结构化旅行 JSON 文本。
    structured_json_text = json.dumps(travel_json, ensure_ascii=False, indent=2)

    return f"""
用户原始需求：
{user_input}

系统识别参数：
- 目的地：{parsed_request["destination"]}
- 旅行天数：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 偏好：{"、".join(parsed_request["preferences"])}

联网事实校验 facts_context：
{facts_context}

结构化 JSON：
{structured_json_text}

请基于上面的结构化 JSON 生成中文 Markdown 攻略。
要求：
1. 不要改变 JSON 中的目的地、天数、预算、偏好和 daily_itinerary。
2. 用户明确提到的所有目的地都必须出现在攻略中，不能只写第一个目的地。
3. 如果是多目的地，先写总览，例如“南京 3 天 + 江西默认 3 天”，再按分段展示。
4. 每日行程必须按 JSON 中的 daily_itinerary 写，不要自由新增重复路线；Day 编号必须连续。
5. 如果 trip_segments 中 days_inferred=true，必须明确说明该目的地天数是系统默认推断。
6. 如果目的地是省份、国家或大区域，必须说明系统选择了具体城市路线。
7. 门票、预约、开放时间、景区政策必须优先依据 facts_context；没有明确搜索结果时写“建议出行前二次确认”。
8. 内容具体、可执行，保留中文名 + 英文/原名。
9. “美食推荐”必须基于 JSON 中的 food_recommendations，每条必须写店名中文名、英文/当地原名、位置、附近景点/区域、人均预算、适合场景、预约提示和地图搜索关键词。
10. 如果无法确认具体地址，不要编造门牌号；写“市中心区域”“靠近某某景点”或“建议以 Google Maps 搜索原名确认”。
11. 必须包含以下二级标题，并保持标题文字完全一致：
## 旅行封面文案
## 详细旅游攻略
## 每日行程
## 美食推荐
## 交通建议
## 预算估算
## 避坑提醒
## 信息来源与更新时间
""".strip()


def build_markdown_from_structured_json(travel_json: dict, parsed_request: dict, facts_context: str) -> str:
    """build_markdown_from_structured_json：当 Markdown 二次生成失败时，用合格 JSON 生成可复制攻略。"""

    # source_updated_at：攻略信息更新时间。
    source_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # itinerary_lines：从结构化 JSON 生成的每日行程 Markdown。
    itinerary_lines = []
    for day_item in travel_json.get("daily_itinerary", []):
        segment_prefix = f"{day_item.get('segment_destination', parsed_request['destination'])}｜"
        itinerary_lines.append(f"### Day {day_item['day']}：{segment_prefix}{day_item['theme']}")
        for slot_key, slot_label in [("morning", "上午"), ("noon", "中午"), ("afternoon", "下午"), ("evening", "晚上")]:
            slot = day_item[slot_key]
            itinerary_lines.append(
                f"- {slot_label}：{slot['place']}｜{slot['reason']}｜{slot['duration']}｜{slot['transport']}；{slot['booking_note']}"
            )

    # food_lines：从结构化 JSON 生成的美食推荐 Markdown。
    food_lines = []
    for food_item in travel_json.get("food_recommendations", []):
        food_title = food_item["name_cn"]
        if food_item.get("name_original") and food_item["name_original"] != food_title:
            food_title = f"{food_title}（{food_item['name_original']}）"
        food_lines.append(
            f"- {food_title}：位置 {food_item['location']}，靠近 {food_item['nearby_spot']}｜"
            f"{food_item['reason']}｜{food_item['budget']}｜{food_item['scene']}｜"
            f"{food_item['booking_note']}｜地图搜索：{food_item['map_keyword']}"
        )

    # preferences_text：用户旅行偏好。
    preferences_text = "、".join(parsed_request["preferences"])

    # segment_overview_text：多目的地分段总览。
    segment_overview_text = " + ".join(
        f"{segment['destination']}{'默认' if segment.get('days_inferred') else ''}{segment['days']}天"
        for segment in parsed_request.get("trip_segments", [])
    )

    # segment_note_lines：多目的地或省份推断说明。
    segment_note_lines = "\n".join(f"- {note}" for note in parsed_request.get("trip_notes", []))

    return f"""
## 旅行封面文案
{parsed_request["destination"]} {parsed_request["days"]} 天 {parsed_request["nights"]} 晚旅行计划：围绕 {preferences_text} 安排路线。

## 详细旅游攻略
- 目的地：{parsed_request["destination"]}
- 行程总览：{segment_overview_text or parsed_request["destination"]}
- 行程长度：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 旅行风格：{preferences_text}
- 说明：本 Markdown 根据已通过校验的结构化 JSON 生成。
{segment_note_lines}

## 每日行程
{chr(10).join(itinerary_lines)}

## 美食推荐
{chr(10).join(food_lines) if food_lines else f"- 请结合每日路线选择附近餐厅，热门餐厅建议提前预约或取号｜人均预算按 {parsed_request['budget']} 控制｜适合午餐和晚餐"}

## 交通建议
- 每天优先围绕同一区域规划，减少跨区往返。
- 景区门票、预约和开放时间建议出行前二次确认。

## 预算估算
- 用户预算：{parsed_request["budget"]}
- 交通、住宿、餐饮、门票体验和机动费用建议按实际日期二次核算。

## 避坑提醒
- 不要把热门景点、热门餐厅和远距离交通挤在同一天。
- 对所有可能变化的信息，建议出行前二次确认。

## 信息来源与更新时间
- 更新时间：{source_updated_at}
- 来源说明：结构化 JSON 已通过程序校验；门票、预约、开放时间等仍需出行前二次确认。
- 联网事实状态：{facts_context.splitlines()[0] if facts_context else "未启用联网搜索"}
""".strip()


def extract_markdown_section_text(markdown_text: str, heading: str) -> str:
    """extract_markdown_section_text：从 Markdown 中提取指定二级标题下的正文。"""

    # normalized_markdown：保证开头有换行，便于正则匹配二级标题。
    normalized_markdown = "\n" + markdown_text.strip()

    # section_pattern：匹配指定二级标题到下一个二级标题之间的内容。
    section_pattern = rf"\n##\s+{re.escape(heading)}\s*\n([\s\S]*?)(?=\n##\s+|\Z)"
    match = re.search(section_pattern, normalized_markdown)
    return match.group(1).strip() if match else ""


def parse_itinerary_day_blocks(markdown_text: str) -> list[dict]:
    """parse_itinerary_day_blocks：解析 Markdown 每日行程区中的 Day 块。"""

    # itinerary_text：每日行程区域 Markdown。
    itinerary_text = extract_markdown_section_text(markdown_text, "每日行程")
    if not itinerary_text:
        return []

    # day_matches：匹配 Day 标题。
    day_matches = list(re.finditer(r"(?m)^###\s*Day\s*([0-9一二两三四五六七八九十]+)\s*[：:]?\s*(.*?)\s*$", itinerary_text))

    # day_blocks：解析后的 Day 数据。
    day_blocks = []
    for index, match in enumerate(day_matches):
        start_index = match.end()
        end_index = day_matches[index + 1].start() if index + 1 < len(day_matches) else len(itinerary_text)
        day_number = parse_chinese_number(match.group(1))
        theme = match.group(2).strip() or f"Day {day_number}"
        body = itinerary_text[start_index:end_index].strip()
        day_blocks.append({"day": day_number, "theme": theme, "body": body})

    return day_blocks


def validate_itinerary_markdown(markdown_text: str, parsed_request: dict) -> list[str]:
    """validate_itinerary_markdown：校验模型生成的每日行程是否满足不重复和天数要求。"""

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    # day_blocks：模型输出中的 Day 块。
    day_blocks = parse_itinerary_day_blocks(markdown_text)

    # validation_errors：行程校验错误列表。
    validation_errors = []

    if len(day_blocks) < expected_days:
        validation_errors.append(f"每日行程只生成了 {len(day_blocks)} 天，用户需要 {expected_days} 天。")
    elif len(day_blocks) > expected_days:
        validation_errors.append(f"每日行程生成了 {len(day_blocks)} 天，用户只需要 {expected_days} 天。")

    # day_number_set：实际出现的 Day 编号集合。
    day_number_set = {day_block["day"] for day_block in day_blocks}
    missing_days = [day for day in range(1, expected_days + 1) if day not in day_number_set]
    if missing_days:
        validation_errors.append(f"缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    # required_slots：每天必须包含的四个时间段。
    required_slots = ["上午", "中午", "下午", "晚上"]

    # generic_terms：不允许反复出现的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # signature_set：用于检查整天内容是否重复。
    signature_set = set()

    # theme_set：用于检查每天主题是否重复。
    theme_set = set()

    # seen_places：用于检查具体地点是否重复安排。
    seen_places = set()

    for day_block in day_blocks[:expected_days]:
        # day_theme：单日主题，必须和其他天不同。
        day_theme = clean_markdown_text(day_block["theme"])
        if day_theme in theme_set:
            validation_errors.append(f"Day {day_block['day']} 主题重复：{day_theme}。")
        theme_set.add(day_theme)

        day_signature_parts = []
        for slot_label in required_slots:
            slot_text = extract_slot_text(day_block["body"], slot_label)
            if not slot_text:
                validation_errors.append(f"Day {day_block['day']} 缺少{slot_label}安排。")
                continue

            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_block['day']} {slot_label}使用了空泛模板词：{slot_text[:40]}")

            # slot_parts：时间段内容应拆成地点、推荐理由、预计耗时、交通或预约提醒。
            slot_parts = [part.strip() for part in re.split(r"[｜|]", slot_text) if part.strip()]
            if len(slot_parts) < 4:
                validation_errors.append(
                    f"Day {day_block['day']} {slot_label}格式不完整，需要“具体地点｜推荐理由｜预计耗时｜交通或预约提醒”。"
                )

            # place：时间段内容中竖线前面的具体地点。
            place = slot_parts[0] if slot_parts else ""
            if place:
                if place in seen_places:
                    validation_errors.append(f"重复安排地点：{place}。")
                seen_places.add(place)

            day_signature_parts.append(slot_text)

        # day_signature：单日四段安排合并后的指纹。
        day_signature = "||".join(day_signature_parts)
        if day_signature and day_signature in signature_set:
            validation_errors.append(f"Day {day_block['day']} 与其他日期的行程内容高度重复。")
        signature_set.add(day_signature)

    return validation_errors


def build_itinerary_retry_prompt(base_prompt: str, validation_errors: list[str], parsed_request: dict) -> str:
    """build_itinerary_retry_prompt：根据校验错误生成二次生成提示词。"""

    # error_text：校验错误说明。
    error_text = "\n".join(f"- {error}" for error in validation_errors)

    return f"""
{base_prompt}

上一次输出的每日行程不合格，必须重新生成完整攻略，重点修复：
{error_text}

强制要求：
1. 必须生成 Day 1 到 Day {parsed_request["days"]}，一天都不能少。
2. 每一天主题必须不同。
3. 不要使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
4. 每个时间段必须写具体地点原名、推荐理由、预计耗时、交通或预约提醒。
5. 不要重复同一景点、同一餐厅、同一区域。
""".strip()


def call_deepseek_api(user_input: str, parsed_request: dict, facts_context: str) -> tuple[str | None, str | None]:
    """call_deepseek_api：使用 OpenAI Python SDK 调用 DeepSeek API 生成攻略文本。"""

    if OpenAI is None:
        return None, "没有安装 openai 依赖，已使用本地演示攻略。"

    # api_key：DeepSeek API Key，从 .env、环境变量或 Streamlit secrets 读取。
    api_key = get_deepseek_api_key()
    if not api_key:
        return None, "未配置 DEEPSEEK_API_KEY，已使用本地演示攻略。"

    # model_name：当前使用的 DeepSeek 模型名称，可通过 DEEPSEEK_MODEL 修改。
    model_name = get_deepseek_model_name()

    # client：OpenAI SDK 客户端，通过 base_url 指向 DeepSeek 服务。
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # instructions：给模型的角色和输出风格要求。
    instructions = """
你是一名资深旅行编辑和行程规划师，擅长把用户的一句话需求整理成清晰、真实、好执行的旅行攻略。
请用中文输出，语气像旅行杂志编辑，但结构要像实用攻略工具。
不要编造实时价格或实时营业状态；涉及价格时用区间估算，并提醒以出行前查询为准。
""".strip()

    # prompt：最终发送给模型的完整提示词。
    prompt = build_ai_prompt(user_input, parsed_request, facts_context)

    try:
        def create_markdown(prompt_text: str) -> str | None:
            """create_markdown：执行一次 DeepSeek Chat Completions 调用并返回 Markdown 文本。"""

            # response：DeepSeek Chat Completions API 返回的大模型结果。
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt_text},
                ],
            )

            # markdown_content：从模型回复中取出的 Markdown 攻略文本。
            markdown_content = response.choices[0].message.content
            return markdown_content.strip() if markdown_content else None

        # markdown_text：从模型回复中取出的 Markdown 攻略文本。
        markdown_text = create_markdown(prompt)
        if not markdown_text:
            return None, "DeepSeek 返回内容为空，已使用本地演示攻略。"

        # validation_errors：第一次生成后的每日行程校验问题。
        validation_errors = validate_itinerary_markdown(markdown_text, parsed_request)
        if not validation_errors:
            return markdown_text, None

        # retry_prompt：校验失败时给模型的二次生成提示词。
        retry_prompt = build_itinerary_retry_prompt(prompt, validation_errors, parsed_request)

        # retry_markdown：二次生成的 Markdown 攻略文本。
        retry_markdown = create_markdown(retry_prompt)
        if not retry_markdown:
            error_summary = "；".join(validation_errors[:4])
            return markdown_text, f"DeepSeek 二次生成返回内容为空，已展示第一次结果；每日行程可能不完整：{error_summary}"

        # retry_errors：二次生成后再次校验每日行程。
        retry_errors = validate_itinerary_markdown(retry_markdown, parsed_request)
        if retry_errors:
            error_summary = "；".join(retry_errors[:4])
            return retry_markdown, f"每日行程校验未完全通过：{error_summary}。页面会显示解析问题，请重新生成或调整输入。"

        return retry_markdown, "检测到第一次每日行程不完整，已自动重新生成并通过校验。"
    except Exception as error:
        return None, f"DeepSeek API 调用失败，已使用本地演示攻略。错误：{error}"


def build_demo_markdown(parsed_request: dict, facts_context: str = "") -> str:
    """build_demo_markdown：没有 API Key 或 API 失败时生成本地演示攻略。"""

    # destination：攻略目的地。
    destination = parsed_request["destination"]

    # days：旅行天数。
    days = parsed_request["days"]

    # nights：住宿晚数。
    nights = parsed_request["nights"]

    # budget：预算档位。
    budget = parsed_request["budget"]

    # budget_exchange_hint：国外目的地预算粗略换算提示。
    budget_exchange_hint = parsed_request.get("budget_exchange_hint")

    # preferences_text：用户旅行偏好。
    preferences_text = "、".join(parsed_request["preferences"])

    # source_updated_at：本地演示攻略的信息更新时间。
    source_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # source_note：本地演示攻略的信息来源说明。
    source_note = (
        "未配置搜索 API 或搜索失败，本地演示攻略没有使用实时联网来源；门票、预约、开放时间建议出行前再次核对。"
        if "联网事实校验状态：已执行" not in facts_context and "联网事实校验状态：缓存命中" not in facts_context
        else "已传入联网搜索 facts_context；具体门票、预约和开放时间请以搜索来源及出行前二次确认为准。"
    )

    return f"""
## 旅行封面文案
{destination} {days} 天 {nights} 晚旅行计划：把 {preferences_text} 放进行程主线，用轻松但不松散的节奏完成一次有记忆点的城市探索。

## 详细旅游攻略
- 目的地：{destination}
- 行程长度：{days} 天 {nights} 晚
- 预算：{budget}
- 旅行风格：{preferences_text}
- 规划思路：第一天熟悉城市动线，第二天深入主题体验，最后一天安排轻量活动和购物补漏。
{f"- 换算提示：{budget_exchange_hint}" if budget_exchange_hint else ""}

## 每日行程
本地演示模式不会生成每日行程模板。请配置 DEEPSEEK_API_KEY 后由 DeepSeek 按目的地、天数、偏好和联网事实生成完整不重复行程。

## 美食推荐
- 本地代表料理：优先选择评分稳定、翻台快、位置靠近行程路线的店｜人均 80-180 人民币(CNY)｜适合第一顿正式餐
- 街区小吃：适合放在下午或夜间，不要把所有排队店集中到同一天｜人均 30-80 人民币(CNY)｜适合边逛边吃
- 甜品或咖啡：适合安排在步行较多的下午，作为休息点｜人均 40-100 人民币(CNY)｜适合拍照和休息
- 预约型餐厅：如果是热门目的地，建议提前 3 到 7 天确认｜人均 180-400 人民币(CNY)｜适合纪念日晚餐

## 交通建议
- 城市内优先使用地铁、公交或官方交通卡，减少频繁打车。
- 每天尽量围绕一个区域规划，避免跨城式来回移动。
- 机场或车站到酒店先查官方线路，再对比打车价格。
- 如果有大件行李，最后一天优先选择寄存点或酒店寄存。

## 预算估算
- 用户预算：{budget}
{f"- {budget_exchange_hint}" if budget_exchange_hint else ""}
- 交通：经济预算约 150-300 人民币(CNY)/人，普通预算约 300-600 人民币(CNY)/人，高预算按实际打车和跨城交通增加。
- 住宿：经济预算约 300-600 人民币(CNY)/晚，普通预算约 600-1200 人民币(CNY)/晚，高预算约 1200 人民币(CNY)/晚以上。
- 餐饮：约 150-350 人民币(CNY)/人/天，热门餐厅和预约餐厅另算。
- 门票体验：约 100-500 人民币(CNY)/人，主题展、乐园、演出费用可能更高。
- 机动费用：建议预留总预算的 10%-20%。

## 避坑提醒
- 不要把热门景点、热门餐厅和远距离交通挤在同一天。
- 不要只看社交平台种草，出发前确认营业时间、预约方式和交通路线。
- 夜景点通常受天气影响明显，建议保留备选方案。
- 购物和伴手礼尽量放在后半程，避免一路背负行李。
- 本攻略为第一版演示内容，真实出行前请再次确认价格、营业时间和交通信息。

## 信息来源与更新时间
- 更新时间：{source_updated_at}
- 来源说明：{source_note}
- 门票、预约、开放时间：根据搜索结果整理，仍需出行前二次确认；如果没有联网结果，请勿将本地演示内容视为实时信息。
""".strip()


def generate_travel_markdown(user_input: str, parsed_request: dict, facts_context: str) -> tuple[str, str | None]:
    """generate_travel_markdown：优先用大模型生成攻略，失败时回退到本地演示攻略。"""

    # ai_markdown：大模型生成的 Markdown 文本。
    ai_markdown, api_message = call_deepseek_api(user_input, parsed_request, facts_context)
    if ai_markdown:
        return ai_markdown, api_message

    # demo_markdown：本地演示 Markdown 文本。
    demo_markdown = build_demo_markdown(parsed_request, facts_context)
    return demo_markdown, api_message


def call_deepseek_markdown_from_json_api(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
    travel_json: dict,
) -> tuple[str | None, str | None]:
    """call_deepseek_markdown_from_json_api：基于合格 JSON 生成 Markdown 攻略。"""

    # instructions：给模型的 Markdown 写作角色要求。
    instructions = """
你是一名资深旅行编辑和行程规划师。请基于给定 JSON 写中文 Markdown 攻略。
不能改变 JSON 中的行程天数、地点、预算和偏好；不要编造实时营业状态。
""".strip()

    # markdown_prompt：基于结构化 JSON 生成 Markdown 的提示词。
    markdown_prompt = build_markdown_from_json_prompt(user_input, parsed_request, facts_context, travel_json)
    return call_deepseek_chat(markdown_prompt, instructions)


def generate_travel_content(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
) -> tuple[str, str | None, dict | None, str | None, list[str]]:
    """generate_travel_content：先生成结构化 JSON，再基于 JSON 生成 Markdown 攻略。"""

    # travel_json：用于页面时间线渲染的结构化旅行数据。
    travel_json, json_raw, json_errors, json_message = call_deepseek_structured_json_api(
        user_input,
        parsed_request,
        facts_context,
    )

    if not travel_json:
        # demo_markdown：结构化 JSON 失败时仍保留页面其他区域，不使用假行程补齐。
        demo_markdown = build_demo_markdown(parsed_request, facts_context)
        error_summary = "；".join(json_errors[:4]) if json_errors else "结构化 JSON 未生成。"
        api_message = f"每日行程 JSON 未通过校验：{error_summary}"
        if json_message and json_message not in api_message:
            api_message = f"{api_message}；{json_message}"
        return demo_markdown, api_message, None, json_raw, json_errors

    # markdown_text：基于合格 JSON 生成的 Markdown 攻略。
    markdown_text, markdown_message = call_deepseek_markdown_from_json_api(
        user_input,
        parsed_request,
        facts_context,
        travel_json,
    )

    # api_messages：需要展示给用户的生成状态说明。
    api_messages = []
    if json_message:
        api_messages.append(json_message)

    if markdown_text:
        if markdown_message:
            api_messages.append(markdown_message)
        return markdown_text, "；".join(api_messages) or None, travel_json, json_raw, []

    # Markdown 二次生成失败时，用已通过校验的 JSON 生成可复制攻略，不生成假行程。
    fallback_markdown = build_markdown_from_structured_json(travel_json, parsed_request, facts_context)
    if markdown_message:
        api_messages.append(f"Markdown 生成失败，已根据合格 JSON 生成可复制攻略：{markdown_message}")
    else:
        api_messages.append("Markdown 生成失败，已根据合格 JSON 生成可复制攻略。")

    return fallback_markdown, "；".join(api_messages), travel_json, json_raw, []


def generate_cover_image_url(parsed_request: dict) -> str:
    """generate_cover_image_url：生成封面图地址，后续可替换为图片生成 API。"""

    # destination：封面图上显示的目的地。
    destination = parsed_request["destination"]

    # preferences_text：封面图上显示的旅行偏好。
    preferences_text = " / ".join(parsed_request["preferences"][:4])

    # cover_svg：使用旅行杂志感 SVG 占位图，保证没有图片 API 时也能显示大封面。
    cover_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
      <defs>
        <linearGradient id="sky" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#111827"/>
          <stop offset="28%" stop-color="#26324f"/>
          <stop offset="62%" stop-color="#92400e"/>
          <stop offset="100%" stop-color="#020617"/>
        </linearGradient>
        <linearGradient id="sunset" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#fef3c7" stop-opacity="0.86"/>
          <stop offset="42%" stop-color="#fb923c" stop-opacity="0.38"/>
          <stop offset="100%" stop-color="#020617" stop-opacity="0"/>
        </linearGradient>
        <linearGradient id="water" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.72"/>
          <stop offset="52%" stop-color="#14b8a6" stop-opacity="0.42"/>
          <stop offset="100%" stop-color="#f97316" stop-opacity="0.48"/>
        </linearGradient>
        <filter id="grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch"/>
          <feColorMatrix type="saturate" values="0"/>
          <feComponentTransfer>
            <feFuncA type="table" tableValues="0 0.18"/>
          </feComponentTransfer>
        </filter>
        <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="24" stdDeviation="24" flood-color="#020617" flood-opacity="0.42"/>
        </filter>
        <clipPath id="photoClip">
          <rect x="760" y="115" width="470" height="560" rx="32"/>
        </clipPath>
      </defs>
      <rect width="1600" height="900" fill="url(#sky)"/>
      <rect width="1600" height="900" fill="url(#sunset)" opacity="0.62"/>
      <rect width="1600" height="900" filter="url(#grain)" opacity="0.36"/>
      <path d="M0 565 C165 470 305 515 440 430 C570 350 690 392 820 315 C1010 205 1190 295 1600 185 L1600 900 L0 900 Z" fill="#0f172a" opacity="0.76"/>
      <path d="M0 625 C230 508 365 610 545 515 C710 428 865 560 1020 468 C1188 368 1365 438 1600 315 L1600 900 L0 900 Z" fill="#1e293b" opacity="0.72"/>
      <path d="M0 690 C260 590 430 710 675 625 C910 544 1060 690 1308 568 C1435 506 1510 520 1600 475 L1600 900 L0 900 Z" fill="url(#water)" opacity="0.78"/>
      <path d="M0 742 C210 700 390 785 640 735 C880 688 1030 790 1285 708 C1430 662 1510 675 1600 642 L1600 900 L0 900 Z" fill="#020617" opacity="0.70"/>
      <g filter="url(#softShadow)" opacity="0.95">
        <rect x="760" y="115" width="470" height="560" rx="32" fill="#f8fafc" opacity="0.92"/>
        <g clip-path="url(#photoClip)">
          <rect x="760" y="115" width="470" height="560" fill="#0f172a"/>
          <rect x="760" y="115" width="470" height="560" fill="url(#sky)" opacity="0.52"/>
          <circle cx="1110" cy="230" r="72" fill="#fde68a" opacity="0.86"/>
          <path d="M760 430 C845 360 910 380 975 320 C1055 245 1115 360 1230 275 L1230 675 L760 675 Z" fill="#334155"/>
          <path d="M760 520 C900 455 960 550 1080 488 C1145 455 1188 470 1230 430 L1230 675 L760 675 Z" fill="#0f766e" opacity="0.7"/>
          <path d="M760 575 C860 540 940 615 1055 558 C1120 524 1170 540 1230 512 L1230 675 L760 675 Z" fill="#0ea5e9" opacity="0.55"/>
          <path d="M860 675 L1018 444 L1135 675 Z" fill="#f8fafc" opacity="0.82"/>
          <path d="M924 675 L1018 500 L1080 675 Z" fill="#f59e0b" opacity="0.55"/>
        </g>
      </g>
      <path d="M1320 170 C1390 210 1425 268 1450 350" stroke="#fde68a" stroke-width="3" stroke-dasharray="12 16" fill="none" opacity="0.55"/>
      <path d="M1450 350 l28 -12 l-20 31 z" fill="#fde68a" opacity="0.75"/>
      <g opacity="0.78">
        <rect x="120" y="640" width="420" height="3" fill="#fde68a"/>
        <rect x="120" y="665" width="315" height="3" fill="#f8fafc" opacity="0.52"/>
        <rect x="120" y="690" width="250" height="3" fill="#f8fafc" opacity="0.34"/>
      </g>
      <text x="126" y="145" fill="#fde68a" font-size="32" font-family="Arial, sans-serif" letter-spacing="6">AI TRAVEL MAGAZINE</text>
      <text x="126" y="232" fill="#ffffff" font-size="72" font-family="Arial, sans-serif" font-weight="700">{html.escape(destination)}</text>
      <text x="130" y="292" fill="#e5e7eb" font-size="30" font-family="Arial, sans-serif">{html.escape(preferences_text)}</text>
    </svg>
    """

    return "data:image/svg+xml;charset=utf-8," + quote(cover_svg)


def split_markdown_sections(markdown_text: str) -> dict:
    """split_markdown_sections：把 Markdown 按二级标题拆成多个展示卡片。"""

    # section_map：保存标题和正文的对应关系。
    section_map = {}

    # normalized_markdown：保证文本开头有换行，方便正则切分。
    normalized_markdown = "\n" + markdown_text.strip()

    # matches：匹配所有“## 标题”和标题后正文。
    matches = re.finditer(r"\n##\s+(.+?)\n([\s\S]*?)(?=\n##\s+|\Z)", normalized_markdown)
    for match in matches:
        title = match.group(1).strip()
        content = match.group(2).strip()
        section_map[title] = content

    return section_map


def clean_markdown_text(markdown_text: str) -> str:
    """clean_markdown_text：清理 Markdown 符号，方便放进自定义 HTML 卡片。"""

    # cleaned_text：去掉列表符号、粗体和多余空格后的文本。
    cleaned_text = re.sub(r"^[\-\*\d\.\s]+", "", markdown_text.strip())
    cleaned_text = re.sub(r"[*`#]+", "", cleaned_text)
    return cleaned_text.strip()


def extract_bullet_items(section_text: str, max_items: int = 6) -> list[str]:
    """extract_bullet_items：从 Markdown 段落中提取列表项。"""

    # item_list：从 Markdown 中提取出的列表内容。
    item_list = []
    for line in section_text.splitlines():
        stripped_line = line.strip()
        if re.match(r"^[-*]\s+", stripped_line) or re.match(r"^\d+[.、]\s+", stripped_line):
            item = clean_markdown_text(stripped_line)
            if item:
                item_list.append(item)

    if not item_list and section_text.strip():
        # fallback_lines：当模型没有使用列表时，按非空行兜底提取。
        fallback_lines = [clean_markdown_text(line) for line in section_text.splitlines() if clean_markdown_text(line)]
        item_list = fallback_lines

    return item_list[:max_items]


def is_markdown_table_separator(line_text: str) -> bool:
    """is_markdown_table_separator：判断是否为 Markdown 表格分隔线。"""

    # normalized_text：去掉空格后的表格行文本。
    normalized_text = line_text.strip().replace(" ", "")
    if not normalized_text:
        return False

    return bool(re.fullmatch(r"\|?[:\-|]+\|?", normalized_text))


def is_budget_noise_line(line_text: str) -> bool:
    """is_budget_noise_line：过滤预算区域中的空行、表头、分隔线和无意义符号行。"""

    # cleaned_line：清理后的单行文本。
    cleaned_line = clean_markdown_text(line_text).strip()
    if not cleaned_line:
        return True

    if is_markdown_table_separator(line_text):
        return True

    # symbol_only：只包含 Markdown 表格符号或横线的行。
    symbol_only = re.sub(r"[\s\|:\-—–]+", "", line_text.strip())
    if not symbol_only:
        return True

    # table_header_words：常见预算表头字段。
    table_header_words = ["项目", "预算", "说明", "费用", "金额", "备注"]
    if "|" in line_text:
        table_cells = [cell.strip() for cell in line_text.strip().strip("|").split("|")]
        normalized_cells = [re.sub(r"\s+", "", cell) for cell in table_cells if cell.strip()]
        if normalized_cells and all(any(word in cell for word in table_header_words) for cell in normalized_cells):
            return True

    return False


def parse_budget_table_rows(section_text: str) -> list[dict]:
    """parse_budget_table_rows：优先把 Markdown 表格解析成预算卡片数据。"""

    def clean_budget_table_cell(cell_text: str) -> str:
        """clean_budget_table_cell：清理预算表格单元格但保留数字金额。"""

        return re.sub(r"[*`#]+", "", cell_text.strip()).strip()

    # budget_items：解析后的预算项目。
    budget_items = []
    for line in section_text.splitlines():
        stripped_line = line.strip()
        if "|" not in stripped_line or is_budget_noise_line(stripped_line):
            continue

        # table_cells：表格单元格，过滤空单元。
        table_cells = [clean_budget_table_cell(cell) for cell in stripped_line.strip().strip("|").split("|")]
        table_cells = [cell for cell in table_cells if cell]
        if len(table_cells) < 2:
            continue

        # title：预算项目名称。
        title = table_cells[0]
        if title in {"项目", "预算", "说明", "费用", "金额", "备注"}:
            continue

        # description_parts：预算金额和说明。
        description_parts = table_cells[1:]
        description = "；".join(description_parts)
        if not description or is_budget_noise_line(description):
            continue

        budget_items.append({"title": title[:18], "description": description})

    return budget_items


def build_budget_items(section_text: str, max_items: int = 6) -> list[dict]:
    """build_budget_items：解析预算估算内容，过滤 Markdown 表头和空卡片。"""

    # table_items：优先解析 Markdown 表格。
    table_items = parse_budget_table_rows(section_text)
    if table_items:
        return table_items[:max_items]

    # raw_items：从列表或普通行提取出的候选预算项。
    raw_items = extract_bullet_items(section_text, max_items=max_items * 2)

    # budget_items：过滤后的预算卡片数据。
    budget_items = []
    for raw_item in raw_items:
        if is_budget_noise_line(raw_item):
            continue

        # title/description：预算项标题和说明。
        title = "预算项"
        description = raw_item
        if "：" in raw_item:
            title, description = raw_item.split("：", 1)
        elif ":" in raw_item:
            title, description = raw_item.split(":", 1)

        title = clean_markdown_text(title)[:18] or "预算项"
        description = clean_markdown_text(description)
        if not description or is_budget_noise_line(description):
            continue

        budget_items.append({"title": title, "description": description})

    return budget_items[:max_items]


def estimate_total_cost(parsed_request: dict) -> str:
    """estimate_total_cost：根据天数和预算档位估算不含大交通的人均总花费。"""

    if parsed_request.get("budget_has_explicit_amount"):
        return f"按 {parsed_request['budget']} 控制"

    # budget_level：用户预算档位。
    budget_level = parsed_request.get("budget_level", parsed_request["budget"])

    # days：旅行天数。
    days = parsed_request["days"]

    # nights：住宿晚数。
    nights = parsed_request["nights"]

    if budget_level == "经济预算":
        day_cost, night_cost = 260, 320
    elif budget_level == "高预算":
        day_cost, night_cost = 980, 1600
    else:
        day_cost, night_cost = 480, 720

    # low_cost：较低估算值。
    low_cost = days * day_cost + nights * night_cost

    # high_cost：较高估算值。
    high_cost = int(low_cost * 1.35)

    return f"约 {low_cost:,}-{high_cost:,} 人民币(CNY)/人"


def infer_trip_pace(parsed_request: dict) -> str:
    """infer_trip_pace：根据天数和偏好推断旅行节奏。"""

    # preferences：用户偏好列表。
    preferences = parsed_request["preferences"]

    if parsed_request["days"] >= 6 or any(item in preferences for item in ["自然", "咖啡", "温泉", "海边"]):
        return "松弛慢旅行"
    if parsed_request["days"] <= 3 and any(item in preferences for item in ["购物", "夜景", "动漫"]):
        return "高效城市探索"
    return "舒适均衡节奏"


def infer_audience(parsed_request: dict) -> str:
    """infer_audience：根据偏好推断适合人群。"""

    # preferences：用户偏好列表。
    preferences = parsed_request["preferences"]

    if "亲子" in preferences:
        return "家庭与亲子出行"
    if any(item in preferences for item in ["动漫", "购物", "夜景"]):
        return "城市玩家与潮流爱好者"
    if any(item in preferences for item in ["自然", "徒步", "海边"]):
        return "自然风景和慢旅行人群"
    return "第一次到访和自由行用户"


def build_summary_metrics(parsed_request: dict) -> dict:
    """build_summary_metrics：生成攻略摘要区需要展示的指标。"""

    # preferences_count：偏好数量，用于生成推荐强度。
    preferences_count = len(parsed_request["preferences"])

    # recommendation_score：推荐强度评分。
    recommendation_score = "4.9 / 5" if preferences_count >= 3 else "4.6 / 5"

    return {
        "推荐强度": recommendation_score,
        "旅行节奏": infer_trip_pace(parsed_request),
        "适合人群": infer_audience(parsed_request),
        "预计总花费": estimate_total_cost(parsed_request),
    }


def extract_slot_text(day_text: str, slot_label: str) -> str:
    """extract_slot_text：从某一天行程中提取上午、中午、下午或晚上的内容。"""

    # slot_pattern：匹配指定时间段的 Markdown 行。
    slot_pattern = rf"(?:^|\n)\s*[-*]?\s*(?:\*\*)?{slot_label}(?:\*\*)?\s*[：:]\s*(.+)"
    match = re.search(slot_pattern, day_text)
    if match:
        return clean_markdown_text(match.group(1))
    return ""


def split_place_and_description(slot_text: str) -> tuple[str, str]:
    """split_place_and_description：把时间段内容拆成地点和说明。"""

    # parts：按中文或英文竖线拆开的地点和说明。
    parts = [part.strip() for part in re.split(r"[｜|]", slot_text, maxsplit=1)]
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]

    return "", slot_text


def build_timeline_days(section_map: dict, parsed_request: dict) -> tuple[list[dict], list[str]]:
    """build_timeline_days：把每日行程 Markdown 转成时间线数据。"""

    # itinerary_text：每日行程 Markdown 内容。
    itinerary_text = section_map.get("每日行程", "").strip()
    if not itinerary_text:
        return [], ["未找到“每日行程”区域。"]

    # timeline_markdown：补回二级标题，复用统一的 Day 块解析函数。
    timeline_markdown = f"## 每日行程\n{itinerary_text}"

    # day_blocks：从 Markdown 中解析出的每日行程块。
    day_blocks = parse_itinerary_day_blocks(timeline_markdown)

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    # validation_errors：时间线渲染前的结构化校验问题。
    validation_errors = []

    if len(day_blocks) != expected_days:
        validation_errors.append(f"模型返回 {len(day_blocks)} 天行程，系统识别用户需要 {expected_days} 天。")

    # day_map：按 Day 编号索引每日行程，方便检查缺失天数。
    day_map = {day_block["day"]: day_block for day_block in day_blocks}
    missing_days = [day for day in range(1, expected_days + 1) if day not in day_map]
    if missing_days:
        validation_errors.append(f"缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    # slot_config：时间线四个固定时段。
    slot_config = [
        {"label": "上午", "time": "09:00 - 11:30", "icon": "AM"},
        {"label": "中午", "time": "12:00 - 13:30", "icon": "NO"},
        {"label": "下午", "time": "14:00 - 17:30", "icon": "PM"},
        {"label": "晚上", "time": "18:30 - 21:30", "icon": "EV"},
    ]

    # generic_terms：不允许出现在时间线里的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # seen_places：已出现地点集合，用于避免同一地点重复安排。
    seen_places = set()

    # seen_themes：已出现主题集合，用于避免每天主题重复。
    seen_themes = set()

    # timeline_days：最终时间线数据。
    timeline_days = []
    for day_number in range(1, expected_days + 1):
        day_block = day_map.get(day_number)
        if not day_block:
            continue

        # day_theme：当前 Day 的主题文本。
        day_theme = clean_markdown_text(day_block["theme"])
        if day_theme in seen_themes:
            validation_errors.append(f"Day {day_number} 主题重复：{day_theme}。")
        seen_themes.add(day_theme)

        # slot_list：单日四个时间段的数据。
        slot_list = []
        for slot in slot_config:
            slot_text = extract_slot_text(day_block["body"], slot["label"])
            if not slot_text:
                validation_errors.append(f"Day {day_number} 缺少{slot['label']}安排。")
                continue

            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_number} {slot['label']}仍包含空泛模板词：{slot_text[:40]}")

            # slot_parts：时间段内容必须包含地点、推荐理由、预计耗时、交通或预约提醒。
            slot_parts = [part.strip() for part in re.split(r"[｜|]", slot_text) if part.strip()]
            if len(slot_parts) < 4:
                validation_errors.append(
                    f"Day {day_number} {slot['label']}未完整包含地点、推荐理由、预计耗时、交通或预约提醒。"
                )
                continue

            place, description = split_place_and_description(slot_text)
            if not place:
                validation_errors.append(f"Day {day_number} {slot['label']}未按“具体地点｜推荐理由｜预计耗时｜交通或预约提醒”格式输出。")
                continue

            if place in seen_places:
                validation_errors.append(f"重复安排地点：{place}。")
            seen_places.add(place)

            slot_list.append(
                {
                    "label": slot["label"],
                    "time": slot["time"],
                    "icon": slot["icon"],
                    "place": place,
                    "description": description,
                }
            )

        timeline_days.append({"title": f"Day {day_number}：{day_block['theme']}", "slots": slot_list})

    if validation_errors:
        return [], validation_errors

    return timeline_days, []


def build_timeline_days_from_json(travel_json: dict | None, parsed_request: dict) -> tuple[list[dict], list[str]]:
    """build_timeline_days_from_json：把结构化 JSON 转成时间线卡片数据。"""

    # validation_errors：结构化 JSON 的校验错误。
    validation_errors = validate_structured_travel_json(travel_json, parsed_request)
    if validation_errors:
        return [], validation_errors

    # slot_config：英文 JSON 字段和页面展示标签的对应关系。
    slot_config = [
        {"key": "morning", "label": "上午", "icon": "AM"},
        {"key": "noon", "label": "中午", "icon": "NO"},
        {"key": "afternoon", "label": "下午", "icon": "PM"},
        {"key": "evening", "label": "晚上", "icon": "EV"},
    ]

    # timeline_days：最终时间线数据。
    timeline_days = []
    for day_item in travel_json.get("daily_itinerary", []):
        # segment_destination：多目的地时展示当前日期所属目的地。
        segment_destination = str(day_item.get("segment_destination", "")).strip()

        # title_prefix：多目的地时间线标题前缀。
        title_prefix = f"{segment_destination}｜" if segment_destination else ""

        # slot_list：单日四个时间段的数据。
        slot_list = []
        for slot in slot_config:
            # slot_data：结构化 JSON 中的时间段对象。
            slot_data = day_item[slot["key"]]
            description = (
                f"{slot_data['original_name']}｜{slot_data['reason']}｜{slot_data['duration']}｜"
                f"{slot_data['transport']}；{slot_data['booking_note']}"
            )
            slot_list.append(
                {
                    "label": slot["label"],
                    "time": slot_data["time"],
                    "icon": slot["icon"],
                    "place": slot_data["place"],
                    "description": description,
                    "original_name": slot_data["original_name"],
                    "reason": slot_data["reason"],
                    "duration": slot_data["duration"],
                    "transport": slot_data["transport"],
                    "booking_note": slot_data["booking_note"],
                }
            )

        timeline_days.append({"title": f"Day {day_item['day']}：{title_prefix}{day_item['theme']}", "slots": slot_list})

    return timeline_days, []


def build_food_cards(section_map: dict, parsed_request: dict) -> list[dict]:
    """build_food_cards：把美食推荐 Markdown 转成美食卡片数据。"""

    # food_items：美食推荐列表。
    food_items = extract_bullet_items(section_map.get("美食推荐", ""), max_items=6)

    if not food_items:
        food_items = [
            "本地代表料理：优先选择路线附近的高评分店｜人均 80-180 元｜适合第一顿正式餐",
            "街区小吃：适合放在下午或夜间，边走边吃更轻松｜人均 30-80 元｜适合探索街区",
            "甜品或咖啡：作为下午休息点，也适合拍照｜人均 40-100 元｜适合慢旅行",
        ]

    # budget_level：预算档位，用于美食卡片的人均预算兜底。
    budget_level = parsed_request.get("budget_level", parsed_request["budget"])

    # default_budget：美食卡片的人均预算兜底。
    default_budget = "人均 80-180 人民币(CNY)" if budget_level != "经济预算" else "人均 30-90 人民币(CNY)"

    # food_cards：最终美食卡片数据。
    food_cards = []
    for item in food_items:
        title = item
        detail = "结合行程路线选择，减少排队和跨区移动。"
        if "：" in item:
            title, detail = item.split("：", 1)
        elif ":" in item:
            title, detail = item.split(":", 1)

        # detail_parts：按竖线拆出的理由、预算和场景。
        detail_parts = [part.strip() for part in re.split(r"[｜|]", detail) if part.strip()]
        reason = detail_parts[0] if detail_parts else detail
        budget = detail_parts[1] if len(detail_parts) > 1 else default_budget
        scene = detail_parts[2] if len(detail_parts) > 2 else "适合穿插在当日行程中"

        food_cards.append(
            {
                "title": clean_markdown_text(title)[:34],
                "reason": clean_markdown_text(reason),
                "budget": clean_markdown_text(budget),
                "scene": clean_markdown_text(scene),
                "location": "位置：建议结合当日行程区域确认",
                "nearby_spot": "当日行程附近",
                "booking_note": "热门时段建议提前确认或预约",
                "map_keyword": clean_markdown_text(title)[:34],
            }
        )

    return food_cards


def build_food_cards_from_json(travel_json: dict | None) -> list[dict]:
    """build_food_cards_from_json：把结构化 JSON 中的美食推荐转成卡片数据。"""

    if not isinstance(travel_json, dict):
        return []

    # food_recommendations：结构化 JSON 中的美食推荐列表。
    food_recommendations = travel_json.get("food_recommendations", [])
    if not isinstance(food_recommendations, list):
        return []

    # food_cards：最终美食卡片数据。
    food_cards = []
    for food_item in food_recommendations[:6]:
        if not isinstance(food_item, dict):
            continue

        # name_cn/name_original：店铺中文名与英文/当地原名。
        name_cn = clean_markdown_text(str(food_item.get("name_cn", "")))
        name_original = clean_markdown_text(str(food_item.get("name_original", "")))
        title = f"{name_cn}（{name_original}）" if name_original and name_original != name_cn else name_cn

        food_cards.append(
            {
                "title": title or "待确认美食点",
                "location": clean_markdown_text(str(food_item.get("location", ""))) or "建议以地图搜索原名确认",
                "nearby_spot": clean_markdown_text(str(food_item.get("nearby_spot", ""))) or "适合穿插在当日行程中",
                "reason": clean_markdown_text(str(food_item.get("reason", ""))) or "结合行程路线选择，减少跨区移动。",
                "budget": clean_markdown_text(str(food_item.get("budget", ""))) or "人均预算待确认",
                "scene": clean_markdown_text(str(food_item.get("scene", ""))) or "适合穿插在当日行程中",
                "booking_note": clean_markdown_text(str(food_item.get("booking_note", ""))) or "热门时段建议提前确认或预约",
                "map_keyword": clean_markdown_text(str(food_item.get("map_keyword", ""))) or title,
            }
        )

    return food_cards


def build_advice_cards(section_text: str, fallback_items: list[str], max_items: int = 4) -> list[dict]:
    """build_advice_cards：把交通建议或避坑提醒转成卡片数据。"""

    # advice_items：从 Markdown 中提取出的建议列表。
    advice_items = extract_bullet_items(section_text, max_items=max_items) or fallback_items

    # advice_cards：最终建议卡片数据。
    advice_cards = []
    for index, item in enumerate(advice_items[:max_items], start=1):
        title = f"建议 {index}"
        description = item
        if "：" in item and len(item.split("：", 1)[0]) <= 16:
            title, description = item.split("：", 1)
        elif ":" in item and len(item.split(":", 1)[0]) <= 16:
            title, description = item.split(":", 1)
        else:
            title = clean_markdown_text(item)[:14]

        advice_cards.append({"title": clean_markdown_text(title), "description": clean_markdown_text(description)})

    return advice_cards


def render_hero() -> None:
    """render_hero：渲染页面顶部的产品标题区。"""

    st.markdown(
        """
        <nav class="top-nav">
            <div class="nav-brand"><span class="brand-mark">T</span><span>TripAgent</span></div>
            <div class="nav-links">
                <span>AI旅行规划</span>
                <span>示例</span>
                <span>反馈</span>
            </div>
        </nav>
        <section class="hero product-hero">
            <div class="hero-layout">
                <div>
                    <div class="eyebrow">AI Private Travel Advisor · Magazine Edition</div>
                    <h1 class="hero-title"><span class="hero-title-line">一句话生成</span><span class="hero-title-line">你的专属旅行路线</span></h1>
                    <p>输入目的地、天数和偏好，AI 为你规划每日行程、美食、预算、交通、天气与避坑提醒。</p>
                    <div class="hero-proof">
                        <span>多目的地连续规划</span>
                        <span>天气与出行提醒</span>
                        <span>Markdown 一键带走</span>
                    </div>
                </div>
                <aside class="hero-panel product-preview">
                    <div class="preview-cover">
                        <div class="preview-cover-label">
                            <span>AI Travel Magazine</span>
                            <strong>Nanjing × Jiangxi<br>7 Days Journey</strong>
                        </div>
                    </div>
                    <div class="preview-steps">
                        <div class="preview-step">
                            <b>01</b>
                            <div><span>Route</span><p>自动拆分多目的地，每天主题不同。</p></div>
                        </div>
                        <div class="preview-step">
                            <b>02</b>
                            <div><span>Food & Weather</span><p>美食、天气、预算和避坑提醒一起整理。</p></div>
                        </div>
                        <div class="preview-step">
                            <b>03</b>
                            <div><span>Export</span><p>生成可复制、可下载的 Markdown 攻略。</p></div>
                        </div>
                    </div>
                </aside>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_box() -> tuple[bool, str]:
    """render_input_box：渲染醒目的自然语言输入框。"""

    with st.container(border=True):
        st.markdown('<p class="input-kicker">Start with one sentence</p>', unsafe_allow_html=True)
        st.markdown('<p class="input-title">告诉我你想怎么旅行</p>', unsafe_allow_html=True)
        st.markdown('<p class="sample-title">选择一个示例，或直接输入你的旅行需求</p>', unsafe_allow_html=True)

        # sample_columns：用于横向排列示例标签按钮。
        sample_columns = st.columns(4)
        for index, sample_prompt in enumerate(SAMPLE_PROMPTS):
            if sample_columns[index].button(sample_prompt["label"], key=f"sample_prompt_{index}"):
                st.session_state["travel_request_input"] = sample_prompt["prompt"]

        with st.form("travel_request_form"):
            # user_input：用户输入的一句话旅行需求。
            user_input = st.text_area(
                label="旅行需求",
                label_visibility="collapsed",
                placeholder="例如：我想去南京游玩3天，然后再去江西游玩，喜欢历史文化、美食和夜景，预算8000",
                key="travel_request_input",
            )

            # submitted：用户是否点击了生成按钮。
            submitted = st.form_submit_button("生成专属旅行方案")

        st.markdown('<p class="hint">不用填复杂表单，一句话就够。没写天数默认 3 天 2 晚；预算数字没写单位时默认人民币 CNY。</p>', unsafe_allow_html=True)

    return submitted, user_input


def render_cover(parsed_request: dict, cover_image_url: str) -> None:
    """render_cover：渲染大封面图区域。"""

    # cover_destination_text：封面专用目的地文本，多目的地用“×”营造旅行杂志标题感。
    cover_destination_text = " × ".join(parsed_request.get("destinations", [])) or parsed_request["destination"]

    # safe_destination：转义后的目的地文本，避免 HTML 注入。
    safe_destination = html.escape(cover_destination_text)

    # safe_preferences：转义后的偏好文本。
    safe_preferences = html.escape("、".join(parsed_request["preferences"]))

    # safe_cover_image_url：转义后的封面图片地址。
    safe_cover_image_url = html.escape(cover_image_url, quote=True)

    # badge_items：封面上展示的旅行关键信息标签。
    badge_items = [
        f"{parsed_request['days']} 天 {parsed_request['nights']} 晚",
        parsed_request["budget"],
        *parsed_request["preferences"][:4],
    ]

    # badge_html：封面标签 HTML。
    badge_html = "".join(f'<span class="cover-badge">{html.escape(item)}</span>' for item in badge_items)

    st.markdown(
        f"""
        <section class="cover-card" style='background-image: url("{safe_cover_image_url}");'>
            <div class="cover-content">
                <div class="label">AI TRAVEL MAGAZINE</div>
                <h2>{safe_destination}</h2>
                <div class="cover-dayline">{parsed_request["days"]} Days Journey · {html.escape(parsed_request["budget"])}</div>
                <p>{safe_preferences} · 由 AI 生成的旅行封面与城市探索计划</p>
                <div class="cover-badges">{badge_html}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_summary_bento(parsed_request: dict) -> None:
    """render_summary_bento：用 bento grid 展示攻略摘要信息。"""

    # preferences_text：用于展示的偏好文本。
    preferences_text = "、".join(parsed_request["preferences"])

    # metric_map：推荐强度、旅行节奏、适合人群和预计总花费。
    metric_map = build_summary_metrics(parsed_request)

    # budget_note：预算卡片中的说明文本。
    budget_note = parsed_request.get("budget_exchange_hint") or "价格为区间估算，出发前需再次确认。"

    st.markdown(
        f"""
        <h2 class="section-heading">攻略摘要</h2>
        <p class="section-subtitle">系统从你的自然语言输入中提取旅行关键参数，并补充可执行的规划指标。</p>
        <div class="bento-grid">
            <div class="bento-card large warm"><span>目的地</span><strong>{html.escape(parsed_request["destination"])}</strong><p>本次攻略围绕城市动线、主题偏好和轻量避坑提醒展开。</p></div>
            <div class="bento-card"><span>旅行天数</span><strong>{parsed_request["days"]} 天 {parsed_request["nights"]} 晚</strong><p>按每日四段式节奏规划。</p></div>
            <div class="bento-card"><span>预算</span><strong>{html.escape(parsed_request["budget"])}</strong><p>{html.escape(budget_note)}</p></div>
            <div class="bento-card large"><span>偏好标签</span><strong>{html.escape(preferences_text)}</strong><p>用于安排主题街区、美食和拍照点。</p></div>
            <div class="bento-card"><span>推荐强度</span><strong>{html.escape(metric_map["推荐强度"])}</strong><p>基于偏好匹配度估算。</p></div>
            <div class="bento-card"><span>旅行节奏</span><strong>{html.escape(metric_map["旅行节奏"])}</strong><p>兼顾体验密度和休息时间。</p></div>
            <div class="bento-card"><span>适合人群</span><strong>{html.escape(metric_map["适合人群"])}</strong><p>可按同行人群继续微调。</p></div>
            <div class="bento-card warm"><span>预计总花费</span><strong>{html.escape(metric_map["预计总花费"])}</strong><p>不含跨城机票或长途交通。</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trip_segments_overview(parsed_request: dict) -> None:
    """render_trip_segments_overview：展示多目的地或推断天数的分段总览。"""

    # trip_segments：系统识别出的目的地分段。
    trip_segments = parsed_request.get("trip_segments", [])
    if not trip_segments:
        return

    # should_show_segments：多目的地或存在默认推断说明时展示分段总览。
    should_show_segments = parsed_request.get("trip_type") == "multi_destination" or bool(parsed_request.get("trip_notes"))
    if not should_show_segments:
        return

    # segment_cards：每个目的地分段的卡片 HTML。
    segment_cards = []
    current_day_start = 1
    for segment in trip_segments:
        # destination：分段目的地。
        destination = str(segment.get("destination", "")).strip()

        # segment_days：分段天数。
        segment_days = int(segment.get("days", DEFAULT_TRAVEL_DAYS))

        # day_range：页面展示的连续 Day 范围。
        day_range = f"Day {current_day_start}-Day {current_day_start + segment_days - 1}"
        current_day_start += segment_days

        # inferred_label：未写天数时明确标注默认推断。
        inferred_label = "默认 " if segment.get("days_inferred") else ""

        # note：分段说明，省份/大区域或默认天数提示。
        note = str(segment.get("note", "")).strip() or "按用户输入的目的地和偏好生成分段路线。"

        segment_cards.append(
            '<article class="segment-card">'
            f'<span>{html.escape(day_range)}</span>'
            f'<strong>{html.escape(destination)} · {inferred_label}{segment_days} 天 {max(0, segment_days - 1)} 晚</strong>'
            f'<p>{html.escape(note)}</p>'
            "</article>"
        )

    # segment_html：无缩进 HTML，避免 Markdown 把 HTML 识别为代码块。
    segment_html = (
        '<h2 class="section-heading">行程分段总览</h2>'
        '<p class="section-subtitle">多目的地会按连续日期拆分；未说明天数的目的地会明确标注默认规划。</p>'
        f'<div class="segment-overview-grid">{"".join(segment_cards)}</div>'
    )
    st.markdown(segment_html, unsafe_allow_html=True)


def render_overview_card(section_map: dict) -> None:
    """render_overview_card：展示详细攻略的简要说明卡片。"""

    # overview_text：详细旅游攻略内容。
    overview_text = section_map.get("详细旅游攻略", "")
    if not overview_text:
        return

    with st.container(border=True):
        st.markdown("### 旅行编辑摘要")
        st.markdown(overview_text)


def render_timeline(
    section_map: dict,
    parsed_request: dict,
    travel_json: dict | None = None,
    json_errors: list[str] | None = None,
    json_raw: str | None = None,
) -> None:
    """render_timeline：用时间线样式展示每日行程。"""

    # timeline_days：优先从结构化 JSON 构建每日行程时间线数据。
    if travel_json:
        timeline_days, timeline_errors = build_timeline_days_from_json(travel_json, parsed_request)
    else:
        timeline_days = []
        timeline_errors = json_errors or ["未生成可用于页面渲染的结构化 JSON。"]

    if timeline_errors:
        st.markdown(
            """
            <h2 class="section-heading">每日行程时间线</h2>
            <p class="section-subtitle">每日行程没有通过完整性校验，因此没有使用默认模板补齐。</p>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.error("每日行程结构不完整。请重新生成，系统不会用重复模板自动补齐。")
            st.markdown("- 请点击页面上方的“生成专属旅行攻略”重新生成。")
            if is_debug_enabled():
                with st.expander("开发者调试信息", expanded=False):
                    for timeline_error in timeline_errors[:8]:
                        st.markdown(f"- {timeline_error}")
                    if len(timeline_errors) > 8:
                        st.markdown(f"- 还有 {len(timeline_errors) - 8} 条校验问题未展示。")
                    if json_raw:
                        st.code(json_raw, language="json")
                    else:
                        st.markdown("未获取到结构化 JSON 原文。")
        return

    # day_html_list：每天独立卡片 HTML。
    day_html_list = []
    for day in timeline_days:
        slot_html_list = []
        for slot in day["slots"]:
            # original_name/reason/duration/transport/booking_note：结构化 JSON 中的时间段详情。
            original_name = html.escape(slot.get("original_name", ""))
            reason = html.escape(slot.get("reason", slot.get("description", "")))
            duration = html.escape(slot.get("duration", ""))
            transport = html.escape(slot.get("transport", ""))
            booking_note = html.escape(slot.get("booking_note", ""))

            # slot_detail_html：分层展示的时间段信息，避免大段文字堆叠。
            slot_detail_html = (
                f'<div class="slot-original">{original_name}</div>'
                f'<p class="slot-desc">{reason}</p>'
                '<div class="slot-meta-grid">'
                f'<div class="slot-meta-item"><strong>耗时</strong><br>{duration}</div>'
                f'<div class="slot-meta-item"><strong>交通</strong><br>{transport}</div>'
                f'<div class="slot-meta-item"><strong>预约/注意</strong><br>{booking_note}</div>'
                "</div>"
            )
            slot_html_list.append(
                '<div class="timeline-slot">'
                f'<div class="slot-icon">{html.escape(slot["icon"])}</div>'
                "<div>"
                f'<div class="slot-time">{html.escape(slot["label"])} · {html.escape(slot["time"])}</div>'
                f'<div class="slot-place">{html.escape(slot["place"])}</div>'
                f"{slot_detail_html}"
                "</div>"
                "</div>"
            )

        day_html_list.append(
            '<article class="timeline-day">'
            f'<h3>{html.escape(day["title"])}</h3>'
            f'{"".join(slot_html_list)}'
            "</article>"
        )

    # timeline_html：无缩进 HTML，避免 Markdown 把 HTML 识别为代码块。
    timeline_html = (
        '<h2 class="section-heading">每日行程时间线</h2>'
        '<p class="section-subtitle">每天拆成上午、中午、下午和晚上四个时间段，便于实际执行。</p>'
        f'<div class="timeline-grid">{"".join(day_html_list)}</div>'
    )

    st.markdown(timeline_html, unsafe_allow_html=True)


def render_food_cards(section_map: dict, parsed_request: dict, travel_json: dict | None = None) -> None:
    """render_food_cards：用卡片展示美食推荐。"""

    # food_cards：美食卡片数据。
    food_cards = build_food_cards_from_json(travel_json) or build_food_cards(section_map, parsed_request)

    # food_html：美食卡片 HTML。
    food_html = ""
    for food in food_cards:
        food_html += f"""
        <article class="food-card">
            <h3>{html.escape(food["title"])}</h3>
            <div class="food-location">📍 位置：{html.escape(food["location"])}，靠近 {html.escape(food["nearby_spot"])}</div>
            <p>{html.escape(food["reason"])}</p>
            <div class="food-map-keyword">地图搜索：{html.escape(food["map_keyword"])}</div>
            <div class="food-meta">
                <span>{html.escape(food["budget"])}</span>
                <span>{html.escape(food["scene"])}</span>
                <span>{html.escape(food["booking_note"])}</span>
            </div>
        </article>
        """

    st.markdown(
        f"""
        <h2 class="section-heading">美食推荐</h2>
        <p class="section-subtitle">把餐饮当成行程体验的一部分，而不是临时补位。</p>
        <div class="food-grid">{food_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_advice_sections(section_map: dict) -> None:
    """render_advice_sections：展示交通建议和避坑提醒。"""

    # transport_fallback：交通建议兜底内容。
    transport_fallback = [
        "城市内优先使用地铁、公交或官方交通卡，减少频繁打车。",
        "每天尽量围绕一个区域规划，避免跨城式来回移动。",
        "机场或车站到酒店先查官方线路，再对比打车价格。",
        "最后一天优先选择寄存点或酒店寄存，减少拖行李时间。",
    ]

    # warning_fallback：避坑提醒兜底内容。
    warning_fallback = [
        "不要把热门景点、热门餐厅和远距离交通挤在同一天。",
        "出发前确认营业时间、预约方式和交通路线。",
        "夜景点受天气影响明显，建议保留备选方案。",
        "购物和伴手礼尽量放在后半程，避免一路背负行李。",
    ]

    # transport_cards：交通建议卡片数据。
    transport_cards = build_advice_cards(section_map.get("交通建议", ""), transport_fallback, max_items=4)

    # warning_cards：避坑提醒卡片数据。
    warning_cards = build_advice_cards(section_map.get("避坑提醒", ""), warning_fallback, max_items=4)

    # budget_items：预算估算列表，过滤 Markdown 表头、分隔线和空项目。
    budget_items = build_budget_items(section_map.get("预算估算", ""), max_items=6)

    transport_html = "".join(
        f"""
        <article class="info-card">
            <div class="card-title-row">
                <div class="info-icon">i</div>
                <h3>{html.escape(card["title"])}</h3>
            </div>
            <p>{html.escape(card["description"])}</p>
        </article>
        """
        for card in transport_cards
    )

    warning_html = "".join(
        f"""
        <article class="warning-card">
            <div class="card-title-row">
                <div class="warning-icon">!</div>
                <h3>{html.escape(card["title"])}</h3>
            </div>
            <p>{html.escape(card["description"])}</p>
        </article>
        """
        for card in warning_cards
    )

    st.markdown(
        f"""
        <h2 class="section-heading">交通建议</h2>
        <p class="section-subtitle">优先减少无效移动，把时间留给真正的体验。</p>
        <div class="info-grid">{transport_html}</div>
        """,
        unsafe_allow_html=True,
    )

    if budget_items:
        # budget_html：预算估算卡片 HTML。
        budget_html = "".join(
            '<article class="budget-card">'
            f'<span>{html.escape(budget_item["title"])}</span>'
            f'<p>{html.escape(budget_item["description"])}</p>'
            "</article>"
            for budget_item in budget_items
        )
        # budget_section_html：无缩进 HTML，避免 Markdown 把卡片识别为代码块。
        budget_section_html = (
            '<h2 class="section-heading">预算估算</h2>'
            '<p class="section-subtitle">把花费拆成交通、住宿、餐饮和机动预算，方便你出发前调整。</p>'
            f'<div class="budget-grid">{budget_html}</div>'
        )
        st.markdown(budget_section_html, unsafe_allow_html=True)

    st.markdown(
        f"""
        <h2 class="section-heading">避坑提醒</h2>
        <p class="section-subtitle">提前规避高概率踩坑点，让行程更稳定。</p>
        <div class="warning-grid">{warning_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_source_info(section_map: dict, generated_at: str | None = None) -> None:
    """render_source_info：展示信息来源与更新时间区域。"""

    # source_text：模型生成的来源与更新时间内容，普通用户界面不直接展示技术状态。
    source_text = section_map.get("信息来源与更新时间", "")

    with st.container(border=True):
        st.markdown('<div class="source-card-title">信息与更新时间</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <p class="source-card-text">本攻略由 AI 根据你的输入和当前可用信息整理生成。</p>
            <p class="source-card-text">更新时间：{html.escape(generated_at or datetime.now().strftime('%Y-%m-%d %H:%M'))}</p>
            <p class="source-card-text">门票、预约、开放时间、交通政策和天气情况可能变化，请出行前以官方渠道和天气 App 为准。</p>
            """,
            unsafe_allow_html=True,
        )
        if source_text and is_debug_enabled():
            with st.expander("开发者调试信息", expanded=False):
                st.markdown(source_text)


def render_weather_section(weather_cards: list[dict] | None) -> None:
    """render_weather_section：渲染天气与出行提醒卡片。"""

    st.markdown('<h2 class="section-heading">天气与出行提醒 🌦️</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">根据最近几天的天气情况整理携带建议，出发前请再用天气 App 复核一次。</p>',
        unsafe_allow_html=True,
    )

    if not weather_cards:
        st.markdown(
            '<div class="weather-fallback">天气信息暂时无法获取，请出行前查看天气 App。</div>',
            unsafe_allow_html=True,
        )
        return

    # weather_card_html_list：所有目的地天气卡片 HTML。
    weather_card_html_list = []
    for weather_card in weather_cards:
        # destination：天气卡片展示的目的地。
        destination = html.escape(str(weather_card.get("destination", "目的地")))

        if weather_card.get("error"):
            weather_card_html_list.append(
                '<article class="weather-card">'
                f'<h3>{destination}｜未来 {WEATHER_FORECAST_DAYS} 天天气</h3>'
                f'<p class="weather-advice">{html.escape(str(weather_card["error"]))}</p>'
                "</article>"
            )
            continue

        # day_html_list：单个目的地内的每日天气 HTML。
        day_html_list = []
        for day_weather in weather_card.get("days", []):
            # will_rain_text：是否可能下雨的展示文本。
            will_rain_text = "可能下雨" if day_weather.get("will_rain") else "降雨风险较低"

            day_html_list.append(
                '<div class="weather-day">'
                f'<div class="weather-icon">{html.escape(str(day_weather.get("weather_icon", "🌦️")))}</div>'
                "<div>"
                f'<div class="weather-date">{html.escape(str(day_weather.get("date", "")))}</div>'
                f'<div class="weather-main">天气：{html.escape(str(day_weather.get("weather_text", "天气待确认")))}</div>'
                '<div class="weather-meta">'
                f'<span>温度：{html.escape(format_weather_value(day_weather.get("temperature_min"), "°C"))} - {html.escape(format_weather_value(day_weather.get("temperature_max"), "°C"))}</span>'
                f'<span>湿度：{html.escape(format_weather_value(day_weather.get("humidity"), "%"))}</span>'
                f'<span>降水概率：{html.escape(format_weather_value(day_weather.get("precipitation_probability"), "%"))}</span>'
                f'<span>{html.escape(will_rain_text)}</span>'
                "</div>"
                f'<p class="weather-advice">建议：{html.escape(str(day_weather.get("advice", "天气信息仅供参考，请出行前查看天气 App。")))}</p>'
                "</div>"
                "</div>"
            )

        weather_card_html_list.append(
            '<article class="weather-card">'
            f'<h3>{destination}｜未来 {WEATHER_FORECAST_DAYS} 天天气</h3>'
            f'{"".join(day_html_list)}'
            "</article>"
        )

    # weather_html：无缩进 HTML，避免 Markdown 把天气卡片识别为代码块。
    weather_html = f'<div class="weather-grid">{"".join(weather_card_html_list)}</div>'
    st.markdown(weather_html, unsafe_allow_html=True)


def render_travel_blessing() -> None:
    """render_travel_blessing：在攻略最后展示温和的旅行祝福语。"""

    st.markdown(
        """
        <div class="blessing-card">
            祝你这次旅行顺利又开心。记得提前确认天气、门票和交通安排，慢慢走、好好看，把喜欢的风景都装进记忆里。祝你旅途愉快呀～ 🌿✨🧳
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_visual_guide(
    markdown_text: str,
    parsed_request: dict,
    travel_json: dict | None = None,
    json_errors: list[str] | None = None,
    json_raw: str | None = None,
    weather_cards: list[dict] | None = None,
    generated_at: str | None = None,
) -> None:
    """render_visual_guide：把 Markdown 攻略渲染成高级卡片式视觉结果。"""

    # section_map：按标题拆分后的攻略内容。
    section_map = split_markdown_sections(markdown_text)

    if not section_map:
        with st.container(border=True):
            st.markdown(markdown_text)
        render_timeline({}, parsed_request, travel_json, json_errors, json_raw)
        render_weather_section(weather_cards)
        render_source_info({}, generated_at)
        render_travel_blessing()
        return

    render_overview_card(section_map)
    render_timeline(section_map, parsed_request, travel_json, json_errors, json_raw)
    render_food_cards(section_map, parsed_request, travel_json)
    render_advice_sections(section_map)
    render_weather_section(weather_cards)
    render_source_info(section_map, generated_at)
    render_travel_blessing()


def render_copy_button(markdown_text: str) -> None:
    """render_copy_button：渲染可复制 Markdown 攻略的按钮。"""

    # markdown_json：安全注入到 JavaScript 的攻略文本。
    markdown_json = json.dumps(markdown_text, ensure_ascii=False)

    st.html(
        f"""
        <style>
        .copy-widget {{
            font-family: Arial, sans-serif;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            padding: 8px 0;
            width: 100%;
        }}
        .copy-widget button {{
            border: 1px solid rgba(246, 199, 111, 0.32);
            border-radius: 999px;
            padding: 12px 18px;
            background: linear-gradient(135deg, #f6c76f, #fb923c);
            color: #17120a;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 12px 28px rgba(251, 146, 60, 0.18);
            width: 100%;
        }}
        .copy-widget span {{
            color: #cbd5e1;
            font-size: 14px;
        }}
        </style>
        <div class="copy-widget">
            <button id="copy-markdown-button">复制攻略</button>
            <span id="copy-markdown-status">一键复制 Markdown 全文。</span>
        </div>
        <script>
        const markdownText = {markdown_json};
        const copyButton = document.getElementById("copy-markdown-button");
        const copyStatus = document.getElementById("copy-markdown-status");
        copyButton.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText(markdownText);
                copyStatus.textContent = "已复制到剪贴板。";
            }} catch (error) {{
                copyStatus.textContent = "复制失败，请手动复制下方原文。";
            }}
        }});
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_result_actions(markdown_text: str) -> None:
    """render_result_actions：在结果底部展示复制、下载和重新生成操作。"""

    with st.container(border=True):
        st.markdown('<div class="result-actions-title">结果操作</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="source-card-text">复制给同行人、下载成 Markdown，或基于当前输入重新生成一版。</p>',
            unsafe_allow_html=True,
        )

        # action_columns：结果操作按钮区域。
        action_columns = st.columns([1.2, 0.8, 0.8])
        with action_columns[0]:
            render_copy_button(markdown_text)
        with action_columns[1]:
            st.download_button(
                label="下载 Markdown",
                data=markdown_text,
                file_name="ai-travel-guide.md",
                mime="text/markdown",
                key="download_markdown_action",
            )
        with action_columns[2]:
            if st.button("重新生成", key="regenerate_result_button"):
                st.session_state["force_regenerate"] = True
                st.rerun()


def render_markdown_source(markdown_text: str) -> None:
    """render_markdown_source：用折叠区域展示 Markdown 原文，并提供复制和下载。"""

    st.markdown('<h2 class="section-heading">Markdown 原文</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">默认收起，适合最后复制到笔记、公众号或行程文档中。</p>', unsafe_allow_html=True)

    with st.expander("查看可复制的 Markdown 攻略", expanded=False):
        # action_columns：复制和下载按钮区域。
        action_columns = st.columns([1, 1])
        with action_columns[0]:
            render_copy_button(markdown_text)
        with action_columns[1]:
            st.download_button(
                label="下载 Markdown 文件",
                data=markdown_text,
                file_name="ai-travel-guide.md",
                mime="text/markdown",
            )

        st.code(markdown_text, language="markdown")


def render_debug_panel(parsed_request: dict) -> None:
    """render_debug_panel：默认折叠展示系统识别出的结构化参数。"""

    if not is_debug_enabled():
        return

    # debug_data：开发调试用的结构化解析结果。
    debug_data = {
        "trip_type": parsed_request.get("trip_type"),
        "destination": parsed_request["destination"],
        "destinations": parsed_request.get("destinations", []),
        "trip_segments": parsed_request.get("trip_segments", []),
        "trip_notes": parsed_request.get("trip_notes", []),
        "days": parsed_request["days"],
        "nights": parsed_request["nights"],
        "budget_amount": parsed_request.get("budget_amount"),
        "currency": parsed_request.get("budget_currency") or "未指定",
        "style": parsed_request.get("style") or parsed_request.get("budget_level"),
        "preferences": "、".join(parsed_request["preferences"]),
    }

    with st.expander("开发者调试信息", expanded=False):
        st.json(debug_data)


def format_public_search_status(search_message: str | None) -> str:
    """format_public_search_status：把内部搜索状态转换为普通用户可理解的提示。"""

    if not search_message:
        return "实时信息未启用，请出行前二次确认。"

    # raw_message：内部搜索状态文案。
    raw_message = str(search_message)
    if "已启用" in raw_message:
        return "已参考当前可用公开信息。"
    if "缓存" in raw_message:
        return "已参考近期可用信息。"
    if "未启用" in raw_message or "未配置" in raw_message:
        return "实时信息未启用，请出行前二次确认。"
    if "额度" in raw_message or "失败" in raw_message or "受限" in raw_message:
        return "实时信息暂时不可用，请出行前二次确认。"

    return "请出行前再次确认门票、预约、开放时间和交通政策。"


def render_search_status(search_message: str | None) -> None:
    """render_search_status：用小标签展示 Tavily 联网搜索状态。"""

    # safe_message：转义后的状态文案，避免 HTML 注入。
    safe_message = html.escape(format_public_search_status(search_message))
    st.markdown(
        f"""
        <div class="search-status-pill">
            <span class="search-status-dot"></span>
            <span>{safe_message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trust_strip(search_message: str | None, generated_at: str | None) -> None:
    """render_trust_strip：展示轻量信任说明，提醒用户核对实时信息。"""

    # search_status_text：联网搜索状态文案。
    search_status_text = format_public_search_status(search_message)

    # generated_time_text：攻略生成时间。
    generated_time_text = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")

    st.markdown(
        f"""
        <div class="trust-strip">
            <article class="trust-card">
                <span>信息提示</span>
                <strong>{html.escape(search_status_text)}</strong>
                <p>如未联网，实时门票、预约和开放时间请出行前再次核对。</p>
            </article>
            <article class="trust-card">
                <span>AI Generated</span>
                <strong>攻略由 AI 生成，仅供参考</strong>
                <p>路线、预算和餐饮建议适合作为初步规划，不替代官方信息。</p>
            </article>
            <article class="trust-card">
                <span>Updated</span>
                <strong>{html.escape(generated_time_text)}</strong>
                <p>门票、预约、开放时间和交通政策请以官方渠道为准。</p>
            </article>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_generation_count() -> int:
    """get_generation_count：读取当前浏览器 session 已生成攻略次数。"""

    return int(st.session_state.get("generation_count", 0))


def render_generation_quota() -> None:
    """render_generation_quota：展示 Beta 测试版当前 session 剩余生成次数。"""

    # remaining_count：当前 session 剩余生成次数。
    remaining_count = max(0, MAX_GENERATIONS_PER_SESSION - get_generation_count())
    st.markdown(
        f"""
        <p class="generation-quota">Beta 测试额度：本会话剩余 {remaining_count}/{MAX_GENERATIONS_PER_SESSION} 次生成。</p>
        """,
        unsafe_allow_html=True,
    )


def build_result_data(user_input: str) -> dict:
    """build_result_data：根据用户输入生成结果数据，但不把完整用户输入保存到 session。"""

    # generated_at：本次攻略生成时间。
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # parsed_request：自然语言解析结果。
    parsed_request = parse_travel_request(user_input)

    # cover_image_url：封面图地址，第一版是本地 SVG 占位。
    cover_image_url = generate_cover_image_url(parsed_request)

    with st.status("正在理解你的旅行需求...", expanded=True) as loading_status:
        time.sleep(0.2)
        if get_bool_config("USE_TAVILY", True) and get_tavily_api_key():
            loading_status.update(label="正在检索目的地最新信息...", state="running")
        else:
            loading_status.update(label="当前未启用联网搜索，正在切换普通生成模式...", state="running")

        # facts_context：联网搜索整理出的事实校验上下文。
        facts_context, source_records, search_message = build_facts_context(parsed_request)
        loading_status.update(label="正在规划每日路线...", state="running")
        # markdown_text/travel_json：最终展示的 Markdown 和页面时间线使用的结构化 JSON。
        markdown_text, api_message, travel_json, json_raw, json_errors = generate_travel_content(
            user_input,
            parsed_request,
            facts_context,
        )
        loading_status.update(label="正在整理美食与交通建议...", state="running")
        time.sleep(0.2)
        # weather_cards：Open-Meteo 免费天气数据，失败不影响攻略生成。
        try:
            weather_cards = build_weather_cards(parsed_request, travel_json)
        except Exception:
            weather_cards = []
        markdown_text = append_weather_and_blessing_to_markdown(markdown_text, weather_cards, generated_at)
        loading_status.update(label="正在生成专属旅行方案...", state="running")
        time.sleep(0.2)
        loading_status.update(label="专属旅行方案已生成", state="complete", expanded=False)

    return {
        "parsed_request": parsed_request,
        "cover_image_url": cover_image_url,
        "markdown_text": markdown_text,
        "api_message": api_message,
        "travel_json": travel_json,
        "json_raw": json_raw,
        "json_errors": json_errors,
        "weather_cards": weather_cards,
        "search_message": search_message,
        "generated_at": generated_at,
    }


def render_result_data(result_data: dict) -> None:
    """render_result_data：渲染已经生成并缓存在当前 session 中的攻略结果。"""

    # parsed_request：结构化旅行参数，不包含任何 API Key。
    parsed_request = result_data["parsed_request"]

    # markdown_text：最终展示和复制的 Markdown 攻略。
    markdown_text = result_data["markdown_text"]

    render_search_status(result_data.get("search_message"))
    render_trust_strip(result_data.get("search_message"), result_data.get("generated_at"))

    if result_data.get("api_message") and is_debug_enabled():
        with st.expander("开发者调试信息", expanded=False):
            st.markdown(html.escape(str(result_data["api_message"])))

    render_debug_panel(parsed_request)
    render_cover(parsed_request, result_data["cover_image_url"])
    render_summary_bento(parsed_request)
    render_trip_segments_overview(parsed_request)
    render_visual_guide(
        markdown_text,
        parsed_request,
        result_data.get("travel_json"),
        result_data.get("json_errors"),
        result_data.get("json_raw"),
        result_data.get("weather_cards"),
        result_data.get("generated_at"),
    )
    render_result_actions(markdown_text)
    render_markdown_source(markdown_text)


def render_result(user_input: str) -> None:
    """render_result：兼容旧调用方式，立即生成并渲染完整旅行攻略。"""

    render_result_data(build_result_data(user_input))


def render_beta_notice() -> None:
    """render_beta_notice：在页面底部展示 Beta 测试版和隐私安全提醒。"""

    st.markdown(
        f"""
        <div class="beta-notice">{html.escape(BETA_NOTICE_TEXT)}</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """main：应用入口函数。"""

    setup_page()
    render_hero()

    # submitted：是否点击生成按钮。
    submitted, user_input = render_input_box()
    render_generation_quota()

    # force_regenerate：结果页“重新生成”按钮触发的再生成动作。
    force_regenerate = bool(st.session_state.pop("force_regenerate", False))

    if submitted or force_regenerate:
        if not user_input.strip():
            st.warning("请先输入一句旅行需求。")
        elif get_generation_count() >= MAX_GENERATIONS_PER_SESSION:
            st.warning(
                f"当前 Beta 测试版每个浏览器会话最多生成 {MAX_GENERATIONS_PER_SESSION} 次攻略。"
                "请刷新浏览器会话或稍后再试。"
            )
        else:
            # generation_count：点击生成才增加次数，页面重绘不会重复消耗 API。
            st.session_state["generation_count"] = get_generation_count() + 1

            # last_result_data：只保存生成后的旅行参数和攻略结果，不保存完整用户输入或任何密钥。
            st.session_state["last_result_data"] = build_result_data(user_input.strip())

    if "last_result_data" in st.session_state:
        render_result_data(st.session_state["last_result_data"])
    else:
        st.markdown(
            """
            <p class="hint">示例输入：我想去东京旅游，喜欢动漫、美食和夜景，预算5000，想玩 3 天 2 晚。</p>
            """,
            unsafe_allow_html=True,
        )

    render_beta_notice()


if __name__ == "__main__":
    main()
import html
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


# DEFAULT_TRAVEL_DAYS：用户没有写旅行天数时，默认按 3 天处理。
DEFAULT_TRAVEL_DAYS = 3

# DEFAULT_TRAVEL_NIGHTS：用户没有写住宿晚数时，默认按 2 晚处理。
DEFAULT_TRAVEL_NIGHTS = 2

# DEFAULT_BUDGET_LEVEL：用户没有写预算时，默认使用普通预算。
DEFAULT_BUDGET_LEVEL = "普通预算"

# DEFAULT_BUDGET_CURRENCY：用户输入预算数字但没有写货币单位时，默认按人民币处理。
DEFAULT_BUDGET_CURRENCY = "CNY"

# DEFAULT_DESTINATION：用户没有写明确目的地时，用于演示的默认目的地。
DEFAULT_DESTINATION = "东京"

# DEEPSEEK_BASE_URL：DeepSeek API 的基础地址，OpenAI SDK 会通过这个地址请求 DeepSeek。
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DEFAULT_DEEPSEEK_MODEL：用户没有在 .env 配置模型时，默认使用的 DeepSeek 模型。
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

# DEFAULT_SEARCH_MAX_RESULTS：每个搜索查询最多保留的结果数量。
DEFAULT_SEARCH_MAX_RESULTS = 3

# DEFAULT_TAVILY_SEARCH_DEPTH：Tavily 默认使用 basic 搜索，控制搜索额度消耗。
DEFAULT_TAVILY_SEARCH_DEPTH = "basic"

# DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE：每份攻略默认最多调用 Tavily 的次数。
DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE = 1

# TAVILY_CACHE_FILE：Tavily 搜索结果本地缓存文件，避免 24 小时内重复消耗额度。
TAVILY_CACHE_FILE = "tavily_cache.json"

# TAVILY_CACHE_TTL_SECONDS：Tavily 缓存有效期，默认 24 小时。
TAVILY_CACHE_TTL_SECONDS = 24 * 60 * 60

# TAVILY_CACHE_PATH：Tavily 缓存文件的绝对路径。
TAVILY_CACHE_PATH = Path(__file__).with_name(TAVILY_CACHE_FILE)

# MAX_GENERATIONS_PER_SESSION：Beta 测试版每个浏览器会话最多生成攻略次数，避免 API 被滥用。
MAX_GENERATIONS_PER_SESSION = 3

# OPEN_METEO_GEOCODING_URL：Open-Meteo 免费地理编码接口，不需要 API Key。
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# OPEN_METEO_FORECAST_URL：Open-Meteo 免费天气预报接口，不需要 API Key。
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WEATHER_FORECAST_DAYS：天气模块默认展示最近几天。
WEATHER_FORECAST_DAYS = 3

# BETA_NOTICE_TEXT：上线前页面底部展示的 Beta 和隐私安全提醒。
BETA_NOTICE_TEXT = "当前为 Beta 测试版。AI 生成内容仅供参考，门票、预约、开放时间、交通政策等信息请以官方渠道为准。请勿输入身份证号、手机号、住址、护照号等敏感个人信息。"

# SAMPLE_PROMPTS：Hero 输入区的示例旅行需求，点击后会自动填入对话框。
SAMPLE_PROMPTS = [
    {"label": "东京3天动漫美食游", "prompt": "我想去东京旅游，喜欢动漫、美食和夜景，预算5000，3 天 2 晚"},
    {"label": "杭州7天舒适游", "prompt": "杭州7日游，想去西湖、灵隐寺、龙井村，也想吃杭州美食和看夜景，预算一万，要求舒适一点"},
    {"label": "南京3天 + 江西4天", "prompt": "南京3天，然后去江西4天，喜欢历史文化、美食和夜景，预算8000"},
    {"label": "大阪京都5日自由行", "prompt": "我想去大阪京都自由行，喜欢历史、美食、购物和拍照，普通预算，5 天 4 晚"},
]


if load_dotenv:
    # load_dotenv：读取本地 .env 文件，方便初学者不用每次手动设置环境变量。
    load_dotenv()


def get_config_value(config_name: str, default_value: str = "") -> str:
    """get_config_value：优先从 .env/环境变量读取配置，其次兼容 Streamlit secrets。"""

    # env_value：load_dotenv 后从系统环境变量中读取到的配置值。
    env_value = os.getenv(config_name)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()

    try:
        # secret_value：Streamlit Cloud 部署时可从 st.secrets 读取的配置值。
        secret_value = st.secrets.get(config_name)
        if secret_value is not None and str(secret_value).strip():
            return str(secret_value).strip()
    except Exception:
        return default_value

    return default_value


def get_bool_config(config_name: str, default_value: bool = False) -> bool:
    """get_bool_config：把环境变量或 secrets 中的开关配置转换成布尔值。"""

    # raw_value：配置原始字符串。
    raw_value = get_config_value(config_name, str(default_value)).strip().lower()
    return raw_value in {"1", "true", "yes", "y", "on", "启用", "是"}


def get_int_config(config_name: str, default_value: int) -> int:
    """get_int_config：读取整数配置，非法值自动使用默认值。"""

    # raw_value：配置原始字符串。
    raw_value = get_config_value(config_name, str(default_value)).strip()
    try:
        return int(raw_value)
    except ValueError:
        return default_value


def setup_page() -> None:
    """setup_page：设置 Streamlit 页面基础信息和自定义样式。"""

    st.set_page_config(
        page_title="AI 旅游攻略 Agent",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # custom_css：控制页面视觉风格，让 Streamlit 默认界面更接近高级旅行杂志和 AI 工具。
    custom_css = """
    <style>
    :root {
        --bg-deep: #05070f;
        --panel: rgba(15, 23, 42, 0.58);
        --panel-strong: rgba(15, 23, 42, 0.82);
        --panel-warm: rgba(120, 75, 32, 0.18);
        --line: rgba(255, 255, 255, 0.14);
        --line-soft: rgba(255, 255, 255, 0.08);
        --text-soft: #cbd5e1;
        --text-muted: #94a3b8;
        --cyan: #38bdf8;
        --mint: #34d399;
        --rose: #fb7185;
        --gold: #f6c76f;
        --champagne: #fdecc8;
        --orange: #fb923c;
        --shadow: 0 28px 90px rgba(0, 0, 0, 0.34);
        --glass-blur: blur(20px);
    }

    *,
    *::before,
    *::after {
        box-sizing: border-box;
    }

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        max-width: 100%;
        overflow-x: hidden;
    }

    [data-testid="stMain"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="column"],
    [data-testid="stForm"],
    [data-testid="stTextArea"],
    [data-testid="stMarkdownContainer"] {
        max-width: 100%;
        min-width: 0;
    }

    img,
    iframe,
    table,
    svg {
        max-width: 100%;
    }

    .stApp {
        color: #f8fafc;
        background:
            linear-gradient(118deg, rgba(246, 199, 111, 0.16) 0%, transparent 24%),
            linear-gradient(242deg, rgba(251, 146, 60, 0.12) 0%, transparent 31%),
            linear-gradient(180deg, rgba(255, 255, 255, 0.045), transparent 24%),
            linear-gradient(145deg, #04060d 0%, #0b1020 36%, #111827 68%, #05070f 100%);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.028) 1px, transparent 1px);
        background-size: 72px 72px;
        mask-image: linear-gradient(180deg, rgba(0,0,0,0.72), transparent 72%);
        opacity: 0.38;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    .block-container {
        width: 100%;
        max-width: 1180px;
        padding-top: 1.35rem;
        padding-bottom: 5rem;
    }

    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2.8rem;
        padding: 0.82rem 1.05rem;
        border: 1px solid rgba(246, 199, 111, 0.16);
        border-radius: 999px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02)),
            rgba(8, 12, 24, 0.62);
        box-shadow: 0 16px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.02rem;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: inline-grid;
        place-items: center;
        color: #111827;
        background: linear-gradient(135deg, var(--gold), #fff7d6 48%, var(--orange));
        box-shadow: 0 0 0 1px rgba(255,255,255,0.32), 0 12px 28px rgba(251, 146, 60, 0.24);
    }

    .nav-links {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: var(--text-soft);
        font-size: 0.92rem;
    }

    .nav-links span {
        padding: 0.45rem 0.72rem;
        border-radius: 999px;
        color: #dbeafe;
    }

    .hero {
        position: relative;
        margin-bottom: 1.75rem;
        padding: 0.25rem 0 0.35rem;
    }

    .hero::after {
        content: "";
        display: block;
        width: min(420px, 68vw);
        height: 1px;
        margin-top: 1.5rem;
        background: linear-gradient(90deg, rgba(246, 199, 111, 0.72), transparent);
    }

    .hero-layout {
        display: grid;
        grid-template-columns: minmax(0, 1.16fr) minmax(280px, 0.84fr);
        gap: 1.35rem;
        align-items: stretch;
        max-width: 100%;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 0.8rem;
        border: 1px solid rgba(246, 199, 111, 0.34);
        border-radius: 999px;
        color: #fde68a;
        background: rgba(120, 75, 32, 0.22);
        font-size: 0.84rem;
        margin-bottom: 1.15rem;
    }

    .hero-proof {
        display: flex;
        flex-wrap: wrap;
        gap: 0.58rem;
        margin-top: 1.15rem;
    }

    .hero-proof span {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        padding: 0.38rem 0.68rem;
        color: #e5e7eb;
        background: rgba(15, 23, 42, 0.42);
        font-size: 0.84rem;
        backdrop-filter: blur(12px);
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2.65rem, 5.8vw, 5.85rem);
        line-height: 0.98;
        letter-spacing: 0;
        max-width: 930px;
        color: #fff7ed;
        text-wrap: balance;
    }

    .hero p {
        margin: 1.15rem 0 0;
        max-width: 720px;
        color: #d1d5db;
        font-size: 1.08rem;
        line-height: 1.85;
    }

    .hero-panel {
        min-width: 0;
        min-height: 100%;
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 1.25rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.035)),
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(251, 146, 60, 0.06));
        box-shadow: var(--shadow);
        backdrop-filter: var(--glass-blur);
        position: relative;
        overflow: hidden;
    }

    .hero-panel::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(90deg, transparent 0, rgba(246, 199, 111, 0.08) 1px, transparent 1px),
            linear-gradient(180deg, transparent 0, rgba(255, 255, 255, 0.045) 1px, transparent 1px);
        background-size: 34px 34px;
        opacity: 0.5;
    }

    .mini-card {
        position: relative;
        z-index: 1;
        border: 1px solid var(--line-soft);
        border-radius: 20px;
        padding: 1rem;
        background: rgba(3, 7, 18, 0.42);
        margin-bottom: 0.85rem;
    }

    .mini-card span {
        display: block;
        color: var(--gold);
        font-size: 0.78rem;
        margin-bottom: 0.45rem;
    }

    .mini-card strong {
        display: block;
        font-size: 1.2rem;
        margin-bottom: 0.35rem;
    }

    .mini-card p {
        margin: 0;
        color: var(--text-muted);
        line-height: 1.55;
        font-size: 0.92rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line) !important;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.105), rgba(255, 255, 255, 0.035)),
            rgba(10, 15, 30, 0.58) !important;
        border-radius: 26px !important;
        box-shadow: var(--shadow);
        backdrop-filter: var(--glass-blur);
    }

    [data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #fff7ed;
        letter-spacing: 0;
    }

    .input-title {
        font-size: 1.12rem;
        color: #fef3c7;
        margin: 0 0 0.35rem;
        font-weight: 700;
    }

    .input-kicker {
        color: var(--gold);
        font-size: 0.78rem;
        text-transform: uppercase;
        margin: 0 0 0.32rem;
    }

    .sample-title {
        color: var(--text-muted);
        font-size: 0.88rem;
        margin: 0.85rem 0 0.45rem;
    }

    .stTextArea textarea {
        min-height: 150px !important;
        border-radius: 22px !important;
        border: 1px solid rgba(246, 199, 111, 0.34) !important;
        background:
            linear-gradient(145deg, rgba(2, 6, 23, 0.78), rgba(15, 23, 42, 0.62)) !important;
        color: #f8fafc !important;
        font-size: 1.03rem !important;
        line-height: 1.65 !important;
        box-shadow:
            inset 0 0 0 1px rgba(255, 255, 255, 0.04),
            0 18px 50px rgba(0, 0, 0, 0.16);
    }

    .stTextArea textarea:focus {
        border-color: rgba(246, 199, 111, 0.88) !important;
        box-shadow: 0 0 0 4px rgba(246, 199, 111, 0.14) !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        width: 100%;
        border: 1px solid rgba(246, 199, 111, 0.26);
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(246, 199, 111, 0.95), rgba(251, 146, 60, 0.92));
        color: #17120a;
        font-weight: 800;
        padding: 0.78rem 1rem;
        box-shadow: 0 14px 34px rgba(251, 146, 60, 0.18);
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        color: #17120a;
        filter: brightness(1.05);
        border-color: rgba(255, 247, 237, 0.55);
    }

    .cover-card {
        width: 100%;
        max-width: 100%;
        aspect-ratio: 16 / 9;
        min-height: 420px;
        border-radius: 30px;
        border: 1px solid rgba(246, 199, 111, 0.24);
        background-size: cover;
        background-position: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 34px 110px rgba(0, 0, 0, 0.45);
        margin: 2.15rem 0 1.45rem;
    }

    .cover-card::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: 1;
        pointer-events: none;
        background:
            linear-gradient(90deg, rgba(246, 199, 111, 0.16) 1px, transparent 1px),
            linear-gradient(180deg, rgba(255, 255, 255, 0.07) 1px, transparent 1px),
            linear-gradient(135deg, transparent 0 62%, rgba(246, 199, 111, 0.12) 62% 63%, transparent 63%);
        background-size: 88px 88px, 88px 88px, 100% 100%;
        opacity: 0.42;
    }

    .cover-card::after {
        content: "";
        position: absolute;
        inset: 0;
        z-index: 0;
        background:
            linear-gradient(90deg, rgba(2, 6, 23, 0.82), rgba(2, 6, 23, 0.28) 58%, rgba(2, 6, 23, 0.68)),
            linear-gradient(180deg, rgba(2, 6, 23, 0.02) 20%, rgba(2, 6, 23, 0.88));
    }

    .cover-content {
        position: absolute;
        inset: auto clamp(1.25rem, 4vw, 3.2rem) clamp(1.25rem, 4vw, 3.2rem) clamp(1.25rem, 4vw, 3.2rem);
        z-index: 2;
    }

    .cover-content .label {
        color: #fde68a;
        font-size: 0.88rem;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .cover-content h2 {
        margin: 0;
        font-size: clamp(2.55rem, 6.2vw, 5.4rem);
        line-height: 0.98;
        letter-spacing: 0;
    }

    .cover-dayline {
        color: var(--champagne);
        font-size: clamp(1rem, 2.2vw, 1.45rem);
        font-weight: 800;
        margin-top: 0.62rem;
    }

    .cover-content p {
        margin: 0.9rem 0 0;
        max-width: 760px;
        color: #f8fafc;
        font-size: 1rem;
        line-height: 1.75;
    }

    .cover-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .cover-badge {
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        color: #fff7ed;
        background: rgba(15, 23, 42, 0.46);
        backdrop-filter: blur(10px);
        font-size: 0.86rem;
    }

    .bento-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        grid-auto-rows: minmax(112px, auto);
        gap: 0.9rem;
        margin: 1rem 0 2rem;
    }

    .bento-card {
        min-width: 0;
        max-width: 100%;
        border: 1px solid var(--line-soft);
        border-radius: 24px;
        padding: 1.05rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.52);
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
        min-height: 112px;
    }

    .bento-card.large {
        grid-column: span 2;
    }

    .bento-card.warm {
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.19), rgba(251, 146, 60, 0.065)),
            rgba(15, 23, 42, 0.52);
    }

    .bento-card span {
        display: block;
        color: #fcd34d;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }

    .bento-card strong {
        display: block;
        color: #f8fafc;
        font-size: clamp(1.05rem, 2.2vw, 1.45rem);
        line-height: 1.18;
        letter-spacing: 0;
    }

    .bento-card p {
        color: var(--text-muted);
        line-height: 1.55;
        margin: 0.55rem 0 0;
        font-size: 0.92rem;
    }

    .segment-overview-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.85rem;
        max-width: 100%;
        margin: -0.45rem 0 1.9rem;
    }

    .segment-card {
        width: 100%;
        min-width: 0;
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 22px;
        padding: 1rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(56, 189, 248, 0.05)),
            rgba(15, 23, 42, 0.48);
        box-shadow: 0 16px 50px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
    }

    .segment-card span {
        display: block;
        color: #fcd34d;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }

    .segment-card strong {
        display: block;
        color: #fff7ed;
        font-size: 1.18rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .segment-card p {
        color: var(--text-muted);
        line-height: 1.58;
        margin: 0.55rem 0 0;
        font-size: 0.9rem;
        overflow-wrap: anywhere;
    }

    .section-heading {
        margin: 2.25rem 0 0.35rem;
        color: #fff7ed;
        font-size: clamp(1.55rem, 3vw, 2.1rem);
        letter-spacing: 0;
    }

    .section-subtitle {
        margin: 0 0 1.1rem;
        color: var(--text-muted);
        line-height: 1.65;
    }

    .timeline-grid,
    .food-grid,
    .info-grid,
    .warning-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.6rem;
        max-width: 100%;
    }

    .timeline-day,
    .food-card,
    .info-card,
    .warning-card {
        width: 100%;
        max-width: 100%;
        min-width: 0;
        border: 1px solid var(--line-soft);
        border-radius: 26px;
        padding: 1.15rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.25);
        backdrop-filter: var(--glass-blur);
    }

    .timeline-day h3,
    .food-card h3,
    .info-card h3,
    .warning-card h3 {
        margin: 0 0 0.85rem;
        color: #fff7ed;
        letter-spacing: 0;
        overflow-wrap: anywhere;
    }

    .card-title-row {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin-bottom: 0.82rem;
    }

    .card-title-row h3 {
        margin: 0;
    }

    .info-icon,
    .warning-icon {
        width: 36px;
        height: 36px;
        border-radius: 13px;
        display: grid;
        place-items: center;
        font-weight: 900;
        flex: 0 0 auto;
    }

    .info-icon {
        color: #082f49;
        background: linear-gradient(135deg, #7dd3fc, #38bdf8);
    }

    .warning-icon {
        color: #17120a;
        background: linear-gradient(135deg, #f6c76f, #fb923c);
    }

    .timeline-slot {
        display: grid;
        grid-template-columns: 44px minmax(0, 1fr);
        gap: 0.85rem;
        padding: 0.82rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .timeline-slot:first-of-type {
        border-top: 0;
        padding-top: 0;
    }

    .slot-icon {
        width: 40px;
        height: 40px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        color: #17120a;
        background: linear-gradient(135deg, #f6c76f, #fb923c);
        font-weight: 900;
        box-shadow: 0 12px 26px rgba(251, 146, 60, 0.18);
    }

    .slot-time {
        color: #fcd34d;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .slot-place {
        color: #f8fafc;
        font-weight: 800;
        margin-bottom: 0.25rem;
        overflow-wrap: anywhere;
    }

    .slot-original {
        color: #fde68a;
        font-size: 0.78rem;
        margin-bottom: 0.38rem;
        opacity: 0.92;
        overflow-wrap: anywhere;
    }

    .slot-desc,
    .food-card p,
    .info-card p,
    .warning-card p {
        color: #cbd5e1;
        line-height: 1.62;
        margin: 0;
        font-size: 0.93rem;
        overflow-wrap: anywhere;
    }

    .slot-meta-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.45rem;
        margin-top: 0.62rem;
    }

    .slot-meta-item {
        border: 1px solid rgba(255, 255, 255, 0.075);
        border-radius: 14px;
        padding: 0.48rem 0.56rem;
        color: #cbd5e1;
        background: rgba(2, 6, 23, 0.26);
        font-size: 0.8rem;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    .slot-meta-item strong {
        color: #fcd34d;
        font-weight: 800;
    }

    .food-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.9rem;
    }

    .food-meta span {
        border: 1px solid rgba(246, 199, 111, 0.24);
        border-radius: 999px;
        padding: 0.32rem 0.58rem;
        color: #fde68a;
        background: rgba(120, 75, 32, 0.18);
        font-size: 0.8rem;
    }

    .food-location {
        color: #9ca3af;
        font-size: 0.82rem;
        line-height: 1.55;
        margin: -0.42rem 0 0.76rem;
    }

    .food-map-keyword {
        color: #fcd34d;
        font-size: 0.78rem;
        line-height: 1.45;
        margin-top: 0.65rem;
        opacity: 0.9;
    }

    .info-card {
        background:
            linear-gradient(145deg, rgba(56, 189, 248, 0.13), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
    }

    .warning-card {
        border-color: rgba(251, 146, 60, 0.22);
        background:
            linear-gradient(145deg, rgba(251, 146, 60, 0.18), rgba(127, 29, 29, 0.10)),
            rgba(15, 23, 42, 0.54);
    }

    .weather-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        max-width: 100%;
        margin-bottom: 1.5rem;
    }

    .weather-card {
        width: 100%;
        max-width: 100%;
        min-width: 0;
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 26px;
        padding: 1.1rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(56, 189, 248, 0.055)),
            rgba(15, 23, 42, 0.54);
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.24);
        backdrop-filter: var(--glass-blur);
    }

    .weather-card h3 {
        margin: 0 0 0.8rem;
        color: #fff7ed;
        letter-spacing: 0;
        overflow-wrap: anywhere;
    }

    .weather-day {
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr);
        gap: 0.82rem;
        padding: 0.86rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .weather-day:first-of-type {
        border-top: 0;
        padding-top: 0;
    }

    .weather-icon {
        width: 40px;
        height: 40px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: rgba(246, 199, 111, 0.15);
        border: 1px solid rgba(246, 199, 111, 0.22);
        font-size: 1.2rem;
    }

    .weather-date {
        color: #fcd34d;
        font-weight: 800;
        font-size: 0.88rem;
        margin-bottom: 0.22rem;
    }

    .weather-main {
        color: #f8fafc;
        font-weight: 800;
        margin-bottom: 0.36rem;
        overflow-wrap: anywhere;
    }

    .weather-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        margin: 0.45rem 0;
    }

    .weather-meta span {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 999px;
        padding: 0.28rem 0.48rem;
        color: #fde68a;
        background: rgba(15, 23, 42, 0.42);
        font-size: 0.78rem;
        overflow-wrap: anywhere;
    }

    .weather-advice {
        color: #cbd5e1;
        line-height: 1.6;
        font-size: 0.9rem;
        margin: 0.55rem 0 0;
        overflow-wrap: anywhere;
    }

    .weather-fallback {
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 22px;
        padding: 1rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.12), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
        color: #cbd5e1;
        line-height: 1.65;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.20);
        backdrop-filter: var(--glass-blur);
    }

    .blessing-card {
        margin: 1.2rem 0 1.6rem;
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 24px;
        padding: 1rem 1.05rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.14), rgba(251, 146, 60, 0.045)),
            rgba(15, 23, 42, 0.56);
        color: #f8fafc;
        line-height: 1.72;
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
    }

    .markdown-actions {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 220px;
        gap: 0.8rem;
        align-items: center;
        margin-bottom: 0.7rem;
    }

    .hint {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.65;
        overflow-wrap: anywhere;
    }

    .search-status-pill {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        max-width: 100%;
        gap: 0.48rem;
        margin: 0.3rem 0 1.1rem;
        padding: 0.5rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(246, 199, 111, 0.22);
        background: rgba(15, 23, 42, 0.48);
        color: #fde68a;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
        backdrop-filter: var(--glass-blur);
        font-size: 0.86rem;
        line-height: 1.4;
    }

    .trust-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.45rem 0 1.15rem;
        max-width: 100%;
    }

    .trust-card {
        min-width: 0;
        border: 1px solid rgba(246, 199, 111, 0.16);
        border-radius: 18px;
        padding: 0.82rem 0.9rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.10), rgba(255, 255, 255, 0.025)),
            rgba(15, 23, 42, 0.48);
        box-shadow: 0 14px 42px rgba(0, 0, 0, 0.18);
        backdrop-filter: var(--glass-blur);
    }

    .trust-card span {
        display: block;
        color: #fcd34d;
        font-size: 0.76rem;
        margin-bottom: 0.32rem;
    }

    .trust-card strong {
        display: block;
        color: #fff7ed;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .trust-card p {
        color: var(--text-muted);
        font-size: 0.84rem;
        line-height: 1.5;
        margin: 0.42rem 0 0;
        overflow-wrap: anywhere;
    }

    .search-status-dot {
        width: 0.46rem;
        height: 0.46rem;
        flex: 0 0 auto;
        border-radius: 999px;
        background: var(--gold);
        box-shadow: 0 0 18px rgba(246, 199, 111, 0.52);
    }

    .generation-quota {
        color: #fde68a;
        font-size: 0.86rem;
        margin-top: 0.55rem;
        opacity: 0.92;
    }

    .result-action-panel {
        margin: 2rem 0 1rem;
        padding: 1.05rem;
        border: 1px solid rgba(246, 199, 111, 0.20);
        border-radius: 24px;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.56);
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.25);
        backdrop-filter: var(--glass-blur);
    }

    .result-action-panel h2 {
        margin: 0 0 0.25rem;
        color: #fff7ed;
        font-size: 1.25rem;
        letter-spacing: 0;
    }

    .result-action-panel p {
        margin: 0;
        color: var(--text-muted);
        line-height: 1.55;
        font-size: 0.9rem;
    }

    .result-action-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.8fr) minmax(0, 0.8fr);
        gap: 0.78rem;
        align-items: center;
        margin-top: 0.9rem;
        max-width: 100%;
    }

    .beta-notice {
        margin-top: 2.2rem;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(246, 199, 111, 0.22);
        border-radius: 20px;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.12), rgba(255, 255, 255, 0.035)),
            rgba(15, 23, 42, 0.58);
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.9rem;
        backdrop-filter: var(--glass-blur);
    }

    div[data-testid="stExpander"] {
        max-width: 100%;
        border: 1px solid var(--line-soft);
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.50);
        backdrop-filter: var(--glass-blur);
    }

    pre,
    code,
    .stCodeBlock {
        max-width: 100%;
        overflow-wrap: anywhere;
    }

    pre {
        white-space: pre-wrap;
    }

    @media (max-width: 768px) {
        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"] {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }

        .block-container {
            max-width: 100% !important;
            padding: 0.85rem 0.78rem 3rem !important;
        }

        .top-nav {
            width: 100%;
            border-radius: 22px;
            align-items: flex-start;
            margin-bottom: 1.45rem;
            padding: 0.72rem 0.78rem;
        }

        .nav-links {
            display: none;
        }

        .hero {
            margin-bottom: 1.2rem;
        }

        .hero-layout,
        .segment-overview-grid,
        .trust-strip,
        .slot-meta-grid,
        .result-action-grid,
        .weather-grid,
        .timeline-grid,
        .food-grid,
        .info-grid,
        .warning-grid,
        .markdown-actions,
        .bento-grid {
            grid-template-columns: minmax(0, 1fr) !important;
            width: 100%;
        }

        .hero h1 {
            font-size: clamp(2rem, 12vw, 3.25rem);
            line-height: 1.04;
        }

        .hero p {
            font-size: 0.96rem;
            line-height: 1.68;
        }

        .hero-panel {
            padding: 0.85rem;
            border-radius: 22px;
        }

        .bento-card.large {
            grid-column: span 1;
        }

        .bento-card,
        .segment-card,
        .trust-card,
        .weather-card,
        .timeline-day,
        .food-card,
        .info-card,
        .warning-card {
            width: 100%;
            padding: 0.92rem;
            border-radius: 20px;
            box-shadow: 0 14px 44px rgba(0, 0, 0, 0.22);
        }

        .cover-card {
            min-height: 240px;
            aspect-ratio: 4 / 3;
            border-radius: 22px;
            margin: 1.25rem 0 1rem;
            background-position: center;
        }

        .cover-content {
            inset: auto 1rem 1rem 1rem;
        }

        .cover-content h2 {
            font-size: clamp(2rem, 12vw, 3.2rem);
        }

        .cover-content p {
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .cover-badge {
            font-size: 0.76rem;
            padding: 0.34rem 0.55rem;
        }

        .section-heading {
            font-size: 1.35rem;
            margin-top: 1.55rem;
        }

        .section-subtitle {
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .timeline-slot {
            grid-template-columns: 36px minmax(0, 1fr);
            gap: 0.68rem;
        }

        .slot-icon {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            font-size: 0.75rem;
        }

        .weather-day {
            grid-template-columns: 36px minmax(0, 1fr);
            gap: 0.68rem;
        }

        .weather-icon {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            font-size: 1rem;
        }

        .food-meta span {
            max-width: 100%;
            white-space: normal;
        }

        .search-status-pill {
            width: 100%;
            align-items: flex-start;
            border-radius: 18px;
        }

        .result-action-panel {
            padding: 0.92rem;
            border-radius: 20px;
        }

        .hero-proof {
            gap: 0.42rem;
        }

        .hero-proof span {
            width: 100%;
            text-align: center;
        }

        [data-testid="column"] {
            width: 100% !important;
            min-width: 0 !important;
            flex: 1 1 100% !important;
        }

        .stTextArea textarea {
            width: 100% !important;
            min-height: 118px !important;
            font-size: 0.95rem !important;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {
            width: 100% !important;
            white-space: normal;
            line-height: 1.35;
        }

        .stApp::before {
            opacity: 0.18;
            background-size: 96px 96px;
        }

        table {
            display: block;
            overflow-x: auto;
        }
    }

    @media (max-width: 520px) {
        .block-container {
            padding-left: 0.62rem !important;
            padding-right: 0.62rem !important;
        }

        .bento-grid {
            grid-template-columns: 1fr;
        }

        .bento-card.large {
            grid-column: span 1;
        }

        .top-nav {
            border-radius: 18px;
        }

        .hero h1 {
            font-size: clamp(1.85rem, 13vw, 2.75rem);
        }

        .cover-card {
            min-height: 218px;
        }

        .cover-content .label {
            font-size: 0.72rem;
            letter-spacing: 0.08rem;
        }

        .cover-dayline {
            font-size: 0.92rem;
        }
    }
    /* =========================
       TripAgent Product UI v2
       ========================= */
    :root {
        --v2-bg-a: #060711;
        --v2-bg-b: #111827;
        --v2-ink: #fff7ed;
        --v2-muted: #a7b0c0;
        --v2-gold: #f7d58a;
        --v2-gold-2: #f59e0b;
        --v2-ice: #9bdcff;
        --v2-card: rgba(12, 18, 34, 0.66);
        --v2-card-strong: rgba(8, 13, 26, 0.82);
        --v2-line: rgba(255, 244, 214, 0.16);
        --v2-line-bright: rgba(247, 213, 138, 0.34);
        --v2-shadow: 0 28px 100px rgba(0, 0, 0, 0.42);
    }

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        width: 100%;
        max-width: 100%;
        overflow-x: hidden !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 16% 8%, rgba(247, 213, 138, 0.22), transparent 28%),
            radial-gradient(circle at 82% 12%, rgba(155, 220, 255, 0.15), transparent 26%),
            radial-gradient(circle at 70% 86%, rgba(245, 158, 11, 0.15), transparent 34%),
            linear-gradient(145deg, #050611 0%, #0b1020 42%, #171923 72%, #060711 100%) !important;
        color: var(--v2-ink);
    }

    .stApp::before {
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(247, 213, 138, 0.035) 1px, transparent 1px) !important;
        background-size: 92px 92px;
        opacity: 0.42;
    }

    .block-container {
        max-width: 1240px !important;
        padding-top: 1rem !important;
    }

    .top-nav {
        position: sticky;
        top: 0.7rem;
        z-index: 5;
        margin-bottom: 1.6rem !important;
        padding: 0.74rem 0.92rem !important;
        border: 1px solid var(--v2-line-bright) !important;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.13), rgba(255, 255, 255, 0.035)),
            rgba(8, 12, 24, 0.78) !important;
        box-shadow: 0 18px 58px rgba(0, 0, 0, 0.34);
    }

    .brand-mark {
        background: linear-gradient(135deg, #fff2bf, #f7d58a 45%, #f59e0b) !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.34), 0 0 34px rgba(247, 213, 138, 0.22) !important;
    }

    .nav-links span {
        color: #f8fafc !important;
        border: 1px solid transparent;
    }

    .nav-links span:hover {
        border-color: rgba(247, 213, 138, 0.2);
        background: rgba(247, 213, 138, 0.08);
    }

    .hero.product-hero {
        position: relative;
        padding: clamp(1.1rem, 3vw, 2rem);
        border: 1px solid rgba(247, 213, 138, 0.20);
        border-radius: 34px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.115), rgba(255, 255, 255, 0.035)),
            radial-gradient(circle at 10% 0%, rgba(247, 213, 138, 0.15), transparent 38%),
            radial-gradient(circle at 90% 12%, rgba(155, 220, 255, 0.10), transparent 32%),
            rgba(9, 14, 28, 0.62);
        box-shadow: var(--v2-shadow);
        backdrop-filter: blur(26px);
        overflow: hidden;
        margin-bottom: 1.2rem;
    }

    .hero.product-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(90deg, transparent 0 8%, rgba(247, 213, 138, 0.12) 8% 8.12%, transparent 8.12%),
            linear-gradient(180deg, transparent 0 18%, rgba(255, 255, 255, 0.08) 18% 18.12%, transparent 18.12%);
        opacity: 0.55;
    }

    .hero.product-hero::after {
        display: none;
    }

    .hero-layout {
        position: relative;
        z-index: 1;
        grid-template-columns: minmax(0, 1.06fr) minmax(330px, 0.94fr) !important;
        gap: clamp(1rem, 2.4vw, 2rem) !important;
        align-items: center !important;
    }

    .eyebrow {
        border-color: rgba(247, 213, 138, 0.36) !important;
        color: #fff2bf !important;
        background: rgba(247, 213, 138, 0.10) !important;
        box-shadow: 0 12px 38px rgba(245, 158, 11, 0.13);
    }

    .hero h1,
    .hero-title {
        max-width: 900px;
        margin: 0.35rem 0 0.85rem !important;
        padding: 0.08rem 0;
        font-size: clamp(3rem, 5.8vw, 4.5rem) !important;
        line-height: 1.1 !important;
        color: transparent !important;
        background: linear-gradient(102deg, #fff7ed 0%, #f7d58a 58%, #9bdcff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        text-wrap: balance;
        overflow-wrap: break-word;
    }

    .hero-title-line {
        display: inline;
    }

    .hero p {
        max-width: 720px;
        color: #d7dbe5 !important;
        font-size: clamp(1rem, 1.8vw, 1.22rem) !important;
        line-height: 1.85 !important;
    }

    .hero-proof span {
        border-color: rgba(247, 213, 138, 0.20) !important;
        background: rgba(255, 255, 255, 0.07) !important;
    }

    .hero-panel.product-preview {
        min-height: 430px;
        border: 1px solid rgba(247, 213, 138, 0.24) !important;
        border-radius: 30px !important;
        padding: 1.1rem !important;
        background:
            radial-gradient(circle at 20% 10%, rgba(247, 213, 138, 0.18), transparent 34%),
            linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.035)),
            rgba(6, 10, 22, 0.72) !important;
        overflow: hidden;
    }

    .preview-cover {
        position: relative;
        min-height: 176px;
        border-radius: 24px;
        border: 1px solid rgba(247, 213, 138, 0.22);
        background:
            linear-gradient(120deg, rgba(2, 6, 23, 0.20), rgba(2, 6, 23, 0.82)),
            radial-gradient(circle at 20% 20%, rgba(247, 213, 138, 0.72), transparent 20%),
            linear-gradient(135deg, #1e293b, #7c2d12 58%, #020617);
        box-shadow: 0 20px 65px rgba(0, 0, 0, 0.32);
        overflow: hidden;
    }

    .preview-cover::after {
        content: "";
        position: absolute;
        inset: 18px;
        border: 1px solid rgba(255, 247, 237, 0.18);
        border-radius: 18px;
    }

    .preview-cover-label {
        position: absolute;
        left: 1rem;
        bottom: 1rem;
        z-index: 1;
    }

    .preview-cover-label span {
        display: block;
        color: #f7d58a;
        font-size: 0.76rem;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .preview-cover-label strong {
        display: block;
        font-size: 1.55rem;
        color: #fff7ed;
        line-height: 1.08;
    }

    .preview-steps {
        display: grid;
        gap: 0.72rem;
        margin-top: 0.92rem;
    }

    .preview-step {
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr);
        gap: 0.72rem;
        align-items: center;
        padding: 0.72rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.055);
    }

    .preview-step b {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 14px;
        color: #17120a;
        background: linear-gradient(135deg, #fff2bf, #f59e0b);
    }

    .preview-step span {
        display: block;
        color: #f7d58a;
        font-size: 0.78rem;
        margin-bottom: 0.18rem;
    }

    .preview-step p {
        margin: 0 !important;
        color: #dbe4f0 !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-kicker) {
        position: relative;
        border: 1px solid rgba(247, 213, 138, 0.25) !important;
        border-radius: 30px !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.034)),
            rgba(8, 13, 26, 0.72) !important;
        box-shadow: 0 24px 86px rgba(0, 0, 0, 0.34), 0 0 0 1px rgba(255,255,255,0.035) inset !important;
        backdrop-filter: blur(26px);
        overflow: hidden;
        padding: 0.4rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-kicker)::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: radial-gradient(circle at 6% 0%, rgba(247, 213, 138, 0.16), transparent 34%);
    }

    .input-kicker {
        color: #f7d58a !important;
        letter-spacing: 0.12rem;
    }

    .input-title {
        color: #fff7ed !important;
        font-size: 1.26rem !important;
    }

    .sample-title,
    .hint {
        color: #aeb8ca !important;
    }

    .stTextArea textarea {
        min-height: 170px !important;
        border-radius: 24px !important;
        border: 1px solid rgba(247, 213, 138, 0.36) !important;
        background:
            linear-gradient(145deg, rgba(2,6,23,0.88), rgba(15,23,42,0.70)) !important;
        box-shadow: 0 18px 58px rgba(0,0,0,0.26), 0 0 0 1px rgba(255,255,255,0.045) inset !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        min-height: 44px;
        border: 1px solid rgba(255, 247, 237, 0.24) !important;
        border-radius: 999px !important;
        background:
            linear-gradient(135deg, #fff2bf 0%, #f7d58a 36%, #f59e0b 100%) !important;
        color: #16120b !important;
        box-shadow: 0 15px 38px rgba(245, 158, 11, 0.22) !important;
        font-weight: 900 !important;
    }

    .cover-card {
        min-height: 520px !important;
        border-radius: 36px !important;
        border: 1px solid rgba(247, 213, 138, 0.30) !important;
        box-shadow: 0 42px 130px rgba(0, 0, 0, 0.52), 0 0 80px rgba(247, 213, 138, 0.08) inset !important;
        isolation: isolate;
    }

    .cover-card::before {
        z-index: 2 !important;
        background:
            linear-gradient(90deg, rgba(247, 213, 138, 0.20) 1px, transparent 1px),
            linear-gradient(180deg, rgba(255,255,255,0.09) 1px, transparent 1px),
            radial-gradient(circle at 88% 18%, rgba(155, 220, 255, 0.18), transparent 28%) !important;
        background-size: 94px 94px, 94px 94px, 100% 100% !important;
        opacity: 0.5 !important;
    }

    .cover-card::after {
        background:
            linear-gradient(90deg, rgba(2, 6, 23, 0.88), rgba(2, 6, 23, 0.34) 56%, rgba(2, 6, 23, 0.78)),
            linear-gradient(180deg, rgba(2, 6, 23, 0.02), rgba(2, 6, 23, 0.88)) !important;
    }

    .cover-content {
        z-index: 3 !important;
    }

    .cover-content .label {
        color: #f7d58a !important;
        letter-spacing: 0.18rem !important;
    }

    .cover-content h2 {
        color: #fff7ed !important;
        font-size: clamp(3rem, 7vw, 6.4rem) !important;
        text-shadow: 0 22px 80px rgba(0,0,0,0.5);
    }

    .cover-dayline {
        color: #fff2bf !important;
        font-size: clamp(1.1rem, 2.4vw, 1.6rem) !important;
    }

    .cover-badge,
    .weather-meta span,
    .food-meta span {
        border-color: rgba(247, 213, 138, 0.24) !important;
        background: rgba(247, 213, 138, 0.10) !important;
        color: #fff2bf !important;
    }

    .section-heading {
        margin-top: 2.6rem !important;
        font-size: clamp(1.7rem, 3vw, 2.35rem) !important;
        color: #fff7ed !important;
    }

    .section-heading::after {
        content: "";
        display: block;
        width: 86px;
        height: 2px;
        margin-top: 0.45rem;
        background: linear-gradient(90deg, #f7d58a, transparent);
    }

    .section-subtitle {
        color: #aeb8ca !important;
        max-width: 820px;
    }

    .bento-grid {
        grid-template-columns: repeat(6, minmax(0, 1fr)) !important;
        gap: 1rem !important;
    }

    .bento-card {
        grid-column: span 2;
        min-height: 148px !important;
        border-radius: 28px !important;
        border: 1px solid rgba(247, 213, 138, 0.18) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.105), rgba(255,255,255,0.026)),
            rgba(10, 16, 31, 0.64) !important;
        box-shadow: 0 22px 76px rgba(0, 0, 0, 0.28) !important;
    }

    .bento-card.large {
        grid-column: span 3 !important;
    }

    .bento-card.warm {
        background:
            radial-gradient(circle at 12% 10%, rgba(247, 213, 138, 0.18), transparent 38%),
            linear-gradient(145deg, rgba(247, 213, 138, 0.14), rgba(255,255,255,0.03)),
            rgba(10, 16, 31, 0.64) !important;
    }

    .bento-card span,
    .segment-card span {
        color: #f7d58a !important;
        letter-spacing: 0.04rem;
        text-transform: uppercase;
    }

    .bento-card strong {
        color: #fff7ed !important;
        font-size: clamp(1.25rem, 2.2vw, 1.68rem) !important;
    }

    .timeline-grid {
        grid-template-columns: minmax(0, 1fr) !important;
        gap: 1.25rem !important;
    }

    .timeline-day {
        position: relative;
        border-radius: 30px !important;
        border: 1px solid rgba(247, 213, 138, 0.18) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.105), rgba(255,255,255,0.026)),
            rgba(10, 16, 31, 0.66) !important;
        box-shadow: 0 24px 86px rgba(0,0,0,0.30) !important;
        padding: 1.25rem 1.25rem 1.25rem 1.45rem !important;
        overflow: hidden;
    }

    .timeline-day::before {
        content: "";
        position: absolute;
        left: 2.05rem;
        top: 4.6rem;
        bottom: 1.4rem;
        width: 1px;
        background: linear-gradient(180deg, #f7d58a, rgba(247, 213, 138, 0.05));
    }

    .timeline-day h3 {
        font-size: clamp(1.2rem, 2vw, 1.55rem) !important;
        color: #fff7ed !important;
        padding-left: 0.2rem;
    }

    .timeline-slot {
        position: relative;
        grid-template-columns: 54px minmax(0, 1fr) !important;
        gap: 1rem !important;
        border-top: 0 !important;
        padding: 0.7rem 0 !important;
    }

    .slot-icon {
        position: relative;
        z-index: 1;
        width: 46px !important;
        height: 46px !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #fff2bf, #f59e0b) !important;
        box-shadow: 0 14px 32px rgba(245, 158, 11, 0.23) !important;
    }

    .timeline-slot > div:nth-child(2) {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        background: rgba(255,255,255,0.045);
        padding: 0.86rem 0.92rem;
    }

    .slot-time,
    .slot-original,
    .weather-date {
        color: #f7d58a !important;
    }

    .slot-place,
    .weather-main {
        color: #fff7ed !important;
    }

    .slot-meta-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    }

    .slot-meta-item {
        border-color: rgba(247, 213, 138, 0.12) !important;
        background: rgba(7, 11, 22, 0.46) !important;
    }

    .food-grid,
    .info-grid,
    .warning-grid,
    .weather-grid,
    .budget-grid {
        gap: 1rem !important;
    }

    .food-card,
    .info-card,
    .warning-card,
    .weather-card,
    .budget-card,
    .segment-card {
        border-radius: 28px !important;
        border: 1px solid rgba(247, 213, 138, 0.17) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.025)),
            rgba(10, 16, 31, 0.64) !important;
        box-shadow: 0 22px 76px rgba(0,0,0,0.28) !important;
    }

    .budget-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        max-width: 100%;
        margin-bottom: 1.6rem;
    }

    .budget-card {
        min-width: 0;
        padding: 1rem;
    }

    .budget-card span {
        display: block;
        color: #f7d58a;
        font-size: 0.78rem;
        letter-spacing: 0.04rem;
        margin-bottom: 0.38rem;
        text-transform: uppercase;
    }

    .budget-card p {
        color: #cbd5e1;
        line-height: 1.62;
        margin: 0;
        overflow-wrap: anywhere;
    }

    .food-card {
        position: relative;
        overflow: hidden;
    }

    .food-card::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: radial-gradient(circle at 88% 10%, rgba(247, 213, 138, 0.13), transparent 28%);
    }

    .food-card h3 {
        position: relative;
        color: #fff7ed !important;
        font-size: 1.22rem;
    }

    .food-location,
    .food-map-keyword {
        position: relative;
        color: #aeb8ca !important;
    }

    .weather-card h3,
    .info-card h3,
    .warning-card h3 {
        color: #fff7ed !important;
    }

    .weather-day {
        border-top-color: rgba(247, 213, 138, 0.10) !important;
    }

    .weather-icon,
    .info-icon,
    .warning-icon {
        background: linear-gradient(135deg, #fff2bf, #f59e0b) !important;
        color: #17120a !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-actions-title),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.source-card-title) {
        border-radius: 28px !important;
        border: 1px solid rgba(247, 213, 138, 0.20) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03)),
            rgba(10, 16, 31, 0.64) !important;
        box-shadow: 0 22px 76px rgba(0,0,0,0.28) !important;
    }

    .result-actions-title,
    .source-card-title {
        color: #fff7ed;
        font-size: 1.25rem;
        font-weight: 900;
        margin: 0 0 0.35rem;
    }

    .source-card-text {
        color: #cbd5e1;
        line-height: 1.65;
        margin: 0.25rem 0;
    }

    .search-status-pill,
    .trust-card,
    .weather-fallback,
    .blessing-card {
        border-color: rgba(247, 213, 138, 0.18) !important;
        background:
            linear-gradient(145deg, rgba(247, 213, 138, 0.10), rgba(255,255,255,0.025)),
            rgba(10, 16, 31, 0.60) !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 0.72rem 0.68rem 3rem !important;
        }

        .top-nav {
            position: static;
            border-radius: 22px !important;
            margin-bottom: 0.9rem !important;
        }

        .hero.product-hero {
            border-radius: 26px;
            padding: 1rem;
        }

        .hero-layout,
        .bento-grid,
        .timeline-grid,
        .food-grid,
        .info-grid,
        .warning-grid,
        .weather-grid,
        .budget-grid,
        .segment-overview-grid,
        .trust-strip,
        .result-action-grid,
        .slot-meta-grid {
            grid-template-columns: minmax(0, 1fr) !important;
            width: 100% !important;
        }

        .hero h1,
        .hero-title {
            max-width: min(100%, 900px);
            margin: 0.55rem 0 1rem !important;
            font-size: clamp(2.375rem, 10.8vw, 2.875rem) !important;
            line-height: 1.12 !important;
            letter-spacing: 0 !important;
        }

        .hero-title-line {
            display: block;
        }

        .hero p {
            font-size: 0.96rem !important;
            line-height: 1.65 !important;
        }

        .hero-panel.product-preview {
            min-height: auto;
        }

        .preview-cover {
            min-height: 150px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-kicker) {
            border-radius: 24px !important;
        }

        .stTextArea textarea {
            min-height: 128px !important;
        }

        .cover-card {
            min-height: 300px !important;
            aspect-ratio: 4 / 3 !important;
            border-radius: 26px !important;
        }

        .cover-content h2 {
            font-size: clamp(2rem, 12vw, 3.3rem) !important;
        }

        .bento-card,
        .bento-card.large {
            grid-column: span 1 !important;
            min-height: auto !important;
        }

        .timeline-day::before {
            left: 1.55rem;
            top: 4.5rem;
        }

        .timeline-slot {
            grid-template-columns: 42px minmax(0, 1fr) !important;
            gap: 0.72rem !important;
        }

        .slot-icon {
            width: 36px !important;
            height: 36px !important;
            border-radius: 13px !important;
            font-size: 0.76rem !important;
        }

        .timeline-slot > div:nth-child(2) {
            padding: 0.72rem;
            border-radius: 17px;
        }

        .weather-day {
            grid-template-columns: 38px minmax(0, 1fr) !important;
        }

        .weather-meta span,
        .food-meta span {
            width: 100%;
        }

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 0 !important;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {
            width: 100% !important;
            white-space: normal !important;
        }
    }

    @media (max-width: 520px) {
        .hero-proof span {
            width: 100%;
        }

        .preview-step {
            grid-template-columns: 36px minmax(0, 1fr);
        }

        .preview-step b {
            width: 34px;
            height: 34px;
        }

        .section-heading {
            font-size: 1.45rem !important;
        }

        .cover-content {
            inset: auto 0.9rem 0.95rem 0.9rem !important;
        }
    }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)


def parse_chinese_number(number_text: str) -> int:
    """parse_chinese_number：把常见中文数字转换成整数。"""

    # chinese_number_map：保存中文数字到阿拉伯数字的对应关系。
    chinese_number_map = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    if number_text.isdigit():
        return int(number_text)

    if number_text == "十":
        return 10

    if number_text.startswith("十"):
        return 10 + chinese_number_map.get(number_text[-1], 0)

    if "十" in number_text:
        # parts：中文数字按“十”拆分后的十位和个位。
        parts = number_text.split("十")
        tens = chinese_number_map.get(parts[0], 1)
        ones = chinese_number_map.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones

    return chinese_number_map.get(number_text, DEFAULT_TRAVEL_DAYS)


def parse_chinese_amount(amount_text: str) -> float | None:
    """parse_chinese_amount：把中文预算金额转换成数字，例如“一万”转成 10000。"""

    # cleaned_amount：清理空格后的中文金额文本。
    cleaned_amount = amount_text.strip()
    if not cleaned_amount:
        return None

    if re.fullmatch(r"[0-9][0-9,]*(?:\.\d+)?", cleaned_amount):
        return float(cleaned_amount.replace(",", ""))

    # chinese_digit_map：中文数字字符和数值的对应关系。
    chinese_digit_map = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    if cleaned_amount.endswith("万"):
        # base_text：中文金额中“万”前面的数字部分。
        base_text = cleaned_amount[:-1]
        if not base_text:
            return 10000.0
        base_value = parse_chinese_number(base_text)
        return float(base_value * 10000)

    if cleaned_amount in chinese_digit_map:
        return float(chinese_digit_map[cleaned_amount])

    if "千" in cleaned_amount:
        # thousand_parts：中文金额按“千”拆分后的千位和余数。
        thousand_parts = cleaned_amount.split("千", 1)
        thousands = parse_chinese_number(thousand_parts[0] or "一") * 1000
        rest = parse_chinese_amount(thousand_parts[1]) if thousand_parts[1] else 0
        return float(thousands + (rest or 0))

    return None


def normalize_currency_unit(currency_text: str | None) -> str:
    """normalize_currency_unit：把用户输入的货币单位统一成标准货币代码。"""

    if not currency_text:
        return DEFAULT_BUDGET_CURRENCY

    # normalized_unit：统一大小写并去除空格后的货币单位。
    normalized_unit = currency_text.strip().upper()

    # currency_alias_map：常见货币表达和标准货币代码的对应关系。
    currency_alias_map = {
        "人民币": "CNY",
        "RMB": "CNY",
        "CNY": "CNY",
        "元": "CNY",
        "块": "CNY",
        "日元": "JPY",
        "日币": "JPY",
        "日圓": "JPY",
        "JPY": "JPY",
        "美元": "USD",
        "美金": "USD",
        "USD": "USD",
        "欧元": "EUR",
        "欧": "EUR",
        "EUR": "EUR",
        "韩元": "KRW",
        "韩币": "KRW",
        "KRW": "KRW",
    }

    return currency_alias_map.get(normalized_unit, DEFAULT_BUDGET_CURRENCY)


def get_currency_name(currency_code: str) -> str:
    """get_currency_name：把标准货币代码转换成中文显示名称。"""

    # currency_name_map：标准货币代码和中文名称的对应关系。
    currency_name_map = {
        "CNY": "人民币",
        "JPY": "日元",
        "USD": "美元",
        "EUR": "欧元",
        "KRW": "韩元",
    }

    return currency_name_map.get(currency_code, currency_code)


def format_budget_amount(amount: float) -> str:
    """format_budget_amount：把预算金额格式化为适合页面展示的文本。"""

    # numeric_amount：兼容 int 和 float 的预算金额数值。
    numeric_amount = float(amount)

    if numeric_amount.is_integer():
        return f"{int(numeric_amount):,}"

    return f"{numeric_amount:,.2f}".rstrip("0").rstrip(".")


def parse_budget_info(cleaned_input: str, budget_level: str) -> dict:
    """parse_budget_info：识别用户输入中的预算金额和货币单位。"""

    # budget_pattern：匹配“预算5000”“预算一万”“预算10万日元”“预算800 USD”等表达。
    budget_pattern = re.compile(
        r"(?:预算|总预算|花费|费用)\s*"
        r"(?:约|大概|大约|控制在|不超过|不超|以内|左右|是|为|:|：)?\s*"
        r"([0-9][0-9,]*(?:\.\d+)?|[零一二两三四五六七八九十百千万]+)\s*"
        r"(万)?\s*"
        r"(人民币|日元|日币|日圓|美元|美金|欧元|欧|韩元|韩币|USD|EUR|JPY|KRW|RMB|CNY|元|块)?",
        re.IGNORECASE,
    )

    # budget_match：预算金额匹配结果。
    budget_match = budget_pattern.search(cleaned_input)
    if not budget_match:
        return {
            "amount": None,
            "currency": None,
            "currency_name": None,
            "display": budget_level,
            "level": budget_level,
            "has_explicit_amount": False,
        }

    # amount_text：预算金额文本。
    amount_text = budget_match.group(1)

    # amount_value：预算金额数值。
    amount_value = parse_chinese_amount(amount_text)
    if amount_value is None:
        amount_value = float(amount_text.replace(",", ""))

    if budget_match.group(2) and "万" not in amount_text:
        amount_value *= 10000

    # currency_code：标准货币代码；用户没写单位时默认 CNY。
    currency_code = normalize_currency_unit(budget_match.group(3))

    # currency_name：中文货币名称。
    currency_name = get_currency_name(currency_code)

    # budget_display：页面和提示词展示的预算文本。
    budget_display = f"{format_budget_amount(amount_value)} {currency_name} ({currency_code})"

    # normalized_amount：整数金额保存为 int，便于 Debug 区显示 10000 而不是 10000.0。
    normalized_amount = int(amount_value) if float(amount_value).is_integer() else amount_value

    return {
        "amount": normalized_amount,
        "currency": currency_code,
        "currency_name": currency_name,
        "display": budget_display,
        "level": budget_level,
        "has_explicit_amount": True,
    }


def infer_destination_currency(destination: str) -> str | None:
    """infer_destination_currency：根据目的地粗略推断当地常用货币。"""

    # destination_currency_keywords：国外目的地关键词和当地货币代码。
    destination_currency_keywords = {
        "JPY": ["日本", "东京", "大阪", "京都", "北海道", "冲绳", "奈良", "福冈", "名古屋", "札幌", "箱根"],
        "KRW": ["韩国", "首尔", "釜山", "济州"],
        "EUR": [
            "欧洲",
            "法国",
            "巴黎",
            "意大利",
            "罗马",
            "米兰",
            "德国",
            "柏林",
            "西班牙",
            "巴塞罗那",
            "荷兰",
            "阿姆斯特丹",
            "葡萄牙",
            "希腊",
            "瑞士",
        ],
        "USD": ["美国", "纽约", "洛杉矶", "旧金山", "西雅图", "夏威夷"],
    }

    for currency_code, keyword_list in destination_currency_keywords.items():
        if any(keyword in destination for keyword in keyword_list):
            return currency_code

    return None


def build_exchange_hint(parsed_request: dict) -> str | None:
    """build_exchange_hint：为国外目的地生成粗略换算提示。"""

    # budget_amount：用户输入的预算金额。
    budget_amount = parsed_request.get("budget_amount")
    if not budget_amount:
        return None

    # destination_currency：根据目的地推断出的当地货币。
    destination_currency = infer_destination_currency(parsed_request["destination"])
    if not destination_currency:
        return None

    # source_currency：用户输入预算的货币代码。
    source_currency = parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY

    # cny_to_currency_rate：人民币到其他货币的粗略换算比例。
    cny_to_currency_rate = {
        "JPY": 20.0,
        "KRW": 190.0,
        "EUR": 0.13,
        "USD": 0.14,
    }

    # currency_to_cny_rate：其他货币到人民币的粗略换算比例。
    currency_to_cny_rate = {
        "JPY": 0.05,
        "KRW": 0.0053,
        "EUR": 7.8,
        "USD": 7.2,
        "CNY": 1.0,
    }

    if source_currency == DEFAULT_BUDGET_CURRENCY and destination_currency in cny_to_currency_rate:
        # converted_amount：人民币预算换算成目的地当地货币的粗略金额。
        converted_amount = budget_amount * cny_to_currency_rate[destination_currency]
        return (
            f"粗略换算：{format_budget_amount(budget_amount)} 人民币约 "
            f"{format_budget_amount(converted_amount)} {get_currency_name(destination_currency)}"
            "，汇率仅供参考，请以出行前实际汇率为准。"
        )

    if source_currency != DEFAULT_BUDGET_CURRENCY and source_currency in currency_to_cny_rate:
        # converted_amount：外币预算换算成人民币的粗略金额。
        converted_amount = budget_amount * currency_to_cny_rate[source_currency]
        return (
            f"粗略换算：{format_budget_amount(budget_amount)} {get_currency_name(source_currency)}约 "
            f"{format_budget_amount(converted_amount)} 人民币"
            "，汇率仅供参考，请以出行前实际汇率为准。"
        )

    return None


def extract_destination(cleaned_input: str) -> str:
    """extract_destination：从用户输入中优先提取明确目的地。"""

    # destination_patterns：从强到弱排列的目的地匹配规则。
    destination_patterns = [
        r"(?:^|[，。,\s])([一-龥A-Za-z]{2,20})\s*(?:[0-9一二两三四五六七八九十]+)\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行)",
        r"想去(?!看看|看一看|看|尝|尝一尝|吃|逛)([一-龥A-Za-z]{2,20}?)(?:旅游|旅行|自由行|游|度假|玩|看|赏|吃|逛|，|。|,|\s|$)",
        r"去(?!看看|看一看|看|尝|尝一尝|吃|逛)([一-龥A-Za-z]{2,20}?)(?:旅游|旅行|自由行|游|度假|玩|看|赏|吃|逛|，|。|,|\s|$)",
        r"(?:^|[，。,\s])([一-龥A-Za-z]{2,20}?)\s*(?:旅游|旅行|自由行|度假|游)",
        r"目的地[:：]\s*([一-龥A-Za-z]{2,20})",
    ]

    # invalid_destination_words：不应被当成目的地的动作词或泛词。
    invalid_destination_words = {
        "看看",
        "看一看",
        "看",
        "尝",
        "尝一尝",
        "美食",
        "夜景",
        "其他景点",
        "景点",
    }

    for pattern in destination_patterns:
        match = re.search(pattern, cleaned_input)
        if not match:
            continue

        # destination：当前规则识别出的目的地。
        destination = match.group(1).strip()
        destination = re.sub(r"^(?:我|我们|本人)?(?:想去|要去|计划去|打算去|去)", "", destination).strip()
        if destination and destination not in invalid_destination_words:
            return destination

    return DEFAULT_DESTINATION


def extract_trip_days(cleaned_input: str) -> tuple[int, int]:
    """extract_trip_days：识别“7日游”“7天”“七日”等明确天数。"""

    # days_patterns：可识别的天数表达。
    days_patterns = [
        r"([0-9一二两三四五六七八九十]+)\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行)",
        r"([0-9一二两三四五六七八九十]+)\s*(?:天|日)(?!元|币)",
    ]

    for pattern in days_patterns:
        match = re.search(pattern, cleaned_input)
        if match:
            days = max(1, parse_chinese_number(match.group(1)))
            return days, max(0, days - 1)

    return DEFAULT_TRAVEL_DAYS, DEFAULT_TRAVEL_NIGHTS


def get_known_destination_names() -> list[str]:
    """get_known_destination_names：返回用于多目的地识别的常见目的地名称。"""

    return [
        "内蒙古",
        "黑龙江",
        "张家界",
        "九寨沟",
        "西双版纳",
        "香格里拉",
        "南京",
        "江西",
        "南昌",
        "景德镇",
        "婺源",
        "庐山",
        "上饶",
        "赣州",
        "上海",
        "苏州",
        "杭州",
        "东京",
        "京都",
        "大阪",
        "广州",
        "深圳",
        "北京",
        "成都",
        "重庆",
        "西安",
        "云南",
        "大理",
        "丽江",
        "昆明",
        "福建",
        "厦门",
        "泉州",
        "福州",
        "广东",
        "海南",
        "三亚",
        "青岛",
        "长沙",
        "武汉",
        "天津",
        "首尔",
        "釜山",
        "济州",
        "日本",
        "韩国",
        "欧洲",
        "法国",
        "巴黎",
        "意大利",
        "罗马",
        "美国",
        "纽约",
        "洛杉矶",
    ]


def get_province_route_note(destination: str) -> str:
    """get_province_route_note：为省份或大区域目的地生成具体城市路线提示。"""

    # province_route_map：省份或大区域到经典城市组合的映射。
    province_route_map = {
        "江西": "你输入的是省份，系统为你选择较经典的江西路线：南昌、景德镇、婺源、庐山、上饶，可根据偏好调整。",
        "云南": "你输入的是省份，系统会优先按昆明、大理、丽江、香格里拉等经典路线规划，可根据偏好调整。",
        "福建": "你输入的是省份，系统会优先按厦门、泉州、福州或武夷山等经典路线规划，可根据偏好调整。",
        "广东": "你输入的是省份，系统会优先按广州、深圳、珠海或潮汕等经典路线规划，可根据偏好调整。",
        "海南": "你输入的是省份，系统会优先按海口、三亚、万宁等经典路线规划，可根据偏好调整。",
        "日本": "你输入的是国家，系统会优先按东京、京都、大阪等经典路线规划，可根据偏好调整。",
        "韩国": "你输入的是国家，系统会优先按首尔、釜山、济州等经典路线规划，可根据偏好调整。",
        "欧洲": "你输入的是大区域，系统会选择适合天数的城市组合，并明确说明推断依据。",
    }

    return province_route_map.get(destination, "")


def get_weather_reference_city(destination: str, travel_json: dict | None = None) -> tuple[str, str]:
    """get_weather_reference_city：把省份或大区域目的地转换为适合查询天气的主要城市。"""

    # destination_city_map：省份、国家或大区域到天气参考城市的映射。
    destination_city_map = {
        "江西": "南昌",
        "云南": "昆明",
        "福建": "厦门",
        "广东": "广州",
        "海南": "海口",
        "日本": "东京",
        "韩国": "首尔",
        "欧洲": "巴黎",
    }

    # city_from_map：从固定映射中得到的参考城市。
    city_from_map = destination_city_map.get(destination)
    if city_from_map:
        return city_from_map, f"{destination}主要城市天气参考：{city_from_map}"

    return destination, destination


def geocode_destination(destination: str) -> dict | None:
    """geocode_destination：使用 Open-Meteo Geocoding API 把目的地转换为经纬度。"""

    if not destination:
        return None

    try:
        # response：Open-Meteo 地理编码接口响应。
        response = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params={
                "name": destination,
                "count": 1,
                "language": "zh",
                "format": "json",
            },
            timeout=8,
        )
        response.raise_for_status()
        # geocode_data：地理编码 JSON 数据。
        geocode_data = response.json()
    except Exception:
        return None

    # result_list：Open-Meteo 返回的候选地点列表。
    result_list = geocode_data.get("results", [])
    if not result_list:
        return None

    # first_result：最匹配的地点。
    first_result = result_list[0]
    latitude = first_result.get("latitude")
    longitude = first_result.get("longitude")
    if latitude is None or longitude is None:
        return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "name": first_result.get("name", destination),
        "country": first_result.get("country", ""),
        "timezone": first_result.get("timezone", "auto"),
    }


def fetch_weather_forecast(latitude: float, longitude: float) -> dict | None:
    """fetch_weather_forecast：使用 Open-Meteo Forecast API 查询未来天气。"""

    try:
        # response：Open-Meteo 天气预报接口响应。
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": ",".join(
                    [
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                        "wind_speed_10m_max",
                    ]
                ),
                "hourly": "relative_humidity_2m",
                "timezone": "auto",
                "forecast_days": WEATHER_FORECAST_DAYS,
            },
            timeout=8,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def map_weather_code(weather_code: int | None) -> tuple[str, str]:
    """map_weather_code：把 Open-Meteo 天气代码转换成中文天气状态和图标。"""

    if weather_code is None:
        return "天气待确认", "🌦️"

    # weather_code_map：WMO 天气代码到中文状态的映射。
    weather_code_map = {
        0: ("晴", "☀️"),
        1: ("大致晴朗", "🌤️"),
        2: ("多云", "🌤️"),
        3: ("阴", "☁️"),
        45: ("有雾", "☁️"),
        48: ("雾凇", "☁️"),
        51: ("小毛毛雨", "🌧️"),
        53: ("毛毛雨", "🌧️"),
        55: ("较强毛毛雨", "🌧️"),
        56: ("冻毛毛雨", "🌧️"),
        57: ("强冻毛毛雨", "🌧️"),
        61: ("小雨", "🌧️"),
        63: ("中雨", "🌧️"),
        65: ("大雨", "🌧️"),
        66: ("冻雨", "🌧️"),
        67: ("强冻雨", "🌧️"),
        71: ("小雪", "🌨️"),
        73: ("中雪", "🌨️"),
        75: ("大雪", "🌨️"),
        77: ("雪粒", "🌨️"),
        80: ("阵雨", "🌧️"),
        81: ("较强阵雨", "🌧️"),
        82: ("强阵雨", "🌧️"),
        85: ("阵雪", "🌨️"),
        86: ("强阵雪", "🌨️"),
        95: ("雷雨", "⛈️"),
        96: ("雷雨伴冰雹", "⛈️"),
        99: ("强雷雨伴冰雹", "⛈️"),
    }

    return weather_code_map.get(weather_code, ("天气待确认", "🌦️"))


def format_weather_date(date_text: str) -> str:
    """format_weather_date：把 YYYY-MM-DD 日期转换成中文短日期。"""

    try:
        # date_value：解析后的日期对象。
        date_value = datetime.fromisoformat(date_text)
        return f"{date_value.month}月{date_value.day}日"
    except ValueError:
        return date_text


def format_weather_value(value: float | int | None, suffix: str) -> str:
    """format_weather_value：格式化天气数值，缺失时不编造。"""

    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{round(value)}{suffix}"
    return f"{value}{suffix}"


def calculate_daily_humidity(weather_data: dict) -> dict[str, int | None]:
    """calculate_daily_humidity：从小时级湿度数据计算每天平均湿度。"""

    # hourly_data：Open-Meteo 小时级天气数据。
    hourly_data = weather_data.get("hourly", {})

    # time_list/humidity_list：小时和相对湿度列表。
    time_list = hourly_data.get("time", [])
    humidity_list = hourly_data.get("relative_humidity_2m", [])

    # humidity_bucket：按日期收集的湿度值。
    humidity_bucket: dict[str, list[float]] = {}
    for time_text, humidity_value in zip(time_list, humidity_list):
        if humidity_value is None or not time_text:
            continue
        date_key = str(time_text).split("T", 1)[0]
        humidity_bucket.setdefault(date_key, []).append(float(humidity_value))

    # humidity_by_date：每天平均湿度。
    humidity_by_date: dict[str, int | None] = {}
    for date_key, values in humidity_bucket.items():
        humidity_by_date[date_key] = round(sum(values) / len(values)) if values else None

    return humidity_by_date


def build_single_weather_advice(weather_item: dict) -> str:
    """build_single_weather_advice：根据单日天气生成携带和出行提醒。"""

    # weather_text：中文天气状态。
    weather_text = str(weather_item.get("weather_text", ""))

    # precipitation_probability：降水概率。
    precipitation_probability = weather_item.get("precipitation_probability")

    # temperature_max/temperature_min/wind_speed：温度和风速。
    temperature_max = weather_item.get("temperature_max")
    temperature_min = weather_item.get("temperature_min")
    wind_speed = weather_item.get("wind_speed")

    # advice_parts：多条提醒合并后的建议。
    advice_parts = []

    # rainy_words：用于判断雨天的关键词。
    rainy_words = ["雨", "小雨", "中雨", "大雨", "阵雨", "雷雨"]
    is_rainy = precipitation_probability is not None and precipitation_probability >= 50
    if is_rainy or any(word in weather_text for word in rainy_words):
        advice_parts.append("可能下雨，建议携带雨伞、防水袋和防滑鞋，户外景点注意地面湿滑。☔")

    if temperature_max is not None and temperature_max >= 30:
        advice_parts.append("气温偏高，注意防晒、补水，尽量避开中午长时间暴晒。")

    if temperature_min is not None and temperature_min <= 10:
        advice_parts.append("早晚偏冷，建议带外套，注意昼夜温差。")

    if "晴" in weather_text:
        advice_parts.append("晴天适合户外游玩，记得准备防晒、墨镜和水。☀️")

    if "多云" in weather_text or "阴" in weather_text:
        advice_parts.append("适合步行游玩，但天气变化仍建议出门前查看实时天气。")

    if wind_speed is not None and wind_speed >= 38:
        advice_parts.append("风力偏大，注意保暖，高处观景或乘船行程要留意安全。🌬️")

    if not advice_parts:
        advice_parts.append("天气信息仅供参考，请出行前查看天气 App。")

    return " ".join(advice_parts)


def build_weather_advice(weather_data: dict) -> list[dict]:
    """build_weather_advice：把 Open-Meteo 天气数据整理成每日天气卡片。"""

    if not isinstance(weather_data, dict):
        return []

    # daily_data：Open-Meteo 每日天气数据。
    daily_data = weather_data.get("daily", {})
    date_list = daily_data.get("time", [])
    if not date_list:
        return []

    # humidity_by_date：按日期计算出的平均湿度。
    humidity_by_date = calculate_daily_humidity(weather_data)

    # weather_items：每日天气卡片数据。
    weather_items = []
    for index, date_text in enumerate(date_list[:WEATHER_FORECAST_DAYS]):
        # weather_code：Open-Meteo WMO 天气代码。
        weather_code_list = daily_data.get("weather_code", [])
        weather_code = weather_code_list[index] if index < len(weather_code_list) else None
        weather_text, weather_icon = map_weather_code(weather_code)

        # temperature_max/min：最高和最低温度。
        temperature_max_list = daily_data.get("temperature_2m_max", [])
        temperature_min_list = daily_data.get("temperature_2m_min", [])
        temperature_max = temperature_max_list[index] if index < len(temperature_max_list) else None
        temperature_min = temperature_min_list[index] if index < len(temperature_min_list) else None

        # precipitation_probability：最大降水概率。
        precipitation_list = daily_data.get("precipitation_probability_max", [])
        precipitation_probability = precipitation_list[index] if index < len(precipitation_list) else None

        # wind_speed：最大风速。
        wind_speed_list = daily_data.get("wind_speed_10m_max", [])
        wind_speed = wind_speed_list[index] if index < len(wind_speed_list) else None
        if wind_speed is not None and wind_speed >= 38:
            weather_icon = "🌬️"
            weather_text = f"{weather_text}，风力偏大"

        # weather_item：单日天气展示数据。
        weather_item = {
            "date": format_weather_date(str(date_text)),
            "raw_date": str(date_text),
            "weather_text": weather_text,
            "weather_icon": weather_icon,
            "temperature_max": temperature_max,
            "temperature_min": temperature_min,
            "humidity": humidity_by_date.get(str(date_text)),
            "precipitation_probability": precipitation_probability,
            "wind_speed": wind_speed,
            "will_rain": bool(
                (precipitation_probability is not None and precipitation_probability >= 50)
                or any(word in weather_text for word in ["雨", "阵雨", "雷雨"])
            ),
        }
        weather_item["advice"] = build_single_weather_advice(weather_item)
        weather_items.append(weather_item)

    return weather_items


def build_weather_cards(parsed_request: dict, travel_json: dict | None = None) -> list[dict]:
    """build_weather_cards：为单目的地或多目的地构建天气模块卡片数据。"""

    # trip_segments：目的地分段列表。
    trip_segments = parsed_request.get("trip_segments") or [{"destination": parsed_request["destination"]}]

    # weather_cards：所有目的地天气卡片。
    weather_cards = []
    seen_weather_destinations = set()
    for segment in trip_segments:
        # destination：当前天气卡片对应的目的地。
        destination = str(segment.get("destination", "")).strip()
        if not destination or destination in seen_weather_destinations:
            continue
        seen_weather_destinations.add(destination)

        # query_city/display_destination：用于查询天气的城市和页面展示标题。
        query_city, display_destination = get_weather_reference_city(destination, travel_json)

        # geocode_result：目的地经纬度。
        geocode_result = geocode_destination(query_city)
        if not geocode_result:
            weather_cards.append(
                {
                    "destination": display_destination,
                    "query_city": query_city,
                    "error": "天气信息暂时无法获取，请出行前查看天气 App。",
                    "days": [],
                }
            )
            continue

        # forecast_data：Open-Meteo 天气预报数据。
        forecast_data = fetch_weather_forecast(geocode_result["latitude"], geocode_result["longitude"])
        if not forecast_data:
            weather_cards.append(
                {
                    "destination": display_destination,
                    "query_city": query_city,
                    "error": "天气信息暂时无法获取，请出行前查看天气 App。",
                    "days": [],
                }
            )
            continue

        # day_weather_items：每日天气卡片数据。
        day_weather_items = build_weather_advice(forecast_data)
        if not day_weather_items:
            weather_cards.append(
                {
                    "destination": display_destination,
                    "query_city": query_city,
                    "error": "天气信息暂时无法获取，请出行前查看天气 App。",
                    "days": [],
                }
            )
            continue

        weather_cards.append(
            {
                "destination": display_destination,
                "query_city": query_city,
                "error": "",
                "days": day_weather_items,
            }
        )

    return weather_cards


def build_weather_markdown(weather_cards: list[dict] | None) -> str:
    """build_weather_markdown：把天气卡片转换成可复制 Markdown。"""

    if not weather_cards:
        return "## 天气与出行提醒 🌦️\n天气信息暂时无法获取，请出行前查看天气 App。"

    # markdown_lines：天气 Markdown 行。
    markdown_lines = ["## 天气与出行提醒 🌦️"]
    for weather_card in weather_cards:
        destination = str(weather_card.get("destination", "目的地"))
        markdown_lines.append(f"### {destination}｜未来 {WEATHER_FORECAST_DAYS} 天天气")
        if weather_card.get("error"):
            markdown_lines.append(str(weather_card["error"]))
            continue

        for day_weather in weather_card.get("days", []):
            will_rain_text = "可能下雨" if day_weather.get("will_rain") else "降雨风险较低"
            markdown_lines.extend(
                [
                    f"#### {day_weather.get('date', '')}",
                    f"- 天气：{day_weather.get('weather_text', '天气待确认')}",
                    (
                        f"- 温度：{format_weather_value(day_weather.get('temperature_min'), '°C')} - "
                        f"{format_weather_value(day_weather.get('temperature_max'), '°C')}"
                    ),
                    f"- 湿度：{format_weather_value(day_weather.get('humidity'), '%')}",
                    f"- 降水概率：{format_weather_value(day_weather.get('precipitation_probability'), '%')}",
                    f"- 是否可能下雨：{will_rain_text}",
                    f"- 建议：{day_weather.get('advice', '天气信息仅供参考，请出行前查看天气 App。')}",
                ]
            )

    return "\n".join(markdown_lines)


def remove_markdown_sections(markdown_text: str, heading_names: set[str]) -> str:
    """remove_markdown_sections：移除指定二级标题章节，避免 Markdown 原文重复。"""

    # markdown_lines：原始 Markdown 行。
    markdown_lines = markdown_text.splitlines()

    # kept_lines：保留下来的 Markdown 行。
    kept_lines = []
    skipping = False
    for line in markdown_lines:
        # heading_match：匹配二级标题。
        heading_match = re.match(r"^##\s+(.+?)\s*$", line)
        if heading_match:
            heading_text = heading_match.group(1).strip()
            skipping = heading_text in heading_names
            if skipping:
                continue
        if not skipping:
            kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def append_weather_and_blessing_to_markdown(markdown_text: str, weather_cards: list[dict] | None, generated_at: str) -> str:
    """append_weather_and_blessing_to_markdown：把天气、更新时间和祝福语追加到 Markdown 末尾。"""

    # cleaned_markdown：移除模型可能生成的旧信息区，避免重复和技术词外露。
    cleaned_markdown = remove_markdown_sections(
        markdown_text,
        {
            "天气与出行提醒",
            "天气与出行提醒 🌦️",
            "信息来源与更新时间",
            "信息与更新时间",
            "旅行祝福语",
        },
    )

    # source_markdown：用户友好的信息与更新时间。
    source_markdown = "\n".join(
        [
            "## 信息与更新时间",
            "本攻略由 AI 根据你的输入和当前可用信息整理生成。",
            f"更新时间：{generated_at}",
            "门票、预约、开放时间、交通政策和天气情况可能变化，请出行前以官方渠道和天气 App 为准。",
        ]
    )

    # blessing_markdown：旅行祝福语。
    blessing_markdown = (
        "## 旅行祝福语\n"
        "祝你这次旅行顺利又开心。记得提前确认天气、门票和交通安排，慢慢走、好好看，"
        "把喜欢的风景都装进记忆里。祝你旅途愉快呀～ 🌿✨🧳"
    )

    return "\n\n".join([cleaned_markdown, build_weather_markdown(weather_cards), source_markdown, blessing_markdown]).strip()


def extract_days_near_destination(text_after_destination: str) -> int | None:
    """extract_days_near_destination：在目的地后方的小片段中提取天数。"""

    # day_match：匹配“玩2天”“游玩3天”“4日”等表达。
    day_match = re.search(
        r"(?:游玩|玩|旅游|旅行|自由行|游)?\s*([0-9一二两三四五六七八九十]+)\s*(?:天|日)(?!元|币)",
        text_after_destination,
    )
    if not day_match:
        return None

    return max(1, parse_chinese_number(day_match.group(1)))


def extract_trip_segments(cleaned_input: str, preferences: list[str]) -> list[dict]:
    """extract_trip_segments：识别单目的地或多目的地分段行程。"""

    # known_destinations：常见目的地词表，按长度降序避免“江西”被更短词截断。
    known_destinations = sorted(get_known_destination_names(), key=len, reverse=True)

    # destination_matches：用户输入中出现的目的地及位置。
    destination_matches = []
    for destination in known_destinations:
        for match in re.finditer(re.escape(destination), cleaned_input):
            destination_matches.append({"destination": destination, "start": match.start(), "end": match.end()})

    destination_matches.sort(key=lambda item: item["start"])

    # deduped_matches：按文本位置去重，避免同一位置重复匹配。
    deduped_matches = []
    occupied_ranges = []
    for item in destination_matches:
        if any(item["start"] >= start and item["end"] <= end for start, end in occupied_ranges):
            continue
        deduped_matches.append(item)
        occupied_ranges.append((item["start"], item["end"]))

    # unique_matches：同一个目的地多次出现时只保留第一次，避免“杭州美食、杭州夜景”被拆成多段。
    unique_matches = []
    seen_destinations = set()
    for item in deduped_matches:
        destination = item["destination"]
        if destination in seen_destinations:
            continue
        unique_matches.append(item)
        seen_destinations.add(destination)
    deduped_matches = unique_matches

    if not deduped_matches:
        destination = extract_destination(cleaned_input)
        days, nights = extract_trip_days(cleaned_input)
        return [
            {
                "destination": destination,
                "days": days,
                "nights": nights,
                "days_inferred": False,
                "preferences": preferences,
                "note": get_province_route_note(destination),
            }
        ]

    # segments：最终分段行程列表。
    segments = []
    for index, item in enumerate(deduped_matches):
        next_start = deduped_matches[index + 1]["start"] if index + 1 < len(deduped_matches) else len(cleaned_input)
        segment_text = cleaned_input[item["end"] : next_start]
        days = extract_days_near_destination(segment_text)
        days_inferred = days is None
        if days_inferred:
            days = DEFAULT_TRAVEL_DAYS

        destination = item["destination"]
        note = get_province_route_note(destination)
        if days_inferred:
            inferred_note = f"用户未说明{destination}游玩天数，系统默认按{DEFAULT_TRAVEL_DAYS}天规划。"
            note = f"{inferred_note} {note}".strip()

        segments.append(
            {
                "destination": destination,
                "days": days,
                "nights": max(0, days - 1),
                "days_inferred": days_inferred,
                "preferences": preferences,
                "note": note,
            }
        )

    # 如果只识别出一个目的地，沿用全局天数解析，避免“杭州7日游”被默认覆盖。
    if len(segments) == 1:
        days, nights = extract_trip_days(cleaned_input)
        explicit_days_found = bool(re.search(r"[0-9一二两三四五六七八九十]+\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行|天|日)", cleaned_input))
        segments[0]["days"] = days
        segments[0]["nights"] = nights
        segments[0]["days_inferred"] = not explicit_days_found
        if segments[0]["days_inferred"]:
            inferred_note = f"用户未说明{segments[0]['destination']}游玩天数，系统默认按{DEFAULT_TRAVEL_DAYS}天规划。"
            segments[0]["note"] = f"{inferred_note} {get_province_route_note(segments[0]['destination'])}".strip()

    return segments


def infer_budget_level(cleaned_input: str) -> str:
    """infer_budget_level：识别用户输入中的预算风格档位。"""

    if re.search(r"穷游|省钱|低预算|便宜|学生党", cleaned_input):
        return "经济预算"

    if re.search(r"舒适一点|舒适|舒服|品质|高端|豪华|不差钱|预算充足", cleaned_input):
        return "舒适预算"

    return DEFAULT_BUDGET_LEVEL


def extract_preferences(cleaned_input: str) -> list[str]:
    """extract_preferences：从用户输入中提取旅行偏好、景点和体验主题。"""

    # preference_keywords：可识别的旅行偏好关键词。
    preference_keywords = [
        "西湖",
        "灵隐寺",
        "美食",
        "夜景",
        "自然",
        "动漫",
        "购物",
        "历史",
        "拍照",
        "文化",
        "博物馆",
        "亲子",
        "海边",
        "徒步",
        "温泉",
        "咖啡",
        "艺术",
    ]

    # preferences：从用户输入中识别出的偏好列表。
    preferences = []
    for keyword in preference_keywords:
        if keyword in cleaned_input and keyword not in preferences:
            preferences.append(keyword)

    if not preferences:
        preferences = ["美食", "拍照"]

    return preferences


def extract_fact_check_spots(parsed_request: dict) -> list[str]:
    """extract_fact_check_spots：提取需要联网校验门票、预约和开放时间的景点。"""

    # generic_preferences：不适合作为具体景点搜索的泛偏好。
    generic_preferences = {
        "美食",
        "夜景",
        "自然",
        "动漫",
        "购物",
        "历史",
        "拍照",
        "文化",
        "博物馆",
        "亲子",
        "海边",
        "徒步",
        "温泉",
        "咖啡",
        "艺术",
    }

    # spot_list：从偏好里筛出的具体景点。
    spot_list = [item for item in parsed_request["preferences"] if item not in generic_preferences]

    if not spot_list:
        spot_list = ["主要景点"]

    return spot_list[:4]


def build_fact_search_queries(parsed_request: dict) -> list[str]:
    """build_fact_search_queries：生成省额度的合并搜索查询，避免每个景点单独搜索。"""

    # destination：目的地名称，多目的地时合并为一个省额度查询。
    destination = " ".join(parsed_request.get("destinations") or [parsed_request["destination"]])

    # spot_list：需要查询的景点列表。
    spot_list = extract_fact_check_spots(parsed_request)

    # important_spots：用户明确提到的重点景点，最多放入 4 个，避免 query 过长。
    important_spots = [spot for spot in spot_list if spot != "主要景点"][:4]

    # spot_text：重点景点文本，没有明确景点时使用热门景点兜底。
    spot_text = " ".join(important_spots) if important_spots else "热门景点"

    # query_list：省额度模式下的合并查询列表；默认只会执行第一条。
    query_list = [
        f"{destination} {spot_text} 旅游 景点 门票 预约 开放时间 最新 交通 政策",
        f"{destination} 官方 旅游 景区规则 门票 预约 开放时间 最新政策",
    ]

    return query_list


def get_tavily_api_key() -> str | None:
    """get_tavily_api_key：读取 Tavily API Key，并忽略示例占位值。"""

    # tavily_api_key：从 .env、环境变量或 Streamlit secrets 读取的 Tavily API Key。
    tavily_api_key = get_config_value("TAVILY_API_KEY", "").strip()
    if not tavily_api_key or tavily_api_key.startswith("tvly-your"):
        return None

    return tavily_api_key


def get_deepseek_api_key() -> str | None:
    """get_deepseek_api_key：读取 DeepSeek API Key，并忽略示例占位值。"""

    # deepseek_api_key：从 .env、环境变量或 Streamlit secrets 读取的 DeepSeek API Key。
    deepseek_api_key = get_config_value("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_api_key or deepseek_api_key.startswith("sk-your"):
        return None

    return deepseek_api_key


def get_deepseek_model_name() -> str:
    """get_deepseek_model_name：读取 DeepSeek 模型名，未配置时使用默认模型。"""

    return get_config_value("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def get_tavily_search_depth() -> str:
    """get_tavily_search_depth：读取 Tavily 搜索深度，并强制使用 basic 省额度模式。"""

    # configured_depth：用户配置的搜索深度；为省额度，任何非 basic 配置都会被降级为 basic。
    configured_depth = get_config_value("TAVILY_SEARCH_DEPTH", DEFAULT_TAVILY_SEARCH_DEPTH).strip().lower()
    return DEFAULT_TAVILY_SEARCH_DEPTH if configured_depth != DEFAULT_TAVILY_SEARCH_DEPTH else configured_depth


def get_tavily_max_searches_per_guide() -> int:
    """get_tavily_max_searches_per_guide：读取每份攻略最多搜索次数，并默认限制为 1 次。"""

    # configured_limit：用户配置的每份攻略最大 Tavily 调用次数。
    configured_limit = get_int_config("TAVILY_MAX_SEARCHES_PER_GUIDE", DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE)
    return max(0, min(configured_limit, DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE))


def normalize_tavily_query(destination: str, query: str) -> str:
    """normalize_tavily_query：把目的地和 query 标准化，用于判断相似搜索并命中缓存。"""

    # token_list：从 query 中提取的中英文关键词。
    token_list = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", query.lower())

    # normalized_tokens：去重排序后的关键词，使词序轻微变化时仍可复用缓存。
    normalized_tokens = sorted(set(token_list))

    return f"{destination.strip().lower()}|{'|'.join(normalized_tokens)}"


def build_tavily_cache_key(destination: str, query: str) -> str:
    """build_tavily_cache_key：为目的地和相似 query 生成稳定缓存 key。"""

    # normalized_query：标准化后的 query 文本。
    normalized_query = normalize_tavily_query(destination, query)
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


def load_tavily_cache() -> dict:
    """load_tavily_cache：读取本地 Tavily 缓存文件。"""

    if not TAVILY_CACHE_PATH.exists():
        return {}

    try:
        with TAVILY_CACHE_PATH.open("r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except Exception:
        return {}


def save_tavily_cache(cache_data: dict) -> None:
    """save_tavily_cache：把 Tavily 搜索结果写入本地缓存文件。"""

    with TAVILY_CACHE_PATH.open("w", encoding="utf-8") as cache_file:
        json.dump(cache_data, cache_file, ensure_ascii=False, indent=2)


def get_cached_tavily_results(destination: str, query: str) -> list[dict] | None:
    """get_cached_tavily_results：读取 24 小时内的 Tavily 缓存结果。"""

    # cache_data：本地缓存文件中的全部数据。
    cache_data = load_tavily_cache()

    # cache_key：当前目的地和 query 对应的缓存 key。
    cache_key = build_tavily_cache_key(destination, query)

    # cached_item：缓存中的单条搜索记录。
    cached_item = cache_data.get(cache_key)
    if not cached_item:
        return None

    # cached_at：缓存写入时间戳。
    cached_at = float(cached_item.get("cached_at", 0))
    if time.time() - cached_at > TAVILY_CACHE_TTL_SECONDS:
        return None

    return cached_item.get("results", [])


def set_cached_tavily_results(destination: str, query: str, results: list[dict]) -> None:
    """set_cached_tavily_results：缓存 Tavily 搜索结果，减少重复搜索消耗。"""

    # cache_data：本地缓存文件中的全部数据。
    cache_data = load_tavily_cache()

    # cache_key：当前目的地和 query 对应的缓存 key。
    cache_key = build_tavily_cache_key(destination, query)
    cache_data[cache_key] = {
        "destination": destination,
        "query": query,
        "cached_at": time.time(),
        "cached_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "results": results,
    }
    save_tavily_cache(cache_data)


def is_tavily_limit_error(error: Exception) -> bool:
    """is_tavily_limit_error：判断 Tavily 错误是否属于额度不足或请求受限。"""

    # error_text：错误文本，兼容 SDK 返回的不同异常格式。
    error_text = str(error).lower()
    limit_keywords = ["429", "quota", "rate limit", "ratelimit", "credits", "credit", "insufficient"]
    return any(keyword in error_text for keyword in limit_keywords)


def call_tavily_search(query: str, parsed_request: dict) -> tuple[list[dict], bool]:
    """call_tavily_search：调用 Tavily SDK 搜索，返回结果和是否命中缓存。"""

    # destination：目的地名称，用于缓存 key。
    destination = parsed_request["destination"]

    # cached_results：24 小时内缓存命中的搜索结果。
    cached_results = get_cached_tavily_results(destination, query)
    if cached_results is not None:
        return cached_results, True

    # tavily_api_key：Tavily API Key，从 .env、环境变量或 Streamlit secrets 读取。
    tavily_api_key = get_tavily_api_key()
    if not tavily_api_key or TavilyClient is None:
        return [], False

    # tavily_client：Tavily Python SDK 客户端。
    tavily_client = TavilyClient(api_key=tavily_api_key)

    # response_data：Tavily SDK 搜索返回结果；不启用 answer/raw/images/auto_parameters，控制额度消耗。
    response_data = tavily_client.search(
        query=query,
        search_depth=get_tavily_search_depth(),
        max_results=DEFAULT_SEARCH_MAX_RESULTS,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        auto_parameters=False,
        timeout=12,
    )

    # results：Tavily 搜索结果列表。
    results = response_data.get("results", [])
    set_cached_tavily_results(destination, query, results)
    return results, False


def build_facts_context(parsed_request: dict) -> tuple[str, list[dict], str | None]:
    """build_facts_context：联网搜索并整理 facts_context，供 DeepSeek 生成攻略时引用。"""

    # searched_at：事实校验执行时间。
    searched_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not get_bool_config("USE_TAVILY", True):
        facts_context = f"""
联网事实校验状态：未启用
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：USE_TAVILY=false
更新时间：{searched_at}
页面提示：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
""".strip()
        return facts_context, [], "未启用联网搜索"

    if TavilyClient is None:
        facts_context = f"""
联网事实校验状态：执行失败
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：未安装 tavily-python
更新时间：{searched_at}
页面提示：联网搜索失败，已切换普通模式。
""".strip()
        return facts_context, [], "联网搜索失败，已切换普通模式"

    # tavily_api_key：Tavily API Key，用于判断是否启用搜索。
    tavily_api_key = get_tavily_api_key()
    if not tavily_api_key:
        facts_context = f"""
联网事实校验状态：未配置
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：未配置 TAVILY_API_KEY
更新时间：{searched_at}
页面提示：未配置 Tavily，当前为普通生成模式。
""".strip()
        return facts_context, [], "未配置 Tavily，当前为普通生成模式"

    # max_searches：每份攻略最多 Tavily 调用次数，默认限制为 1 次。
    max_searches = get_tavily_max_searches_per_guide()
    if max_searches <= 0:
        facts_context = f"""
联网事实校验状态：未启用
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：TAVILY_MAX_SEARCHES_PER_GUIDE=0
更新时间：{searched_at}
页面提示：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
""".strip()
        return facts_context, [], "未启用联网搜索"

    # query_list：本次事实校验需要执行的搜索查询。
    query_list = build_fact_search_queries(parsed_request)[:max_searches]

    # source_records：用于展示和传给模型的搜索结果。
    source_records = []

    # used_cache：本次搜索是否命中过本地缓存。
    used_cache = False

    # context_blocks：facts_context 中的文本块。
    context_blocks = [
        "联网事实校验状态：已执行",
        f"更新时间：{searched_at}",
        f"搜索模式：Tavily basic，省额度模式，每份攻略最多 {max_searches} 次搜索。",
        "使用范围：门票、预约规则、开放时间、交通政策等易变化信息只能基于以下搜索结果整理。",
        "注意：根据搜索结果整理，仍需出行前二次确认。",
    ]

    try:
        for query in query_list:
            # results：单个 query 的网页搜索结果。
            results, cache_hit = call_tavily_search(query, parsed_request)
            used_cache = used_cache or cache_hit
            context_blocks.append(f"\n### 搜索查询：{query}")
            context_blocks.append(f"- 结果来源：{'本地 24 小时缓存' if cache_hit else 'Tavily basic 搜索'}")

            if not results:
                context_blocks.append("- 未查到可用结果。")
                continue

            for result in results:
                # title：搜索结果标题。
                title = result.get("title", "未命名来源")

                # url：搜索结果链接。
                url = result.get("url", "")

                # content：搜索结果摘要内容。
                content = result.get("content", "") or result.get("snippet", "")
                content = re.sub(r"\s+", " ", content).strip()

                source_records.append({"query": query, "title": title, "url": url, "content": content})
                context_blocks.append(f"- 标题：{title}\n  链接：{url}\n  摘要：{content[:360]}")
    except Exception as error:
        if is_tavily_limit_error(error):
            facts_context = f"""
联网事实校验状态：额度不足
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
更新时间：{searched_at}
错误：{error}
要求：门票、预约、开放时间、交通政策等易变化信息不可编造；请写“建议出行前二次确认”。
""".strip()
            return facts_context, source_records, "Tavily 额度不足，已切换普通模式"

        facts_context = f"""
联网事实校验状态：执行失败
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
更新时间：{searched_at}
错误：{error}
要求：门票、预约、开放时间、交通政策等易变化信息不可编造；如果 facts_context 没有查到，请写“建议出行前再次核对”。
""".strip()
        return facts_context, source_records, "联网搜索失败，已切换普通模式"

    if not source_records:
        context_blocks.append("\n结论：未查到足够搜索结果。不要编造门票、预约、开放时间，请提示建议出行前再次核对。")

    if used_cache:
        context_blocks[0] = "联网事实校验状态：缓存命中"
        return "\n".join(context_blocks), source_records, "使用缓存搜索结果"

    return "\n".join(context_blocks), source_records, "已启用联网搜索"


def parse_travel_request(user_input: str) -> dict:
    """parse_travel_request：从用户的一句话里提取目的地、天数、预算和偏好。"""

    # cleaned_input：去掉多余空格后的用户输入。
    cleaned_input = user_input.strip()

    # budget_level：识别到的预算档位。
    budget_level = infer_budget_level(cleaned_input)

    # budget_info：识别到的预算金额、货币单位和展示文本。
    budget_info = parse_budget_info(cleaned_input, budget_level)

    # preferences：从用户输入中识别出的偏好列表。
    preferences = extract_preferences(cleaned_input)

    # trip_segments：单目的地或多目的地分段行程。
    trip_segments = extract_trip_segments(cleaned_input, preferences)

    # destination_list：所有识别到的目的地。
    destination_list = [segment["destination"] for segment in trip_segments]

    # destination：用于页面总览展示的目的地文本。
    destination = " + ".join(destination_list)

    # days：总旅行天数，多目的地时为各段天数之和。
    days = sum(segment["days"] for segment in trip_segments)

    # nights：总住宿晚数，按连续旅行粗略估算。
    nights = max(0, days - 1)

    # nights_match：如果用户明确写了总住宿晚数，则尊重用户输入。
    nights_match = re.search(r"([0-9一二两三四五六七八九十]+)\s*晚", cleaned_input)
    if nights_match:
        nights = max(0, parse_chinese_number(nights_match.group(1)))

    # trip_type：单目的地或多目的地。
    trip_type = "multi_destination" if len(trip_segments) > 1 else "single_destination"

    # trip_notes：多目的地或省份默认推断提示。
    trip_notes = [segment["note"] for segment in trip_segments if segment.get("note")]

    # parsed_request：最终返回给页面和大模型的结构化旅行需求。
    parsed_request = {
        "trip_type": trip_type,
        "destination": destination,
        "destinations": destination_list,
        "trip_segments": trip_segments,
        "trip_notes": trip_notes,
        "days": days,
        "nights": nights,
        "budget": budget_info["display"],
        "budget_level": budget_info["level"],
        "style": budget_info["level"].replace("预算", ""),
        "budget_amount": budget_info["amount"],
        "budget_currency": budget_info["currency"],
        "budget_currency_name": budget_info["currency_name"],
        "budget_has_explicit_amount": budget_info["has_explicit_amount"],
        "preferences": preferences,
    }

    # budget_exchange_hint：国外目的地的粗略换算提示。
    parsed_request["budget_exchange_hint"] = build_exchange_hint(parsed_request)

    return parsed_request


def build_ai_prompt(user_input: str, parsed_request: dict, facts_context: str) -> str:
    """build_ai_prompt：把用户输入、解析结果和联网事实上下文整理成给大模型的提示词。"""

    # preferences_text：把偏好列表合并成适合模型阅读的字符串。
    preferences_text = "、".join(parsed_request["preferences"])

    # exchange_hint_text：国外目的地预算粗略换算提示。
    exchange_hint_text = parsed_request.get("budget_exchange_hint") or "无"

    # budget_currency_text：预算货币单位说明。
    budget_currency_text = (
        f"{parsed_request['budget_currency_name']} ({parsed_request['budget_currency']})"
        if parsed_request.get("budget_currency")
        else "未指定具体金额"
    )

    # search_enabled：是否拿到了可用于事实校验的联网或缓存结果。
    search_enabled = "联网事实校验状态：已执行" in facts_context or "联网事实校验状态：缓存命中" in facts_context

    if search_enabled:
        # fact_rules：启用联网搜索时，对易变化信息使用 facts_context 的强约束规则。
        fact_rules = """
16. 门票、预约规则、开放时间、景区政策、交通政策等易变化信息，必须优先依据 facts_context 写。
17. 如果 facts_context 没有明确说明对应信息，不能编造，请写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。
18. 对门票、预约、开放时间这类信息，必须标注“根据搜索结果整理，仍需出行前二次确认”。
19. 必须增加“信息来源与更新时间”区域，列出来源标题、链接和更新时间；如果搜索结果没有覆盖某项信息，也要说明未查到。
20. 避坑提醒要明确、实用。
21. 必须包含以下二级标题，并保持标题文字完全一致：
""".strip()
    else:
        # fact_rules：普通生成模式下不使用联网结果，但提醒用户二次确认易变化信息。
        fact_rules = """
16. 当前未启用联网搜索，请按普通生成模式输出攻略。
17. DeepSeek 不能编造最新门票、预约规则、开放时间、景区政策或交通政策。
18. 对所有可能变化的信息，必须写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。
19. 必须增加“信息来源与更新时间”区域，并写明：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
20. 避坑提醒要明确、实用。
21. 必须包含以下二级标题，并保持标题文字完全一致：
""".strip()

    return f"""
用户原始需求：
{user_input}

联网事实校验 facts_context：
{facts_context}

系统已识别：
- 目的地：{parsed_request["destination"]}
- 目的地分段：{json.dumps(parsed_request.get("trip_segments", []), ensure_ascii=False)}
- 旅行天数：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 预算单位：{budget_currency_text}
- 预算换算提示：{exchange_hint_text}
- 偏好：{preferences_text}

请生成一份中文旅行攻略，要求：
1. 使用 Markdown。
2. 内容具体、可执行，不要泛泛而谈。
3. 每日行程必须严格生成 {parsed_request["days"]} 天，从 Day 1 到 Day {parsed_request["days"]}，不能少生成，也不能只生成 3 天。
4. 每一天必须有不同主题，标题格式必须是“### Day 1：主题名”，例如“### Day 1：西湖经典路线”。
5. 用户明确提到的景点和偏好必须优先安排：{preferences_text}。
6. 如果用户提到的景点不足以填满全部天数，请根据目的地、偏好、预算和 facts_context 补充适合景点；搜索结果不足时请写“不确定，请出行前核对”，不要编造实时事实。
7. 每天必须包含“上午”“中午”“下午”“晚上”四个时间段。
8. 每个时间段必须写成：- 上午：具体地点｜推荐理由｜预计耗时｜交通或预约提醒。
9. 不要反复使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
10. 同一景点、同一餐厅、同一区域不要重复出现，除非用户明确要求。
11. 美食推荐建议写成：- 名称：推荐理由｜人均预算｜适合场景。
12. 预算估算要分交通、住宿、餐饮、门票体验、机动费用，并明确预算单位。
13. 如果用户输入了预算数字但没有货币单位，必须按人民币 CNY 理解，不要按目的地当地货币理解。
14. 如果用户明确写了美元、日元、欧元、韩元或 USD/EUR/JPY/KRW，必须尊重用户输入的货币单位。
15. 如果存在“预算换算提示”，请在预算估算中补充这条提示，并说明汇率仅供参考。
{fact_rules}
## 旅行封面文案
## 详细旅游攻略
## 每日行程
## 美食推荐
## 交通建议
## 预算估算
## 避坑提醒
## 信息来源与更新时间
""".strip()


def call_deepseek_chat(prompt: str, instructions: str) -> tuple[str | None, str | None]:
    """call_deepseek_chat：执行一次 DeepSeek Chat Completions 调用并返回文本。"""

    if OpenAI is None:
        return None, "没有安装 openai 依赖，已使用本地演示攻略。"

    # api_key：DeepSeek API Key，从 .env、环境变量或 Streamlit secrets 读取。
    api_key = get_deepseek_api_key()
    if not api_key:
        return None, "未配置 DEEPSEEK_API_KEY，已使用本地演示攻略。"

    # model_name：当前使用的 DeepSeek 模型名称，可通过 DEEPSEEK_MODEL 修改。
    model_name = get_deepseek_model_name()

    try:
        # client：OpenAI SDK 客户端，通过 base_url 指向 DeepSeek 服务。
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

        # response：DeepSeek Chat Completions API 返回的大模型结果。
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        )

        # response_text：从模型回复中取出的文本。
        response_text = response.choices[0].message.content
        if not response_text:
            return None, "DeepSeek 返回内容为空，已使用本地演示攻略。"

        return response_text.strip(), None
    except Exception as error:
        return None, f"DeepSeek API 调用失败，已使用本地演示攻略。错误：{error}"


def build_structured_json_prompt(user_input: str, parsed_request: dict, facts_context: str) -> str:
    """build_structured_json_prompt：生成结构化旅行 JSON 的 DeepSeek 提示词。"""

    # preferences_json：用户偏好 JSON 文本，确保用户输入优先。
    preferences_json = json.dumps(parsed_request["preferences"], ensure_ascii=False)

    # trip_segments_json：分段目的地 JSON 文本，确保多目的地不被忽略。
    trip_segments_json = json.dumps(parsed_request.get("trip_segments", []), ensure_ascii=False, indent=2)

    # search_enabled：是否有可用于事实校验的 Tavily 搜索结果。
    search_enabled = "联网事实校验状态：已执行" in facts_context or "联网事实校验状态：缓存命中" in facts_context

    # fact_rule_text：联网事实约束文本。
    fact_rule_text = (
        "门票、预约、开放时间、景区政策、交通政策必须优先依据 facts_context；如果 facts_context 没有明确说明，不要编造，写“建议出行前二次确认”。"
        if search_enabled
        else "当前未启用联网搜索，不能编造最新门票、预约、开放时间、景区政策或交通政策；必须写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。"
    )

    return f"""
用户原始需求：
{user_input}

联网事实校验 facts_context：
{facts_context}

系统已识别参数，必须严格使用，不能被模型猜测或默认值覆盖：
- destination: {parsed_request["destination"]}
- trip_type: {parsed_request.get("trip_type", "single_destination")}
- destinations: {json.dumps(parsed_request.get("destinations", [parsed_request["destination"]]), ensure_ascii=False)}
- trip_segments: {trip_segments_json}
- days: {parsed_request["days"]}
- nights: {parsed_request["nights"]}
- budget_amount: {parsed_request.get("budget_amount")}
- currency: {parsed_request.get("budget_currency") or "CNY"}
- budget_level: {parsed_request.get("style") or parsed_request.get("budget_level")}
- preferences: {preferences_json}

请只返回合法 JSON，不要输出 Markdown，不要解释，不要使用代码块。
JSON 顶层必须包含：
trip_type, destination, destinations, total_days, days, nights, budget, preferences, trip_segments, daily_itinerary, food_recommendations

budget 必须包含：
amount, currency, level

trip_segments 必须和系统识别参数一致。
每个 trip_segments 对象必须包含：
destination, days, nights, days_inferred, theme, note

daily_itinerary 必须 exactly {parsed_request["days"]} 天，从 day 1 到 day {parsed_request["days"]}。
每天必须包含：
day, segment_destination, theme, morning, noon, afternoon, evening

morning/noon/afternoon/evening 每个对象必须包含以下非空字段：
time, place, original_name, reason, duration, transport, booking_note

food_recommendations 必须包含 4-6 家店或小吃点。
每个美食推荐对象必须包含以下非空字段：
name_cn, name_original, location, nearby_spot, reason, budget, scene, booking_note, map_keyword

写作规则：
1. 用户明确提到的景点和偏好必须优先安排：{preferences_json}。
2. 用户明确提到的所有目的地必须出现在 trip_segments 和 daily_itinerary 中，不允许只生成第一个目的地。
3. 如果 days_inferred=true，必须在 note 中说明“用户未说明该目的地天数，系统默认按3天规划”。
4. 如果目的地是省份、国家或大区域，不要泛泛写省名/国家名；必须推荐具体城市路线，并在 note 中说明推断依据。
5. 多目的地 daily_itinerary 的 day 必须全程连续编号，不能每段都从 Day 1 重新开始。
6. 每一天主题必须不同，不能重复。
7. 同一景点、同一餐厅、同一区域不要重复安排。
8. 不要使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
9. place 必须是具体地点，original_name 必须包含中文名和英文/原名；没有英文名时写中文原名。
10. reason、transport、booking_note 必须具体，不能空泛。
11. 美食推荐如果是国外目的地，name_original 必须尽量保留英文名、当地语言原名或常用地图搜索名。
12. 如果无法确认具体地址，不要编造门牌号；location 可以写“市中心区域”“靠近某某景点”“建议以 Google Maps 搜索原名确认”。
13. map_keyword 必须适合复制到 Google Maps / Apple Maps / 百度地图 / 高德地图搜索。
14. {fact_rule_text}
""".strip()


def extract_json_text(model_output: str) -> str:
    """extract_json_text：从模型输出中提取 JSON 文本，兼容代码块和前后解释。"""

    # fenced_match：匹配 ```json 代码块中的 JSON。
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", model_output, flags=re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    # start_index/end_index：提取第一个对象到最后一个对象之间的内容。
    start_index = model_output.find("{")
    end_index = model_output.rfind("}")
    if start_index >= 0 and end_index > start_index:
        return model_output[start_index : end_index + 1].strip()

    return model_output.strip()


def parse_structured_json_output(model_output: str | None) -> tuple[dict | None, list[str]]:
    """parse_structured_json_output：把模型原始输出解析为 JSON 对象。"""

    if not model_output:
        return None, ["模型没有返回 JSON 内容。"]

    # json_text：提取后的 JSON 文本。
    json_text = extract_json_text(model_output)
    try:
        # parsed_json：解析后的 JSON 对象。
        parsed_json = json.loads(json_text)
    except json.JSONDecodeError as error:
        return None, [f"JSON 解析失败：{error}"]

    if not isinstance(parsed_json, dict):
        return None, ["JSON 顶层必须是对象。"]

    return parsed_json, []


def validate_structured_travel_json(travel_json: dict | None, parsed_request: dict) -> list[str]:
    """validate_structured_travel_json：校验每日行程 JSON 是否完整、准确、不重复。"""

    if not isinstance(travel_json, dict):
        return ["结构化结果不是 JSON 对象。"]

    # validation_errors：JSON 校验错误列表。
    validation_errors = []

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    if travel_json.get("destination") != parsed_request["destination"]:
        validation_errors.append(f"destination 不一致，应为 {parsed_request['destination']}。")

    if travel_json.get("days") != expected_days:
        validation_errors.append(f"days 不一致，应为 {expected_days}。")

    if travel_json.get("total_days") is not None and travel_json.get("total_days") != expected_days:
        validation_errors.append(f"total_days 不一致，应为 {expected_days}。")

    if travel_json.get("nights") != parsed_request["nights"]:
        validation_errors.append(f"nights 不一致，应为 {parsed_request['nights']}。")

    # expected_destinations：系统识别出的所有目的地。
    expected_destinations = parsed_request.get("destinations") or [parsed_request["destination"]]

    # destination_json：模型返回的目的地数组。
    destination_json = travel_json.get("destinations")
    if isinstance(destination_json, list):
        missing_destinations = [destination for destination in expected_destinations if destination not in destination_json]
        if missing_destinations:
            validation_errors.append(f"destinations 缺少目的地：{'、'.join(missing_destinations)}。")

    # trip_segments_json：模型返回的分段行程。
    trip_segments_json = travel_json.get("trip_segments")
    if not isinstance(trip_segments_json, list):
        validation_errors.append("trip_segments 必须是数组。")
    else:
        segment_map = {segment.get("destination"): segment for segment in trip_segments_json if isinstance(segment, dict)}
        for expected_segment in parsed_request.get("trip_segments", []):
            destination = expected_segment["destination"]
            segment = segment_map.get(destination)
            if not segment:
                validation_errors.append(f"trip_segments 缺少目的地：{destination}。")
                continue
            if segment.get("days") != expected_segment["days"]:
                validation_errors.append(f"{destination} days 不一致，应为 {expected_segment['days']}。")
            if segment.get("nights") != expected_segment["nights"]:
                validation_errors.append(f"{destination} nights 不一致，应为 {expected_segment['nights']}。")
            if "days_inferred" not in segment:
                validation_errors.append(f"{destination} 缺少 days_inferred 字段。")
            if not str(segment.get("theme", "")).strip():
                validation_errors.append(f"{destination} theme 不能为空。")
            if expected_segment.get("days_inferred") and not str(segment.get("note", "")).strip():
                validation_errors.append(f"{destination} 是默认推断天数，note 不能为空。")

    # budget_json：模型返回的预算对象。
    budget_json = travel_json.get("budget")
    if not isinstance(budget_json, dict):
        validation_errors.append("budget 必须是对象。")
    else:
        if parsed_request.get("budget_has_explicit_amount") and budget_json.get("amount") != parsed_request.get("budget_amount"):
            validation_errors.append(f"budget.amount 不一致，应为 {parsed_request.get('budget_amount')}。")
        if parsed_request.get("budget_currency") and budget_json.get("currency") != parsed_request.get("budget_currency"):
            validation_errors.append(f"budget.currency 不一致，应为 {parsed_request.get('budget_currency')}。")
        if not str(budget_json.get("level", "")).strip():
            validation_errors.append("budget.level 不能为空。")

    # preferences_json：模型返回的偏好列表。
    preferences_json = travel_json.get("preferences")
    if not isinstance(preferences_json, list):
        validation_errors.append("preferences 必须是数组。")
    else:
        missing_preferences = [preference for preference in parsed_request["preferences"] if preference not in preferences_json]
        if missing_preferences:
            validation_errors.append(f"preferences 缺少用户明确偏好：{'、'.join(missing_preferences)}。")

    # daily_itinerary：模型返回的每日行程数组。
    daily_itinerary = travel_json.get("daily_itinerary")
    if not isinstance(daily_itinerary, list):
        return validation_errors + ["daily_itinerary 必须是数组。"]

    if len(daily_itinerary) != expected_days:
        validation_errors.append(f"daily_itinerary 必须 exactly {expected_days} 天，当前为 {len(daily_itinerary)} 天。")

    # required_slots：每天必须包含的四个时间段。
    required_slots = ["morning", "noon", "afternoon", "evening"]

    # required_slot_fields：每个时间段对象必须包含的字段。
    required_slot_fields = ["time", "place", "original_name", "reason", "duration", "transport", "booking_note"]

    # generic_terms：不允许出现的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # seen_days/themes/places/segment_destinations：用于检查编号、主题、地点和目的地覆盖。
    seen_days = set()
    seen_themes = set()
    seen_places = set()
    seen_segment_destinations = set()

    for day_index, day_item in enumerate(daily_itinerary, start=1):
        if not isinstance(day_item, dict):
            validation_errors.append(f"Day {day_index} 必须是对象。")
            continue

        day_number = day_item.get("day")
        if day_number != day_index:
            validation_errors.append(f"Day {day_index} 的 day 字段应为 {day_index}，当前为 {day_number}。")
        if day_number in seen_days:
            validation_errors.append(f"Day 编号重复：{day_number}。")
        seen_days.add(day_number)

        segment_destination = str(day_item.get("segment_destination", "")).strip()
        if not segment_destination:
            validation_errors.append(f"Day {day_index} segment_destination 不能为空。")
        elif segment_destination not in expected_destinations:
            validation_errors.append(f"Day {day_index} segment_destination 不在用户目的地中：{segment_destination}。")
        seen_segment_destinations.add(segment_destination)

        theme = str(day_item.get("theme", "")).strip()
        if not theme:
            validation_errors.append(f"Day {day_index} theme 不能为空。")
        elif theme in seen_themes:
            validation_errors.append(f"Day {day_index} theme 重复：{theme}。")
        seen_themes.add(theme)

        for slot_name in required_slots:
            # slot_data：单个时间段对象。
            slot_data = day_item.get(slot_name)
            if not isinstance(slot_data, dict):
                validation_errors.append(f"Day {day_index} 缺少 {slot_name} 对象。")
                continue

            for field_name in required_slot_fields:
                field_value = slot_data.get(field_name)
                if field_value is None or not str(field_value).strip():
                    validation_errors.append(f"Day {day_index} {slot_name}.{field_name} 不能为空。")

            slot_text = " ".join(str(slot_data.get(field_name, "")) for field_name in required_slot_fields)
            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_index} {slot_name} 包含空泛模板词。")

            place = str(slot_data.get("place", "")).strip()
            if place:
                if place in seen_places:
                    validation_errors.append(f"重复安排地点：{place}。")
                seen_places.add(place)

    missing_days = [day for day in range(1, expected_days + 1) if day not in seen_days]
    if missing_days:
        validation_errors.append(f"daily_itinerary 缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    missing_segment_destinations = [
        destination for destination in expected_destinations if destination not in seen_segment_destinations
    ]
    if missing_segment_destinations:
        validation_errors.append(f"daily_itinerary 未覆盖目的地：{'、'.join(missing_segment_destinations)}。")

    # food_recommendations：模型返回的美食推荐数组。
    food_recommendations = travel_json.get("food_recommendations")
    if not isinstance(food_recommendations, list):
        validation_errors.append("food_recommendations 必须是数组。")
    elif not food_recommendations:
        validation_errors.append("food_recommendations 不能为空。")
    else:
        # required_food_fields：每个美食推荐对象必须包含的字段。
        required_food_fields = [
            "name_cn",
            "name_original",
            "location",
            "nearby_spot",
            "reason",
            "budget",
            "scene",
            "booking_note",
            "map_keyword",
        ]

        # seen_food_names：用于检查店铺名称是否重复。
        seen_food_names = set()
        for food_index, food_item in enumerate(food_recommendations, start=1):
            if not isinstance(food_item, dict):
                validation_errors.append(f"food_recommendations 第 {food_index} 项必须是对象。")
                continue

            for field_name in required_food_fields:
                field_value = food_item.get(field_name)
                if field_value is None or not str(field_value).strip():
                    validation_errors.append(f"food_recommendations 第 {food_index} 项 {field_name} 不能为空。")

            food_name_key = f"{food_item.get('name_cn', '')}|{food_item.get('name_original', '')}".strip()
            if food_name_key in seen_food_names:
                validation_errors.append(f"重复推荐店铺：{food_item.get('name_cn', '')}。")
            seen_food_names.add(food_name_key)

    return validation_errors


def normalize_structured_travel_json(travel_json: dict, parsed_request: dict) -> dict:
    """normalize_structured_travel_json：用系统识别参数覆盖 JSON 顶层关键字段，保证用户输入优先。"""

    # normalized_json：复制后的结构化结果。
    normalized_json = dict(travel_json)
    normalized_json["trip_type"] = parsed_request.get("trip_type", "single_destination")
    normalized_json["destination"] = parsed_request["destination"]
    normalized_json["destinations"] = parsed_request.get("destinations", [parsed_request["destination"]])
    normalized_json["total_days"] = parsed_request["days"]
    normalized_json["days"] = parsed_request["days"]
    normalized_json["nights"] = parsed_request["nights"]
    normalized_json["preferences"] = parsed_request["preferences"]
    normalized_json["trip_segments"] = [
        {
            "destination": segment["destination"],
            "days": segment["days"],
            "nights": segment["nights"],
            "days_inferred": segment.get("days_inferred", False),
            "theme": segment.get("theme") or get_province_route_note(segment["destination"]) or f"{segment['destination']}分段行程",
            "note": segment.get("note", ""),
        }
        for segment in parsed_request.get("trip_segments", [])
    ]
    normalized_json["budget"] = {
        "amount": parsed_request.get("budget_amount"),
        "currency": parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY,
        "level": parsed_request.get("style") or parsed_request.get("budget_level"),
    }
    return normalized_json


def build_json_repair_prompt(
    original_output: str,
    validation_errors: list[str],
    parsed_request: dict,
    facts_context: str,
) -> str:
    """build_json_repair_prompt：根据校验错误要求 DeepSeek 只修复 JSON。"""

    # error_text：校验错误说明。
    error_text = "\n".join(f"- {error}" for error in validation_errors)

    return f"""
你刚才返回的每日行程 JSON 没有通过校验，错误如下：
{error_text}

请基于原始内容修复 JSON。
必须返回合法 JSON。
必须包含 exactly {parsed_request["days"]} 天。
必须包含 trip_type/destination/destinations/total_days/trip_segments/daily_itinerary。
trip_segments 必须等于系统识别分段：
{json.dumps(parsed_request.get("trip_segments", []), ensure_ascii=False, indent=2)}
daily_itinerary 每天必须包含 segment_destination，且必须覆盖所有目的地。
每天必须包含 morning/noon/afternoon/evening。
每个时间段必须包含 time/place/original_name/reason/duration/transport/booking_note。
food_recommendations 必须包含 4-6 项。
每个美食推荐必须包含 name_cn/name_original/location/nearby_spot/reason/budget/scene/booking_note/map_keyword。
必须保留系统识别参数：
- destination: {parsed_request["destination"]}
- destinations: {json.dumps(parsed_request.get("destinations", [parsed_request["destination"]]), ensure_ascii=False)}
- total_days: {parsed_request["days"]}
- days: {parsed_request["days"]}
- nights: {parsed_request["nights"]}
- budget_amount: {parsed_request.get("budget_amount")}
- currency: {parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY}
- budget_level: {parsed_request.get("style") or parsed_request.get("budget_level")}
- preferences: {json.dumps(parsed_request["preferences"], ensure_ascii=False)}

联网事实校验 facts_context：
{facts_context}

原始输出：
{original_output}

不要输出 Markdown，不要解释，只返回 JSON。
""".strip()


def call_deepseek_structured_json_api(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
) -> tuple[dict | None, str | None, list[str], str | None]:
    """call_deepseek_structured_json_api：生成并校验结构化旅行 JSON，失败时自动修复一次。"""

    # instructions：要求模型只返回 JSON 的系统提示。
    instructions = """
你是旅行规划结构化数据生成器。只返回合法 JSON，不输出 Markdown，不解释。
所有用户明确输入的目的地、天数、预算、货币和偏好优先级最高。
不要编造门票、预约、开放时间、景区政策等实时信息。
""".strip()

    # json_prompt：第一次生成结构化 JSON 的提示词。
    json_prompt = build_structured_json_prompt(user_input, parsed_request, facts_context)

    # raw_output：第一次模型原始输出。
    raw_output, api_message = call_deepseek_chat(json_prompt, instructions)
    if not raw_output:
        return None, raw_output, [api_message or "DeepSeek 没有返回结构化 JSON。"], api_message

    # travel_json：第一次解析出的 JSON。
    travel_json, parse_errors = parse_structured_json_output(raw_output)
    validation_errors = parse_errors or validate_structured_travel_json(travel_json, parsed_request)
    if not validation_errors and travel_json:
        return normalize_structured_travel_json(travel_json, parsed_request), raw_output, [], None

    # repair_prompt：JSON 校验失败后的修复提示。
    repair_prompt = build_json_repair_prompt(raw_output, validation_errors, parsed_request, facts_context)

    # repaired_output：修复后的模型原始输出。
    repaired_output, repair_message = call_deepseek_chat(repair_prompt, instructions)
    if not repaired_output:
        return None, raw_output, validation_errors + [repair_message or "DeepSeek JSON 修复没有返回内容。"], repair_message

    # repaired_json：修复后解析出的 JSON。
    repaired_json, repair_parse_errors = parse_structured_json_output(repaired_output)
    repair_errors = repair_parse_errors or validate_structured_travel_json(repaired_json, parsed_request)
    if repair_errors:
        return None, repaired_output, repair_errors, "每日行程 JSON 修复后仍未通过校验。"

    return normalize_structured_travel_json(repaired_json, parsed_request), repaired_output, [], "每日行程 JSON 第一次未通过校验，已自动修复。"


def build_markdown_from_json_prompt(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
    travel_json: dict,
) -> str:
    """build_markdown_from_json_prompt：基于结构化 JSON 生成 Markdown 攻略提示词。"""

    # structured_json_text：结构化旅行 JSON 文本。
    structured_json_text = json.dumps(travel_json, ensure_ascii=False, indent=2)

    return f"""
用户原始需求：
{user_input}

系统识别参数：
- 目的地：{parsed_request["destination"]}
- 旅行天数：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 偏好：{"、".join(parsed_request["preferences"])}

联网事实校验 facts_context：
{facts_context}

结构化 JSON：
{structured_json_text}

请基于上面的结构化 JSON 生成中文 Markdown 攻略。
要求：
1. 不要改变 JSON 中的目的地、天数、预算、偏好和 daily_itinerary。
2. 用户明确提到的所有目的地都必须出现在攻略中，不能只写第一个目的地。
3. 如果是多目的地，先写总览，例如“南京 3 天 + 江西默认 3 天”，再按分段展示。
4. 每日行程必须按 JSON 中的 daily_itinerary 写，不要自由新增重复路线；Day 编号必须连续。
5. 如果 trip_segments 中 days_inferred=true，必须明确说明该目的地天数是系统默认推断。
6. 如果目的地是省份、国家或大区域，必须说明系统选择了具体城市路线。
7. 门票、预约、开放时间、景区政策必须优先依据 facts_context；没有明确搜索结果时写“建议出行前二次确认”。
8. 内容具体、可执行，保留中文名 + 英文/原名。
9. “美食推荐”必须基于 JSON 中的 food_recommendations，每条必须写店名中文名、英文/当地原名、位置、附近景点/区域、人均预算、适合场景、预约提示和地图搜索关键词。
10. 如果无法确认具体地址，不要编造门牌号；写“市中心区域”“靠近某某景点”或“建议以 Google Maps 搜索原名确认”。
11. 必须包含以下二级标题，并保持标题文字完全一致：
## 旅行封面文案
## 详细旅游攻略
## 每日行程
## 美食推荐
## 交通建议
## 预算估算
## 避坑提醒
## 信息来源与更新时间
""".strip()


def build_markdown_from_structured_json(travel_json: dict, parsed_request: dict, facts_context: str) -> str:
    """build_markdown_from_structured_json：当 Markdown 二次生成失败时，用合格 JSON 生成可复制攻略。"""

    # source_updated_at：攻略信息更新时间。
    source_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # itinerary_lines：从结构化 JSON 生成的每日行程 Markdown。
    itinerary_lines = []
    for day_item in travel_json.get("daily_itinerary", []):
        segment_prefix = f"{day_item.get('segment_destination', parsed_request['destination'])}｜"
        itinerary_lines.append(f"### Day {day_item['day']}：{segment_prefix}{day_item['theme']}")
        for slot_key, slot_label in [("morning", "上午"), ("noon", "中午"), ("afternoon", "下午"), ("evening", "晚上")]:
            slot = day_item[slot_key]
            itinerary_lines.append(
                f"- {slot_label}：{slot['place']}｜{slot['reason']}｜{slot['duration']}｜{slot['transport']}；{slot['booking_note']}"
            )

    # food_lines：从结构化 JSON 生成的美食推荐 Markdown。
    food_lines = []
    for food_item in travel_json.get("food_recommendations", []):
        food_title = food_item["name_cn"]
        if food_item.get("name_original") and food_item["name_original"] != food_title:
            food_title = f"{food_title}（{food_item['name_original']}）"
        food_lines.append(
            f"- {food_title}：位置 {food_item['location']}，靠近 {food_item['nearby_spot']}｜"
            f"{food_item['reason']}｜{food_item['budget']}｜{food_item['scene']}｜"
            f"{food_item['booking_note']}｜地图搜索：{food_item['map_keyword']}"
        )

    # preferences_text：用户旅行偏好。
    preferences_text = "、".join(parsed_request["preferences"])

    # segment_overview_text：多目的地分段总览。
    segment_overview_text = " + ".join(
        f"{segment['destination']}{'默认' if segment.get('days_inferred') else ''}{segment['days']}天"
        for segment in parsed_request.get("trip_segments", [])
    )

    # segment_note_lines：多目的地或省份推断说明。
    segment_note_lines = "\n".join(f"- {note}" for note in parsed_request.get("trip_notes", []))

    return f"""
## 旅行封面文案
{parsed_request["destination"]} {parsed_request["days"]} 天 {parsed_request["nights"]} 晚旅行计划：围绕 {preferences_text} 安排路线。

## 详细旅游攻略
- 目的地：{parsed_request["destination"]}
- 行程总览：{segment_overview_text or parsed_request["destination"]}
- 行程长度：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 旅行风格：{preferences_text}
- 说明：本 Markdown 根据已通过校验的结构化 JSON 生成。
{segment_note_lines}

## 每日行程
{chr(10).join(itinerary_lines)}

## 美食推荐
{chr(10).join(food_lines) if food_lines else f"- 请结合每日路线选择附近餐厅，热门餐厅建议提前预约或取号｜人均预算按 {parsed_request['budget']} 控制｜适合午餐和晚餐"}

## 交通建议
- 每天优先围绕同一区域规划，减少跨区往返。
- 景区门票、预约和开放时间建议出行前二次确认。

## 预算估算
- 用户预算：{parsed_request["budget"]}
- 交通、住宿、餐饮、门票体验和机动费用建议按实际日期二次核算。

## 避坑提醒
- 不要把热门景点、热门餐厅和远距离交通挤在同一天。
- 对所有可能变化的信息，建议出行前二次确认。

## 信息来源与更新时间
- 更新时间：{source_updated_at}
- 来源说明：结构化 JSON 已通过程序校验；门票、预约、开放时间等仍需出行前二次确认。
- 联网事实状态：{facts_context.splitlines()[0] if facts_context else "未启用联网搜索"}
""".strip()


def extract_markdown_section_text(markdown_text: str, heading: str) -> str:
    """extract_markdown_section_text：从 Markdown 中提取指定二级标题下的正文。"""

    # normalized_markdown：保证开头有换行，便于正则匹配二级标题。
    normalized_markdown = "\n" + markdown_text.strip()

    # section_pattern：匹配指定二级标题到下一个二级标题之间的内容。
    section_pattern = rf"\n##\s+{re.escape(heading)}\s*\n([\s\S]*?)(?=\n##\s+|\Z)"
    match = re.search(section_pattern, normalized_markdown)
    return match.group(1).strip() if match else ""


def parse_itinerary_day_blocks(markdown_text: str) -> list[dict]:
    """parse_itinerary_day_blocks：解析 Markdown 每日行程区中的 Day 块。"""

    # itinerary_text：每日行程区域 Markdown。
    itinerary_text = extract_markdown_section_text(markdown_text, "每日行程")
    if not itinerary_text:
        return []

    # day_matches：匹配 Day 标题。
    day_matches = list(re.finditer(r"(?m)^###\s*Day\s*([0-9一二两三四五六七八九十]+)\s*[：:]?\s*(.*?)\s*$", itinerary_text))

    # day_blocks：解析后的 Day 数据。
    day_blocks = []
    for index, match in enumerate(day_matches):
        start_index = match.end()
        end_index = day_matches[index + 1].start() if index + 1 < len(day_matches) else len(itinerary_text)
        day_number = parse_chinese_number(match.group(1))
        theme = match.group(2).strip() or f"Day {day_number}"
        body = itinerary_text[start_index:end_index].strip()
        day_blocks.append({"day": day_number, "theme": theme, "body": body})

    return day_blocks


def validate_itinerary_markdown(markdown_text: str, parsed_request: dict) -> list[str]:
    """validate_itinerary_markdown：校验模型生成的每日行程是否满足不重复和天数要求。"""

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    # day_blocks：模型输出中的 Day 块。
    day_blocks = parse_itinerary_day_blocks(markdown_text)

    # validation_errors：行程校验错误列表。
    validation_errors = []

    if len(day_blocks) < expected_days:
        validation_errors.append(f"每日行程只生成了 {len(day_blocks)} 天，用户需要 {expected_days} 天。")
    elif len(day_blocks) > expected_days:
        validation_errors.append(f"每日行程生成了 {len(day_blocks)} 天，用户只需要 {expected_days} 天。")

    # day_number_set：实际出现的 Day 编号集合。
    day_number_set = {day_block["day"] for day_block in day_blocks}
    missing_days = [day for day in range(1, expected_days + 1) if day not in day_number_set]
    if missing_days:
        validation_errors.append(f"缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    # required_slots：每天必须包含的四个时间段。
    required_slots = ["上午", "中午", "下午", "晚上"]

    # generic_terms：不允许反复出现的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # signature_set：用于检查整天内容是否重复。
    signature_set = set()

    # theme_set：用于检查每天主题是否重复。
    theme_set = set()

    # seen_places：用于检查具体地点是否重复安排。
    seen_places = set()

    for day_block in day_blocks[:expected_days]:
        # day_theme：单日主题，必须和其他天不同。
        day_theme = clean_markdown_text(day_block["theme"])
        if day_theme in theme_set:
            validation_errors.append(f"Day {day_block['day']} 主题重复：{day_theme}。")
        theme_set.add(day_theme)

        day_signature_parts = []
        for slot_label in required_slots:
            slot_text = extract_slot_text(day_block["body"], slot_label)
            if not slot_text:
                validation_errors.append(f"Day {day_block['day']} 缺少{slot_label}安排。")
                continue

            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_block['day']} {slot_label}使用了空泛模板词：{slot_text[:40]}")

            # slot_parts：时间段内容应拆成地点、推荐理由、预计耗时、交通或预约提醒。
            slot_parts = [part.strip() for part in re.split(r"[｜|]", slot_text) if part.strip()]
            if len(slot_parts) < 4:
                validation_errors.append(
                    f"Day {day_block['day']} {slot_label}格式不完整，需要“具体地点｜推荐理由｜预计耗时｜交通或预约提醒”。"
                )

            # place：时间段内容中竖线前面的具体地点。
            place = slot_parts[0] if slot_parts else ""
            if place:
                if place in seen_places:
                    validation_errors.append(f"重复安排地点：{place}。")
                seen_places.add(place)

            day_signature_parts.append(slot_text)

        # day_signature：单日四段安排合并后的指纹。
        day_signature = "||".join(day_signature_parts)
        if day_signature and day_signature in signature_set:
            validation_errors.append(f"Day {day_block['day']} 与其他日期的行程内容高度重复。")
        signature_set.add(day_signature)

    return validation_errors


def build_itinerary_retry_prompt(base_prompt: str, validation_errors: list[str], parsed_request: dict) -> str:
    """build_itinerary_retry_prompt：根据校验错误生成二次生成提示词。"""

    # error_text：校验错误说明。
    error_text = "\n".join(f"- {error}" for error in validation_errors)

    return f"""
{base_prompt}

上一次输出的每日行程不合格，必须重新生成完整攻略，重点修复：
{error_text}

强制要求：
1. 必须生成 Day 1 到 Day {parsed_request["days"]}，一天都不能少。
2. 每一天主题必须不同。
3. 不要使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
4. 每个时间段必须写具体地点原名、推荐理由、预计耗时、交通或预约提醒。
5. 不要重复同一景点、同一餐厅、同一区域。
""".strip()


def call_deepseek_api(user_input: str, parsed_request: dict, facts_context: str) -> tuple[str | None, str | None]:
    """call_deepseek_api：使用 OpenAI Python SDK 调用 DeepSeek API 生成攻略文本。"""

    if OpenAI is None:
        return None, "没有安装 openai 依赖，已使用本地演示攻略。"

    # api_key：DeepSeek API Key，从 .env、环境变量或 Streamlit secrets 读取。
    api_key = get_deepseek_api_key()
    if not api_key:
        return None, "未配置 DEEPSEEK_API_KEY，已使用本地演示攻略。"

    # model_name：当前使用的 DeepSeek 模型名称，可通过 DEEPSEEK_MODEL 修改。
    model_name = get_deepseek_model_name()

    # client：OpenAI SDK 客户端，通过 base_url 指向 DeepSeek 服务。
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # instructions：给模型的角色和输出风格要求。
    instructions = """
你是一名资深旅行编辑和行程规划师，擅长把用户的一句话需求整理成清晰、真实、好执行的旅行攻略。
请用中文输出，语气像旅行杂志编辑，但结构要像实用攻略工具。
不要编造实时价格或实时营业状态；涉及价格时用区间估算，并提醒以出行前查询为准。
""".strip()

    # prompt：最终发送给模型的完整提示词。
    prompt = build_ai_prompt(user_input, parsed_request, facts_context)

    try:
        def create_markdown(prompt_text: str) -> str | None:
            """create_markdown：执行一次 DeepSeek Chat Completions 调用并返回 Markdown 文本。"""

            # response：DeepSeek Chat Completions API 返回的大模型结果。
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt_text},
                ],
            )

            # markdown_content：从模型回复中取出的 Markdown 攻略文本。
            markdown_content = response.choices[0].message.content
            return markdown_content.strip() if markdown_content else None

        # markdown_text：从模型回复中取出的 Markdown 攻略文本。
        markdown_text = create_markdown(prompt)
        if not markdown_text:
            return None, "DeepSeek 返回内容为空，已使用本地演示攻略。"

        # validation_errors：第一次生成后的每日行程校验问题。
        validation_errors = validate_itinerary_markdown(markdown_text, parsed_request)
        if not validation_errors:
            return markdown_text, None

        # retry_prompt：校验失败时给模型的二次生成提示词。
        retry_prompt = build_itinerary_retry_prompt(prompt, validation_errors, parsed_request)

        # retry_markdown：二次生成的 Markdown 攻略文本。
        retry_markdown = create_markdown(retry_prompt)
        if not retry_markdown:
            error_summary = "；".join(validation_errors[:4])
            return markdown_text, f"DeepSeek 二次生成返回内容为空，已展示第一次结果；每日行程可能不完整：{error_summary}"

        # retry_errors：二次生成后再次校验每日行程。
        retry_errors = validate_itinerary_markdown(retry_markdown, parsed_request)
        if retry_errors:
            error_summary = "；".join(retry_errors[:4])
            return retry_markdown, f"每日行程校验未完全通过：{error_summary}。页面会显示解析问题，请重新生成或调整输入。"

        return retry_markdown, "检测到第一次每日行程不完整，已自动重新生成并通过校验。"
    except Exception as error:
        return None, f"DeepSeek API 调用失败，已使用本地演示攻略。错误：{error}"


def build_demo_markdown(parsed_request: dict, facts_context: str = "") -> str:
    """build_demo_markdown：没有 API Key 或 API 失败时生成本地演示攻略。"""

    # destination：攻略目的地。
    destination = parsed_request["destination"]

    # days：旅行天数。
    days = parsed_request["days"]

    # nights：住宿晚数。
    nights = parsed_request["nights"]

    # budget：预算档位。
    budget = parsed_request["budget"]

    # budget_exchange_hint：国外目的地预算粗略换算提示。
    budget_exchange_hint = parsed_request.get("budget_exchange_hint")

    # preferences_text：用户旅行偏好。
    preferences_text = "、".join(parsed_request["preferences"])

    # source_updated_at：本地演示攻略的信息更新时间。
    source_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # source_note：本地演示攻略的信息来源说明。
    source_note = (
        "未配置搜索 API 或搜索失败，本地演示攻略没有使用实时联网来源；门票、预约、开放时间建议出行前再次核对。"
        if "联网事实校验状态：已执行" not in facts_context and "联网事实校验状态：缓存命中" not in facts_context
        else "已传入联网搜索 facts_context；具体门票、预约和开放时间请以搜索来源及出行前二次确认为准。"
    )

    return f"""
## 旅行封面文案
{destination} {days} 天 {nights} 晚旅行计划：把 {preferences_text} 放进行程主线，用轻松但不松散的节奏完成一次有记忆点的城市探索。

## 详细旅游攻略
- 目的地：{destination}
- 行程长度：{days} 天 {nights} 晚
- 预算：{budget}
- 旅行风格：{preferences_text}
- 规划思路：第一天熟悉城市动线，第二天深入主题体验，最后一天安排轻量活动和购物补漏。
{f"- 换算提示：{budget_exchange_hint}" if budget_exchange_hint else ""}

## 每日行程
本地演示模式不会生成每日行程模板。请配置 DEEPSEEK_API_KEY 后由 DeepSeek 按目的地、天数、偏好和联网事实生成完整不重复行程。

## 美食推荐
- 本地代表料理：优先选择评分稳定、翻台快、位置靠近行程路线的店｜人均 80-180 人民币(CNY)｜适合第一顿正式餐
- 街区小吃：适合放在下午或夜间，不要把所有排队店集中到同一天｜人均 30-80 人民币(CNY)｜适合边逛边吃
- 甜品或咖啡：适合安排在步行较多的下午，作为休息点｜人均 40-100 人民币(CNY)｜适合拍照和休息
- 预约型餐厅：如果是热门目的地，建议提前 3 到 7 天确认｜人均 180-400 人民币(CNY)｜适合纪念日晚餐

## 交通建议
- 城市内优先使用地铁、公交或官方交通卡，减少频繁打车。
- 每天尽量围绕一个区域规划，避免跨城式来回移动。
- 机场或车站到酒店先查官方线路，再对比打车价格。
- 如果有大件行李，最后一天优先选择寄存点或酒店寄存。

## 预算估算
- 用户预算：{budget}
{f"- {budget_exchange_hint}" if budget_exchange_hint else ""}
- 交通：经济预算约 150-300 人民币(CNY)/人，普通预算约 300-600 人民币(CNY)/人，高预算按实际打车和跨城交通增加。
- 住宿：经济预算约 300-600 人民币(CNY)/晚，普通预算约 600-1200 人民币(CNY)/晚，高预算约 1200 人民币(CNY)/晚以上。
- 餐饮：约 150-350 人民币(CNY)/人/天，热门餐厅和预约餐厅另算。
- 门票体验：约 100-500 人民币(CNY)/人，主题展、乐园、演出费用可能更高。
- 机动费用：建议预留总预算的 10%-20%。

## 避坑提醒
- 不要把热门景点、热门餐厅和远距离交通挤在同一天。
- 不要只看社交平台种草，出发前确认营业时间、预约方式和交通路线。
- 夜景点通常受天气影响明显，建议保留备选方案。
- 购物和伴手礼尽量放在后半程，避免一路背负行李。
- 本攻略为第一版演示内容，真实出行前请再次确认价格、营业时间和交通信息。

## 信息来源与更新时间
- 更新时间：{source_updated_at}
- 来源说明：{source_note}
- 门票、预约、开放时间：根据搜索结果整理，仍需出行前二次确认；如果没有联网结果，请勿将本地演示内容视为实时信息。
""".strip()


def generate_travel_markdown(user_input: str, parsed_request: dict, facts_context: str) -> tuple[str, str | None]:
    """generate_travel_markdown：优先用大模型生成攻略，失败时回退到本地演示攻略。"""

    # ai_markdown：大模型生成的 Markdown 文本。
    ai_markdown, api_message = call_deepseek_api(user_input, parsed_request, facts_context)
    if ai_markdown:
        return ai_markdown, api_message

    # demo_markdown：本地演示 Markdown 文本。
    demo_markdown = build_demo_markdown(parsed_request, facts_context)
    return demo_markdown, api_message


def call_deepseek_markdown_from_json_api(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
    travel_json: dict,
) -> tuple[str | None, str | None]:
    """call_deepseek_markdown_from_json_api：基于合格 JSON 生成 Markdown 攻略。"""

    # instructions：给模型的 Markdown 写作角色要求。
    instructions = """
你是一名资深旅行编辑和行程规划师。请基于给定 JSON 写中文 Markdown 攻略。
不能改变 JSON 中的行程天数、地点、预算和偏好；不要编造实时营业状态。
""".strip()

    # markdown_prompt：基于结构化 JSON 生成 Markdown 的提示词。
    markdown_prompt = build_markdown_from_json_prompt(user_input, parsed_request, facts_context, travel_json)
    return call_deepseek_chat(markdown_prompt, instructions)


def generate_travel_content(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
) -> tuple[str, str | None, dict | None, str | None, list[str]]:
    """generate_travel_content：先生成结构化 JSON，再基于 JSON 生成 Markdown 攻略。"""

    # travel_json：用于页面时间线渲染的结构化旅行数据。
    travel_json, json_raw, json_errors, json_message = call_deepseek_structured_json_api(
        user_input,
        parsed_request,
        facts_context,
    )

    if not travel_json:
        # demo_markdown：结构化 JSON 失败时仍保留页面其他区域，不使用假行程补齐。
        demo_markdown = build_demo_markdown(parsed_request, facts_context)
        error_summary = "；".join(json_errors[:4]) if json_errors else "结构化 JSON 未生成。"
        api_message = f"每日行程 JSON 未通过校验：{error_summary}"
        if json_message and json_message not in api_message:
            api_message = f"{api_message}；{json_message}"
        return demo_markdown, api_message, None, json_raw, json_errors

    # markdown_text：基于合格 JSON 生成的 Markdown 攻略。
    markdown_text, markdown_message = call_deepseek_markdown_from_json_api(
        user_input,
        parsed_request,
        facts_context,
        travel_json,
    )

    # api_messages：需要展示给用户的生成状态说明。
    api_messages = []
    if json_message:
        api_messages.append(json_message)

    if markdown_text:
        if markdown_message:
            api_messages.append(markdown_message)
        return markdown_text, "；".join(api_messages) or None, travel_json, json_raw, []

    # Markdown 二次生成失败时，用已通过校验的 JSON 生成可复制攻略，不生成假行程。
    fallback_markdown = build_markdown_from_structured_json(travel_json, parsed_request, facts_context)
    if markdown_message:
        api_messages.append(f"Markdown 生成失败，已根据合格 JSON 生成可复制攻略：{markdown_message}")
    else:
        api_messages.append("Markdown 生成失败，已根据合格 JSON 生成可复制攻略。")

    return fallback_markdown, "；".join(api_messages), travel_json, json_raw, []


def generate_cover_image_url(parsed_request: dict) -> str:
    """generate_cover_image_url：生成封面图地址，后续可替换为图片生成 API。"""

    # destination：封面图上显示的目的地。
    destination = parsed_request["destination"]

    # preferences_text：封面图上显示的旅行偏好。
    preferences_text = " / ".join(parsed_request["preferences"][:4])

    # cover_svg：使用旅行杂志感 SVG 占位图，保证没有图片 API 时也能显示大封面。
    cover_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
      <defs>
        <linearGradient id="sky" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#111827"/>
          <stop offset="28%" stop-color="#26324f"/>
          <stop offset="62%" stop-color="#92400e"/>
          <stop offset="100%" stop-color="#020617"/>
        </linearGradient>
        <linearGradient id="sunset" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#fef3c7" stop-opacity="0.86"/>
          <stop offset="42%" stop-color="#fb923c" stop-opacity="0.38"/>
          <stop offset="100%" stop-color="#020617" stop-opacity="0"/>
        </linearGradient>
        <linearGradient id="water" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.72"/>
          <stop offset="52%" stop-color="#14b8a6" stop-opacity="0.42"/>
          <stop offset="100%" stop-color="#f97316" stop-opacity="0.48"/>
        </linearGradient>
        <filter id="grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch"/>
          <feColorMatrix type="saturate" values="0"/>
          <feComponentTransfer>
            <feFuncA type="table" tableValues="0 0.18"/>
          </feComponentTransfer>
        </filter>
        <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="24" stdDeviation="24" flood-color="#020617" flood-opacity="0.42"/>
        </filter>
        <clipPath id="photoClip">
          <rect x="760" y="115" width="470" height="560" rx="32"/>
        </clipPath>
      </defs>
      <rect width="1600" height="900" fill="url(#sky)"/>
      <rect width="1600" height="900" fill="url(#sunset)" opacity="0.62"/>
      <rect width="1600" height="900" filter="url(#grain)" opacity="0.36"/>
      <path d="M0 565 C165 470 305 515 440 430 C570 350 690 392 820 315 C1010 205 1190 295 1600 185 L1600 900 L0 900 Z" fill="#0f172a" opacity="0.76"/>
      <path d="M0 625 C230 508 365 610 545 515 C710 428 865 560 1020 468 C1188 368 1365 438 1600 315 L1600 900 L0 900 Z" fill="#1e293b" opacity="0.72"/>
      <path d="M0 690 C260 590 430 710 675 625 C910 544 1060 690 1308 568 C1435 506 1510 520 1600 475 L1600 900 L0 900 Z" fill="url(#water)" opacity="0.78"/>
      <path d="M0 742 C210 700 390 785 640 735 C880 688 1030 790 1285 708 C1430 662 1510 675 1600 642 L1600 900 L0 900 Z" fill="#020617" opacity="0.70"/>
      <g filter="url(#softShadow)" opacity="0.95">
        <rect x="760" y="115" width="470" height="560" rx="32" fill="#f8fafc" opacity="0.92"/>
        <g clip-path="url(#photoClip)">
          <rect x="760" y="115" width="470" height="560" fill="#0f172a"/>
          <rect x="760" y="115" width="470" height="560" fill="url(#sky)" opacity="0.52"/>
          <circle cx="1110" cy="230" r="72" fill="#fde68a" opacity="0.86"/>
          <path d="M760 430 C845 360 910 380 975 320 C1055 245 1115 360 1230 275 L1230 675 L760 675 Z" fill="#334155"/>
          <path d="M760 520 C900 455 960 550 1080 488 C1145 455 1188 470 1230 430 L1230 675 L760 675 Z" fill="#0f766e" opacity="0.7"/>
          <path d="M760 575 C860 540 940 615 1055 558 C1120 524 1170 540 1230 512 L1230 675 L760 675 Z" fill="#0ea5e9" opacity="0.55"/>
          <path d="M860 675 L1018 444 L1135 675 Z" fill="#f8fafc" opacity="0.82"/>
          <path d="M924 675 L1018 500 L1080 675 Z" fill="#f59e0b" opacity="0.55"/>
        </g>
      </g>
      <path d="M1320 170 C1390 210 1425 268 1450 350" stroke="#fde68a" stroke-width="3" stroke-dasharray="12 16" fill="none" opacity="0.55"/>
      <path d="M1450 350 l28 -12 l-20 31 z" fill="#fde68a" opacity="0.75"/>
      <g opacity="0.78">
        <rect x="120" y="640" width="420" height="3" fill="#fde68a"/>
        <rect x="120" y="665" width="315" height="3" fill="#f8fafc" opacity="0.52"/>
        <rect x="120" y="690" width="250" height="3" fill="#f8fafc" opacity="0.34"/>
      </g>
      <text x="126" y="145" fill="#fde68a" font-size="32" font-family="Arial, sans-serif" letter-spacing="6">AI TRAVEL MAGAZINE</text>
      <text x="126" y="232" fill="#ffffff" font-size="72" font-family="Arial, sans-serif" font-weight="700">{html.escape(destination)}</text>
      <text x="130" y="292" fill="#e5e7eb" font-size="30" font-family="Arial, sans-serif">{html.escape(preferences_text)}</text>
    </svg>
    """

    return "data:image/svg+xml;charset=utf-8," + quote(cover_svg)


def split_markdown_sections(markdown_text: str) -> dict:
    """split_markdown_sections：把 Markdown 按二级标题拆成多个展示卡片。"""

    # section_map：保存标题和正文的对应关系。
    section_map = {}

    # normalized_markdown：保证文本开头有换行，方便正则切分。
    normalized_markdown = "\n" + markdown_text.strip()

    # matches：匹配所有“## 标题”和标题后正文。
    matches = re.finditer(r"\n##\s+(.+?)\n([\s\S]*?)(?=\n##\s+|\Z)", normalized_markdown)
    for match in matches:
        title = match.group(1).strip()
        content = match.group(2).strip()
        section_map[title] = content

    return section_map


def clean_markdown_text(markdown_text: str) -> str:
    """clean_markdown_text：清理 Markdown 符号，方便放进自定义 HTML 卡片。"""

    # cleaned_text：去掉列表符号、粗体和多余空格后的文本。
    cleaned_text = re.sub(r"^[\-\*\d\.\s]+", "", markdown_text.strip())
    cleaned_text = re.sub(r"[*`#]+", "", cleaned_text)
    return cleaned_text.strip()


def extract_bullet_items(section_text: str, max_items: int = 6) -> list[str]:
    """extract_bullet_items：从 Markdown 段落中提取列表项。"""

    # item_list：从 Markdown 中提取出的列表内容。
    item_list = []
    for line in section_text.splitlines():
        stripped_line = line.strip()
        if re.match(r"^[-*]\s+", stripped_line) or re.match(r"^\d+[.、]\s+", stripped_line):
            item = clean_markdown_text(stripped_line)
            if item:
                item_list.append(item)

    if not item_list and section_text.strip():
        # fallback_lines：当模型没有使用列表时，按非空行兜底提取。
        fallback_lines = [clean_markdown_text(line) for line in section_text.splitlines() if clean_markdown_text(line)]
        item_list = fallback_lines

    return item_list[:max_items]


def estimate_total_cost(parsed_request: dict) -> str:
    """estimate_total_cost：根据天数和预算档位估算不含大交通的人均总花费。"""

    if parsed_request.get("budget_has_explicit_amount"):
        return f"按 {parsed_request['budget']} 控制"

    # budget_level：用户预算档位。
    budget_level = parsed_request.get("budget_level", parsed_request["budget"])

    # days：旅行天数。
    days = parsed_request["days"]

    # nights：住宿晚数。
    nights = parsed_request["nights"]

    if budget_level == "经济预算":
        day_cost, night_cost = 260, 320
    elif budget_level == "高预算":
        day_cost, night_cost = 980, 1600
    else:
        day_cost, night_cost = 480, 720

    # low_cost：较低估算值。
    low_cost = days * day_cost + nights * night_cost

    # high_cost：较高估算值。
    high_cost = int(low_cost * 1.35)

    return f"约 {low_cost:,}-{high_cost:,} 人民币(CNY)/人"


def infer_trip_pace(parsed_request: dict) -> str:
    """infer_trip_pace：根据天数和偏好推断旅行节奏。"""

    # preferences：用户偏好列表。
    preferences = parsed_request["preferences"]

    if parsed_request["days"] >= 6 or any(item in preferences for item in ["自然", "咖啡", "温泉", "海边"]):
        return "松弛慢旅行"
    if parsed_request["days"] <= 3 and any(item in preferences for item in ["购物", "夜景", "动漫"]):
        return "高效城市探索"
    return "舒适均衡节奏"


def infer_audience(parsed_request: dict) -> str:
    """infer_audience：根据偏好推断适合人群。"""

    # preferences：用户偏好列表。
    preferences = parsed_request["preferences"]

    if "亲子" in preferences:
        return "家庭与亲子出行"
    if any(item in preferences for item in ["动漫", "购物", "夜景"]):
        return "城市玩家与潮流爱好者"
    if any(item in preferences for item in ["自然", "徒步", "海边"]):
        return "自然风景和慢旅行人群"
    return "第一次到访和自由行用户"


def build_summary_metrics(parsed_request: dict) -> dict:
    """build_summary_metrics：生成攻略摘要区需要展示的指标。"""

    # preferences_count：偏好数量，用于生成推荐强度。
    preferences_count = len(parsed_request["preferences"])

    # recommendation_score：推荐强度评分。
    recommendation_score = "4.9 / 5" if preferences_count >= 3 else "4.6 / 5"

    return {
        "推荐强度": recommendation_score,
        "旅行节奏": infer_trip_pace(parsed_request),
        "适合人群": infer_audience(parsed_request),
        "预计总花费": estimate_total_cost(parsed_request),
    }


def extract_slot_text(day_text: str, slot_label: str) -> str:
    """extract_slot_text：从某一天行程中提取上午、中午、下午或晚上的内容。"""

    # slot_pattern：匹配指定时间段的 Markdown 行。
    slot_pattern = rf"(?:^|\n)\s*[-*]?\s*(?:\*\*)?{slot_label}(?:\*\*)?\s*[：:]\s*(.+)"
    match = re.search(slot_pattern, day_text)
    if match:
        return clean_markdown_text(match.group(1))
    return ""


def split_place_and_description(slot_text: str) -> tuple[str, str]:
    """split_place_and_description：把时间段内容拆成地点和说明。"""

    # parts：按中文或英文竖线拆开的地点和说明。
    parts = [part.strip() for part in re.split(r"[｜|]", slot_text, maxsplit=1)]
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]

    return "", slot_text


def build_timeline_days(section_map: dict, parsed_request: dict) -> tuple[list[dict], list[str]]:
    """build_timeline_days：把每日行程 Markdown 转成时间线数据。"""

    # itinerary_text：每日行程 Markdown 内容。
    itinerary_text = section_map.get("每日行程", "").strip()
    if not itinerary_text:
        return [], ["未找到“每日行程”区域。"]

    # timeline_markdown：补回二级标题，复用统一的 Day 块解析函数。
    timeline_markdown = f"## 每日行程\n{itinerary_text}"

    # day_blocks：从 Markdown 中解析出的每日行程块。
    day_blocks = parse_itinerary_day_blocks(timeline_markdown)

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    # validation_errors：时间线渲染前的结构化校验问题。
    validation_errors = []

    if len(day_blocks) != expected_days:
        validation_errors.append(f"模型返回 {len(day_blocks)} 天行程，系统识别用户需要 {expected_days} 天。")

    # day_map：按 Day 编号索引每日行程，方便检查缺失天数。
    day_map = {day_block["day"]: day_block for day_block in day_blocks}
    missing_days = [day for day in range(1, expected_days + 1) if day not in day_map]
    if missing_days:
        validation_errors.append(f"缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    # slot_config：时间线四个固定时段。
    slot_config = [
        {"label": "上午", "time": "09:00 - 11:30", "icon": "AM"},
        {"label": "中午", "time": "12:00 - 13:30", "icon": "NO"},
        {"label": "下午", "time": "14:00 - 17:30", "icon": "PM"},
        {"label": "晚上", "time": "18:30 - 21:30", "icon": "EV"},
    ]

    # generic_terms：不允许出现在时间线里的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # seen_places：已出现地点集合，用于避免同一地点重复安排。
    seen_places = set()

    # seen_themes：已出现主题集合，用于避免每天主题重复。
    seen_themes = set()

    # timeline_days：最终时间线数据。
    timeline_days = []
    for day_number in range(1, expected_days + 1):
        day_block = day_map.get(day_number)
        if not day_block:
            continue

        # day_theme：当前 Day 的主题文本。
        day_theme = clean_markdown_text(day_block["theme"])
        if day_theme in seen_themes:
            validation_errors.append(f"Day {day_number} 主题重复：{day_theme}。")
        seen_themes.add(day_theme)

        # slot_list：单日四个时间段的数据。
        slot_list = []
        for slot in slot_config:
            slot_text = extract_slot_text(day_block["body"], slot["label"])
            if not slot_text:
                validation_errors.append(f"Day {day_number} 缺少{slot['label']}安排。")
                continue

            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_number} {slot['label']}仍包含空泛模板词：{slot_text[:40]}")

            # slot_parts：时间段内容必须包含地点、推荐理由、预计耗时、交通或预约提醒。
            slot_parts = [part.strip() for part in re.split(r"[｜|]", slot_text) if part.strip()]
            if len(slot_parts) < 4:
                validation_errors.append(
                    f"Day {day_number} {slot['label']}未完整包含地点、推荐理由、预计耗时、交通或预约提醒。"
                )
                continue

            place, description = split_place_and_description(slot_text)
            if not place:
                validation_errors.append(f"Day {day_number} {slot['label']}未按“具体地点｜推荐理由｜预计耗时｜交通或预约提醒”格式输出。")
                continue

            if place in seen_places:
                validation_errors.append(f"重复安排地点：{place}。")
            seen_places.add(place)

            slot_list.append(
                {
                    "label": slot["label"],
                    "time": slot["time"],
                    "icon": slot["icon"],
                    "place": place,
                    "description": description,
                }
            )

        timeline_days.append({"title": f"Day {day_number}：{day_block['theme']}", "slots": slot_list})

    if validation_errors:
        return [], validation_errors

    return timeline_days, []


def build_timeline_days_from_json(travel_json: dict | None, parsed_request: dict) -> tuple[list[dict], list[str]]:
    """build_timeline_days_from_json：把结构化 JSON 转成时间线卡片数据。"""

    # validation_errors：结构化 JSON 的校验错误。
    validation_errors = validate_structured_travel_json(travel_json, parsed_request)
    if validation_errors:
        return [], validation_errors

    # slot_config：英文 JSON 字段和页面展示标签的对应关系。
    slot_config = [
        {"key": "morning", "label": "上午", "icon": "AM"},
        {"key": "noon", "label": "中午", "icon": "NO"},
        {"key": "afternoon", "label": "下午", "icon": "PM"},
        {"key": "evening", "label": "晚上", "icon": "EV"},
    ]

    # timeline_days：最终时间线数据。
    timeline_days = []
    for day_item in travel_json.get("daily_itinerary", []):
        # segment_destination：多目的地时展示当前日期所属目的地。
        segment_destination = str(day_item.get("segment_destination", "")).strip()

        # title_prefix：多目的地时间线标题前缀。
        title_prefix = f"{segment_destination}｜" if segment_destination else ""

        # slot_list：单日四个时间段的数据。
        slot_list = []
        for slot in slot_config:
            # slot_data：结构化 JSON 中的时间段对象。
            slot_data = day_item[slot["key"]]
            description = (
                f"{slot_data['original_name']}｜{slot_data['reason']}｜{slot_data['duration']}｜"
                f"{slot_data['transport']}；{slot_data['booking_note']}"
            )
            slot_list.append(
                {
                    "label": slot["label"],
                    "time": slot_data["time"],
                    "icon": slot["icon"],
                    "place": slot_data["place"],
                    "description": description,
                    "original_name": slot_data["original_name"],
                    "reason": slot_data["reason"],
                    "duration": slot_data["duration"],
                    "transport": slot_data["transport"],
                    "booking_note": slot_data["booking_note"],
                }
            )

        timeline_days.append({"title": f"Day {day_item['day']}：{title_prefix}{day_item['theme']}", "slots": slot_list})

    return timeline_days, []


def build_food_cards(section_map: dict, parsed_request: dict) -> list[dict]:
    """build_food_cards：把美食推荐 Markdown 转成美食卡片数据。"""

    # food_items：美食推荐列表。
    food_items = extract_bullet_items(section_map.get("美食推荐", ""), max_items=6)

    if not food_items:
        food_items = [
            "本地代表料理：优先选择路线附近的高评分店｜人均 80-180 元｜适合第一顿正式餐",
            "街区小吃：适合放在下午或夜间，边走边吃更轻松｜人均 30-80 元｜适合探索街区",
            "甜品或咖啡：作为下午休息点，也适合拍照｜人均 40-100 元｜适合慢旅行",
        ]

    # budget_level：预算档位，用于美食卡片的人均预算兜底。
    budget_level = parsed_request.get("budget_level", parsed_request["budget"])

    # default_budget：美食卡片的人均预算兜底。
    default_budget = "人均 80-180 人民币(CNY)" if budget_level != "经济预算" else "人均 30-90 人民币(CNY)"

    # food_cards：最终美食卡片数据。
    food_cards = []
    for item in food_items:
        title = item
        detail = "结合行程路线选择，减少排队和跨区移动。"
        if "：" in item:
            title, detail = item.split("：", 1)
        elif ":" in item:
            title, detail = item.split(":", 1)

        # detail_parts：按竖线拆出的理由、预算和场景。
        detail_parts = [part.strip() for part in re.split(r"[｜|]", detail) if part.strip()]
        reason = detail_parts[0] if detail_parts else detail
        budget = detail_parts[1] if len(detail_parts) > 1 else default_budget
        scene = detail_parts[2] if len(detail_parts) > 2 else "适合穿插在当日行程中"

        food_cards.append(
            {
                "title": clean_markdown_text(title)[:34],
                "reason": clean_markdown_text(reason),
                "budget": clean_markdown_text(budget),
                "scene": clean_markdown_text(scene),
                "location": "位置：建议结合当日行程区域确认",
                "nearby_spot": "当日行程附近",
                "booking_note": "热门时段建议提前确认或预约",
                "map_keyword": clean_markdown_text(title)[:34],
            }
        )

    return food_cards


def build_food_cards_from_json(travel_json: dict | None) -> list[dict]:
    """build_food_cards_from_json：把结构化 JSON 中的美食推荐转成卡片数据。"""

    if not isinstance(travel_json, dict):
        return []

    # food_recommendations：结构化 JSON 中的美食推荐列表。
    food_recommendations = travel_json.get("food_recommendations", [])
    if not isinstance(food_recommendations, list):
        return []

    # food_cards：最终美食卡片数据。
    food_cards = []
    for food_item in food_recommendations[:6]:
        if not isinstance(food_item, dict):
            continue

        # name_cn/name_original：店铺中文名与英文/当地原名。
        name_cn = clean_markdown_text(str(food_item.get("name_cn", "")))
        name_original = clean_markdown_text(str(food_item.get("name_original", "")))
        title = f"{name_cn}（{name_original}）" if name_original and name_original != name_cn else name_cn

        food_cards.append(
            {
                "title": title or "待确认美食点",
                "location": clean_markdown_text(str(food_item.get("location", ""))) or "建议以地图搜索原名确认",
                "nearby_spot": clean_markdown_text(str(food_item.get("nearby_spot", ""))) or "适合穿插在当日行程中",
                "reason": clean_markdown_text(str(food_item.get("reason", ""))) or "结合行程路线选择，减少跨区移动。",
                "budget": clean_markdown_text(str(food_item.get("budget", ""))) or "人均预算待确认",
                "scene": clean_markdown_text(str(food_item.get("scene", ""))) or "适合穿插在当日行程中",
                "booking_note": clean_markdown_text(str(food_item.get("booking_note", ""))) or "热门时段建议提前确认或预约",
                "map_keyword": clean_markdown_text(str(food_item.get("map_keyword", ""))) or title,
            }
        )

    return food_cards


def build_advice_cards(section_text: str, fallback_items: list[str], max_items: int = 4) -> list[dict]:
    """build_advice_cards：把交通建议或避坑提醒转成卡片数据。"""

    # advice_items：从 Markdown 中提取出的建议列表。
    advice_items = extract_bullet_items(section_text, max_items=max_items) or fallback_items

    # advice_cards：最终建议卡片数据。
    advice_cards = []
    for index, item in enumerate(advice_items[:max_items], start=1):
        title = f"建议 {index}"
        description = item
        if "：" in item and len(item.split("：", 1)[0]) <= 16:
            title, description = item.split("：", 1)
        elif ":" in item and len(item.split(":", 1)[0]) <= 16:
            title, description = item.split(":", 1)
        else:
            title = clean_markdown_text(item)[:14]

        advice_cards.append({"title": clean_markdown_text(title), "description": clean_markdown_text(description)})

    return advice_cards


def render_hero() -> None:
    """render_hero：渲染页面顶部的产品标题区。"""

    st.markdown(
        """
        <nav class="top-nav">
            <div class="nav-brand"><span class="brand-mark">T</span><span>TripAgent</span></div>
            <div class="nav-links">
                <span>AI旅行规划</span>
                <span>示例</span>
                <span>反馈</span>
            </div>
        </nav>
        <section class="hero product-hero">
            <div class="hero-layout">
                <div>
                    <div class="eyebrow">AI Private Travel Advisor · Magazine Edition</div>
                    <h1 class="hero-title"><span class="hero-title-line">一句话生成</span><span class="hero-title-line">你的专属旅行路线</span></h1>
                    <p>输入目的地、天数和偏好，AI 为你规划每日行程、美食、预算、交通、天气与避坑提醒。</p>
                    <div class="hero-proof">
                        <span>多目的地连续规划</span>
                        <span>天气与出行提醒</span>
                        <span>Markdown 一键带走</span>
                    </div>
                </div>
                <aside class="hero-panel product-preview">
                    <div class="preview-cover">
                        <div class="preview-cover-label">
                            <span>AI Travel Magazine</span>
                            <strong>Nanjing × Jiangxi<br>7 Days Journey</strong>
                        </div>
                    </div>
                    <div class="preview-steps">
                        <div class="preview-step">
                            <b>01</b>
                            <div><span>Route</span><p>自动拆分多目的地，每天主题不同。</p></div>
                        </div>
                        <div class="preview-step">
                            <b>02</b>
                            <div><span>Food & Weather</span><p>美食、天气、预算和避坑提醒一起整理。</p></div>
                        </div>
                        <div class="preview-step">
                            <b>03</b>
                            <div><span>Export</span><p>生成可复制、可下载的 Markdown 攻略。</p></div>
                        </div>
                    </div>
                </aside>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_box() -> tuple[bool, str]:
    """render_input_box：渲染醒目的自然语言输入框。"""

    with st.container(border=True):
        st.markdown('<p class="input-kicker">Start with one sentence</p>', unsafe_allow_html=True)
        st.markdown('<p class="input-title">告诉我你想怎么旅行</p>', unsafe_allow_html=True)
        st.markdown('<p class="sample-title">选择一个示例，或直接输入你的旅行需求</p>', unsafe_allow_html=True)

        # sample_columns：用于横向排列示例标签按钮。
        sample_columns = st.columns(4)
        for index, sample_prompt in enumerate(SAMPLE_PROMPTS):
            if sample_columns[index].button(sample_prompt["label"], key=f"sample_prompt_{index}"):
                st.session_state["travel_request_input"] = sample_prompt["prompt"]

        with st.form("travel_request_form"):
            # user_input：用户输入的一句话旅行需求。
            user_input = st.text_area(
                label="旅行需求",
                label_visibility="collapsed",
                placeholder="例如：我想去南京游玩3天，然后再去江西游玩，喜欢历史文化、美食和夜景，预算8000",
                key="travel_request_input",
            )

            # submitted：用户是否点击了生成按钮。
            submitted = st.form_submit_button("生成专属旅行方案")

        st.markdown('<p class="hint">不用填复杂表单，一句话就够。没写天数默认 3 天 2 晚；预算数字没写单位时默认人民币 CNY。</p>', unsafe_allow_html=True)

    return submitted, user_input


def render_cover(parsed_request: dict, cover_image_url: str) -> None:
    """render_cover：渲染大封面图区域。"""

    # cover_destination_text：封面专用目的地文本，多目的地用“×”营造旅行杂志标题感。
    cover_destination_text = " × ".join(parsed_request.get("destinations", [])) or parsed_request["destination"]

    # safe_destination：转义后的目的地文本，避免 HTML 注入。
    safe_destination = html.escape(cover_destination_text)

    # safe_preferences：转义后的偏好文本。
    safe_preferences = html.escape("、".join(parsed_request["preferences"]))

    # safe_cover_image_url：转义后的封面图片地址。
    safe_cover_image_url = html.escape(cover_image_url, quote=True)

    # badge_items：封面上展示的旅行关键信息标签。
    badge_items = [
        f"{parsed_request['days']} 天 {parsed_request['nights']} 晚",
        parsed_request["budget"],
        *parsed_request["preferences"][:4],
    ]

    # badge_html：封面标签 HTML。
    badge_html = "".join(f'<span class="cover-badge">{html.escape(item)}</span>' for item in badge_items)

    st.markdown(
        f"""
        <section class="cover-card" style='background-image: url("{safe_cover_image_url}");'>
            <div class="cover-content">
                <div class="label">AI TRAVEL MAGAZINE</div>
                <h2>{safe_destination}</h2>
                <div class="cover-dayline">{parsed_request["days"]} Days Journey · {html.escape(parsed_request["budget"])}</div>
                <p>{safe_preferences} · 由 AI 生成的旅行封面与城市探索计划</p>
                <div class="cover-badges">{badge_html}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_summary_bento(parsed_request: dict) -> None:
    """render_summary_bento：用 bento grid 展示攻略摘要信息。"""

    # preferences_text：用于展示的偏好文本。
    preferences_text = "、".join(parsed_request["preferences"])

    # metric_map：推荐强度、旅行节奏、适合人群和预计总花费。
    metric_map = build_summary_metrics(parsed_request)

    # budget_note：预算卡片中的说明文本。
    budget_note = parsed_request.get("budget_exchange_hint") or "价格为区间估算，出发前需再次确认。"

    st.markdown(
        f"""
        <h2 class="section-heading">攻略摘要</h2>
        <p class="section-subtitle">系统从你的自然语言输入中提取旅行关键参数，并补充可执行的规划指标。</p>
        <div class="bento-grid">
            <div class="bento-card large warm"><span>目的地</span><strong>{html.escape(parsed_request["destination"])}</strong><p>本次攻略围绕城市动线、主题偏好和轻量避坑提醒展开。</p></div>
            <div class="bento-card"><span>旅行天数</span><strong>{parsed_request["days"]} 天 {parsed_request["nights"]} 晚</strong><p>按每日四段式节奏规划。</p></div>
            <div class="bento-card"><span>预算</span><strong>{html.escape(parsed_request["budget"])}</strong><p>{html.escape(budget_note)}</p></div>
            <div class="bento-card large"><span>偏好标签</span><strong>{html.escape(preferences_text)}</strong><p>用于安排主题街区、美食和拍照点。</p></div>
            <div class="bento-card"><span>推荐强度</span><strong>{html.escape(metric_map["推荐强度"])}</strong><p>基于偏好匹配度估算。</p></div>
            <div class="bento-card"><span>旅行节奏</span><strong>{html.escape(metric_map["旅行节奏"])}</strong><p>兼顾体验密度和休息时间。</p></div>
            <div class="bento-card"><span>适合人群</span><strong>{html.escape(metric_map["适合人群"])}</strong><p>可按同行人群继续微调。</p></div>
            <div class="bento-card warm"><span>预计总花费</span><strong>{html.escape(metric_map["预计总花费"])}</strong><p>不含跨城机票或长途交通。</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trip_segments_overview(parsed_request: dict) -> None:
    """render_trip_segments_overview：展示多目的地或推断天数的分段总览。"""

    # trip_segments：系统识别出的目的地分段。
    trip_segments = parsed_request.get("trip_segments", [])
    if not trip_segments:
        return

    # should_show_segments：多目的地或存在默认推断说明时展示分段总览。
    should_show_segments = parsed_request.get("trip_type") == "multi_destination" or bool(parsed_request.get("trip_notes"))
    if not should_show_segments:
        return

    # segment_cards：每个目的地分段的卡片 HTML。
    segment_cards = []
    current_day_start = 1
    for segment in trip_segments:
        # destination：分段目的地。
        destination = str(segment.get("destination", "")).strip()

        # segment_days：分段天数。
        segment_days = int(segment.get("days", DEFAULT_TRAVEL_DAYS))

        # day_range：页面展示的连续 Day 范围。
        day_range = f"Day {current_day_start}-Day {current_day_start + segment_days - 1}"
        current_day_start += segment_days

        # inferred_label：未写天数时明确标注默认推断。
        inferred_label = "默认 " if segment.get("days_inferred") else ""

        # note：分段说明，省份/大区域或默认天数提示。
        note = str(segment.get("note", "")).strip() or "按用户输入的目的地和偏好生成分段路线。"

        segment_cards.append(
            f"""
            <article class="segment-card">
                <span>{html.escape(day_range)}</span>
                <strong>{html.escape(destination)} · {inferred_label}{segment_days} 天 {max(0, segment_days - 1)} 晚</strong>
                <p>{html.escape(note)}</p>
            </article>
            """
        )

    st.markdown(
        f"""
        <h2 class="section-heading">行程分段总览</h2>
        <p class="section-subtitle">多目的地会按连续日期拆分；未说明天数的目的地会明确标注默认规划。</p>
        <div class="segment-overview-grid">{"".join(segment_cards)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_card(section_map: dict) -> None:
    """render_overview_card：展示详细攻略的简要说明卡片。"""

    # overview_text：详细旅游攻略内容。
    overview_text = section_map.get("详细旅游攻略", "")
    if not overview_text:
        return

    with st.container(border=True):
        st.markdown("### 旅行编辑摘要")
        st.markdown(overview_text)


def render_timeline(
    section_map: dict,
    parsed_request: dict,
    travel_json: dict | None = None,
    json_errors: list[str] | None = None,
    json_raw: str | None = None,
) -> None:
    """render_timeline：用时间线样式展示每日行程。"""

    # timeline_days：优先从结构化 JSON 构建每日行程时间线数据。
    if travel_json:
        timeline_days, timeline_errors = build_timeline_days_from_json(travel_json, parsed_request)
    else:
        timeline_days = []
        timeline_errors = json_errors or ["未生成可用于页面渲染的结构化 JSON。"]

    if timeline_errors:
        st.markdown(
            """
            <h2 class="section-heading">每日行程时间线</h2>
            <p class="section-subtitle">每日行程 JSON 没有通过结构化校验，因此没有使用默认模板补齐。</p>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.error("每日行程解析失败或天数不足。请重新生成，系统不会用重复模板自动补齐。")
            for timeline_error in timeline_errors[:8]:
                st.markdown(f"- {timeline_error}")
            if len(timeline_errors) > 8:
                st.markdown(f"- 还有 {len(timeline_errors) - 8} 条校验问题未展示。")
            st.markdown("- 请点击页面上方的“生成专属旅行攻略”重新生成。")
        with st.expander("查看模型返回原文", expanded=False):
            if json_raw:
                st.code(json_raw, language="json")
            else:
                st.markdown("未获取到结构化 JSON 原文。")
        return

    # day_html_list：每天独立卡片 HTML。
    day_html_list = []
    for day in timeline_days:
        slot_html_list = []
        for slot in day["slots"]:
            # original_name/reason/duration/transport/booking_note：结构化 JSON 中的时间段详情。
            original_name = html.escape(slot.get("original_name", ""))
            reason = html.escape(slot.get("reason", slot.get("description", "")))
            duration = html.escape(slot.get("duration", ""))
            transport = html.escape(slot.get("transport", ""))
            booking_note = html.escape(slot.get("booking_note", ""))

            # slot_detail_html：分层展示的时间段信息，避免大段文字堆叠。
            slot_detail_html = (
                f'<div class="slot-original">{original_name}</div>'
                f'<p class="slot-desc">{reason}</p>'
                '<div class="slot-meta-grid">'
                f'<div class="slot-meta-item"><strong>耗时</strong><br>{duration}</div>'
                f'<div class="slot-meta-item"><strong>交通</strong><br>{transport}</div>'
                f'<div class="slot-meta-item"><strong>预约/注意</strong><br>{booking_note}</div>'
                "</div>"
            )
            slot_html_list.append(
                '<div class="timeline-slot">'
                f'<div class="slot-icon">{html.escape(slot["icon"])}</div>'
                "<div>"
                f'<div class="slot-time">{html.escape(slot["label"])} · {html.escape(slot["time"])}</div>'
                f'<div class="slot-place">{html.escape(slot["place"])}</div>'
                f"{slot_detail_html}"
                "</div>"
                "</div>"
            )

        day_html_list.append(
            '<article class="timeline-day">'
            f'<h3>{html.escape(day["title"])}</h3>'
            f'{"".join(slot_html_list)}'
            "</article>"
        )

    # timeline_html：无缩进 HTML，避免 Markdown 把 HTML 识别为代码块。
    timeline_html = (
        '<h2 class="section-heading">每日行程时间线</h2>'
        '<p class="section-subtitle">每天拆成上午、中午、下午和晚上四个时间段，便于实际执行。</p>'
        f'<div class="timeline-grid">{"".join(day_html_list)}</div>'
    )

    st.markdown(timeline_html, unsafe_allow_html=True)


def render_food_cards(section_map: dict, parsed_request: dict, travel_json: dict | None = None) -> None:
    """render_food_cards：用卡片展示美食推荐。"""

    # food_cards：美食卡片数据。
    food_cards = build_food_cards_from_json(travel_json) or build_food_cards(section_map, parsed_request)

    # food_html：美食卡片 HTML。
    food_html = ""
    for food in food_cards:
        food_html += f"""
        <article class="food-card">
            <h3>{html.escape(food["title"])}</h3>
            <div class="food-location">📍 位置：{html.escape(food["location"])}，靠近 {html.escape(food["nearby_spot"])}</div>
            <p>{html.escape(food["reason"])}</p>
            <div class="food-map-keyword">地图搜索：{html.escape(food["map_keyword"])}</div>
            <div class="food-meta">
                <span>{html.escape(food["budget"])}</span>
                <span>{html.escape(food["scene"])}</span>
                <span>{html.escape(food["booking_note"])}</span>
            </div>
        </article>
        """

    st.markdown(
        f"""
        <h2 class="section-heading">美食推荐</h2>
        <p class="section-subtitle">把餐饮当成行程体验的一部分，而不是临时补位。</p>
        <div class="food-grid">{food_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_advice_sections(section_map: dict) -> None:
    """render_advice_sections：展示交通建议和避坑提醒。"""

    # transport_fallback：交通建议兜底内容。
    transport_fallback = [
        "城市内优先使用地铁、公交或官方交通卡，减少频繁打车。",
        "每天尽量围绕一个区域规划，避免跨城式来回移动。",
        "机场或车站到酒店先查官方线路，再对比打车价格。",
        "最后一天优先选择寄存点或酒店寄存，减少拖行李时间。",
    ]

    # warning_fallback：避坑提醒兜底内容。
    warning_fallback = [
        "不要把热门景点、热门餐厅和远距离交通挤在同一天。",
        "出发前确认营业时间、预约方式和交通路线。",
        "夜景点受天气影响明显，建议保留备选方案。",
        "购物和伴手礼尽量放在后半程，避免一路背负行李。",
    ]

    # transport_cards：交通建议卡片数据。
    transport_cards = build_advice_cards(section_map.get("交通建议", ""), transport_fallback, max_items=4)

    # warning_cards：避坑提醒卡片数据。
    warning_cards = build_advice_cards(section_map.get("避坑提醒", ""), warning_fallback, max_items=4)

    # budget_items：预算估算列表。
    budget_items = extract_bullet_items(section_map.get("预算估算", ""), max_items=4)

    transport_html = "".join(
        f"""
        <article class="info-card">
            <div class="card-title-row">
                <div class="info-icon">i</div>
                <h3>{html.escape(card["title"])}</h3>
            </div>
            <p>{html.escape(card["description"])}</p>
        </article>
        """
        for card in transport_cards
    )

    warning_html = "".join(
        f"""
        <article class="warning-card">
            <div class="card-title-row">
                <div class="warning-icon">!</div>
                <h3>{html.escape(card["title"])}</h3>
            </div>
            <p>{html.escape(card["description"])}</p>
        </article>
        """
        for card in warning_cards
    )

    st.markdown(
        f"""
        <h2 class="section-heading">交通建议</h2>
        <p class="section-subtitle">优先减少无效移动，把时间留给真正的体验。</p>
        <div class="info-grid">{transport_html}</div>
        """,
        unsafe_allow_html=True,
    )

    if budget_items:
        # budget_html：预算估算卡片 HTML。
        budget_html = "".join(
            f"""
            <article class="budget-card">
                <span>Budget Note {index}</span>
                <p>{html.escape(budget_item)}</p>
            </article>
            """
            for index, budget_item in enumerate(budget_items, start=1)
        )
        st.markdown(
            f"""
            <h2 class="section-heading">预算估算</h2>
            <p class="section-subtitle">把花费拆成交通、住宿、餐饮和机动预算，方便你出发前调整。</p>
            <div class="budget-grid">{budget_html}</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <h2 class="section-heading">避坑提醒</h2>
        <p class="section-subtitle">提前规避高概率踩坑点，让行程更稳定。</p>
        <div class="warning-grid">{warning_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_source_info(section_map: dict, generated_at: str | None = None) -> None:
    """render_source_info：展示信息来源与更新时间区域。"""

    # source_text：模型生成的来源与更新时间内容，普通用户界面不直接展示技术状态。
    source_text = section_map.get("信息来源与更新时间", "")

    with st.container(border=True):
        st.markdown('<div class="source-card-title">信息与更新时间</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <p class="source-card-text">本攻略由 AI 根据你的输入和当前可用信息整理生成。</p>
            <p class="source-card-text">更新时间：{html.escape(generated_at or datetime.now().strftime('%Y-%m-%d %H:%M'))}</p>
            <p class="source-card-text">门票、预约、开放时间、交通政策和天气情况可能变化，请出行前以官方渠道和天气 App 为准。</p>
            """,
            unsafe_allow_html=True,
        )
        if source_text:
            with st.expander("开发者调试信息", expanded=False):
                st.markdown(source_text)


def render_weather_section(weather_cards: list[dict] | None) -> None:
    """render_weather_section：渲染天气与出行提醒卡片。"""

    st.markdown('<h2 class="section-heading">天气与出行提醒 🌦️</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">根据最近几天的天气情况整理携带建议，出发前请再用天气 App 复核一次。</p>',
        unsafe_allow_html=True,
    )

    if not weather_cards:
        st.markdown(
            """
            <div class="weather-fallback">天气信息暂时无法获取，请出行前查看天气 App。</div>
            """,
            unsafe_allow_html=True,
        )
        return

    # weather_card_html_list：所有目的地天气卡片 HTML。
    weather_card_html_list = []
    for weather_card in weather_cards:
        # destination：天气卡片展示的目的地。
        destination = html.escape(str(weather_card.get("destination", "目的地")))

        if weather_card.get("error"):
            weather_card_html_list.append(
                f"""
                <article class="weather-card">
                    <h3>{destination}｜未来 {WEATHER_FORECAST_DAYS} 天天气</h3>
                    <p class="weather-advice">{html.escape(str(weather_card["error"]))}</p>
                </article>
                """
            )
            continue

        # day_html_list：单个目的地内的每日天气 HTML。
        day_html_list = []
        for day_weather in weather_card.get("days", []):
            # will_rain_text：是否可能下雨的展示文本。
            will_rain_text = "可能下雨" if day_weather.get("will_rain") else "降雨风险较低"

            day_html_list.append(
                f"""
                <div class="weather-day">
                    <div class="weather-icon">{html.escape(str(day_weather.get("weather_icon", "🌦️")))}</div>
                    <div>
                        <div class="weather-date">{html.escape(str(day_weather.get("date", "")))}</div>
                        <div class="weather-main">天气：{html.escape(str(day_weather.get("weather_text", "天气待确认")))}</div>
                        <div class="weather-meta">
                            <span>温度：{html.escape(format_weather_value(day_weather.get("temperature_min"), "°C"))} - {html.escape(format_weather_value(day_weather.get("temperature_max"), "°C"))}</span>
                            <span>湿度：{html.escape(format_weather_value(day_weather.get("humidity"), "%"))}</span>
                            <span>降水概率：{html.escape(format_weather_value(day_weather.get("precipitation_probability"), "%"))}</span>
                            <span>{html.escape(will_rain_text)}</span>
                        </div>
                        <p class="weather-advice">建议：{html.escape(str(day_weather.get("advice", "天气信息仅供参考，请出行前查看天气 App。")))}</p>
                    </div>
                </div>
                """
            )

        weather_card_html_list.append(
            f"""
            <article class="weather-card">
                <h3>{destination}｜未来 {WEATHER_FORECAST_DAYS} 天天气</h3>
                {"".join(day_html_list)}
            </article>
            """
        )

    st.markdown(
        f"""
        <div class="weather-grid">{"".join(weather_card_html_list)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_travel_blessing() -> None:
    """render_travel_blessing：在攻略最后展示温和的旅行祝福语。"""

    st.markdown(
        """
        <div class="blessing-card">
            祝你这次旅行顺利又开心。记得提前确认天气、门票和交通安排，慢慢走、好好看，把喜欢的风景都装进记忆里。祝你旅途愉快呀～ 🌿✨🧳
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_visual_guide(
    markdown_text: str,
    parsed_request: dict,
    travel_json: dict | None = None,
    json_errors: list[str] | None = None,
    json_raw: str | None = None,
    weather_cards: list[dict] | None = None,
    generated_at: str | None = None,
) -> None:
    """render_visual_guide：把 Markdown 攻略渲染成高级卡片式视觉结果。"""

    # section_map：按标题拆分后的攻略内容。
    section_map = split_markdown_sections(markdown_text)

    if not section_map:
        with st.container(border=True):
            st.markdown(markdown_text)
        render_timeline({}, parsed_request, travel_json, json_errors, json_raw)
        render_weather_section(weather_cards)
        render_source_info({}, generated_at)
        render_travel_blessing()
        return

    render_overview_card(section_map)
    render_timeline(section_map, parsed_request, travel_json, json_errors, json_raw)
    render_food_cards(section_map, parsed_request, travel_json)
    render_advice_sections(section_map)
    render_weather_section(weather_cards)
    render_source_info(section_map, generated_at)
    render_travel_blessing()


def render_copy_button(markdown_text: str) -> None:
    """render_copy_button：渲染可复制 Markdown 攻略的按钮。"""

    # markdown_json：安全注入到 JavaScript 的攻略文本。
    markdown_json = json.dumps(markdown_text, ensure_ascii=False)

    st.html(
        f"""
        <style>
        .copy-widget {{
            font-family: Arial, sans-serif;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            padding: 8px 0;
            width: 100%;
        }}
        .copy-widget button {{
            border: 1px solid rgba(246, 199, 111, 0.32);
            border-radius: 999px;
            padding: 12px 18px;
            background: linear-gradient(135deg, #f6c76f, #fb923c);
            color: #17120a;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 12px 28px rgba(251, 146, 60, 0.18);
            width: 100%;
        }}
        .copy-widget span {{
            color: #cbd5e1;
            font-size: 14px;
        }}
        </style>
        <div class="copy-widget">
            <button id="copy-markdown-button">复制攻略</button>
            <span id="copy-markdown-status">一键复制 Markdown 全文。</span>
        </div>
        <script>
        const markdownText = {markdown_json};
        const copyButton = document.getElementById("copy-markdown-button");
        const copyStatus = document.getElementById("copy-markdown-status");
        copyButton.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText(markdownText);
                copyStatus.textContent = "已复制到剪贴板。";
            }} catch (error) {{
                copyStatus.textContent = "复制失败，请手动复制下方原文。";
            }}
        }});
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_result_actions(markdown_text: str) -> None:
    """render_result_actions：在结果底部展示复制、下载和重新生成操作。"""

    with st.container(border=True):
        st.markdown('<div class="result-actions-title">结果操作</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="source-card-text">复制给同行人、下载成 Markdown，或基于当前输入重新生成一版。</p>',
            unsafe_allow_html=True,
        )

        # action_columns：结果操作按钮区域。
        action_columns = st.columns([1.2, 0.8, 0.8])
        with action_columns[0]:
            render_copy_button(markdown_text)
        with action_columns[1]:
            st.download_button(
                label="下载 Markdown",
                data=markdown_text,
                file_name="ai-travel-guide.md",
                mime="text/markdown",
                key="download_markdown_action",
            )
        with action_columns[2]:
            if st.button("重新生成", key="regenerate_result_button"):
                st.session_state["force_regenerate"] = True
                st.rerun()


def render_markdown_source(markdown_text: str) -> None:
    """render_markdown_source：用折叠区域展示 Markdown 原文，并提供复制和下载。"""

    st.markdown('<h2 class="section-heading">Markdown 原文</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">默认收起，适合最后复制到笔记、公众号或行程文档中。</p>', unsafe_allow_html=True)

    with st.expander("查看可复制的 Markdown 攻略", expanded=False):
        # action_columns：复制和下载按钮区域。
        action_columns = st.columns([1, 1])
        with action_columns[0]:
            render_copy_button(markdown_text)
        with action_columns[1]:
            st.download_button(
                label="下载 Markdown 文件",
                data=markdown_text,
                file_name="ai-travel-guide.md",
                mime="text/markdown",
            )

        st.code(markdown_text, language="markdown")


def render_debug_panel(parsed_request: dict) -> None:
    """render_debug_panel：默认折叠展示系统识别出的结构化参数。"""

    # debug_data：开发调试用的结构化解析结果。
    debug_data = {
        "trip_type": parsed_request.get("trip_type"),
        "destination": parsed_request["destination"],
        "destinations": parsed_request.get("destinations", []),
        "trip_segments": parsed_request.get("trip_segments", []),
        "trip_notes": parsed_request.get("trip_notes", []),
        "days": parsed_request["days"],
        "nights": parsed_request["nights"],
        "budget_amount": parsed_request.get("budget_amount"),
        "currency": parsed_request.get("budget_currency") or "未指定",
        "style": parsed_request.get("style") or parsed_request.get("budget_level"),
        "preferences": "、".join(parsed_request["preferences"]),
    }

    with st.expander("Debug：查看系统识别参数", expanded=False):
        st.json(debug_data)


def format_public_search_status(search_message: str | None) -> str:
    """format_public_search_status：把内部搜索状态转换为普通用户可理解的提示。"""

    if not search_message:
        return "实时信息未启用，请出行前二次确认。"

    # raw_message：内部搜索状态文案。
    raw_message = str(search_message)
    if "已启用" in raw_message:
        return "已参考当前可用公开信息。"
    if "缓存" in raw_message:
        return "已参考近期可用信息。"
    if "未启用" in raw_message or "未配置" in raw_message:
        return "实时信息未启用，请出行前二次确认。"
    if "额度" in raw_message or "失败" in raw_message or "受限" in raw_message:
        return "实时信息暂时不可用，请出行前二次确认。"

    return "请出行前再次确认门票、预约、开放时间和交通政策。"


def render_search_status(search_message: str | None) -> None:
    """render_search_status：用小标签展示 Tavily 联网搜索状态。"""

    # safe_message：转义后的状态文案，避免 HTML 注入。
    safe_message = html.escape(format_public_search_status(search_message))
    st.markdown(
        f"""
        <div class="search-status-pill">
            <span class="search-status-dot"></span>
            <span>{safe_message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trust_strip(search_message: str | None, generated_at: str | None) -> None:
    """render_trust_strip：展示轻量信任说明，提醒用户核对实时信息。"""

    # search_status_text：联网搜索状态文案。
    search_status_text = format_public_search_status(search_message)

    # generated_time_text：攻略生成时间。
    generated_time_text = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")

    st.markdown(
        f"""
        <div class="trust-strip">
            <article class="trust-card">
                <span>Search Status</span>
                <strong>{html.escape(search_status_text)}</strong>
                <p>如未联网，实时门票、预约和开放时间请出行前再次核对。</p>
            </article>
            <article class="trust-card">
                <span>AI Generated</span>
                <strong>攻略由 AI 生成，仅供参考</strong>
                <p>路线、预算和餐饮建议适合作为初步规划，不替代官方信息。</p>
            </article>
            <article class="trust-card">
                <span>Updated</span>
                <strong>{html.escape(generated_time_text)}</strong>
                <p>门票、预约、开放时间和交通政策请以官方渠道为准。</p>
            </article>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_generation_count() -> int:
    """get_generation_count：读取当前浏览器 session 已生成攻略次数。"""

    return int(st.session_state.get("generation_count", 0))


def render_generation_quota() -> None:
    """render_generation_quota：展示 Beta 测试版当前 session 剩余生成次数。"""

    # remaining_count：当前 session 剩余生成次数。
    remaining_count = max(0, MAX_GENERATIONS_PER_SESSION - get_generation_count())
    st.markdown(
        f"""
        <p class="generation-quota">Beta 测试额度：本会话剩余 {remaining_count}/{MAX_GENERATIONS_PER_SESSION} 次生成。</p>
        """,
        unsafe_allow_html=True,
    )


def build_result_data(user_input: str) -> dict:
    """build_result_data：根据用户输入生成结果数据，但不把完整用户输入保存到 session。"""

    # generated_at：本次攻略生成时间。
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # parsed_request：自然语言解析结果。
    parsed_request = parse_travel_request(user_input)

    # cover_image_url：封面图地址，第一版是本地 SVG 占位。
    cover_image_url = generate_cover_image_url(parsed_request)

    with st.status("正在理解你的旅行需求...", expanded=True) as loading_status:
        time.sleep(0.2)
        if get_bool_config("USE_TAVILY", True) and get_tavily_api_key():
            loading_status.update(label="正在检索目的地最新信息...", state="running")
        else:
            loading_status.update(label="当前未启用联网搜索，正在切换普通生成模式...", state="running")

        # facts_context：联网搜索整理出的事实校验上下文。
        facts_context, source_records, search_message = build_facts_context(parsed_request)
        loading_status.update(label="正在规划每日路线...", state="running")
        # markdown_text/travel_json：最终展示的 Markdown 和页面时间线使用的结构化 JSON。
        markdown_text, api_message, travel_json, json_raw, json_errors = generate_travel_content(
            user_input,
            parsed_request,
            facts_context,
        )
        loading_status.update(label="正在整理美食与交通建议...", state="running")
        time.sleep(0.2)
        # weather_cards：Open-Meteo 免费天气数据，失败不影响攻略生成。
        try:
            weather_cards = build_weather_cards(parsed_request, travel_json)
        except Exception:
            weather_cards = []
        markdown_text = append_weather_and_blessing_to_markdown(markdown_text, weather_cards, generated_at)
        loading_status.update(label="正在生成专属旅行方案...", state="running")
        time.sleep(0.2)
        loading_status.update(label="专属旅行方案已生成", state="complete", expanded=False)

    return {
        "parsed_request": parsed_request,
        "cover_image_url": cover_image_url,
        "markdown_text": markdown_text,
        "api_message": api_message,
        "travel_json": travel_json,
        "json_raw": json_raw,
        "json_errors": json_errors,
        "weather_cards": weather_cards,
        "search_message": search_message,
        "generated_at": generated_at,
    }


def render_result_data(result_data: dict) -> None:
    """render_result_data：渲染已经生成并缓存在当前 session 中的攻略结果。"""

    # parsed_request：结构化旅行参数，不包含任何 API Key。
    parsed_request = result_data["parsed_request"]

    # markdown_text：最终展示和复制的 Markdown 攻略。
    markdown_text = result_data["markdown_text"]

    render_search_status(result_data.get("search_message"))
    render_trust_strip(result_data.get("search_message"), result_data.get("generated_at"))

    if result_data.get("api_message"):
        st.info(result_data["api_message"])

    render_debug_panel(parsed_request)
    render_cover(parsed_request, result_data["cover_image_url"])
    render_summary_bento(parsed_request)
    render_trip_segments_overview(parsed_request)
    render_visual_guide(
        markdown_text,
        parsed_request,
        result_data.get("travel_json"),
        result_data.get("json_errors"),
        result_data.get("json_raw"),
        result_data.get("weather_cards"),
        result_data.get("generated_at"),
    )
    render_result_actions(markdown_text)
    render_markdown_source(markdown_text)


def render_result(user_input: str) -> None:
    """render_result：兼容旧调用方式，立即生成并渲染完整旅行攻略。"""

    render_result_data(build_result_data(user_input))


def render_beta_notice() -> None:
    """render_beta_notice：在页面底部展示 Beta 测试版和隐私安全提醒。"""

    st.markdown(
        f"""
        <div class="beta-notice">{html.escape(BETA_NOTICE_TEXT)}</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """main：应用入口函数。"""

    setup_page()
    render_hero()

    # submitted：是否点击生成按钮。
    submitted, user_input = render_input_box()
    render_generation_quota()

    # force_regenerate：结果页“重新生成”按钮触发的再生成动作。
    force_regenerate = bool(st.session_state.pop("force_regenerate", False))

    if submitted or force_regenerate:
        if not user_input.strip():
            st.warning("请先输入一句旅行需求。")
        elif get_generation_count() >= MAX_GENERATIONS_PER_SESSION:
            st.warning(
                f"当前 Beta 测试版每个浏览器会话最多生成 {MAX_GENERATIONS_PER_SESSION} 次攻略。"
                "请刷新浏览器会话或稍后再试。"
            )
        else:
            # generation_count：点击生成才增加次数，页面重绘不会重复消耗 API。
            st.session_state["generation_count"] = get_generation_count() + 1

            # last_result_data：只保存生成后的旅行参数和攻略结果，不保存完整用户输入或任何密钥。
            st.session_state["last_result_data"] = build_result_data(user_input.strip())

    if "last_result_data" in st.session_state:
        render_result_data(st.session_state["last_result_data"])
    else:
        st.markdown(
            """
            <p class="hint">示例输入：我想去东京旅游，喜欢动漫、美食和夜景，预算5000，想玩 3 天 2 晚。</p>
            """,
            unsafe_allow_html=True,
        )

    render_beta_notice()


if __name__ == "__main__":
    main()
import html
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


# DEFAULT_TRAVEL_DAYS：用户没有写旅行天数时，默认按 3 天处理。
DEFAULT_TRAVEL_DAYS = 3

# DEFAULT_TRAVEL_NIGHTS：用户没有写住宿晚数时，默认按 2 晚处理。
DEFAULT_TRAVEL_NIGHTS = 2

# DEFAULT_BUDGET_LEVEL：用户没有写预算时，默认使用普通预算。
DEFAULT_BUDGET_LEVEL = "普通预算"

# DEFAULT_BUDGET_CURRENCY：用户输入预算数字但没有写货币单位时，默认按人民币处理。
DEFAULT_BUDGET_CURRENCY = "CNY"

# DEFAULT_DESTINATION：用户没有写明确目的地时，用于演示的默认目的地。
DEFAULT_DESTINATION = "东京"

# DEEPSEEK_BASE_URL：DeepSeek API 的基础地址，OpenAI SDK 会通过这个地址请求 DeepSeek。
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DEFAULT_DEEPSEEK_MODEL：用户没有在 .env 配置模型时，默认使用的 DeepSeek 模型。
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

# DEFAULT_SEARCH_MAX_RESULTS：每个搜索查询最多保留的结果数量。
DEFAULT_SEARCH_MAX_RESULTS = 3

# DEFAULT_TAVILY_SEARCH_DEPTH：Tavily 默认使用 basic 搜索，控制搜索额度消耗。
DEFAULT_TAVILY_SEARCH_DEPTH = "basic"

# DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE：每份攻略默认最多调用 Tavily 的次数。
DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE = 1

# TAVILY_CACHE_FILE：Tavily 搜索结果本地缓存文件，避免 24 小时内重复消耗额度。
TAVILY_CACHE_FILE = "tavily_cache.json"

# TAVILY_CACHE_TTL_SECONDS：Tavily 缓存有效期，默认 24 小时。
TAVILY_CACHE_TTL_SECONDS = 24 * 60 * 60

# TAVILY_CACHE_PATH：Tavily 缓存文件的绝对路径。
TAVILY_CACHE_PATH = Path(__file__).with_name(TAVILY_CACHE_FILE)

# MAX_GENERATIONS_PER_SESSION：Beta 测试版每个浏览器会话最多生成攻略次数，避免 API 被滥用。
MAX_GENERATIONS_PER_SESSION = 3

# OPEN_METEO_GEOCODING_URL：Open-Meteo 免费地理编码接口，不需要 API Key。
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# OPEN_METEO_FORECAST_URL：Open-Meteo 免费天气预报接口，不需要 API Key。
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WEATHER_FORECAST_DAYS：天气模块默认展示最近几天。
WEATHER_FORECAST_DAYS = 3

# BETA_NOTICE_TEXT：上线前页面底部展示的 Beta 和隐私安全提醒。
BETA_NOTICE_TEXT = "当前为 Beta 测试版。AI 生成内容仅供参考，门票、预约、开放时间、交通政策等信息请以官方渠道为准。请勿输入身份证号、手机号、住址、护照号等敏感个人信息。"

# SAMPLE_PROMPTS：Hero 输入区的示例旅行需求，点击后会自动填入对话框。
SAMPLE_PROMPTS = [
    {"label": "东京3天动漫美食游", "prompt": "我想去东京旅游，喜欢动漫、美食和夜景，预算5000，3 天 2 晚"},
    {"label": "杭州7天舒适游", "prompt": "杭州7日游，想去西湖、灵隐寺、龙井村，也想吃杭州美食和看夜景，预算一万，要求舒适一点"},
    {"label": "南京3天 + 江西4天", "prompt": "南京3天，然后去江西4天，喜欢历史文化、美食和夜景，预算8000"},
    {"label": "大阪京都5日自由行", "prompt": "我想去大阪京都自由行，喜欢历史、美食、购物和拍照，普通预算，5 天 4 晚"},
]


if load_dotenv:
    # load_dotenv：读取本地 .env 文件，方便初学者不用每次手动设置环境变量。
    load_dotenv()


def get_config_value(config_name: str, default_value: str = "") -> str:
    """get_config_value：优先从 .env/环境变量读取配置，其次兼容 Streamlit secrets。"""

    # env_value：load_dotenv 后从系统环境变量中读取到的配置值。
    env_value = os.getenv(config_name)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()

    try:
        # secret_value：Streamlit Cloud 部署时可从 st.secrets 读取的配置值。
        secret_value = st.secrets.get(config_name)
        if secret_value is not None and str(secret_value).strip():
            return str(secret_value).strip()
    except Exception:
        return default_value

    return default_value


def get_bool_config(config_name: str, default_value: bool = False) -> bool:
    """get_bool_config：把环境变量或 secrets 中的开关配置转换成布尔值。"""

    # raw_value：配置原始字符串。
    raw_value = get_config_value(config_name, str(default_value)).strip().lower()
    return raw_value in {"1", "true", "yes", "y", "on", "启用", "是"}


def get_int_config(config_name: str, default_value: int) -> int:
    """get_int_config：读取整数配置，非法值自动使用默认值。"""

    # raw_value：配置原始字符串。
    raw_value = get_config_value(config_name, str(default_value)).strip()
    try:
        return int(raw_value)
    except ValueError:
        return default_value


def setup_page() -> None:
    """setup_page：设置 Streamlit 页面基础信息和自定义样式。"""

    st.set_page_config(
        page_title="AI 旅游攻略 Agent",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # custom_css：控制页面视觉风格，让 Streamlit 默认界面更接近高级旅行杂志和 AI 工具。
    custom_css = """
    <style>
    :root {
        --bg-deep: #05070f;
        --panel: rgba(15, 23, 42, 0.58);
        --panel-strong: rgba(15, 23, 42, 0.82);
        --panel-warm: rgba(120, 75, 32, 0.18);
        --line: rgba(255, 255, 255, 0.14);
        --line-soft: rgba(255, 255, 255, 0.08);
        --text-soft: #cbd5e1;
        --text-muted: #94a3b8;
        --cyan: #38bdf8;
        --mint: #34d399;
        --rose: #fb7185;
        --gold: #f6c76f;
        --champagne: #fdecc8;
        --orange: #fb923c;
        --shadow: 0 28px 90px rgba(0, 0, 0, 0.34);
        --glass-blur: blur(20px);
    }

    *,
    *::before,
    *::after {
        box-sizing: border-box;
    }

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        max-width: 100%;
        overflow-x: hidden;
    }

    [data-testid="stMain"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="column"],
    [data-testid="stForm"],
    [data-testid="stTextArea"],
    [data-testid="stMarkdownContainer"] {
        max-width: 100%;
        min-width: 0;
    }

    img,
    iframe,
    table,
    svg {
        max-width: 100%;
    }

    .stApp {
        color: #f8fafc;
        background:
            linear-gradient(118deg, rgba(246, 199, 111, 0.16) 0%, transparent 24%),
            linear-gradient(242deg, rgba(251, 146, 60, 0.12) 0%, transparent 31%),
            linear-gradient(180deg, rgba(255, 255, 255, 0.045), transparent 24%),
            linear-gradient(145deg, #04060d 0%, #0b1020 36%, #111827 68%, #05070f 100%);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.028) 1px, transparent 1px);
        background-size: 72px 72px;
        mask-image: linear-gradient(180deg, rgba(0,0,0,0.72), transparent 72%);
        opacity: 0.38;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    .block-container {
        width: 100%;
        max-width: 1180px;
        padding-top: 1.35rem;
        padding-bottom: 5rem;
    }

    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2.8rem;
        padding: 0.82rem 1.05rem;
        border: 1px solid rgba(246, 199, 111, 0.16);
        border-radius: 999px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02)),
            rgba(8, 12, 24, 0.62);
        box-shadow: 0 16px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.02rem;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: inline-grid;
        place-items: center;
        color: #111827;
        background: linear-gradient(135deg, var(--gold), #fff7d6 48%, var(--orange));
        box-shadow: 0 0 0 1px rgba(255,255,255,0.32), 0 12px 28px rgba(251, 146, 60, 0.24);
    }

    .nav-links {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: var(--text-soft);
        font-size: 0.92rem;
    }

    .nav-links span {
        padding: 0.45rem 0.72rem;
        border-radius: 999px;
        color: #dbeafe;
    }

    .hero {
        position: relative;
        margin-bottom: 1.75rem;
        padding: 0.25rem 0 0.35rem;
    }

    .hero::after {
        content: "";
        display: block;
        width: min(420px, 68vw);
        height: 1px;
        margin-top: 1.5rem;
        background: linear-gradient(90deg, rgba(246, 199, 111, 0.72), transparent);
    }

    .hero-layout {
        display: grid;
        grid-template-columns: minmax(0, 1.16fr) minmax(280px, 0.84fr);
        gap: 1.35rem;
        align-items: stretch;
        max-width: 100%;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 0.8rem;
        border: 1px solid rgba(246, 199, 111, 0.34);
        border-radius: 999px;
        color: #fde68a;
        background: rgba(120, 75, 32, 0.22);
        font-size: 0.84rem;
        margin-bottom: 1.15rem;
    }

    .hero-proof {
        display: flex;
        flex-wrap: wrap;
        gap: 0.58rem;
        margin-top: 1.15rem;
    }

    .hero-proof span {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        padding: 0.38rem 0.68rem;
        color: #e5e7eb;
        background: rgba(15, 23, 42, 0.42);
        font-size: 0.84rem;
        backdrop-filter: blur(12px);
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2.65rem, 5.8vw, 5.85rem);
        line-height: 0.98;
        letter-spacing: 0;
        max-width: 930px;
        color: #fff7ed;
        text-wrap: balance;
    }

    .hero p {
        margin: 1.15rem 0 0;
        max-width: 720px;
        color: #d1d5db;
        font-size: 1.08rem;
        line-height: 1.85;
    }

    .hero-panel {
        min-width: 0;
        min-height: 100%;
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 1.25rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.035)),
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(251, 146, 60, 0.06));
        box-shadow: var(--shadow);
        backdrop-filter: var(--glass-blur);
        position: relative;
        overflow: hidden;
    }

    .hero-panel::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(90deg, transparent 0, rgba(246, 199, 111, 0.08) 1px, transparent 1px),
            linear-gradient(180deg, transparent 0, rgba(255, 255, 255, 0.045) 1px, transparent 1px);
        background-size: 34px 34px;
        opacity: 0.5;
    }

    .mini-card {
        position: relative;
        z-index: 1;
        border: 1px solid var(--line-soft);
        border-radius: 20px;
        padding: 1rem;
        background: rgba(3, 7, 18, 0.42);
        margin-bottom: 0.85rem;
    }

    .mini-card span {
        display: block;
        color: var(--gold);
        font-size: 0.78rem;
        margin-bottom: 0.45rem;
    }

    .mini-card strong {
        display: block;
        font-size: 1.2rem;
        margin-bottom: 0.35rem;
    }

    .mini-card p {
        margin: 0;
        color: var(--text-muted);
        line-height: 1.55;
        font-size: 0.92rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line) !important;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.105), rgba(255, 255, 255, 0.035)),
            rgba(10, 15, 30, 0.58) !important;
        border-radius: 26px !important;
        box-shadow: var(--shadow);
        backdrop-filter: var(--glass-blur);
    }

    [data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #fff7ed;
        letter-spacing: 0;
    }

    .input-title {
        font-size: 1.12rem;
        color: #fef3c7;
        margin: 0 0 0.35rem;
        font-weight: 700;
    }

    .input-kicker {
        color: var(--gold);
        font-size: 0.78rem;
        text-transform: uppercase;
        margin: 0 0 0.32rem;
    }

    .sample-title {
        color: var(--text-muted);
        font-size: 0.88rem;
        margin: 0.85rem 0 0.45rem;
    }

    .stTextArea textarea {
        min-height: 150px !important;
        border-radius: 22px !important;
        border: 1px solid rgba(246, 199, 111, 0.34) !important;
        background:
            linear-gradient(145deg, rgba(2, 6, 23, 0.78), rgba(15, 23, 42, 0.62)) !important;
        color: #f8fafc !important;
        font-size: 1.03rem !important;
        line-height: 1.65 !important;
        box-shadow:
            inset 0 0 0 1px rgba(255, 255, 255, 0.04),
            0 18px 50px rgba(0, 0, 0, 0.16);
    }

    .stTextArea textarea:focus {
        border-color: rgba(246, 199, 111, 0.88) !important;
        box-shadow: 0 0 0 4px rgba(246, 199, 111, 0.14) !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        width: 100%;
        border: 1px solid rgba(246, 199, 111, 0.26);
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(246, 199, 111, 0.95), rgba(251, 146, 60, 0.92));
        color: #17120a;
        font-weight: 800;
        padding: 0.78rem 1rem;
        box-shadow: 0 14px 34px rgba(251, 146, 60, 0.18);
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        color: #17120a;
        filter: brightness(1.05);
        border-color: rgba(255, 247, 237, 0.55);
    }

    .cover-card {
        width: 100%;
        max-width: 100%;
        aspect-ratio: 16 / 9;
        min-height: 420px;
        border-radius: 30px;
        border: 1px solid rgba(246, 199, 111, 0.24);
        background-size: cover;
        background-position: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 34px 110px rgba(0, 0, 0, 0.45);
        margin: 2.15rem 0 1.45rem;
    }

    .cover-card::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: 1;
        pointer-events: none;
        background:
            linear-gradient(90deg, rgba(246, 199, 111, 0.16) 1px, transparent 1px),
            linear-gradient(180deg, rgba(255, 255, 255, 0.07) 1px, transparent 1px),
            linear-gradient(135deg, transparent 0 62%, rgba(246, 199, 111, 0.12) 62% 63%, transparent 63%);
        background-size: 88px 88px, 88px 88px, 100% 100%;
        opacity: 0.42;
    }

    .cover-card::after {
        content: "";
        position: absolute;
        inset: 0;
        z-index: 0;
        background:
            linear-gradient(90deg, rgba(2, 6, 23, 0.82), rgba(2, 6, 23, 0.28) 58%, rgba(2, 6, 23, 0.68)),
            linear-gradient(180deg, rgba(2, 6, 23, 0.02) 20%, rgba(2, 6, 23, 0.88));
    }

    .cover-content {
        position: absolute;
        inset: auto clamp(1.25rem, 4vw, 3.2rem) clamp(1.25rem, 4vw, 3.2rem) clamp(1.25rem, 4vw, 3.2rem);
        z-index: 2;
    }

    .cover-content .label {
        color: #fde68a;
        font-size: 0.88rem;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .cover-content h2 {
        margin: 0;
        font-size: clamp(2.55rem, 6.2vw, 5.4rem);
        line-height: 0.98;
        letter-spacing: 0;
    }

    .cover-dayline {
        color: var(--champagne);
        font-size: clamp(1rem, 2.2vw, 1.45rem);
        font-weight: 800;
        margin-top: 0.62rem;
    }

    .cover-content p {
        margin: 0.9rem 0 0;
        max-width: 760px;
        color: #f8fafc;
        font-size: 1rem;
        line-height: 1.75;
    }

    .cover-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .cover-badge {
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        color: #fff7ed;
        background: rgba(15, 23, 42, 0.46);
        backdrop-filter: blur(10px);
        font-size: 0.86rem;
    }

    .bento-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        grid-auto-rows: minmax(112px, auto);
        gap: 0.9rem;
        margin: 1rem 0 2rem;
    }

    .bento-card {
        min-width: 0;
        max-width: 100%;
        border: 1px solid var(--line-soft);
        border-radius: 24px;
        padding: 1.05rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.52);
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
        min-height: 112px;
    }

    .bento-card.large {
        grid-column: span 2;
    }

    .bento-card.warm {
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.19), rgba(251, 146, 60, 0.065)),
            rgba(15, 23, 42, 0.52);
    }

    .bento-card span {
        display: block;
        color: #fcd34d;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }

    .bento-card strong {
        display: block;
        color: #f8fafc;
        font-size: clamp(1.05rem, 2.2vw, 1.45rem);
        line-height: 1.18;
        letter-spacing: 0;
    }

    .bento-card p {
        color: var(--text-muted);
        line-height: 1.55;
        margin: 0.55rem 0 0;
        font-size: 0.92rem;
    }

    .segment-overview-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.85rem;
        max-width: 100%;
        margin: -0.45rem 0 1.9rem;
    }

    .segment-card {
        width: 100%;
        min-width: 0;
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 22px;
        padding: 1rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(56, 189, 248, 0.05)),
            rgba(15, 23, 42, 0.48);
        box-shadow: 0 16px 50px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
    }

    .segment-card span {
        display: block;
        color: #fcd34d;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }

    .segment-card strong {
        display: block;
        color: #fff7ed;
        font-size: 1.18rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .segment-card p {
        color: var(--text-muted);
        line-height: 1.58;
        margin: 0.55rem 0 0;
        font-size: 0.9rem;
        overflow-wrap: anywhere;
    }

    .section-heading {
        margin: 2.25rem 0 0.35rem;
        color: #fff7ed;
        font-size: clamp(1.55rem, 3vw, 2.1rem);
        letter-spacing: 0;
    }

    .section-subtitle {
        margin: 0 0 1.1rem;
        color: var(--text-muted);
        line-height: 1.65;
    }

    .timeline-grid,
    .food-grid,
    .info-grid,
    .warning-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.6rem;
        max-width: 100%;
    }

    .timeline-day,
    .food-card,
    .info-card,
    .warning-card {
        width: 100%;
        max-width: 100%;
        min-width: 0;
        border: 1px solid var(--line-soft);
        border-radius: 26px;
        padding: 1.15rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.25);
        backdrop-filter: var(--glass-blur);
    }

    .timeline-day h3,
    .food-card h3,
    .info-card h3,
    .warning-card h3 {
        margin: 0 0 0.85rem;
        color: #fff7ed;
        letter-spacing: 0;
        overflow-wrap: anywhere;
    }

    .card-title-row {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin-bottom: 0.82rem;
    }

    .card-title-row h3 {
        margin: 0;
    }

    .info-icon,
    .warning-icon {
        width: 36px;
        height: 36px;
        border-radius: 13px;
        display: grid;
        place-items: center;
        font-weight: 900;
        flex: 0 0 auto;
    }

    .info-icon {
        color: #082f49;
        background: linear-gradient(135deg, #7dd3fc, #38bdf8);
    }

    .warning-icon {
        color: #17120a;
        background: linear-gradient(135deg, #f6c76f, #fb923c);
    }

    .timeline-slot {
        display: grid;
        grid-template-columns: 44px minmax(0, 1fr);
        gap: 0.85rem;
        padding: 0.82rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .timeline-slot:first-of-type {
        border-top: 0;
        padding-top: 0;
    }

    .slot-icon {
        width: 40px;
        height: 40px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        color: #17120a;
        background: linear-gradient(135deg, #f6c76f, #fb923c);
        font-weight: 900;
        box-shadow: 0 12px 26px rgba(251, 146, 60, 0.18);
    }

    .slot-time {
        color: #fcd34d;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .slot-place {
        color: #f8fafc;
        font-weight: 800;
        margin-bottom: 0.25rem;
        overflow-wrap: anywhere;
    }

    .slot-original {
        color: #fde68a;
        font-size: 0.78rem;
        margin-bottom: 0.38rem;
        opacity: 0.92;
        overflow-wrap: anywhere;
    }

    .slot-desc,
    .food-card p,
    .info-card p,
    .warning-card p {
        color: #cbd5e1;
        line-height: 1.62;
        margin: 0;
        font-size: 0.93rem;
        overflow-wrap: anywhere;
    }

    .slot-meta-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.45rem;
        margin-top: 0.62rem;
    }

    .slot-meta-item {
        border: 1px solid rgba(255, 255, 255, 0.075);
        border-radius: 14px;
        padding: 0.48rem 0.56rem;
        color: #cbd5e1;
        background: rgba(2, 6, 23, 0.26);
        font-size: 0.8rem;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    .slot-meta-item strong {
        color: #fcd34d;
        font-weight: 800;
    }

    .food-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.9rem;
    }

    .food-meta span {
        border: 1px solid rgba(246, 199, 111, 0.24);
        border-radius: 999px;
        padding: 0.32rem 0.58rem;
        color: #fde68a;
        background: rgba(120, 75, 32, 0.18);
        font-size: 0.8rem;
    }

    .food-location {
        color: #9ca3af;
        font-size: 0.82rem;
        line-height: 1.55;
        margin: -0.42rem 0 0.76rem;
    }

    .food-map-keyword {
        color: #fcd34d;
        font-size: 0.78rem;
        line-height: 1.45;
        margin-top: 0.65rem;
        opacity: 0.9;
    }

    .info-card {
        background:
            linear-gradient(145deg, rgba(56, 189, 248, 0.13), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
    }

    .warning-card {
        border-color: rgba(251, 146, 60, 0.22);
        background:
            linear-gradient(145deg, rgba(251, 146, 60, 0.18), rgba(127, 29, 29, 0.10)),
            rgba(15, 23, 42, 0.54);
    }

    .weather-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        max-width: 100%;
        margin-bottom: 1.5rem;
    }

    .weather-card {
        width: 100%;
        max-width: 100%;
        min-width: 0;
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 26px;
        padding: 1.1rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(56, 189, 248, 0.055)),
            rgba(15, 23, 42, 0.54);
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.24);
        backdrop-filter: var(--glass-blur);
    }

    .weather-card h3 {
        margin: 0 0 0.8rem;
        color: #fff7ed;
        letter-spacing: 0;
        overflow-wrap: anywhere;
    }

    .weather-day {
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr);
        gap: 0.82rem;
        padding: 0.86rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .weather-day:first-of-type {
        border-top: 0;
        padding-top: 0;
    }

    .weather-icon {
        width: 40px;
        height: 40px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: rgba(246, 199, 111, 0.15);
        border: 1px solid rgba(246, 199, 111, 0.22);
        font-size: 1.2rem;
    }

    .weather-date {
        color: #fcd34d;
        font-weight: 800;
        font-size: 0.88rem;
        margin-bottom: 0.22rem;
    }

    .weather-main {
        color: #f8fafc;
        font-weight: 800;
        margin-bottom: 0.36rem;
        overflow-wrap: anywhere;
    }

    .weather-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        margin: 0.45rem 0;
    }

    .weather-meta span {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 999px;
        padding: 0.28rem 0.48rem;
        color: #fde68a;
        background: rgba(15, 23, 42, 0.42);
        font-size: 0.78rem;
        overflow-wrap: anywhere;
    }

    .weather-advice {
        color: #cbd5e1;
        line-height: 1.6;
        font-size: 0.9rem;
        margin: 0.55rem 0 0;
        overflow-wrap: anywhere;
    }

    .weather-fallback {
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 22px;
        padding: 1rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.12), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
        color: #cbd5e1;
        line-height: 1.65;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.20);
        backdrop-filter: var(--glass-blur);
    }

    .blessing-card {
        margin: 1.2rem 0 1.6rem;
        border: 1px solid rgba(246, 199, 111, 0.18);
        border-radius: 24px;
        padding: 1rem 1.05rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.14), rgba(251, 146, 60, 0.045)),
            rgba(15, 23, 42, 0.56);
        color: #f8fafc;
        line-height: 1.72;
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
    }

    .markdown-actions {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 220px;
        gap: 0.8rem;
        align-items: center;
        margin-bottom: 0.7rem;
    }

    .hint {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.65;
        overflow-wrap: anywhere;
    }

    .search-status-pill {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        max-width: 100%;
        gap: 0.48rem;
        margin: 0.3rem 0 1.1rem;
        padding: 0.5rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(246, 199, 111, 0.22);
        background: rgba(15, 23, 42, 0.48);
        color: #fde68a;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
        backdrop-filter: var(--glass-blur);
        font-size: 0.86rem;
        line-height: 1.4;
    }

    .trust-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.45rem 0 1.15rem;
        max-width: 100%;
    }

    .trust-card {
        min-width: 0;
        border: 1px solid rgba(246, 199, 111, 0.16);
        border-radius: 18px;
        padding: 0.82rem 0.9rem;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.10), rgba(255, 255, 255, 0.025)),
            rgba(15, 23, 42, 0.48);
        box-shadow: 0 14px 42px rgba(0, 0, 0, 0.18);
        backdrop-filter: var(--glass-blur);
    }

    .trust-card span {
        display: block;
        color: #fcd34d;
        font-size: 0.76rem;
        margin-bottom: 0.32rem;
    }

    .trust-card strong {
        display: block;
        color: #fff7ed;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .trust-card p {
        color: var(--text-muted);
        font-size: 0.84rem;
        line-height: 1.5;
        margin: 0.42rem 0 0;
        overflow-wrap: anywhere;
    }

    .search-status-dot {
        width: 0.46rem;
        height: 0.46rem;
        flex: 0 0 auto;
        border-radius: 999px;
        background: var(--gold);
        box-shadow: 0 0 18px rgba(246, 199, 111, 0.52);
    }

    .generation-quota {
        color: #fde68a;
        font-size: 0.86rem;
        margin-top: 0.55rem;
        opacity: 0.92;
    }

    .result-action-panel {
        margin: 2rem 0 1rem;
        padding: 1.05rem;
        border: 1px solid rgba(246, 199, 111, 0.20);
        border-radius: 24px;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.13), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.56);
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.25);
        backdrop-filter: var(--glass-blur);
    }

    .result-action-panel h2 {
        margin: 0 0 0.25rem;
        color: #fff7ed;
        font-size: 1.25rem;
        letter-spacing: 0;
    }

    .result-action-panel p {
        margin: 0;
        color: var(--text-muted);
        line-height: 1.55;
        font-size: 0.9rem;
    }

    .result-action-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.8fr) minmax(0, 0.8fr);
        gap: 0.78rem;
        align-items: center;
        margin-top: 0.9rem;
        max-width: 100%;
    }

    .beta-notice {
        margin-top: 2.2rem;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(246, 199, 111, 0.22);
        border-radius: 20px;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.12), rgba(255, 255, 255, 0.035)),
            rgba(15, 23, 42, 0.58);
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.9rem;
        backdrop-filter: var(--glass-blur);
    }

    div[data-testid="stExpander"] {
        max-width: 100%;
        border: 1px solid var(--line-soft);
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.50);
        backdrop-filter: var(--glass-blur);
    }

    pre,
    code,
    .stCodeBlock {
        max-width: 100%;
        overflow-wrap: anywhere;
    }

    pre {
        white-space: pre-wrap;
    }

    @media (max-width: 768px) {
        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"] {
            max-width: 100% !important;
            overflow-x: hidden !important;
        }

        .block-container {
            max-width: 100% !important;
            padding: 0.85rem 0.78rem 3rem !important;
        }

        .top-nav {
            width: 100%;
            border-radius: 22px;
            align-items: flex-start;
            margin-bottom: 1.45rem;
            padding: 0.72rem 0.78rem;
        }

        .nav-links {
            display: none;
        }

        .hero {
            margin-bottom: 1.2rem;
        }

        .hero-layout,
        .segment-overview-grid,
        .trust-strip,
        .slot-meta-grid,
        .result-action-grid,
        .weather-grid,
        .timeline-grid,
        .food-grid,
        .info-grid,
        .warning-grid,
        .markdown-actions,
        .bento-grid {
            grid-template-columns: minmax(0, 1fr) !important;
            width: 100%;
        }

        .hero h1 {
            font-size: clamp(2rem, 12vw, 3.25rem);
            line-height: 1.04;
        }

        .hero p {
            font-size: 0.96rem;
            line-height: 1.68;
        }

        .hero-panel {
            padding: 0.85rem;
            border-radius: 22px;
        }

        .bento-card.large {
            grid-column: span 1;
        }

        .bento-card,
        .segment-card,
        .trust-card,
        .weather-card,
        .timeline-day,
        .food-card,
        .info-card,
        .warning-card {
            width: 100%;
            padding: 0.92rem;
            border-radius: 20px;
            box-shadow: 0 14px 44px rgba(0, 0, 0, 0.22);
        }

        .cover-card {
            min-height: 240px;
            aspect-ratio: 4 / 3;
            border-radius: 22px;
            margin: 1.25rem 0 1rem;
            background-position: center;
        }

        .cover-content {
            inset: auto 1rem 1rem 1rem;
        }

        .cover-content h2 {
            font-size: clamp(2rem, 12vw, 3.2rem);
        }

        .cover-content p {
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .cover-badge {
            font-size: 0.76rem;
            padding: 0.34rem 0.55rem;
        }

        .section-heading {
            font-size: 1.35rem;
            margin-top: 1.55rem;
        }

        .section-subtitle {
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .timeline-slot {
            grid-template-columns: 36px minmax(0, 1fr);
            gap: 0.68rem;
        }

        .slot-icon {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            font-size: 0.75rem;
        }

        .weather-day {
            grid-template-columns: 36px minmax(0, 1fr);
            gap: 0.68rem;
        }

        .weather-icon {
            width: 34px;
            height: 34px;
            border-radius: 12px;
            font-size: 1rem;
        }

        .food-meta span {
            max-width: 100%;
            white-space: normal;
        }

        .search-status-pill {
            width: 100%;
            align-items: flex-start;
            border-radius: 18px;
        }

        .result-action-panel {
            padding: 0.92rem;
            border-radius: 20px;
        }

        .hero-proof {
            gap: 0.42rem;
        }

        .hero-proof span {
            width: 100%;
            text-align: center;
        }

        [data-testid="column"] {
            width: 100% !important;
            min-width: 0 !important;
            flex: 1 1 100% !important;
        }

        .stTextArea textarea {
            width: 100% !important;
            min-height: 118px !important;
            font-size: 0.95rem !important;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {
            width: 100% !important;
            white-space: normal;
            line-height: 1.35;
        }

        .stApp::before {
            opacity: 0.18;
            background-size: 96px 96px;
        }

        table {
            display: block;
            overflow-x: auto;
        }
    }

    @media (max-width: 520px) {
        .block-container {
            padding-left: 0.62rem !important;
            padding-right: 0.62rem !important;
        }

        .bento-grid {
            grid-template-columns: 1fr;
        }

        .bento-card.large {
            grid-column: span 1;
        }

        .top-nav {
            border-radius: 18px;
        }

        .hero h1 {
            font-size: clamp(1.85rem, 13vw, 2.75rem);
        }

        .cover-card {
            min-height: 218px;
        }

        .cover-content .label {
            font-size: 0.72rem;
            letter-spacing: 0.08rem;
        }

        .cover-dayline {
            font-size: 0.92rem;
        }
    }
    /* =========================
       TripAgent Product UI v2
       ========================= */
    :root {
        --v2-bg-a: #060711;
        --v2-bg-b: #111827;
        --v2-ink: #fff7ed;
        --v2-muted: #a7b0c0;
        --v2-gold: #f7d58a;
        --v2-gold-2: #f59e0b;
        --v2-ice: #9bdcff;
        --v2-card: rgba(12, 18, 34, 0.66);
        --v2-card-strong: rgba(8, 13, 26, 0.82);
        --v2-line: rgba(255, 244, 214, 0.16);
        --v2-line-bright: rgba(247, 213, 138, 0.34);
        --v2-shadow: 0 28px 100px rgba(0, 0, 0, 0.42);
    }

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        width: 100%;
        max-width: 100%;
        overflow-x: hidden !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 16% 8%, rgba(247, 213, 138, 0.22), transparent 28%),
            radial-gradient(circle at 82% 12%, rgba(155, 220, 255, 0.15), transparent 26%),
            radial-gradient(circle at 70% 86%, rgba(245, 158, 11, 0.15), transparent 34%),
            linear-gradient(145deg, #050611 0%, #0b1020 42%, #171923 72%, #060711 100%) !important;
        color: var(--v2-ink);
    }

    .stApp::before {
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(247, 213, 138, 0.035) 1px, transparent 1px) !important;
        background-size: 92px 92px;
        opacity: 0.42;
    }

    .block-container {
        max-width: 1240px !important;
        padding-top: 1rem !important;
    }

    .top-nav {
        position: sticky;
        top: 0.7rem;
        z-index: 5;
        margin-bottom: 1.6rem !important;
        padding: 0.74rem 0.92rem !important;
        border: 1px solid var(--v2-line-bright) !important;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.13), rgba(255, 255, 255, 0.035)),
            rgba(8, 12, 24, 0.78) !important;
        box-shadow: 0 18px 58px rgba(0, 0, 0, 0.34);
    }

    .brand-mark {
        background: linear-gradient(135deg, #fff2bf, #f7d58a 45%, #f59e0b) !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.34), 0 0 34px rgba(247, 213, 138, 0.22) !important;
    }

    .nav-links span {
        color: #f8fafc !important;
        border: 1px solid transparent;
    }

    .nav-links span:hover {
        border-color: rgba(247, 213, 138, 0.2);
        background: rgba(247, 213, 138, 0.08);
    }

    .hero.product-hero {
        position: relative;
        padding: clamp(1.1rem, 3vw, 2rem);
        border: 1px solid rgba(247, 213, 138, 0.20);
        border-radius: 34px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.115), rgba(255, 255, 255, 0.035)),
            radial-gradient(circle at 10% 0%, rgba(247, 213, 138, 0.15), transparent 38%),
            radial-gradient(circle at 90% 12%, rgba(155, 220, 255, 0.10), transparent 32%),
            rgba(9, 14, 28, 0.62);
        box-shadow: var(--v2-shadow);
        backdrop-filter: blur(26px);
        overflow: hidden;
        margin-bottom: 1.2rem;
    }

    .hero.product-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(90deg, transparent 0 8%, rgba(247, 213, 138, 0.12) 8% 8.12%, transparent 8.12%),
            linear-gradient(180deg, transparent 0 18%, rgba(255, 255, 255, 0.08) 18% 18.12%, transparent 18.12%);
        opacity: 0.55;
    }

    .hero.product-hero::after {
        display: none;
    }

    .hero-layout {
        position: relative;
        z-index: 1;
        grid-template-columns: minmax(0, 1.06fr) minmax(330px, 0.94fr) !important;
        gap: clamp(1rem, 2.4vw, 2rem) !important;
        align-items: center !important;
    }

    .eyebrow {
        border-color: rgba(247, 213, 138, 0.36) !important;
        color: #fff2bf !important;
        background: rgba(247, 213, 138, 0.10) !important;
        box-shadow: 0 12px 38px rgba(245, 158, 11, 0.13);
    }

    .hero h1 {
        max-width: 880px;
        font-size: clamp(3rem, 7.2vw, 6.9rem) !important;
        line-height: 0.92 !important;
        color: transparent !important;
        background: linear-gradient(102deg, #fff7ed 0%, #f7d58a 58%, #9bdcff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        text-wrap: balance;
    }

    .hero p {
        max-width: 720px;
        color: #d7dbe5 !important;
        font-size: clamp(1rem, 1.8vw, 1.22rem) !important;
        line-height: 1.85 !important;
    }

    .hero-proof span {
        border-color: rgba(247, 213, 138, 0.20) !important;
        background: rgba(255, 255, 255, 0.07) !important;
    }

    .hero-panel.product-preview {
        min-height: 430px;
        border: 1px solid rgba(247, 213, 138, 0.24) !important;
        border-radius: 30px !important;
        padding: 1.1rem !important;
        background:
            radial-gradient(circle at 20% 10%, rgba(247, 213, 138, 0.18), transparent 34%),
            linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.035)),
            rgba(6, 10, 22, 0.72) !important;
        overflow: hidden;
    }

    .preview-cover {
        position: relative;
        min-height: 176px;
        border-radius: 24px;
        border: 1px solid rgba(247, 213, 138, 0.22);
        background:
            linear-gradient(120deg, rgba(2, 6, 23, 0.20), rgba(2, 6, 23, 0.82)),
            radial-gradient(circle at 20% 20%, rgba(247, 213, 138, 0.72), transparent 20%),
            linear-gradient(135deg, #1e293b, #7c2d12 58%, #020617);
        box-shadow: 0 20px 65px rgba(0, 0, 0, 0.32);
        overflow: hidden;
    }

    .preview-cover::after {
        content: "";
        position: absolute;
        inset: 18px;
        border: 1px solid rgba(255, 247, 237, 0.18);
        border-radius: 18px;
    }

    .preview-cover-label {
        position: absolute;
        left: 1rem;
        bottom: 1rem;
        z-index: 1;
    }

    .preview-cover-label span {
        display: block;
        color: #f7d58a;
        font-size: 0.76rem;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .preview-cover-label strong {
        display: block;
        font-size: 1.55rem;
        color: #fff7ed;
        line-height: 1.08;
    }

    .preview-steps {
        display: grid;
        gap: 0.72rem;
        margin-top: 0.92rem;
    }

    .preview-step {
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr);
        gap: 0.72rem;
        align-items: center;
        padding: 0.72rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.055);
    }

    .preview-step b {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 14px;
        color: #17120a;
        background: linear-gradient(135deg, #fff2bf, #f59e0b);
    }

    .preview-step span {
        display: block;
        color: #f7d58a;
        font-size: 0.78rem;
        margin-bottom: 0.18rem;
    }

    .preview-step p {
        margin: 0 !important;
        color: #dbe4f0 !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-kicker) {
        position: relative;
        border: 1px solid rgba(247, 213, 138, 0.25) !important;
        border-radius: 30px !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.034)),
            rgba(8, 13, 26, 0.72) !important;
        box-shadow: 0 24px 86px rgba(0, 0, 0, 0.34), 0 0 0 1px rgba(255,255,255,0.035) inset !important;
        backdrop-filter: blur(26px);
        overflow: hidden;
        padding: 0.4rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-kicker)::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: radial-gradient(circle at 6% 0%, rgba(247, 213, 138, 0.16), transparent 34%);
    }

    .input-kicker {
        color: #f7d58a !important;
        letter-spacing: 0.12rem;
    }

    .input-title {
        color: #fff7ed !important;
        font-size: 1.26rem !important;
    }

    .sample-title,
    .hint {
        color: #aeb8ca !important;
    }

    .stTextArea textarea {
        min-height: 170px !important;
        border-radius: 24px !important;
        border: 1px solid rgba(247, 213, 138, 0.36) !important;
        background:
            linear-gradient(145deg, rgba(2,6,23,0.88), rgba(15,23,42,0.70)) !important;
        box-shadow: 0 18px 58px rgba(0,0,0,0.26), 0 0 0 1px rgba(255,255,255,0.045) inset !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        min-height: 44px;
        border: 1px solid rgba(255, 247, 237, 0.24) !important;
        border-radius: 999px !important;
        background:
            linear-gradient(135deg, #fff2bf 0%, #f7d58a 36%, #f59e0b 100%) !important;
        color: #16120b !important;
        box-shadow: 0 15px 38px rgba(245, 158, 11, 0.22) !important;
        font-weight: 900 !important;
    }

    .cover-card {
        min-height: 520px !important;
        border-radius: 36px !important;
        border: 1px solid rgba(247, 213, 138, 0.30) !important;
        box-shadow: 0 42px 130px rgba(0, 0, 0, 0.52), 0 0 80px rgba(247, 213, 138, 0.08) inset !important;
        isolation: isolate;
    }

    .cover-card::before {
        z-index: 2 !important;
        background:
            linear-gradient(90deg, rgba(247, 213, 138, 0.20) 1px, transparent 1px),
            linear-gradient(180deg, rgba(255,255,255,0.09) 1px, transparent 1px),
            radial-gradient(circle at 88% 18%, rgba(155, 220, 255, 0.18), transparent 28%) !important;
        background-size: 94px 94px, 94px 94px, 100% 100% !important;
        opacity: 0.5 !important;
    }

    .cover-card::after {
        background:
            linear-gradient(90deg, rgba(2, 6, 23, 0.88), rgba(2, 6, 23, 0.34) 56%, rgba(2, 6, 23, 0.78)),
            linear-gradient(180deg, rgba(2, 6, 23, 0.02), rgba(2, 6, 23, 0.88)) !important;
    }

    .cover-content {
        z-index: 3 !important;
    }

    .cover-content .label {
        color: #f7d58a !important;
        letter-spacing: 0.18rem !important;
    }

    .cover-content h2 {
        color: #fff7ed !important;
        font-size: clamp(3rem, 7vw, 6.4rem) !important;
        text-shadow: 0 22px 80px rgba(0,0,0,0.5);
    }

    .cover-dayline {
        color: #fff2bf !important;
        font-size: clamp(1.1rem, 2.4vw, 1.6rem) !important;
    }

    .cover-badge,
    .weather-meta span,
    .food-meta span {
        border-color: rgba(247, 213, 138, 0.24) !important;
        background: rgba(247, 213, 138, 0.10) !important;
        color: #fff2bf !important;
    }

    .section-heading {
        margin-top: 2.6rem !important;
        font-size: clamp(1.7rem, 3vw, 2.35rem) !important;
        color: #fff7ed !important;
    }

    .section-heading::after {
        content: "";
        display: block;
        width: 86px;
        height: 2px;
        margin-top: 0.45rem;
        background: linear-gradient(90deg, #f7d58a, transparent);
    }

    .section-subtitle {
        color: #aeb8ca !important;
        max-width: 820px;
    }

    .bento-grid {
        grid-template-columns: repeat(6, minmax(0, 1fr)) !important;
        gap: 1rem !important;
    }

    .bento-card {
        grid-column: span 2;
        min-height: 148px !important;
        border-radius: 28px !important;
        border: 1px solid rgba(247, 213, 138, 0.18) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.105), rgba(255,255,255,0.026)),
            rgba(10, 16, 31, 0.64) !important;
        box-shadow: 0 22px 76px rgba(0, 0, 0, 0.28) !important;
    }

    .bento-card.large {
        grid-column: span 3 !important;
    }

    .bento-card.warm {
        background:
            radial-gradient(circle at 12% 10%, rgba(247, 213, 138, 0.18), transparent 38%),
            linear-gradient(145deg, rgba(247, 213, 138, 0.14), rgba(255,255,255,0.03)),
            rgba(10, 16, 31, 0.64) !important;
    }

    .bento-card span,
    .segment-card span {
        color: #f7d58a !important;
        letter-spacing: 0.04rem;
        text-transform: uppercase;
    }

    .bento-card strong {
        color: #fff7ed !important;
        font-size: clamp(1.25rem, 2.2vw, 1.68rem) !important;
    }

    .timeline-grid {
        grid-template-columns: minmax(0, 1fr) !important;
        gap: 1.25rem !important;
    }

    .timeline-day {
        position: relative;
        border-radius: 30px !important;
        border: 1px solid rgba(247, 213, 138, 0.18) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.105), rgba(255,255,255,0.026)),
            rgba(10, 16, 31, 0.66) !important;
        box-shadow: 0 24px 86px rgba(0,0,0,0.30) !important;
        padding: 1.25rem 1.25rem 1.25rem 1.45rem !important;
        overflow: hidden;
    }

    .timeline-day::before {
        content: "";
        position: absolute;
        left: 2.05rem;
        top: 4.6rem;
        bottom: 1.4rem;
        width: 1px;
        background: linear-gradient(180deg, #f7d58a, rgba(247, 213, 138, 0.05));
    }

    .timeline-day h3 {
        font-size: clamp(1.2rem, 2vw, 1.55rem) !important;
        color: #fff7ed !important;
        padding-left: 0.2rem;
    }

    .timeline-slot {
        position: relative;
        grid-template-columns: 54px minmax(0, 1fr) !important;
        gap: 1rem !important;
        border-top: 0 !important;
        padding: 0.7rem 0 !important;
    }

    .slot-icon {
        position: relative;
        z-index: 1;
        width: 46px !important;
        height: 46px !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #fff2bf, #f59e0b) !important;
        box-shadow: 0 14px 32px rgba(245, 158, 11, 0.23) !important;
    }

    .timeline-slot > div:nth-child(2) {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        background: rgba(255,255,255,0.045);
        padding: 0.86rem 0.92rem;
    }

    .slot-time,
    .slot-original,
    .weather-date {
        color: #f7d58a !important;
    }

    .slot-place,
    .weather-main {
        color: #fff7ed !important;
    }

    .slot-meta-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    }

    .slot-meta-item {
        border-color: rgba(247, 213, 138, 0.12) !important;
        background: rgba(7, 11, 22, 0.46) !important;
    }

    .food-grid,
    .info-grid,
    .warning-grid,
    .weather-grid,
    .budget-grid {
        gap: 1rem !important;
    }

    .food-card,
    .info-card,
    .warning-card,
    .weather-card,
    .budget-card,
    .segment-card {
        border-radius: 28px !important;
        border: 1px solid rgba(247, 213, 138, 0.17) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.025)),
            rgba(10, 16, 31, 0.64) !important;
        box-shadow: 0 22px 76px rgba(0,0,0,0.28) !important;
    }

    .budget-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        max-width: 100%;
        margin-bottom: 1.6rem;
    }

    .budget-card {
        min-width: 0;
        padding: 1rem;
    }

    .budget-card span {
        display: block;
        color: #f7d58a;
        font-size: 0.78rem;
        letter-spacing: 0.04rem;
        margin-bottom: 0.38rem;
        text-transform: uppercase;
    }

    .budget-card p {
        color: #cbd5e1;
        line-height: 1.62;
        margin: 0;
        overflow-wrap: anywhere;
    }

    .food-card {
        position: relative;
        overflow: hidden;
    }

    .food-card::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: radial-gradient(circle at 88% 10%, rgba(247, 213, 138, 0.13), transparent 28%);
    }

    .food-card h3 {
        position: relative;
        color: #fff7ed !important;
        font-size: 1.22rem;
    }

    .food-location,
    .food-map-keyword {
        position: relative;
        color: #aeb8ca !important;
    }

    .weather-card h3,
    .info-card h3,
    .warning-card h3 {
        color: #fff7ed !important;
    }

    .weather-day {
        border-top-color: rgba(247, 213, 138, 0.10) !important;
    }

    .weather-icon,
    .info-icon,
    .warning-icon {
        background: linear-gradient(135deg, #fff2bf, #f59e0b) !important;
        color: #17120a !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.result-actions-title),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.source-card-title) {
        border-radius: 28px !important;
        border: 1px solid rgba(247, 213, 138, 0.20) !important;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03)),
            rgba(10, 16, 31, 0.64) !important;
        box-shadow: 0 22px 76px rgba(0,0,0,0.28) !important;
    }

    .result-actions-title,
    .source-card-title {
        color: #fff7ed;
        font-size: 1.25rem;
        font-weight: 900;
        margin: 0 0 0.35rem;
    }

    .source-card-text {
        color: #cbd5e1;
        line-height: 1.65;
        margin: 0.25rem 0;
    }

    .search-status-pill,
    .trust-card,
    .weather-fallback,
    .blessing-card {
        border-color: rgba(247, 213, 138, 0.18) !important;
        background:
            linear-gradient(145deg, rgba(247, 213, 138, 0.10), rgba(255,255,255,0.025)),
            rgba(10, 16, 31, 0.60) !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 0.72rem 0.68rem 3rem !important;
        }

        .top-nav {
            position: static;
            border-radius: 22px !important;
            margin-bottom: 0.9rem !important;
        }

        .hero.product-hero {
            border-radius: 26px;
            padding: 1rem;
        }

        .hero-layout,
        .bento-grid,
        .timeline-grid,
        .food-grid,
        .info-grid,
        .warning-grid,
        .weather-grid,
        .budget-grid,
        .segment-overview-grid,
        .trust-strip,
        .result-action-grid,
        .slot-meta-grid {
            grid-template-columns: minmax(0, 1fr) !important;
            width: 100% !important;
        }

        .hero h1 {
            font-size: clamp(2.35rem, 13vw, 3.6rem) !important;
        }

        .hero p {
            font-size: 0.96rem !important;
            line-height: 1.65 !important;
        }

        .hero-panel.product-preview {
            min-height: auto;
        }

        .preview-cover {
            min-height: 150px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.input-kicker) {
            border-radius: 24px !important;
        }

        .stTextArea textarea {
            min-height: 128px !important;
        }

        .cover-card {
            min-height: 300px !important;
            aspect-ratio: 4 / 3 !important;
            border-radius: 26px !important;
        }

        .cover-content h2 {
            font-size: clamp(2rem, 12vw, 3.3rem) !important;
        }

        .bento-card,
        .bento-card.large {
            grid-column: span 1 !important;
            min-height: auto !important;
        }

        .timeline-day::before {
            left: 1.55rem;
            top: 4.5rem;
        }

        .timeline-slot {
            grid-template-columns: 42px minmax(0, 1fr) !important;
            gap: 0.72rem !important;
        }

        .slot-icon {
            width: 36px !important;
            height: 36px !important;
            border-radius: 13px !important;
            font-size: 0.76rem !important;
        }

        .timeline-slot > div:nth-child(2) {
            padding: 0.72rem;
            border-radius: 17px;
        }

        .weather-day {
            grid-template-columns: 38px minmax(0, 1fr) !important;
        }

        .weather-meta span,
        .food-meta span {
            width: 100%;
        }

        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 0 !important;
        }

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {
            width: 100% !important;
            white-space: normal !important;
        }
    }

    @media (max-width: 520px) {
        .hero-proof span {
            width: 100%;
        }

        .preview-step {
            grid-template-columns: 36px minmax(0, 1fr);
        }

        .preview-step b {
            width: 34px;
            height: 34px;
        }

        .section-heading {
            font-size: 1.45rem !important;
        }

        .cover-content {
            inset: auto 0.9rem 0.95rem 0.9rem !important;
        }
    }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)


def parse_chinese_number(number_text: str) -> int:
    """parse_chinese_number：把常见中文数字转换成整数。"""

    # chinese_number_map：保存中文数字到阿拉伯数字的对应关系。
    chinese_number_map = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    if number_text.isdigit():
        return int(number_text)

    if number_text == "十":
        return 10

    if number_text.startswith("十"):
        return 10 + chinese_number_map.get(number_text[-1], 0)

    if "十" in number_text:
        # parts：中文数字按“十”拆分后的十位和个位。
        parts = number_text.split("十")
        tens = chinese_number_map.get(parts[0], 1)
        ones = chinese_number_map.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones

    return chinese_number_map.get(number_text, DEFAULT_TRAVEL_DAYS)


def parse_chinese_amount(amount_text: str) -> float | None:
    """parse_chinese_amount：把中文预算金额转换成数字，例如“一万”转成 10000。"""

    # cleaned_amount：清理空格后的中文金额文本。
    cleaned_amount = amount_text.strip()
    if not cleaned_amount:
        return None

    if re.fullmatch(r"[0-9][0-9,]*(?:\.\d+)?", cleaned_amount):
        return float(cleaned_amount.replace(",", ""))

    # chinese_digit_map：中文数字字符和数值的对应关系。
    chinese_digit_map = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    if cleaned_amount.endswith("万"):
        # base_text：中文金额中“万”前面的数字部分。
        base_text = cleaned_amount[:-1]
        if not base_text:
            return 10000.0
        base_value = parse_chinese_number(base_text)
        return float(base_value * 10000)

    if cleaned_amount in chinese_digit_map:
        return float(chinese_digit_map[cleaned_amount])

    if "千" in cleaned_amount:
        # thousand_parts：中文金额按“千”拆分后的千位和余数。
        thousand_parts = cleaned_amount.split("千", 1)
        thousands = parse_chinese_number(thousand_parts[0] or "一") * 1000
        rest = parse_chinese_amount(thousand_parts[1]) if thousand_parts[1] else 0
        return float(thousands + (rest or 0))

    return None


def normalize_currency_unit(currency_text: str | None) -> str:
    """normalize_currency_unit：把用户输入的货币单位统一成标准货币代码。"""

    if not currency_text:
        return DEFAULT_BUDGET_CURRENCY

    # normalized_unit：统一大小写并去除空格后的货币单位。
    normalized_unit = currency_text.strip().upper()

    # currency_alias_map：常见货币表达和标准货币代码的对应关系。
    currency_alias_map = {
        "人民币": "CNY",
        "RMB": "CNY",
        "CNY": "CNY",
        "元": "CNY",
        "块": "CNY",
        "日元": "JPY",
        "日币": "JPY",
        "日圓": "JPY",
        "JPY": "JPY",
        "美元": "USD",
        "美金": "USD",
        "USD": "USD",
        "欧元": "EUR",
        "欧": "EUR",
        "EUR": "EUR",
        "韩元": "KRW",
        "韩币": "KRW",
        "KRW": "KRW",
    }

    return currency_alias_map.get(normalized_unit, DEFAULT_BUDGET_CURRENCY)


def get_currency_name(currency_code: str) -> str:
    """get_currency_name：把标准货币代码转换成中文显示名称。"""

    # currency_name_map：标准货币代码和中文名称的对应关系。
    currency_name_map = {
        "CNY": "人民币",
        "JPY": "日元",
        "USD": "美元",
        "EUR": "欧元",
        "KRW": "韩元",
    }

    return currency_name_map.get(currency_code, currency_code)


def format_budget_amount(amount: float) -> str:
    """format_budget_amount：把预算金额格式化为适合页面展示的文本。"""

    # numeric_amount：兼容 int 和 float 的预算金额数值。
    numeric_amount = float(amount)

    if numeric_amount.is_integer():
        return f"{int(numeric_amount):,}"

    return f"{numeric_amount:,.2f}".rstrip("0").rstrip(".")


def parse_budget_info(cleaned_input: str, budget_level: str) -> dict:
    """parse_budget_info：识别用户输入中的预算金额和货币单位。"""

    # budget_pattern：匹配“预算5000”“预算一万”“预算10万日元”“预算800 USD”等表达。
    budget_pattern = re.compile(
        r"(?:预算|总预算|花费|费用)\s*"
        r"(?:约|大概|大约|控制在|不超过|不超|以内|左右|是|为|:|：)?\s*"
        r"([0-9][0-9,]*(?:\.\d+)?|[零一二两三四五六七八九十百千万]+)\s*"
        r"(万)?\s*"
        r"(人民币|日元|日币|日圓|美元|美金|欧元|欧|韩元|韩币|USD|EUR|JPY|KRW|RMB|CNY|元|块)?",
        re.IGNORECASE,
    )

    # budget_match：预算金额匹配结果。
    budget_match = budget_pattern.search(cleaned_input)
    if not budget_match:
        return {
            "amount": None,
            "currency": None,
            "currency_name": None,
            "display": budget_level,
            "level": budget_level,
            "has_explicit_amount": False,
        }

    # amount_text：预算金额文本。
    amount_text = budget_match.group(1)

    # amount_value：预算金额数值。
    amount_value = parse_chinese_amount(amount_text)
    if amount_value is None:
        amount_value = float(amount_text.replace(",", ""))

    if budget_match.group(2) and "万" not in amount_text:
        amount_value *= 10000

    # currency_code：标准货币代码；用户没写单位时默认 CNY。
    currency_code = normalize_currency_unit(budget_match.group(3))

    # currency_name：中文货币名称。
    currency_name = get_currency_name(currency_code)

    # budget_display：页面和提示词展示的预算文本。
    budget_display = f"{format_budget_amount(amount_value)} {currency_name} ({currency_code})"

    # normalized_amount：整数金额保存为 int，便于 Debug 区显示 10000 而不是 10000.0。
    normalized_amount = int(amount_value) if float(amount_value).is_integer() else amount_value

    return {
        "amount": normalized_amount,
        "currency": currency_code,
        "currency_name": currency_name,
        "display": budget_display,
        "level": budget_level,
        "has_explicit_amount": True,
    }


def infer_destination_currency(destination: str) -> str | None:
    """infer_destination_currency：根据目的地粗略推断当地常用货币。"""

    # destination_currency_keywords：国外目的地关键词和当地货币代码。
    destination_currency_keywords = {
        "JPY": ["日本", "东京", "大阪", "京都", "北海道", "冲绳", "奈良", "福冈", "名古屋", "札幌", "箱根"],
        "KRW": ["韩国", "首尔", "釜山", "济州"],
        "EUR": [
            "欧洲",
            "法国",
            "巴黎",
            "意大利",
            "罗马",
            "米兰",
            "德国",
            "柏林",
            "西班牙",
            "巴塞罗那",
            "荷兰",
            "阿姆斯特丹",
            "葡萄牙",
            "希腊",
            "瑞士",
        ],
        "USD": ["美国", "纽约", "洛杉矶", "旧金山", "西雅图", "夏威夷"],
    }

    for currency_code, keyword_list in destination_currency_keywords.items():
        if any(keyword in destination for keyword in keyword_list):
            return currency_code

    return None


def build_exchange_hint(parsed_request: dict) -> str | None:
    """build_exchange_hint：为国外目的地生成粗略换算提示。"""

    # budget_amount：用户输入的预算金额。
    budget_amount = parsed_request.get("budget_amount")
    if not budget_amount:
        return None

    # destination_currency：根据目的地推断出的当地货币。
    destination_currency = infer_destination_currency(parsed_request["destination"])
    if not destination_currency:
        return None

    # source_currency：用户输入预算的货币代码。
    source_currency = parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY

    # cny_to_currency_rate：人民币到其他货币的粗略换算比例。
    cny_to_currency_rate = {
        "JPY": 20.0,
        "KRW": 190.0,
        "EUR": 0.13,
        "USD": 0.14,
    }

    # currency_to_cny_rate：其他货币到人民币的粗略换算比例。
    currency_to_cny_rate = {
        "JPY": 0.05,
        "KRW": 0.0053,
        "EUR": 7.8,
        "USD": 7.2,
        "CNY": 1.0,
    }

    if source_currency == DEFAULT_BUDGET_CURRENCY and destination_currency in cny_to_currency_rate:
        # converted_amount：人民币预算换算成目的地当地货币的粗略金额。
        converted_amount = budget_amount * cny_to_currency_rate[destination_currency]
        return (
            f"粗略换算：{format_budget_amount(budget_amount)} 人民币约 "
            f"{format_budget_amount(converted_amount)} {get_currency_name(destination_currency)}"
            "，汇率仅供参考，请以出行前实际汇率为准。"
        )

    if source_currency != DEFAULT_BUDGET_CURRENCY and source_currency in currency_to_cny_rate:
        # converted_amount：外币预算换算成人民币的粗略金额。
        converted_amount = budget_amount * currency_to_cny_rate[source_currency]
        return (
            f"粗略换算：{format_budget_amount(budget_amount)} {get_currency_name(source_currency)}约 "
            f"{format_budget_amount(converted_amount)} 人民币"
            "，汇率仅供参考，请以出行前实际汇率为准。"
        )

    return None


def extract_destination(cleaned_input: str) -> str:
    """extract_destination：从用户输入中优先提取明确目的地。"""

    # destination_patterns：从强到弱排列的目的地匹配规则。
    destination_patterns = [
        r"(?:^|[，。,\s])([一-龥A-Za-z]{2,20})\s*(?:[0-9一二两三四五六七八九十]+)\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行)",
        r"想去(?!看看|看一看|看|尝|尝一尝|吃|逛)([一-龥A-Za-z]{2,20}?)(?:旅游|旅行|自由行|游|度假|玩|看|赏|吃|逛|，|。|,|\s|$)",
        r"去(?!看看|看一看|看|尝|尝一尝|吃|逛)([一-龥A-Za-z]{2,20}?)(?:旅游|旅行|自由行|游|度假|玩|看|赏|吃|逛|，|。|,|\s|$)",
        r"(?:^|[，。,\s])([一-龥A-Za-z]{2,20}?)\s*(?:旅游|旅行|自由行|度假|游)",
        r"目的地[:：]\s*([一-龥A-Za-z]{2,20})",
    ]

    # invalid_destination_words：不应被当成目的地的动作词或泛词。
    invalid_destination_words = {
        "看看",
        "看一看",
        "看",
        "尝",
        "尝一尝",
        "美食",
        "夜景",
        "其他景点",
        "景点",
    }

    for pattern in destination_patterns:
        match = re.search(pattern, cleaned_input)
        if not match:
            continue

        # destination：当前规则识别出的目的地。
        destination = match.group(1).strip()
        destination = re.sub(r"^(?:我|我们|本人)?(?:想去|要去|计划去|打算去|去)", "", destination).strip()
        if destination and destination not in invalid_destination_words:
            return destination

    return DEFAULT_DESTINATION


def extract_trip_days(cleaned_input: str) -> tuple[int, int]:
    """extract_trip_days：识别“7日游”“7天”“七日”等明确天数。"""

    # days_patterns：可识别的天数表达。
    days_patterns = [
        r"([0-9一二两三四五六七八九十]+)\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行)",
        r"([0-9一二两三四五六七八九十]+)\s*(?:天|日)(?!元|币)",
    ]

    for pattern in days_patterns:
        match = re.search(pattern, cleaned_input)
        if match:
            days = max(1, parse_chinese_number(match.group(1)))
            return days, max(0, days - 1)

    return DEFAULT_TRAVEL_DAYS, DEFAULT_TRAVEL_NIGHTS


def get_known_destination_names() -> list[str]:
    """get_known_destination_names：返回用于多目的地识别的常见目的地名称。"""

    return [
        "内蒙古",
        "黑龙江",
        "张家界",
        "九寨沟",
        "西双版纳",
        "香格里拉",
        "南京",
        "江西",
        "南昌",
        "景德镇",
        "婺源",
        "庐山",
        "上饶",
        "赣州",
        "上海",
        "苏州",
        "杭州",
        "东京",
        "京都",
        "大阪",
        "广州",
        "深圳",
        "北京",
        "成都",
        "重庆",
        "西安",
        "云南",
        "大理",
        "丽江",
        "昆明",
        "福建",
        "厦门",
        "泉州",
        "福州",
        "广东",
        "海南",
        "三亚",
        "青岛",
        "长沙",
        "武汉",
        "天津",
        "首尔",
        "釜山",
        "济州",
        "日本",
        "韩国",
        "欧洲",
        "法国",
        "巴黎",
        "意大利",
        "罗马",
        "美国",
        "纽约",
        "洛杉矶",
    ]


def get_province_route_note(destination: str) -> str:
    """get_province_route_note：为省份或大区域目的地生成具体城市路线提示。"""

    # province_route_map：省份或大区域到经典城市组合的映射。
    province_route_map = {
        "江西": "你输入的是省份，系统为你选择较经典的江西路线：南昌、景德镇、婺源、庐山、上饶，可根据偏好调整。",
        "云南": "你输入的是省份，系统会优先按昆明、大理、丽江、香格里拉等经典路线规划，可根据偏好调整。",
        "福建": "你输入的是省份，系统会优先按厦门、泉州、福州或武夷山等经典路线规划，可根据偏好调整。",
        "广东": "你输入的是省份，系统会优先按广州、深圳、珠海或潮汕等经典路线规划，可根据偏好调整。",
        "海南": "你输入的是省份，系统会优先按海口、三亚、万宁等经典路线规划，可根据偏好调整。",
        "日本": "你输入的是国家，系统会优先按东京、京都、大阪等经典路线规划，可根据偏好调整。",
        "韩国": "你输入的是国家，系统会优先按首尔、釜山、济州等经典路线规划，可根据偏好调整。",
        "欧洲": "你输入的是大区域，系统会选择适合天数的城市组合，并明确说明推断依据。",
    }

    return province_route_map.get(destination, "")


def get_weather_reference_city(destination: str, travel_json: dict | None = None) -> tuple[str, str]:
    """get_weather_reference_city：把省份或大区域目的地转换为适合查询天气的主要城市。"""

    # destination_city_map：省份、国家或大区域到天气参考城市的映射。
    destination_city_map = {
        "江西": "南昌",
        "云南": "昆明",
        "福建": "厦门",
        "广东": "广州",
        "海南": "海口",
        "日本": "东京",
        "韩国": "首尔",
        "欧洲": "巴黎",
    }

    # city_from_map：从固定映射中得到的参考城市。
    city_from_map = destination_city_map.get(destination)
    if city_from_map:
        return city_from_map, f"{destination}主要城市天气参考：{city_from_map}"

    return destination, destination


def geocode_destination(destination: str) -> dict | None:
    """geocode_destination：使用 Open-Meteo Geocoding API 把目的地转换为经纬度。"""

    if not destination:
        return None

    try:
        # response：Open-Meteo 地理编码接口响应。
        response = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params={
                "name": destination,
                "count": 1,
                "language": "zh",
                "format": "json",
            },
            timeout=8,
        )
        response.raise_for_status()
        # geocode_data：地理编码 JSON 数据。
        geocode_data = response.json()
    except Exception:
        return None

    # result_list：Open-Meteo 返回的候选地点列表。
    result_list = geocode_data.get("results", [])
    if not result_list:
        return None

    # first_result：最匹配的地点。
    first_result = result_list[0]
    latitude = first_result.get("latitude")
    longitude = first_result.get("longitude")
    if latitude is None or longitude is None:
        return None

    return {
        "latitude": latitude,
        "longitude": longitude,
        "name": first_result.get("name", destination),
        "country": first_result.get("country", ""),
        "timezone": first_result.get("timezone", "auto"),
    }


def fetch_weather_forecast(latitude: float, longitude: float) -> dict | None:
    """fetch_weather_forecast：使用 Open-Meteo Forecast API 查询未来天气。"""

    try:
        # response：Open-Meteo 天气预报接口响应。
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": ",".join(
                    [
                        "weather_code",
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                        "wind_speed_10m_max",
                    ]
                ),
                "hourly": "relative_humidity_2m",
                "timezone": "auto",
                "forecast_days": WEATHER_FORECAST_DAYS,
            },
            timeout=8,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def map_weather_code(weather_code: int | None) -> tuple[str, str]:
    """map_weather_code：把 Open-Meteo 天气代码转换成中文天气状态和图标。"""

    if weather_code is None:
        return "天气待确认", "🌦️"

    # weather_code_map：WMO 天气代码到中文状态的映射。
    weather_code_map = {
        0: ("晴", "☀️"),
        1: ("大致晴朗", "🌤️"),
        2: ("多云", "🌤️"),
        3: ("阴", "☁️"),
        45: ("有雾", "☁️"),
        48: ("雾凇", "☁️"),
        51: ("小毛毛雨", "🌧️"),
        53: ("毛毛雨", "🌧️"),
        55: ("较强毛毛雨", "🌧️"),
        56: ("冻毛毛雨", "🌧️"),
        57: ("强冻毛毛雨", "🌧️"),
        61: ("小雨", "🌧️"),
        63: ("中雨", "🌧️"),
        65: ("大雨", "🌧️"),
        66: ("冻雨", "🌧️"),
        67: ("强冻雨", "🌧️"),
        71: ("小雪", "🌨️"),
        73: ("中雪", "🌨️"),
        75: ("大雪", "🌨️"),
        77: ("雪粒", "🌨️"),
        80: ("阵雨", "🌧️"),
        81: ("较强阵雨", "🌧️"),
        82: ("强阵雨", "🌧️"),
        85: ("阵雪", "🌨️"),
        86: ("强阵雪", "🌨️"),
        95: ("雷雨", "⛈️"),
        96: ("雷雨伴冰雹", "⛈️"),
        99: ("强雷雨伴冰雹", "⛈️"),
    }

    return weather_code_map.get(weather_code, ("天气待确认", "🌦️"))


def format_weather_date(date_text: str) -> str:
    """format_weather_date：把 YYYY-MM-DD 日期转换成中文短日期。"""

    try:
        # date_value：解析后的日期对象。
        date_value = datetime.fromisoformat(date_text)
        return f"{date_value.month}月{date_value.day}日"
    except ValueError:
        return date_text


def format_weather_value(value: float | int | None, suffix: str) -> str:
    """format_weather_value：格式化天气数值，缺失时不编造。"""

    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{round(value)}{suffix}"
    return f"{value}{suffix}"


def calculate_daily_humidity(weather_data: dict) -> dict[str, int | None]:
    """calculate_daily_humidity：从小时级湿度数据计算每天平均湿度。"""

    # hourly_data：Open-Meteo 小时级天气数据。
    hourly_data = weather_data.get("hourly", {})

    # time_list/humidity_list：小时和相对湿度列表。
    time_list = hourly_data.get("time", [])
    humidity_list = hourly_data.get("relative_humidity_2m", [])

    # humidity_bucket：按日期收集的湿度值。
    humidity_bucket: dict[str, list[float]] = {}
    for time_text, humidity_value in zip(time_list, humidity_list):
        if humidity_value is None or not time_text:
            continue
        date_key = str(time_text).split("T", 1)[0]
        humidity_bucket.setdefault(date_key, []).append(float(humidity_value))

    # humidity_by_date：每天平均湿度。
    humidity_by_date: dict[str, int | None] = {}
    for date_key, values in humidity_bucket.items():
        humidity_by_date[date_key] = round(sum(values) / len(values)) if values else None

    return humidity_by_date


def build_single_weather_advice(weather_item: dict) -> str:
    """build_single_weather_advice：根据单日天气生成携带和出行提醒。"""

    # weather_text：中文天气状态。
    weather_text = str(weather_item.get("weather_text", ""))

    # precipitation_probability：降水概率。
    precipitation_probability = weather_item.get("precipitation_probability")

    # temperature_max/temperature_min/wind_speed：温度和风速。
    temperature_max = weather_item.get("temperature_max")
    temperature_min = weather_item.get("temperature_min")
    wind_speed = weather_item.get("wind_speed")

    # advice_parts：多条提醒合并后的建议。
    advice_parts = []

    # rainy_words：用于判断雨天的关键词。
    rainy_words = ["雨", "小雨", "中雨", "大雨", "阵雨", "雷雨"]
    is_rainy = precipitation_probability is not None and precipitation_probability >= 50
    if is_rainy or any(word in weather_text for word in rainy_words):
        advice_parts.append("可能下雨，建议携带雨伞、防水袋和防滑鞋，户外景点注意地面湿滑。☔")

    if temperature_max is not None and temperature_max >= 30:
        advice_parts.append("气温偏高，注意防晒、补水，尽量避开中午长时间暴晒。")

    if temperature_min is not None and temperature_min <= 10:
        advice_parts.append("早晚偏冷，建议带外套，注意昼夜温差。")

    if "晴" in weather_text:
        advice_parts.append("晴天适合户外游玩，记得准备防晒、墨镜和水。☀️")

    if "多云" in weather_text or "阴" in weather_text:
        advice_parts.append("适合步行游玩，但天气变化仍建议出门前查看实时天气。")

    if wind_speed is not None and wind_speed >= 38:
        advice_parts.append("风力偏大，注意保暖，高处观景或乘船行程要留意安全。🌬️")

    if not advice_parts:
        advice_parts.append("天气信息仅供参考，请出行前查看天气 App。")

    return " ".join(advice_parts)


def build_weather_advice(weather_data: dict) -> list[dict]:
    """build_weather_advice：把 Open-Meteo 天气数据整理成每日天气卡片。"""

    if not isinstance(weather_data, dict):
        return []

    # daily_data：Open-Meteo 每日天气数据。
    daily_data = weather_data.get("daily", {})
    date_list = daily_data.get("time", [])
    if not date_list:
        return []

    # humidity_by_date：按日期计算出的平均湿度。
    humidity_by_date = calculate_daily_humidity(weather_data)

    # weather_items：每日天气卡片数据。
    weather_items = []
    for index, date_text in enumerate(date_list[:WEATHER_FORECAST_DAYS]):
        # weather_code：Open-Meteo WMO 天气代码。
        weather_code_list = daily_data.get("weather_code", [])
        weather_code = weather_code_list[index] if index < len(weather_code_list) else None
        weather_text, weather_icon = map_weather_code(weather_code)

        # temperature_max/min：最高和最低温度。
        temperature_max_list = daily_data.get("temperature_2m_max", [])
        temperature_min_list = daily_data.get("temperature_2m_min", [])
        temperature_max = temperature_max_list[index] if index < len(temperature_max_list) else None
        temperature_min = temperature_min_list[index] if index < len(temperature_min_list) else None

        # precipitation_probability：最大降水概率。
        precipitation_list = daily_data.get("precipitation_probability_max", [])
        precipitation_probability = precipitation_list[index] if index < len(precipitation_list) else None

        # wind_speed：最大风速。
        wind_speed_list = daily_data.get("wind_speed_10m_max", [])
        wind_speed = wind_speed_list[index] if index < len(wind_speed_list) else None
        if wind_speed is not None and wind_speed >= 38:
            weather_icon = "🌬️"
            weather_text = f"{weather_text}，风力偏大"

        # weather_item：单日天气展示数据。
        weather_item = {
            "date": format_weather_date(str(date_text)),
            "raw_date": str(date_text),
            "weather_text": weather_text,
            "weather_icon": weather_icon,
            "temperature_max": temperature_max,
            "temperature_min": temperature_min,
            "humidity": humidity_by_date.get(str(date_text)),
            "precipitation_probability": precipitation_probability,
            "wind_speed": wind_speed,
            "will_rain": bool(
                (precipitation_probability is not None and precipitation_probability >= 50)
                or any(word in weather_text for word in ["雨", "阵雨", "雷雨"])
            ),
        }
        weather_item["advice"] = build_single_weather_advice(weather_item)
        weather_items.append(weather_item)

    return weather_items


def build_weather_cards(parsed_request: dict, travel_json: dict | None = None) -> list[dict]:
    """build_weather_cards：为单目的地或多目的地构建天气模块卡片数据。"""

    # trip_segments：目的地分段列表。
    trip_segments = parsed_request.get("trip_segments") or [{"destination": parsed_request["destination"]}]

    # weather_cards：所有目的地天气卡片。
    weather_cards = []
    seen_weather_destinations = set()
    for segment in trip_segments:
        # destination：当前天气卡片对应的目的地。
        destination = str(segment.get("destination", "")).strip()
        if not destination or destination in seen_weather_destinations:
            continue
        seen_weather_destinations.add(destination)

        # query_city/display_destination：用于查询天气的城市和页面展示标题。
        query_city, display_destination = get_weather_reference_city(destination, travel_json)

        # geocode_result：目的地经纬度。
        geocode_result = geocode_destination(query_city)
        if not geocode_result:
            weather_cards.append(
                {
                    "destination": display_destination,
                    "query_city": query_city,
                    "error": "天气信息暂时无法获取，请出行前查看天气 App。",
                    "days": [],
                }
            )
            continue

        # forecast_data：Open-Meteo 天气预报数据。
        forecast_data = fetch_weather_forecast(geocode_result["latitude"], geocode_result["longitude"])
        if not forecast_data:
            weather_cards.append(
                {
                    "destination": display_destination,
                    "query_city": query_city,
                    "error": "天气信息暂时无法获取，请出行前查看天气 App。",
                    "days": [],
                }
            )
            continue

        # day_weather_items：每日天气卡片数据。
        day_weather_items = build_weather_advice(forecast_data)
        if not day_weather_items:
            weather_cards.append(
                {
                    "destination": display_destination,
                    "query_city": query_city,
                    "error": "天气信息暂时无法获取，请出行前查看天气 App。",
                    "days": [],
                }
            )
            continue

        weather_cards.append(
            {
                "destination": display_destination,
                "query_city": query_city,
                "error": "",
                "days": day_weather_items,
            }
        )

    return weather_cards


def build_weather_markdown(weather_cards: list[dict] | None) -> str:
    """build_weather_markdown：把天气卡片转换成可复制 Markdown。"""

    if not weather_cards:
        return "## 天气与出行提醒 🌦️\n天气信息暂时无法获取，请出行前查看天气 App。"

    # markdown_lines：天气 Markdown 行。
    markdown_lines = ["## 天气与出行提醒 🌦️"]
    for weather_card in weather_cards:
        destination = str(weather_card.get("destination", "目的地"))
        markdown_lines.append(f"### {destination}｜未来 {WEATHER_FORECAST_DAYS} 天天气")
        if weather_card.get("error"):
            markdown_lines.append(str(weather_card["error"]))
            continue

        for day_weather in weather_card.get("days", []):
            will_rain_text = "可能下雨" if day_weather.get("will_rain") else "降雨风险较低"
            markdown_lines.extend(
                [
                    f"#### {day_weather.get('date', '')}",
                    f"- 天气：{day_weather.get('weather_text', '天气待确认')}",
                    (
                        f"- 温度：{format_weather_value(day_weather.get('temperature_min'), '°C')} - "
                        f"{format_weather_value(day_weather.get('temperature_max'), '°C')}"
                    ),
                    f"- 湿度：{format_weather_value(day_weather.get('humidity'), '%')}",
                    f"- 降水概率：{format_weather_value(day_weather.get('precipitation_probability'), '%')}",
                    f"- 是否可能下雨：{will_rain_text}",
                    f"- 建议：{day_weather.get('advice', '天气信息仅供参考，请出行前查看天气 App。')}",
                ]
            )

    return "\n".join(markdown_lines)


def remove_markdown_sections(markdown_text: str, heading_names: set[str]) -> str:
    """remove_markdown_sections：移除指定二级标题章节，避免 Markdown 原文重复。"""

    # markdown_lines：原始 Markdown 行。
    markdown_lines = markdown_text.splitlines()

    # kept_lines：保留下来的 Markdown 行。
    kept_lines = []
    skipping = False
    for line in markdown_lines:
        # heading_match：匹配二级标题。
        heading_match = re.match(r"^##\s+(.+?)\s*$", line)
        if heading_match:
            heading_text = heading_match.group(1).strip()
            skipping = heading_text in heading_names
            if skipping:
                continue
        if not skipping:
            kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def append_weather_and_blessing_to_markdown(markdown_text: str, weather_cards: list[dict] | None, generated_at: str) -> str:
    """append_weather_and_blessing_to_markdown：把天气、更新时间和祝福语追加到 Markdown 末尾。"""

    # cleaned_markdown：移除模型可能生成的旧信息区，避免重复和技术词外露。
    cleaned_markdown = remove_markdown_sections(
        markdown_text,
        {
            "天气与出行提醒",
            "天气与出行提醒 🌦️",
            "信息来源与更新时间",
            "信息与更新时间",
            "旅行祝福语",
        },
    )

    # source_markdown：用户友好的信息与更新时间。
    source_markdown = "\n".join(
        [
            "## 信息与更新时间",
            "本攻略由 AI 根据你的输入和当前可用信息整理生成。",
            f"更新时间：{generated_at}",
            "门票、预约、开放时间、交通政策和天气情况可能变化，请出行前以官方渠道和天气 App 为准。",
        ]
    )

    # blessing_markdown：旅行祝福语。
    blessing_markdown = (
        "## 旅行祝福语\n"
        "祝你这次旅行顺利又开心。记得提前确认天气、门票和交通安排，慢慢走、好好看，"
        "把喜欢的风景都装进记忆里。祝你旅途愉快呀～ 🌿✨🧳"
    )

    return "\n\n".join([cleaned_markdown, build_weather_markdown(weather_cards), source_markdown, blessing_markdown]).strip()


def extract_days_near_destination(text_after_destination: str) -> int | None:
    """extract_days_near_destination：在目的地后方的小片段中提取天数。"""

    # day_match：匹配“玩2天”“游玩3天”“4日”等表达。
    day_match = re.search(
        r"(?:游玩|玩|旅游|旅行|自由行|游)?\s*([0-9一二两三四五六七八九十]+)\s*(?:天|日)(?!元|币)",
        text_after_destination,
    )
    if not day_match:
        return None

    return max(1, parse_chinese_number(day_match.group(1)))


def extract_trip_segments(cleaned_input: str, preferences: list[str]) -> list[dict]:
    """extract_trip_segments：识别单目的地或多目的地分段行程。"""

    # known_destinations：常见目的地词表，按长度降序避免“江西”被更短词截断。
    known_destinations = sorted(get_known_destination_names(), key=len, reverse=True)

    # destination_matches：用户输入中出现的目的地及位置。
    destination_matches = []
    for destination in known_destinations:
        for match in re.finditer(re.escape(destination), cleaned_input):
            destination_matches.append({"destination": destination, "start": match.start(), "end": match.end()})

    destination_matches.sort(key=lambda item: item["start"])

    # deduped_matches：按文本位置去重，避免同一位置重复匹配。
    deduped_matches = []
    occupied_ranges = []
    for item in destination_matches:
        if any(item["start"] >= start and item["end"] <= end for start, end in occupied_ranges):
            continue
        deduped_matches.append(item)
        occupied_ranges.append((item["start"], item["end"]))

    # unique_matches：同一个目的地多次出现时只保留第一次，避免“杭州美食、杭州夜景”被拆成多段。
    unique_matches = []
    seen_destinations = set()
    for item in deduped_matches:
        destination = item["destination"]
        if destination in seen_destinations:
            continue
        unique_matches.append(item)
        seen_destinations.add(destination)
    deduped_matches = unique_matches

    if not deduped_matches:
        destination = extract_destination(cleaned_input)
        days, nights = extract_trip_days(cleaned_input)
        return [
            {
                "destination": destination,
                "days": days,
                "nights": nights,
                "days_inferred": False,
                "preferences": preferences,
                "note": get_province_route_note(destination),
            }
        ]

    # segments：最终分段行程列表。
    segments = []
    for index, item in enumerate(deduped_matches):
        next_start = deduped_matches[index + 1]["start"] if index + 1 < len(deduped_matches) else len(cleaned_input)
        segment_text = cleaned_input[item["end"] : next_start]
        days = extract_days_near_destination(segment_text)
        days_inferred = days is None
        if days_inferred:
            days = DEFAULT_TRAVEL_DAYS

        destination = item["destination"]
        note = get_province_route_note(destination)
        if days_inferred:
            inferred_note = f"用户未说明{destination}游玩天数，系统默认按{DEFAULT_TRAVEL_DAYS}天规划。"
            note = f"{inferred_note} {note}".strip()

        segments.append(
            {
                "destination": destination,
                "days": days,
                "nights": max(0, days - 1),
                "days_inferred": days_inferred,
                "preferences": preferences,
                "note": note,
            }
        )

    # 如果只识别出一个目的地，沿用全局天数解析，避免“杭州7日游”被默认覆盖。
    if len(segments) == 1:
        days, nights = extract_trip_days(cleaned_input)
        explicit_days_found = bool(re.search(r"[0-9一二两三四五六七八九十]+\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行|天|日)", cleaned_input))
        segments[0]["days"] = days
        segments[0]["nights"] = nights
        segments[0]["days_inferred"] = not explicit_days_found
        if segments[0]["days_inferred"]:
            inferred_note = f"用户未说明{segments[0]['destination']}游玩天数，系统默认按{DEFAULT_TRAVEL_DAYS}天规划。"
            segments[0]["note"] = f"{inferred_note} {get_province_route_note(segments[0]['destination'])}".strip()

    return segments


def infer_budget_level(cleaned_input: str) -> str:
    """infer_budget_level：识别用户输入中的预算风格档位。"""

    if re.search(r"穷游|省钱|低预算|便宜|学生党", cleaned_input):
        return "经济预算"

    if re.search(r"舒适一点|舒适|舒服|品质|高端|豪华|不差钱|预算充足", cleaned_input):
        return "舒适预算"

    return DEFAULT_BUDGET_LEVEL


def extract_preferences(cleaned_input: str) -> list[str]:
    """extract_preferences：从用户输入中提取旅行偏好、景点和体验主题。"""

    # preference_keywords：可识别的旅行偏好关键词。
    preference_keywords = [
        "西湖",
        "灵隐寺",
        "美食",
        "夜景",
        "自然",
        "动漫",
        "购物",
        "历史",
        "拍照",
        "文化",
        "博物馆",
        "亲子",
        "海边",
        "徒步",
        "温泉",
        "咖啡",
        "艺术",
    ]

    # preferences：从用户输入中识别出的偏好列表。
    preferences = []
    for keyword in preference_keywords:
        if keyword in cleaned_input and keyword not in preferences:
            preferences.append(keyword)

    if not preferences:
        preferences = ["美食", "拍照"]

    return preferences


def extract_fact_check_spots(parsed_request: dict) -> list[str]:
    """extract_fact_check_spots：提取需要联网校验门票、预约和开放时间的景点。"""

    # generic_preferences：不适合作为具体景点搜索的泛偏好。
    generic_preferences = {
        "美食",
        "夜景",
        "自然",
        "动漫",
        "购物",
        "历史",
        "拍照",
        "文化",
        "博物馆",
        "亲子",
        "海边",
        "徒步",
        "温泉",
        "咖啡",
        "艺术",
    }

    # spot_list：从偏好里筛出的具体景点。
    spot_list = [item for item in parsed_request["preferences"] if item not in generic_preferences]

    if not spot_list:
        spot_list = ["主要景点"]

    return spot_list[:4]


def build_fact_search_queries(parsed_request: dict) -> list[str]:
    """build_fact_search_queries：生成省额度的合并搜索查询，避免每个景点单独搜索。"""

    # destination：目的地名称，多目的地时合并为一个省额度查询。
    destination = " ".join(parsed_request.get("destinations") or [parsed_request["destination"]])

    # spot_list：需要查询的景点列表。
    spot_list = extract_fact_check_spots(parsed_request)

    # important_spots：用户明确提到的重点景点，最多放入 4 个，避免 query 过长。
    important_spots = [spot for spot in spot_list if spot != "主要景点"][:4]

    # spot_text：重点景点文本，没有明确景点时使用热门景点兜底。
    spot_text = " ".join(important_spots) if important_spots else "热门景点"

    # query_list：省额度模式下的合并查询列表；默认只会执行第一条。
    query_list = [
        f"{destination} {spot_text} 旅游 景点 门票 预约 开放时间 最新 交通 政策",
        f"{destination} 官方 旅游 景区规则 门票 预约 开放时间 最新政策",
    ]

    return query_list


def get_tavily_api_key() -> str | None:
    """get_tavily_api_key：读取 Tavily API Key，并忽略示例占位值。"""

    # tavily_api_key：从 .env、环境变量或 Streamlit secrets 读取的 Tavily API Key。
    tavily_api_key = get_config_value("TAVILY_API_KEY", "").strip()
    if not tavily_api_key or tavily_api_key.startswith("tvly-your"):
        return None

    return tavily_api_key


def get_deepseek_api_key() -> str | None:
    """get_deepseek_api_key：读取 DeepSeek API Key，并忽略示例占位值。"""

    # deepseek_api_key：从 .env、环境变量或 Streamlit secrets 读取的 DeepSeek API Key。
    deepseek_api_key = get_config_value("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_api_key or deepseek_api_key.startswith("sk-your"):
        return None

    return deepseek_api_key


def get_deepseek_model_name() -> str:
    """get_deepseek_model_name：读取 DeepSeek 模型名，未配置时使用默认模型。"""

    return get_config_value("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def get_tavily_search_depth() -> str:
    """get_tavily_search_depth：读取 Tavily 搜索深度，并强制使用 basic 省额度模式。"""

    # configured_depth：用户配置的搜索深度；为省额度，任何非 basic 配置都会被降级为 basic。
    configured_depth = get_config_value("TAVILY_SEARCH_DEPTH", DEFAULT_TAVILY_SEARCH_DEPTH).strip().lower()
    return DEFAULT_TAVILY_SEARCH_DEPTH if configured_depth != DEFAULT_TAVILY_SEARCH_DEPTH else configured_depth


def get_tavily_max_searches_per_guide() -> int:
    """get_tavily_max_searches_per_guide：读取每份攻略最多搜索次数，并默认限制为 1 次。"""

    # configured_limit：用户配置的每份攻略最大 Tavily 调用次数。
    configured_limit = get_int_config("TAVILY_MAX_SEARCHES_PER_GUIDE", DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE)
    return max(0, min(configured_limit, DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE))


def normalize_tavily_query(destination: str, query: str) -> str:
    """normalize_tavily_query：把目的地和 query 标准化，用于判断相似搜索并命中缓存。"""

    # token_list：从 query 中提取的中英文关键词。
    token_list = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", query.lower())

    # normalized_tokens：去重排序后的关键词，使词序轻微变化时仍可复用缓存。
    normalized_tokens = sorted(set(token_list))

    return f"{destination.strip().lower()}|{'|'.join(normalized_tokens)}"


def build_tavily_cache_key(destination: str, query: str) -> str:
    """build_tavily_cache_key：为目的地和相似 query 生成稳定缓存 key。"""

    # normalized_query：标准化后的 query 文本。
    normalized_query = normalize_tavily_query(destination, query)
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


def load_tavily_cache() -> dict:
    """load_tavily_cache：读取本地 Tavily 缓存文件。"""

    if not TAVILY_CACHE_PATH.exists():
        return {}

    try:
        with TAVILY_CACHE_PATH.open("r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except Exception:
        return {}


def save_tavily_cache(cache_data: dict) -> None:
    """save_tavily_cache：把 Tavily 搜索结果写入本地缓存文件。"""

    with TAVILY_CACHE_PATH.open("w", encoding="utf-8") as cache_file:
        json.dump(cache_data, cache_file, ensure_ascii=False, indent=2)


def get_cached_tavily_results(destination: str, query: str) -> list[dict] | None:
    """get_cached_tavily_results：读取 24 小时内的 Tavily 缓存结果。"""

    # cache_data：本地缓存文件中的全部数据。
    cache_data = load_tavily_cache()

    # cache_key：当前目的地和 query 对应的缓存 key。
    cache_key = build_tavily_cache_key(destination, query)

    # cached_item：缓存中的单条搜索记录。
    cached_item = cache_data.get(cache_key)
    if not cached_item:
        return None

    # cached_at：缓存写入时间戳。
    cached_at = float(cached_item.get("cached_at", 0))
    if time.time() - cached_at > TAVILY_CACHE_TTL_SECONDS:
        return None

    return cached_item.get("results", [])


def set_cached_tavily_results(destination: str, query: str, results: list[dict]) -> None:
    """set_cached_tavily_results：缓存 Tavily 搜索结果，减少重复搜索消耗。"""

    # cache_data：本地缓存文件中的全部数据。
    cache_data = load_tavily_cache()

    # cache_key：当前目的地和 query 对应的缓存 key。
    cache_key = build_tavily_cache_key(destination, query)
    cache_data[cache_key] = {
        "destination": destination,
        "query": query,
        "cached_at": time.time(),
        "cached_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "results": results,
    }
    save_tavily_cache(cache_data)


def is_tavily_limit_error(error: Exception) -> bool:
    """is_tavily_limit_error：判断 Tavily 错误是否属于额度不足或请求受限。"""

    # error_text：错误文本，兼容 SDK 返回的不同异常格式。
    error_text = str(error).lower()
    limit_keywords = ["429", "quota", "rate limit", "ratelimit", "credits", "credit", "insufficient"]
    return any(keyword in error_text for keyword in limit_keywords)


def call_tavily_search(query: str, parsed_request: dict) -> tuple[list[dict], bool]:
    """call_tavily_search：调用 Tavily SDK 搜索，返回结果和是否命中缓存。"""

    # destination：目的地名称，用于缓存 key。
    destination = parsed_request["destination"]

    # cached_results：24 小时内缓存命中的搜索结果。
    cached_results = get_cached_tavily_results(destination, query)
    if cached_results is not None:
        return cached_results, True

    # tavily_api_key：Tavily API Key，从 .env、环境变量或 Streamlit secrets 读取。
    tavily_api_key = get_tavily_api_key()
    if not tavily_api_key or TavilyClient is None:
        return [], False

    # tavily_client：Tavily Python SDK 客户端。
    tavily_client = TavilyClient(api_key=tavily_api_key)

    # response_data：Tavily SDK 搜索返回结果；不启用 answer/raw/images/auto_parameters，控制额度消耗。
    response_data = tavily_client.search(
        query=query,
        search_depth=get_tavily_search_depth(),
        max_results=DEFAULT_SEARCH_MAX_RESULTS,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        auto_parameters=False,
        timeout=12,
    )

    # results：Tavily 搜索结果列表。
    results = response_data.get("results", [])
    set_cached_tavily_results(destination, query, results)
    return results, False


def build_facts_context(parsed_request: dict) -> tuple[str, list[dict], str | None]:
    """build_facts_context：联网搜索并整理 facts_context，供 DeepSeek 生成攻略时引用。"""

    # searched_at：事实校验执行时间。
    searched_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not get_bool_config("USE_TAVILY", True):
        facts_context = f"""
联网事实校验状态：未启用
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：USE_TAVILY=false
更新时间：{searched_at}
页面提示：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
""".strip()
        return facts_context, [], "未启用联网搜索"

    if TavilyClient is None:
        facts_context = f"""
联网事实校验状态：执行失败
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：未安装 tavily-python
更新时间：{searched_at}
页面提示：联网搜索失败，已切换普通模式。
""".strip()
        return facts_context, [], "联网搜索失败，已切换普通模式"

    # tavily_api_key：Tavily API Key，用于判断是否启用搜索。
    tavily_api_key = get_tavily_api_key()
    if not tavily_api_key:
        facts_context = f"""
联网事实校验状态：未配置
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：未配置 TAVILY_API_KEY
更新时间：{searched_at}
页面提示：未配置 Tavily，当前为普通生成模式。
""".strip()
        return facts_context, [], "未配置 Tavily，当前为普通生成模式"

    # max_searches：每份攻略最多 Tavily 调用次数，默认限制为 1 次。
    max_searches = get_tavily_max_searches_per_guide()
    if max_searches <= 0:
        facts_context = f"""
联网事实校验状态：未启用
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：TAVILY_MAX_SEARCHES_PER_GUIDE=0
更新时间：{searched_at}
页面提示：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
""".strip()
        return facts_context, [], "未启用联网搜索"

    # query_list：本次事实校验需要执行的搜索查询。
    query_list = build_fact_search_queries(parsed_request)[:max_searches]

    # source_records：用于展示和传给模型的搜索结果。
    source_records = []

    # used_cache：本次搜索是否命中过本地缓存。
    used_cache = False

    # context_blocks：facts_context 中的文本块。
    context_blocks = [
        "联网事实校验状态：已执行",
        f"更新时间：{searched_at}",
        f"搜索模式：Tavily basic，省额度模式，每份攻略最多 {max_searches} 次搜索。",
        "使用范围：门票、预约规则、开放时间、交通政策等易变化信息只能基于以下搜索结果整理。",
        "注意：根据搜索结果整理，仍需出行前二次确认。",
    ]

    try:
        for query in query_list:
            # results：单个 query 的网页搜索结果。
            results, cache_hit = call_tavily_search(query, parsed_request)
            used_cache = used_cache or cache_hit
            context_blocks.append(f"\n### 搜索查询：{query}")
            context_blocks.append(f"- 结果来源：{'本地 24 小时缓存' if cache_hit else 'Tavily basic 搜索'}")

            if not results:
                context_blocks.append("- 未查到可用结果。")
                continue

            for result in results:
                # title：搜索结果标题。
                title = result.get("title", "未命名来源")

                # url：搜索结果链接。
                url = result.get("url", "")

                # content：搜索结果摘要内容。
                content = result.get("content", "") or result.get("snippet", "")
                content = re.sub(r"\s+", " ", content).strip()

                source_records.append({"query": query, "title": title, "url": url, "content": content})
                context_blocks.append(f"- 标题：{title}\n  链接：{url}\n  摘要：{content[:360]}")
    except Exception as error:
        if is_tavily_limit_error(error):
            facts_context = f"""
联网事实校验状态：额度不足
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
更新时间：{searched_at}
错误：{error}
要求：门票、预约、开放时间、交通政策等易变化信息不可编造；请写“建议出行前二次确认”。
""".strip()
            return facts_context, source_records, "Tavily 额度不足，已切换普通模式"

        facts_context = f"""
联网事实校验状态：执行失败
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
更新时间：{searched_at}
错误：{error}
要求：门票、预约、开放时间、交通政策等易变化信息不可编造；如果 facts_context 没有查到，请写“建议出行前再次核对”。
""".strip()
        return facts_context, source_records, "联网搜索失败，已切换普通模式"

    if not source_records:
        context_blocks.append("\n结论：未查到足够搜索结果。不要编造门票、预约、开放时间，请提示建议出行前再次核对。")

    if used_cache:
        context_blocks[0] = "联网事实校验状态：缓存命中"
        return "\n".join(context_blocks), source_records, "使用缓存搜索结果"

    return "\n".join(context_blocks), source_records, "已启用联网搜索"


def parse_travel_request(user_input: str) -> dict:
    """parse_travel_request：从用户的一句话里提取目的地、天数、预算和偏好。"""

    # cleaned_input：去掉多余空格后的用户输入。
    cleaned_input = user_input.strip()

    # budget_level：识别到的预算档位。
    budget_level = infer_budget_level(cleaned_input)

    # budget_info：识别到的预算金额、货币单位和展示文本。
    budget_info = parse_budget_info(cleaned_input, budget_level)

    # preferences：从用户输入中识别出的偏好列表。
    preferences = extract_preferences(cleaned_input)

    # trip_segments：单目的地或多目的地分段行程。
    trip_segments = extract_trip_segments(cleaned_input, preferences)

    # destination_list：所有识别到的目的地。
    destination_list = [segment["destination"] for segment in trip_segments]

    # destination：用于页面总览展示的目的地文本。
    destination = " + ".join(destination_list)

    # days：总旅行天数，多目的地时为各段天数之和。
    days = sum(segment["days"] for segment in trip_segments)

    # nights：总住宿晚数，按连续旅行粗略估算。
    nights = max(0, days - 1)

    # nights_match：如果用户明确写了总住宿晚数，则尊重用户输入。
    nights_match = re.search(r"([0-9一二两三四五六七八九十]+)\s*晚", cleaned_input)
    if nights_match:
        nights = max(0, parse_chinese_number(nights_match.group(1)))

    # trip_type：单目的地或多目的地。
    trip_type = "multi_destination" if len(trip_segments) > 1 else "single_destination"

    # trip_notes：多目的地或省份默认推断提示。
    trip_notes = [segment["note"] for segment in trip_segments if segment.get("note")]

    # parsed_request：最终返回给页面和大模型的结构化旅行需求。
    parsed_request = {
        "trip_type": trip_type,
        "destination": destination,
        "destinations": destination_list,
        "trip_segments": trip_segments,
        "trip_notes": trip_notes,
        "days": days,
        "nights": nights,
        "budget": budget_info["display"],
        "budget_level": budget_info["level"],
        "style": budget_info["level"].replace("预算", ""),
        "budget_amount": budget_info["amount"],
        "budget_currency": budget_info["currency"],
        "budget_currency_name": budget_info["currency_name"],
        "budget_has_explicit_amount": budget_info["has_explicit_amount"],
        "preferences": preferences,
    }

    # budget_exchange_hint：国外目的地的粗略换算提示。
    parsed_request["budget_exchange_hint"] = build_exchange_hint(parsed_request)

    return parsed_request


def build_ai_prompt(user_input: str, parsed_request: dict, facts_context: str) -> str:
    """build_ai_prompt：把用户输入、解析结果和联网事实上下文整理成给大模型的提示词。"""

    # preferences_text：把偏好列表合并成适合模型阅读的字符串。
    preferences_text = "、".join(parsed_request["preferences"])

    # exchange_hint_text：国外目的地预算粗略换算提示。
    exchange_hint_text = parsed_request.get("budget_exchange_hint") or "无"

    # budget_currency_text：预算货币单位说明。
    budget_currency_text = (
        f"{parsed_request['budget_currency_name']} ({parsed_request['budget_currency']})"
        if parsed_request.get("budget_currency")
        else "未指定具体金额"
    )

    # search_enabled：是否拿到了可用于事实校验的联网或缓存结果。
    search_enabled = "联网事实校验状态：已执行" in facts_context or "联网事实校验状态：缓存命中" in facts_context

    if search_enabled:
        # fact_rules：启用联网搜索时，对易变化信息使用 facts_context 的强约束规则。
        fact_rules = """
16. 门票、预约规则、开放时间、景区政策、交通政策等易变化信息，必须优先依据 facts_context 写。
17. 如果 facts_context 没有明确说明对应信息，不能编造，请写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。
18. 对门票、预约、开放时间这类信息，必须标注“根据搜索结果整理，仍需出行前二次确认”。
19. 必须增加“信息来源与更新时间”区域，列出来源标题、链接和更新时间；如果搜索结果没有覆盖某项信息，也要说明未查到。
20. 避坑提醒要明确、实用。
21. 必须包含以下二级标题，并保持标题文字完全一致：
""".strip()
    else:
        # fact_rules：普通生成模式下不使用联网结果，但提醒用户二次确认易变化信息。
        fact_rules = """
16. 当前未启用联网搜索，请按普通生成模式输出攻略。
17. DeepSeek 不能编造最新门票、预约规则、开放时间、景区政策或交通政策。
18. 对所有可能变化的信息，必须写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。
19. 必须增加“信息来源与更新时间”区域，并写明：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
20. 避坑提醒要明确、实用。
21. 必须包含以下二级标题，并保持标题文字完全一致：
""".strip()

    return f"""
用户原始需求：
{user_input}

联网事实校验 facts_context：
{facts_context}

系统已识别：
- 目的地：{parsed_request["destination"]}
- 目的地分段：{json.dumps(parsed_request.get("trip_segments", []), ensure_ascii=False)}
- 旅行天数：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 预算单位：{budget_currency_text}
- 预算换算提示：{exchange_hint_text}
- 偏好：{preferences_text}

请生成一份中文旅行攻略，要求：
1. 使用 Markdown。
2. 内容具体、可执行，不要泛泛而谈。
3. 每日行程必须严格生成 {parsed_request["days"]} 天，从 Day 1 到 Day {parsed_request["days"]}，不能少生成，也不能只生成 3 天。
4. 每一天必须有不同主题，标题格式必须是“### Day 1：主题名”，例如“### Day 1：西湖经典路线”。
5. 用户明确提到的景点和偏好必须优先安排：{preferences_text}。
6. 如果用户提到的景点不足以填满全部天数，请根据目的地、偏好、预算和 facts_context 补充适合景点；搜索结果不足时请写“不确定，请出行前核对”，不要编造实时事实。
7. 每天必须包含“上午”“中午”“下午”“晚上”四个时间段。
8. 每个时间段必须写成：- 上午：具体地点｜推荐理由｜预计耗时｜交通或预约提醒。
9. 不要反复使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
10. 同一景点、同一餐厅、同一区域不要重复出现，除非用户明确要求。
11. 美食推荐建议写成：- 名称：推荐理由｜人均预算｜适合场景。
12. 预算估算要分交通、住宿、餐饮、门票体验、机动费用，并明确预算单位。
13. 如果用户输入了预算数字但没有货币单位，必须按人民币 CNY 理解，不要按目的地当地货币理解。
14. 如果用户明确写了美元、日元、欧元、韩元或 USD/EUR/JPY/KRW，必须尊重用户输入的货币单位。
15. 如果存在“预算换算提示”，请在预算估算中补充这条提示，并说明汇率仅供参考。
{fact_rules}
## 旅行封面文案
## 详细旅游攻略
## 每日行程
## 美食推荐
## 交通建议
## 预算估算
## 避坑提醒
## 信息来源与更新时间
""".strip()


def call_deepseek_chat(prompt: str, instructions: str) -> tuple[str | None, str | None]:
    """call_deepseek_chat：执行一次 DeepSeek Chat Completions 调用并返回文本。"""

    if OpenAI is None:
        return None, "没有安装 openai 依赖，已使用本地演示攻略。"

    # api_key：DeepSeek API Key，从 .env、环境变量或 Streamlit secrets 读取。
    api_key = get_deepseek_api_key()
    if not api_key:
        return None, "未配置 DEEPSEEK_API_KEY，已使用本地演示攻略。"

    # model_name：当前使用的 DeepSeek 模型名称，可通过 DEEPSEEK_MODEL 修改。
    model_name = get_deepseek_model_name()

    try:
        # client：OpenAI SDK 客户端，通过 base_url 指向 DeepSeek 服务。
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

        # response：DeepSeek Chat Completions API 返回的大模型结果。
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        )

        # response_text：从模型回复中取出的文本。
        response_text = response.choices[0].message.content
        if not response_text:
            return None, "DeepSeek 返回内容为空，已使用本地演示攻略。"

        return response_text.strip(), None
    except Exception as error:
        return None, f"DeepSeek API 调用失败，已使用本地演示攻略。错误：{error}"


def build_structured_json_prompt(user_input: str, parsed_request: dict, facts_context: str) -> str:
    """build_structured_json_prompt：生成结构化旅行 JSON 的 DeepSeek 提示词。"""

    # preferences_json：用户偏好 JSON 文本，确保用户输入优先。
    preferences_json = json.dumps(parsed_request["preferences"], ensure_ascii=False)

    # trip_segments_json：分段目的地 JSON 文本，确保多目的地不被忽略。
    trip_segments_json = json.dumps(parsed_request.get("trip_segments", []), ensure_ascii=False, indent=2)

    # search_enabled：是否有可用于事实校验的 Tavily 搜索结果。
    search_enabled = "联网事实校验状态：已执行" in facts_context or "联网事实校验状态：缓存命中" in facts_context

    # fact_rule_text：联网事实约束文本。
    fact_rule_text = (
        "门票、预约、开放时间、景区政策、交通政策必须优先依据 facts_context；如果 facts_context 没有明确说明，不要编造，写“建议出行前二次确认”。"
        if search_enabled
        else "当前未启用联网搜索，不能编造最新门票、预约、开放时间、景区政策或交通政策；必须写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。"
    )

    return f"""
用户原始需求：
{user_input}

联网事实校验 facts_context：
{facts_context}

系统已识别参数，必须严格使用，不能被模型猜测或默认值覆盖：
- destination: {parsed_request["destination"]}
- trip_type: {parsed_request.get("trip_type", "single_destination")}
- destinations: {json.dumps(parsed_request.get("destinations", [parsed_request["destination"]]), ensure_ascii=False)}
- trip_segments: {trip_segments_json}
- days: {parsed_request["days"]}
- nights: {parsed_request["nights"]}
- budget_amount: {parsed_request.get("budget_amount")}
- currency: {parsed_request.get("budget_currency") or "CNY"}
- budget_level: {parsed_request.get("style") or parsed_request.get("budget_level")}
- preferences: {preferences_json}

请只返回合法 JSON，不要输出 Markdown，不要解释，不要使用代码块。
JSON 顶层必须包含：
trip_type, destination, destinations, total_days, days, nights, budget, preferences, trip_segments, daily_itinerary, food_recommendations

budget 必须包含：
amount, currency, level

trip_segments 必须和系统识别参数一致。
每个 trip_segments 对象必须包含：
destination, days, nights, days_inferred, theme, note

daily_itinerary 必须 exactly {parsed_request["days"]} 天，从 day 1 到 day {parsed_request["days"]}。
每天必须包含：
day, segment_destination, theme, morning, noon, afternoon, evening

morning/noon/afternoon/evening 每个对象必须包含以下非空字段：
time, place, original_name, reason, duration, transport, booking_note

food_recommendations 必须包含 4-6 家店或小吃点。
每个美食推荐对象必须包含以下非空字段：
name_cn, name_original, location, nearby_spot, reason, budget, scene, booking_note, map_keyword

写作规则：
1. 用户明确提到的景点和偏好必须优先安排：{preferences_json}。
2. 用户明确提到的所有目的地必须出现在 trip_segments 和 daily_itinerary 中，不允许只生成第一个目的地。
3. 如果 days_inferred=true，必须在 note 中说明“用户未说明该目的地天数，系统默认按3天规划”。
4. 如果目的地是省份、国家或大区域，不要泛泛写省名/国家名；必须推荐具体城市路线，并在 note 中说明推断依据。
5. 多目的地 daily_itinerary 的 day 必须全程连续编号，不能每段都从 Day 1 重新开始。
6. 每一天主题必须不同，不能重复。
7. 同一景点、同一餐厅、同一区域不要重复安排。
8. 不要使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
9. place 必须是具体地点，original_name 必须包含中文名和英文/原名；没有英文名时写中文原名。
10. reason、transport、booking_note 必须具体，不能空泛。
11. 美食推荐如果是国外目的地，name_original 必须尽量保留英文名、当地语言原名或常用地图搜索名。
12. 如果无法确认具体地址，不要编造门牌号；location 可以写“市中心区域”“靠近某某景点”“建议以 Google Maps 搜索原名确认”。
13. map_keyword 必须适合复制到 Google Maps / Apple Maps / 百度地图 / 高德地图搜索。
14. {fact_rule_text}
""".strip()


def extract_json_text(model_output: str) -> str:
    """extract_json_text：从模型输出中提取 JSON 文本，兼容代码块和前后解释。"""

    # fenced_match：匹配 ```json 代码块中的 JSON。
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", model_output, flags=re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    # start_index/end_index：提取第一个对象到最后一个对象之间的内容。
    start_index = model_output.find("{")
    end_index = model_output.rfind("}")
    if start_index >= 0 and end_index > start_index:
        return model_output[start_index : end_index + 1].strip()

    return model_output.strip()


def parse_structured_json_output(model_output: str | None) -> tuple[dict | None, list[str]]:
    """parse_structured_json_output：把模型原始输出解析为 JSON 对象。"""

    if not model_output:
        return None, ["模型没有返回 JSON 内容。"]

    # json_text：提取后的 JSON 文本。
    json_text = extract_json_text(model_output)
    try:
        # parsed_json：解析后的 JSON 对象。
        parsed_json = json.loads(json_text)
    except json.JSONDecodeError as error:
        return None, [f"JSON 解析失败：{error}"]

    if not isinstance(parsed_json, dict):
        return None, ["JSON 顶层必须是对象。"]

    return parsed_json, []


def validate_structured_travel_json(travel_json: dict | None, parsed_request: dict) -> list[str]:
    """validate_structured_travel_json：校验每日行程 JSON 是否完整、准确、不重复。"""

    if not isinstance(travel_json, dict):
        return ["结构化结果不是 JSON 对象。"]

    # validation_errors：JSON 校验错误列表。
    validation_errors = []

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    if travel_json.get("destination") != parsed_request["destination"]:
        validation_errors.append(f"destination 不一致，应为 {parsed_request['destination']}。")

    if travel_json.get("days") != expected_days:
        validation_errors.append(f"days 不一致，应为 {expected_days}。")

    if travel_json.get("total_days") is not None and travel_json.get("total_days") != expected_days:
        validation_errors.append(f"total_days 不一致，应为 {expected_days}。")

    if travel_json.get("nights") != parsed_request["nights"]:
        validation_errors.append(f"nights 不一致，应为 {parsed_request['nights']}。")

    # expected_destinations：系统识别出的所有目的地。
    expected_destinations = parsed_request.get("destinations") or [parsed_request["destination"]]

    # destination_json：模型返回的目的地数组。
    destination_json = travel_json.get("destinations")
    if isinstance(destination_json, list):
        missing_destinations = [destination for destination in expected_destinations if destination not in destination_json]
        if missing_destinations:
            validation_errors.append(f"destinations 缺少目的地：{'、'.join(missing_destinations)}。")

    # trip_segments_json：模型返回的分段行程。
    trip_segments_json = travel_json.get("trip_segments")
    if not isinstance(trip_segments_json, list):
        validation_errors.append("trip_segments 必须是数组。")
    else:
        segment_map = {segment.get("destination"): segment for segment in trip_segments_json if isinstance(segment, dict)}
        for expected_segment in parsed_request.get("trip_segments", []):
            destination = expected_segment["destination"]
            segment = segment_map.get(destination)
            if not segment:
                validation_errors.append(f"trip_segments 缺少目的地：{destination}。")
                continue
            if segment.get("days") != expected_segment["days"]:
                validation_errors.append(f"{destination} days 不一致，应为 {expected_segment['days']}。")
            if segment.get("nights") != expected_segment["nights"]:
                validation_errors.append(f"{destination} nights 不一致，应为 {expected_segment['nights']}。")
            if "days_inferred" not in segment:
                validation_errors.append(f"{destination} 缺少 days_inferred 字段。")
            if not str(segment.get("theme", "")).strip():
                validation_errors.append(f"{destination} theme 不能为空。")
            if expected_segment.get("days_inferred") and not str(segment.get("note", "")).strip():
                validation_errors.append(f"{destination} 是默认推断天数，note 不能为空。")

    # budget_json：模型返回的预算对象。
    budget_json = travel_json.get("budget")
    if not isinstance(budget_json, dict):
        validation_errors.append("budget 必须是对象。")
    else:
        if parsed_request.get("budget_has_explicit_amount") and budget_json.get("amount") != parsed_request.get("budget_amount"):
            validation_errors.append(f"budget.amount 不一致，应为 {parsed_request.get('budget_amount')}。")
        if parsed_request.get("budget_currency") and budget_json.get("currency") != parsed_request.get("budget_currency"):
            validation_errors.append(f"budget.currency 不一致，应为 {parsed_request.get('budget_currency')}。")
        if not str(budget_json.get("level", "")).strip():
            validation_errors.append("budget.level 不能为空。")

    # preferences_json：模型返回的偏好列表。
    preferences_json = travel_json.get("preferences")
    if not isinstance(preferences_json, list):
        validation_errors.append("preferences 必须是数组。")
    else:
        missing_preferences = [preference for preference in parsed_request["preferences"] if preference not in preferences_json]
        if missing_preferences:
            validation_errors.append(f"preferences 缺少用户明确偏好：{'、'.join(missing_preferences)}。")

    # daily_itinerary：模型返回的每日行程数组。
    daily_itinerary = travel_json.get("daily_itinerary")
    if not isinstance(daily_itinerary, list):
        return validation_errors + ["daily_itinerary 必须是数组。"]

    if len(daily_itinerary) != expected_days:
        validation_errors.append(f"daily_itinerary 必须 exactly {expected_days} 天，当前为 {len(daily_itinerary)} 天。")

    # required_slots：每天必须包含的四个时间段。
    required_slots = ["morning", "noon", "afternoon", "evening"]

    # required_slot_fields：每个时间段对象必须包含的字段。
    required_slot_fields = ["time", "place", "original_name", "reason", "duration", "transport", "booking_note"]

    # generic_terms：不允许出现的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # seen_days/themes/places/segment_destinations：用于检查编号、主题、地点和目的地覆盖。
    seen_days = set()
    seen_themes = set()
    seen_places = set()
    seen_segment_destinations = set()

    for day_index, day_item in enumerate(daily_itinerary, start=1):
        if not isinstance(day_item, dict):
            validation_errors.append(f"Day {day_index} 必须是对象。")
            continue

        day_number = day_item.get("day")
        if day_number != day_index:
            validation_errors.append(f"Day {day_index} 的 day 字段应为 {day_index}，当前为 {day_number}。")
        if day_number in seen_days:
            validation_errors.append(f"Day 编号重复：{day_number}。")
        seen_days.add(day_number)

        segment_destination = str(day_item.get("segment_destination", "")).strip()
        if not segment_destination:
            validation_errors.append(f"Day {day_index} segment_destination 不能为空。")
        elif segment_destination not in expected_destinations:
            validation_errors.append(f"Day {day_index} segment_destination 不在用户目的地中：{segment_destination}。")
        seen_segment_destinations.add(segment_destination)

        theme = str(day_item.get("theme", "")).strip()
        if not theme:
            validation_errors.append(f"Day {day_index} theme 不能为空。")
        elif theme in seen_themes:
            validation_errors.append(f"Day {day_index} theme 重复：{theme}。")
        seen_themes.add(theme)

        for slot_name in required_slots:
            # slot_data：单个时间段对象。
            slot_data = day_item.get(slot_name)
            if not isinstance(slot_data, dict):
                validation_errors.append(f"Day {day_index} 缺少 {slot_name} 对象。")
                continue

            for field_name in required_slot_fields:
                field_value = slot_data.get(field_name)
                if field_value is None or not str(field_value).strip():
                    validation_errors.append(f"Day {day_index} {slot_name}.{field_name} 不能为空。")

            slot_text = " ".join(str(slot_data.get(field_name, "")) for field_name in required_slot_fields)
            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_index} {slot_name} 包含空泛模板词。")

            place = str(slot_data.get("place", "")).strip()
            if place:
                if place in seen_places:
                    validation_errors.append(f"重复安排地点：{place}。")
                seen_places.add(place)

    missing_days = [day for day in range(1, expected_days + 1) if day not in seen_days]
    if missing_days:
        validation_errors.append(f"daily_itinerary 缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    missing_segment_destinations = [
        destination for destination in expected_destinations if destination not in seen_segment_destinations
    ]
    if missing_segment_destinations:
        validation_errors.append(f"daily_itinerary 未覆盖目的地：{'、'.join(missing_segment_destinations)}。")

    # food_recommendations：模型返回的美食推荐数组。
    food_recommendations = travel_json.get("food_recommendations")
    if not isinstance(food_recommendations, list):
        validation_errors.append("food_recommendations 必须是数组。")
    elif not food_recommendations:
        validation_errors.append("food_recommendations 不能为空。")
    else:
        # required_food_fields：每个美食推荐对象必须包含的字段。
        required_food_fields = [
            "name_cn",
            "name_original",
            "location",
            "nearby_spot",
            "reason",
            "budget",
            "scene",
            "booking_note",
            "map_keyword",
        ]

        # seen_food_names：用于检查店铺名称是否重复。
        seen_food_names = set()
        for food_index, food_item in enumerate(food_recommendations, start=1):
            if not isinstance(food_item, dict):
                validation_errors.append(f"food_recommendations 第 {food_index} 项必须是对象。")
                continue

            for field_name in required_food_fields:
                field_value = food_item.get(field_name)
                if field_value is None or not str(field_value).strip():
                    validation_errors.append(f"food_recommendations 第 {food_index} 项 {field_name} 不能为空。")

            food_name_key = f"{food_item.get('name_cn', '')}|{food_item.get('name_original', '')}".strip()
            if food_name_key in seen_food_names:
                validation_errors.append(f"重复推荐店铺：{food_item.get('name_cn', '')}。")
            seen_food_names.add(food_name_key)

    return validation_errors


def normalize_structured_travel_json(travel_json: dict, parsed_request: dict) -> dict:
    """normalize_structured_travel_json：用系统识别参数覆盖 JSON 顶层关键字段，保证用户输入优先。"""

    # normalized_json：复制后的结构化结果。
    normalized_json = dict(travel_json)
    normalized_json["trip_type"] = parsed_request.get("trip_type", "single_destination")
    normalized_json["destination"] = parsed_request["destination"]
    normalized_json["destinations"] = parsed_request.get("destinations", [parsed_request["destination"]])
    normalized_json["total_days"] = parsed_request["days"]
    normalized_json["days"] = parsed_request["days"]
    normalized_json["nights"] = parsed_request["nights"]
    normalized_json["preferences"] = parsed_request["preferences"]
    normalized_json["trip_segments"] = [
        {
            "destination": segment["destination"],
            "days": segment["days"],
            "nights": segment["nights"],
            "days_inferred": segment.get("days_inferred", False),
            "theme": segment.get("theme") or get_province_route_note(segment["destination"]) or f"{segment['destination']}分段行程",
            "note": segment.get("note", ""),
        }
        for segment in parsed_request.get("trip_segments", [])
    ]
    normalized_json["budget"] = {
        "amount": parsed_request.get("budget_amount"),
        "currency": parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY,
        "level": parsed_request.get("style") or parsed_request.get("budget_level"),
    }
    return normalized_json


def build_json_repair_prompt(
    original_output: str,
    validation_errors: list[str],
    parsed_request: dict,
    facts_context: str,
) -> str:
    """build_json_repair_prompt：根据校验错误要求 DeepSeek 只修复 JSON。"""

    # error_text：校验错误说明。
    error_text = "\n".join(f"- {error}" for error in validation_errors)

    return f"""
你刚才返回的每日行程 JSON 没有通过校验，错误如下：
{error_text}

请基于原始内容修复 JSON。
必须返回合法 JSON。
必须包含 exactly {parsed_request["days"]} 天。
必须包含 trip_type/destination/destinations/total_days/trip_segments/daily_itinerary。
trip_segments 必须等于系统识别分段：
{json.dumps(parsed_request.get("trip_segments", []), ensure_ascii=False, indent=2)}
daily_itinerary 每天必须包含 segment_destination，且必须覆盖所有目的地。
每天必须包含 morning/noon/afternoon/evening。
每个时间段必须包含 time/place/original_name/reason/duration/transport/booking_note。
food_recommendations 必须包含 4-6 项。
每个美食推荐必须包含 name_cn/name_original/location/nearby_spot/reason/budget/scene/booking_note/map_keyword。
必须保留系统识别参数：
- destination: {parsed_request["destination"]}
- destinations: {json.dumps(parsed_request.get("destinations", [parsed_request["destination"]]), ensure_ascii=False)}
- total_days: {parsed_request["days"]}
- days: {parsed_request["days"]}
- nights: {parsed_request["nights"]}
- budget_amount: {parsed_request.get("budget_amount")}
- currency: {parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY}
- budget_level: {parsed_request.get("style") or parsed_request.get("budget_level")}
- preferences: {json.dumps(parsed_request["preferences"], ensure_ascii=False)}

联网事实校验 facts_context：
{facts_context}

原始输出：
{original_output}

不要输出 Markdown，不要解释，只返回 JSON。
""".strip()


def call_deepseek_structured_json_api(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
) -> tuple[dict | None, str | None, list[str], str | None]:
    """call_deepseek_structured_json_api：生成并校验结构化旅行 JSON，失败时自动修复一次。"""

    # instructions：要求模型只返回 JSON 的系统提示。
    instructions = """
你是旅行规划结构化数据生成器。只返回合法 JSON，不输出 Markdown，不解释。
所有用户明确输入的目的地、天数、预算、货币和偏好优先级最高。
不要编造门票、预约、开放时间、景区政策等实时信息。
""".strip()

    # json_prompt：第一次生成结构化 JSON 的提示词。
    json_prompt = build_structured_json_prompt(user_input, parsed_request, facts_context)

    # raw_output：第一次模型原始输出。
    raw_output, api_message = call_deepseek_chat(json_prompt, instructions)
    if not raw_output:
        return None, raw_output, [api_message or "DeepSeek 没有返回结构化 JSON。"], api_message

    # travel_json：第一次解析出的 JSON。
    travel_json, parse_errors = parse_structured_json_output(raw_output)
    validation_errors = parse_errors or validate_structured_travel_json(travel_json, parsed_request)
    if not validation_errors and travel_json:
        return normalize_structured_travel_json(travel_json, parsed_request), raw_output, [], None

    # repair_prompt：JSON 校验失败后的修复提示。
    repair_prompt = build_json_repair_prompt(raw_output, validation_errors, parsed_request, facts_context)

    # repaired_output：修复后的模型原始输出。
    repaired_output, repair_message = call_deepseek_chat(repair_prompt, instructions)
    if not repaired_output:
        return None, raw_output, validation_errors + [repair_message or "DeepSeek JSON 修复没有返回内容。"], repair_message

    # repaired_json：修复后解析出的 JSON。
    repaired_json, repair_parse_errors = parse_structured_json_output(repaired_output)
    repair_errors = repair_parse_errors or validate_structured_travel_json(repaired_json, parsed_request)
    if repair_errors:
        return None, repaired_output, repair_errors, "每日行程 JSON 修复后仍未通过校验。"

    return normalize_structured_travel_json(repaired_json, parsed_request), repaired_output, [], "每日行程 JSON 第一次未通过校验，已自动修复。"


def build_markdown_from_json_prompt(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
    travel_json: dict,
) -> str:
    """build_markdown_from_json_prompt：基于结构化 JSON 生成 Markdown 攻略提示词。"""

    # structured_json_text：结构化旅行 JSON 文本。
    structured_json_text = json.dumps(travel_json, ensure_ascii=False, indent=2)

    return f"""
用户原始需求：
{user_input}

系统识别参数：
- 目的地：{parsed_request["destination"]}
- 旅行天数：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 偏好：{"、".join(parsed_request["preferences"])}

联网事实校验 facts_context：
{facts_context}

结构化 JSON：
{structured_json_text}

请基于上面的结构化 JSON 生成中文 Markdown 攻略。
要求：
1. 不要改变 JSON 中的目的地、天数、预算、偏好和 daily_itinerary。
2. 用户明确提到的所有目的地都必须出现在攻略中，不能只写第一个目的地。
3. 如果是多目的地，先写总览，例如“南京 3 天 + 江西默认 3 天”，再按分段展示。
4. 每日行程必须按 JSON 中的 daily_itinerary 写，不要自由新增重复路线；Day 编号必须连续。
5. 如果 trip_segments 中 days_inferred=true，必须明确说明该目的地天数是系统默认推断。
6. 如果目的地是省份、国家或大区域，必须说明系统选择了具体城市路线。
7. 门票、预约、开放时间、景区政策必须优先依据 facts_context；没有明确搜索结果时写“建议出行前二次确认”。
8. 内容具体、可执行，保留中文名 + 英文/原名。
9. “美食推荐”必须基于 JSON 中的 food_recommendations，每条必须写店名中文名、英文/当地原名、位置、附近景点/区域、人均预算、适合场景、预约提示和地图搜索关键词。
10. 如果无法确认具体地址，不要编造门牌号；写“市中心区域”“靠近某某景点”或“建议以 Google Maps 搜索原名确认”。
11. 必须包含以下二级标题，并保持标题文字完全一致：
## 旅行封面文案
## 详细旅游攻略
## 每日行程
## 美食推荐
## 交通建议
## 预算估算
## 避坑提醒
## 信息来源与更新时间
""".strip()


def build_markdown_from_structured_json(travel_json: dict, parsed_request: dict, facts_context: str) -> str:
    """build_markdown_from_structured_json：当 Markdown 二次生成失败时，用合格 JSON 生成可复制攻略。"""

    # source_updated_at：攻略信息更新时间。
    source_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # itinerary_lines：从结构化 JSON 生成的每日行程 Markdown。
    itinerary_lines = []
    for day_item in travel_json.get("daily_itinerary", []):
        segment_prefix = f"{day_item.get('segment_destination', parsed_request['destination'])}｜"
        itinerary_lines.append(f"### Day {day_item['day']}：{segment_prefix}{day_item['theme']}")
        for slot_key, slot_label in [("morning", "上午"), ("noon", "中午"), ("afternoon", "下午"), ("evening", "晚上")]:
            slot = day_item[slot_key]
            itinerary_lines.append(
                f"- {slot_label}：{slot['place']}｜{slot['reason']}｜{slot['duration']}｜{slot['transport']}；{slot['booking_note']}"
            )

    # food_lines：从结构化 JSON 生成的美食推荐 Markdown。
    food_lines = []
    for food_item in travel_json.get("food_recommendations", []):
        food_title = food_item["name_cn"]
        if food_item.get("name_original") and food_item["name_original"] != food_title:
            food_title = f"{food_title}（{food_item['name_original']}）"
        food_lines.append(
            f"- {food_title}：位置 {food_item['location']}，靠近 {food_item['nearby_spot']}｜"
            f"{food_item['reason']}｜{food_item['budget']}｜{food_item['scene']}｜"
            f"{food_item['booking_note']}｜地图搜索：{food_item['map_keyword']}"
        )

    # preferences_text：用户旅行偏好。
    preferences_text = "、".join(parsed_request["preferences"])

    # segment_overview_text：多目的地分段总览。
    segment_overview_text = " + ".join(
        f"{segment['destination']}{'默认' if segment.get('days_inferred') else ''}{segment['days']}天"
        for segment in parsed_request.get("trip_segments", [])
    )

    # segment_note_lines：多目的地或省份推断说明。
    segment_note_lines = "\n".join(f"- {note}" for note in parsed_request.get("trip_notes", []))

    return f"""
## 旅行封面文案
{parsed_request["destination"]} {parsed_request["days"]} 天 {parsed_request["nights"]} 晚旅行计划：围绕 {preferences_text} 安排路线。

## 详细旅游攻略
- 目的地：{parsed_request["destination"]}
- 行程总览：{segment_overview_text or parsed_request["destination"]}
- 行程长度：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 旅行风格：{preferences_text}
- 说明：本 Markdown 根据已通过校验的结构化 JSON 生成。
{segment_note_lines}

## 每日行程
{chr(10).join(itinerary_lines)}

## 美食推荐
{chr(10).join(food_lines) if food_lines else f"- 请结合每日路线选择附近餐厅，热门餐厅建议提前预约或取号｜人均预算按 {parsed_request['budget']} 控制｜适合午餐和晚餐"}

## 交通建议
- 每天优先围绕同一区域规划，减少跨区往返。
- 景区门票、预约和开放时间建议出行前二次确认。

## 预算估算
- 用户预算：{parsed_request["budget"]}
- 交通、住宿、餐饮、门票体验和机动费用建议按实际日期二次核算。

## 避坑提醒
- 不要把热门景点、热门餐厅和远距离交通挤在同一天。
- 对所有可能变化的信息，建议出行前二次确认。

## 信息来源与更新时间
- 更新时间：{source_updated_at}
- 来源说明：结构化 JSON 已通过程序校验；门票、预约、开放时间等仍需出行前二次确认。
- 联网事实状态：{facts_context.splitlines()[0] if facts_context else "未启用联网搜索"}
""".strip()


def extract_markdown_section_text(markdown_text: str, heading: str) -> str:
    """extract_markdown_section_text：从 Markdown 中提取指定二级标题下的正文。"""

    # normalized_markdown：保证开头有换行，便于正则匹配二级标题。
    normalized_markdown = "\n" + markdown_text.strip()

    # section_pattern：匹配指定二级标题到下一个二级标题之间的内容。
    section_pattern = rf"\n##\s+{re.escape(heading)}\s*\n([\s\S]*?)(?=\n##\s+|\Z)"
    match = re.search(section_pattern, normalized_markdown)
    return match.group(1).strip() if match else ""


def parse_itinerary_day_blocks(markdown_text: str) -> list[dict]:
    """parse_itinerary_day_blocks：解析 Markdown 每日行程区中的 Day 块。"""

    # itinerary_text：每日行程区域 Markdown。
    itinerary_text = extract_markdown_section_text(markdown_text, "每日行程")
    if not itinerary_text:
        return []

    # day_matches：匹配 Day 标题。
    day_matches = list(re.finditer(r"(?m)^###\s*Day\s*([0-9一二两三四五六七八九十]+)\s*[：:]?\s*(.*?)\s*$", itinerary_text))

    # day_blocks：解析后的 Day 数据。
    day_blocks = []
    for index, match in enumerate(day_matches):
        start_index = match.end()
        end_index = day_matches[index + 1].start() if index + 1 < len(day_matches) else len(itinerary_text)
        day_number = parse_chinese_number(match.group(1))
        theme = match.group(2).strip() or f"Day {day_number}"
        body = itinerary_text[start_index:end_index].strip()
        day_blocks.append({"day": day_number, "theme": theme, "body": body})

    return day_blocks


def validate_itinerary_markdown(markdown_text: str, parsed_request: dict) -> list[str]:
    """validate_itinerary_markdown：校验模型生成的每日行程是否满足不重复和天数要求。"""

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    # day_blocks：模型输出中的 Day 块。
    day_blocks = parse_itinerary_day_blocks(markdown_text)

    # validation_errors：行程校验错误列表。
    validation_errors = []

    if len(day_blocks) < expected_days:
        validation_errors.append(f"每日行程只生成了 {len(day_blocks)} 天，用户需要 {expected_days} 天。")
    elif len(day_blocks) > expected_days:
        validation_errors.append(f"每日行程生成了 {len(day_blocks)} 天，用户只需要 {expected_days} 天。")

    # day_number_set：实际出现的 Day 编号集合。
    day_number_set = {day_block["day"] for day_block in day_blocks}
    missing_days = [day for day in range(1, expected_days + 1) if day not in day_number_set]
    if missing_days:
        validation_errors.append(f"缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    # required_slots：每天必须包含的四个时间段。
    required_slots = ["上午", "中午", "下午", "晚上"]

    # generic_terms：不允许反复出现的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # signature_set：用于检查整天内容是否重复。
    signature_set = set()

    # theme_set：用于检查每天主题是否重复。
    theme_set = set()

    # seen_places：用于检查具体地点是否重复安排。
    seen_places = set()

    for day_block in day_blocks[:expected_days]:
        # day_theme：单日主题，必须和其他天不同。
        day_theme = clean_markdown_text(day_block["theme"])
        if day_theme in theme_set:
            validation_errors.append(f"Day {day_block['day']} 主题重复：{day_theme}。")
        theme_set.add(day_theme)

        day_signature_parts = []
        for slot_label in required_slots:
            slot_text = extract_slot_text(day_block["body"], slot_label)
            if not slot_text:
                validation_errors.append(f"Day {day_block['day']} 缺少{slot_label}安排。")
                continue

            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_block['day']} {slot_label}使用了空泛模板词：{slot_text[:40]}")

            # slot_parts：时间段内容应拆成地点、推荐理由、预计耗时、交通或预约提醒。
            slot_parts = [part.strip() for part in re.split(r"[｜|]", slot_text) if part.strip()]
            if len(slot_parts) < 4:
                validation_errors.append(
                    f"Day {day_block['day']} {slot_label}格式不完整，需要“具体地点｜推荐理由｜预计耗时｜交通或预约提醒”。"
                )

            # place：时间段内容中竖线前面的具体地点。
            place = slot_parts[0] if slot_parts else ""
            if place:
                if place in seen_places:
                    validation_errors.append(f"重复安排地点：{place}。")
                seen_places.add(place)

            day_signature_parts.append(slot_text)

        # day_signature：单日四段安排合并后的指纹。
        day_signature = "||".join(day_signature_parts)
        if day_signature and day_signature in signature_set:
            validation_errors.append(f"Day {day_block['day']} 与其他日期的行程内容高度重复。")
        signature_set.add(day_signature)

    return validation_errors


def build_itinerary_retry_prompt(base_prompt: str, validation_errors: list[str], parsed_request: dict) -> str:
    """build_itinerary_retry_prompt：根据校验错误生成二次生成提示词。"""

    # error_text：校验错误说明。
    error_text = "\n".join(f"- {error}" for error in validation_errors)

    return f"""
{base_prompt}

上一次输出的每日行程不合格，必须重新生成完整攻略，重点修复：
{error_text}

强制要求：
1. 必须生成 Day 1 到 Day {parsed_request["days"]}，一天都不能少。
2. 每一天主题必须不同。
3. 不要使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
4. 每个时间段必须写具体地点原名、推荐理由、预计耗时、交通或预约提醒。
5. 不要重复同一景点、同一餐厅、同一区域。
""".strip()


def call_deepseek_api(user_input: str, parsed_request: dict, facts_context: str) -> tuple[str | None, str | None]:
    """call_deepseek_api：使用 OpenAI Python SDK 调用 DeepSeek API 生成攻略文本。"""

    if OpenAI is None:
        return None, "没有安装 openai 依赖，已使用本地演示攻略。"

    # api_key：DeepSeek API Key，从 .env、环境变量或 Streamlit secrets 读取。
    api_key = get_deepseek_api_key()
    if not api_key:
        return None, "未配置 DEEPSEEK_API_KEY，已使用本地演示攻略。"

    # model_name：当前使用的 DeepSeek 模型名称，可通过 DEEPSEEK_MODEL 修改。
    model_name = get_deepseek_model_name()

    # client：OpenAI SDK 客户端，通过 base_url 指向 DeepSeek 服务。
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # instructions：给模型的角色和输出风格要求。
    instructions = """
你是一名资深旅行编辑和行程规划师，擅长把用户的一句话需求整理成清晰、真实、好执行的旅行攻略。
请用中文输出，语气像旅行杂志编辑，但结构要像实用攻略工具。
不要编造实时价格或实时营业状态；涉及价格时用区间估算，并提醒以出行前查询为准。
""".strip()

    # prompt：最终发送给模型的完整提示词。
    prompt = build_ai_prompt(user_input, parsed_request, facts_context)

    try:
        def create_markdown(prompt_text: str) -> str | None:
            """create_markdown：执行一次 DeepSeek Chat Completions 调用并返回 Markdown 文本。"""

            # response：DeepSeek Chat Completions API 返回的大模型结果。
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt_text},
                ],
            )

            # markdown_content：从模型回复中取出的 Markdown 攻略文本。
            markdown_content = response.choices[0].message.content
            return markdown_content.strip() if markdown_content else None

        # markdown_text：从模型回复中取出的 Markdown 攻略文本。
        markdown_text = create_markdown(prompt)
        if not markdown_text:
            return None, "DeepSeek 返回内容为空，已使用本地演示攻略。"

        # validation_errors：第一次生成后的每日行程校验问题。
        validation_errors = validate_itinerary_markdown(markdown_text, parsed_request)
        if not validation_errors:
            return markdown_text, None

        # retry_prompt：校验失败时给模型的二次生成提示词。
        retry_prompt = build_itinerary_retry_prompt(prompt, validation_errors, parsed_request)

        # retry_markdown：二次生成的 Markdown 攻略文本。
        retry_markdown = create_markdown(retry_prompt)
        if not retry_markdown:
            error_summary = "；".join(validation_errors[:4])
            return markdown_text, f"DeepSeek 二次生成返回内容为空，已展示第一次结果；每日行程可能不完整：{error_summary}"

        # retry_errors：二次生成后再次校验每日行程。
        retry_errors = validate_itinerary_markdown(retry_markdown, parsed_request)
        if retry_errors:
            error_summary = "；".join(retry_errors[:4])
            return retry_markdown, f"每日行程校验未完全通过：{error_summary}。页面会显示解析问题，请重新生成或调整输入。"

        return retry_markdown, "检测到第一次每日行程不完整，已自动重新生成并通过校验。"
    except Exception as error:
        return None, f"DeepSeek API 调用失败，已使用本地演示攻略。错误：{error}"


def build_demo_markdown(parsed_request: dict, facts_context: str = "") -> str:
    """build_demo_markdown：没有 API Key 或 API 失败时生成本地演示攻略。"""

    # destination：攻略目的地。
    destination = parsed_request["destination"]

    # days：旅行天数。
    days = parsed_request["days"]

    # nights：住宿晚数。
    nights = parsed_request["nights"]

    # budget：预算档位。
    budget = parsed_request["budget"]

    # budget_exchange_hint：国外目的地预算粗略换算提示。
    budget_exchange_hint = parsed_request.get("budget_exchange_hint")

    # preferences_text：用户旅行偏好。
    preferences_text = "、".join(parsed_request["preferences"])

    # source_updated_at：本地演示攻略的信息更新时间。
    source_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # source_note：本地演示攻略的信息来源说明。
    source_note = (
        "未配置搜索 API 或搜索失败，本地演示攻略没有使用实时联网来源；门票、预约、开放时间建议出行前再次核对。"
        if "联网事实校验状态：已执行" not in facts_context and "联网事实校验状态：缓存命中" not in facts_context
        else "已传入联网搜索 facts_context；具体门票、预约和开放时间请以搜索来源及出行前二次确认为准。"
    )

    return f"""
## 旅行封面文案
{destination} {days} 天 {nights} 晚旅行计划：把 {preferences_text} 放进行程主线，用轻松但不松散的节奏完成一次有记忆点的城市探索。

## 详细旅游攻略
- 目的地：{destination}
- 行程长度：{days} 天 {nights} 晚
- 预算：{budget}
- 旅行风格：{preferences_text}
- 规划思路：第一天熟悉城市动线，第二天深入主题体验，最后一天安排轻量活动和购物补漏。
{f"- 换算提示：{budget_exchange_hint}" if budget_exchange_hint else ""}

## 每日行程
本地演示模式不会生成每日行程模板。请配置 DEEPSEEK_API_KEY 后由 DeepSeek 按目的地、天数、偏好和联网事实生成完整不重复行程。

## 美食推荐
- 本地代表料理：优先选择评分稳定、翻台快、位置靠近行程路线的店｜人均 80-180 人民币(CNY)｜适合第一顿正式餐
- 街区小吃：适合放在下午或夜间，不要把所有排队店集中到同一天｜人均 30-80 人民币(CNY)｜适合边逛边吃
- 甜品或咖啡：适合安排在步行较多的下午，作为休息点｜人均 40-100 人民币(CNY)｜适合拍照和休息
- 预约型餐厅：如果是热门目的地，建议提前 3 到 7 天确认｜人均 180-400 人民币(CNY)｜适合纪念日晚餐

## 交通建议
- 城市内优先使用地铁、公交或官方交通卡，减少频繁打车。
- 每天尽量围绕一个区域规划，避免跨城式来回移动。
- 机场或车站到酒店先查官方线路，再对比打车价格。
- 如果有大件行李，最后一天优先选择寄存点或酒店寄存。

## 预算估算
- 用户预算：{budget}
{f"- {budget_exchange_hint}" if budget_exchange_hint else ""}
- 交通：经济预算约 150-300 人民币(CNY)/人，普通预算约 300-600 人民币(CNY)/人，高预算按实际打车和跨城交通增加。
- 住宿：经济预算约 300-600 人民币(CNY)/晚，普通预算约 600-1200 人民币(CNY)/晚，高预算约 1200 人民币(CNY)/晚以上。
- 餐饮：约 150-350 人民币(CNY)/人/天，热门餐厅和预约餐厅另算。
- 门票体验：约 100-500 人民币(CNY)/人，主题展、乐园、演出费用可能更高。
- 机动费用：建议预留总预算的 10%-20%。

## 避坑提醒
- 不要把热门景点、热门餐厅和远距离交通挤在同一天。
- 不要只看社交平台种草，出发前确认营业时间、预约方式和交通路线。
- 夜景点通常受天气影响明显，建议保留备选方案。
- 购物和伴手礼尽量放在后半程，避免一路背负行李。
- 本攻略为第一版演示内容，真实出行前请再次确认价格、营业时间和交通信息。

## 信息来源与更新时间
- 更新时间：{source_updated_at}
- 来源说明：{source_note}
- 门票、预约、开放时间：根据搜索结果整理，仍需出行前二次确认；如果没有联网结果，请勿将本地演示内容视为实时信息。
""".strip()


def generate_travel_markdown(user_input: str, parsed_request: dict, facts_context: str) -> tuple[str, str | None]:
    """generate_travel_markdown：优先用大模型生成攻略，失败时回退到本地演示攻略。"""

    # ai_markdown：大模型生成的 Markdown 文本。
    ai_markdown, api_message = call_deepseek_api(user_input, parsed_request, facts_context)
    if ai_markdown:
        return ai_markdown, api_message

    # demo_markdown：本地演示 Markdown 文本。
    demo_markdown = build_demo_markdown(parsed_request, facts_context)
    return demo_markdown, api_message


def call_deepseek_markdown_from_json_api(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
    travel_json: dict,
) -> tuple[str | None, str | None]:
    """call_deepseek_markdown_from_json_api：基于合格 JSON 生成 Markdown 攻略。"""

    # instructions：给模型的 Markdown 写作角色要求。
    instructions = """
你是一名资深旅行编辑和行程规划师。请基于给定 JSON 写中文 Markdown 攻略。
不能改变 JSON 中的行程天数、地点、预算和偏好；不要编造实时营业状态。
""".strip()

    # markdown_prompt：基于结构化 JSON 生成 Markdown 的提示词。
    markdown_prompt = build_markdown_from_json_prompt(user_input, parsed_request, facts_context, travel_json)
    return call_deepseek_chat(markdown_prompt, instructions)


def generate_travel_content(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
) -> tuple[str, str | None, dict | None, str | None, list[str]]:
    """generate_travel_content：先生成结构化 JSON，再基于 JSON 生成 Markdown 攻略。"""

    # travel_json：用于页面时间线渲染的结构化旅行数据。
    travel_json, json_raw, json_errors, json_message = call_deepseek_structured_json_api(
        user_input,
        parsed_request,
        facts_context,
    )

    if not travel_json:
        # demo_markdown：结构化 JSON 失败时仍保留页面其他区域，不使用假行程补齐。
        demo_markdown = build_demo_markdown(parsed_request, facts_context)
        error_summary = "；".join(json_errors[:4]) if json_errors else "结构化 JSON 未生成。"
        api_message = f"每日行程 JSON 未通过校验：{error_summary}"
        if json_message and json_message not in api_message:
            api_message = f"{api_message}；{json_message}"
        return demo_markdown, api_message, None, json_raw, json_errors

    # markdown_text：基于合格 JSON 生成的 Markdown 攻略。
    markdown_text, markdown_message = call_deepseek_markdown_from_json_api(
        user_input,
        parsed_request,
        facts_context,
        travel_json,
    )

    # api_messages：需要展示给用户的生成状态说明。
    api_messages = []
    if json_message:
        api_messages.append(json_message)

    if markdown_text:
        if markdown_message:
            api_messages.append(markdown_message)
        return markdown_text, "；".join(api_messages) or None, travel_json, json_raw, []

    # Markdown 二次生成失败时，用已通过校验的 JSON 生成可复制攻略，不生成假行程。
    fallback_markdown = build_markdown_from_structured_json(travel_json, parsed_request, facts_context)
    if markdown_message:
        api_messages.append(f"Markdown 生成失败，已根据合格 JSON 生成可复制攻略：{markdown_message}")
    else:
        api_messages.append("Markdown 生成失败，已根据合格 JSON 生成可复制攻略。")

    return fallback_markdown, "；".join(api_messages), travel_json, json_raw, []


def generate_cover_image_url(parsed_request: dict) -> str:
    """generate_cover_image_url：生成封面图地址，后续可替换为图片生成 API。"""

    # destination：封面图上显示的目的地。
    destination = parsed_request["destination"]

    # preferences_text：封面图上显示的旅行偏好。
    preferences_text = " / ".join(parsed_request["preferences"][:4])

    # cover_svg：使用旅行杂志感 SVG 占位图，保证没有图片 API 时也能显示大封面。
    cover_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
      <defs>
        <linearGradient id="sky" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#111827"/>
          <stop offset="28%" stop-color="#26324f"/>
          <stop offset="62%" stop-color="#92400e"/>
          <stop offset="100%" stop-color="#020617"/>
        </linearGradient>
        <linearGradient id="sunset" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#fef3c7" stop-opacity="0.86"/>
          <stop offset="42%" stop-color="#fb923c" stop-opacity="0.38"/>
          <stop offset="100%" stop-color="#020617" stop-opacity="0"/>
        </linearGradient>
        <linearGradient id="water" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.72"/>
          <stop offset="52%" stop-color="#14b8a6" stop-opacity="0.42"/>
          <stop offset="100%" stop-color="#f97316" stop-opacity="0.48"/>
        </linearGradient>
        <filter id="grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch"/>
          <feColorMatrix type="saturate" values="0"/>
          <feComponentTransfer>
            <feFuncA type="table" tableValues="0 0.18"/>
          </feComponentTransfer>
        </filter>
        <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="24" stdDeviation="24" flood-color="#020617" flood-opacity="0.42"/>
        </filter>
        <clipPath id="photoClip">
          <rect x="760" y="115" width="470" height="560" rx="32"/>
        </clipPath>
      </defs>
      <rect width="1600" height="900" fill="url(#sky)"/>
      <rect width="1600" height="900" fill="url(#sunset)" opacity="0.62"/>
      <rect width="1600" height="900" filter="url(#grain)" opacity="0.36"/>
      <path d="M0 565 C165 470 305 515 440 430 C570 350 690 392 820 315 C1010 205 1190 295 1600 185 L1600 900 L0 900 Z" fill="#0f172a" opacity="0.76"/>
      <path d="M0 625 C230 508 365 610 545 515 C710 428 865 560 1020 468 C1188 368 1365 438 1600 315 L1600 900 L0 900 Z" fill="#1e293b" opacity="0.72"/>
      <path d="M0 690 C260 590 430 710 675 625 C910 544 1060 690 1308 568 C1435 506 1510 520 1600 475 L1600 900 L0 900 Z" fill="url(#water)" opacity="0.78"/>
      <path d="M0 742 C210 700 390 785 640 735 C880 688 1030 790 1285 708 C1430 662 1510 675 1600 642 L1600 900 L0 900 Z" fill="#020617" opacity="0.70"/>
      <g filter="url(#softShadow)" opacity="0.95">
        <rect x="760" y="115" width="470" height="560" rx="32" fill="#f8fafc" opacity="0.92"/>
        <g clip-path="url(#photoClip)">
          <rect x="760" y="115" width="470" height="560" fill="#0f172a"/>
          <rect x="760" y="115" width="470" height="560" fill="url(#sky)" opacity="0.52"/>
          <circle cx="1110" cy="230" r="72" fill="#fde68a" opacity="0.86"/>
          <path d="M760 430 C845 360 910 380 975 320 C1055 245 1115 360 1230 275 L1230 675 L760 675 Z" fill="#334155"/>
          <path d="M760 520 C900 455 960 550 1080 488 C1145 455 1188 470 1230 430 L1230 675 L760 675 Z" fill="#0f766e" opacity="0.7"/>
          <path d="M760 575 C860 540 940 615 1055 558 C1120 524 1170 540 1230 512 L1230 675 L760 675 Z" fill="#0ea5e9" opacity="0.55"/>
          <path d="M860 675 L1018 444 L1135 675 Z" fill="#f8fafc" opacity="0.82"/>
          <path d="M924 675 L1018 500 L1080 675 Z" fill="#f59e0b" opacity="0.55"/>
        </g>
      </g>
      <path d="M1320 170 C1390 210 1425 268 1450 350" stroke="#fde68a" stroke-width="3" stroke-dasharray="12 16" fill="none" opacity="0.55"/>
      <path d="M1450 350 l28 -12 l-20 31 z" fill="#fde68a" opacity="0.75"/>
      <g opacity="0.78">
        <rect x="120" y="640" width="420" height="3" fill="#fde68a"/>
        <rect x="120" y="665" width="315" height="3" fill="#f8fafc" opacity="0.52"/>
        <rect x="120" y="690" width="250" height="3" fill="#f8fafc" opacity="0.34"/>
      </g>
      <text x="126" y="145" fill="#fde68a" font-size="32" font-family="Arial, sans-serif" letter-spacing="6">AI TRAVEL MAGAZINE</text>
      <text x="126" y="232" fill="#ffffff" font-size="72" font-family="Arial, sans-serif" font-weight="700">{html.escape(destination)}</text>
      <text x="130" y="292" fill="#e5e7eb" font-size="30" font-family="Arial, sans-serif">{html.escape(preferences_text)}</text>
    </svg>
    """

    return "data:image/svg+xml;charset=utf-8," + quote(cover_svg)


def split_markdown_sections(markdown_text: str) -> dict:
    """split_markdown_sections：把 Markdown 按二级标题拆成多个展示卡片。"""

    # section_map：保存标题和正文的对应关系。
    section_map = {}

    # normalized_markdown：保证文本开头有换行，方便正则切分。
    normalized_markdown = "\n" + markdown_text.strip()

    # matches：匹配所有“## 标题”和标题后正文。
    matches = re.finditer(r"\n##\s+(.+?)\n([\s\S]*?)(?=\n##\s+|\Z)", normalized_markdown)
    for match in matches:
        title = match.group(1).strip()
        content = match.group(2).strip()
        section_map[title] = content

    return section_map


def clean_markdown_text(markdown_text: str) -> str:
    """clean_markdown_text：清理 Markdown 符号，方便放进自定义 HTML 卡片。"""

    # cleaned_text：去掉列表符号、粗体和多余空格后的文本。
    cleaned_text = re.sub(r"^[\-\*\d\.\s]+", "", markdown_text.strip())
    cleaned_text = re.sub(r"[*`#]+", "", cleaned_text)
    return cleaned_text.strip()


def extract_bullet_items(section_text: str, max_items: int = 6) -> list[str]:
    """extract_bullet_items：从 Markdown 段落中提取列表项。"""

    # item_list：从 Markdown 中提取出的列表内容。
    item_list = []
    for line in section_text.splitlines():
        stripped_line = line.strip()
        if re.match(r"^[-*]\s+", stripped_line) or re.match(r"^\d+[.、]\s+", stripped_line):
            item = clean_markdown_text(stripped_line)
            if item:
                item_list.append(item)

    if not item_list and section_text.strip():
        # fallback_lines：当模型没有使用列表时，按非空行兜底提取。
        fallback_lines = [clean_markdown_text(line) for line in section_text.splitlines() if clean_markdown_text(line)]
        item_list = fallback_lines

    return item_list[:max_items]


def estimate_total_cost(parsed_request: dict) -> str:
    """estimate_total_cost：根据天数和预算档位估算不含大交通的人均总花费。"""

    if parsed_request.get("budget_has_explicit_amount"):
        return f"按 {parsed_request['budget']} 控制"

    # budget_level：用户预算档位。
    budget_level = parsed_request.get("budget_level", parsed_request["budget"])

    # days：旅行天数。
    days = parsed_request["days"]

    # nights：住宿晚数。
    nights = parsed_request["nights"]

    if budget_level == "经济预算":
        day_cost, night_cost = 260, 320
    elif budget_level == "高预算":
        day_cost, night_cost = 980, 1600
    else:
        day_cost, night_cost = 480, 720

    # low_cost：较低估算值。
    low_cost = days * day_cost + nights * night_cost

    # high_cost：较高估算值。
    high_cost = int(low_cost * 1.35)

    return f"约 {low_cost:,}-{high_cost:,} 人民币(CNY)/人"


def infer_trip_pace(parsed_request: dict) -> str:
    """infer_trip_pace：根据天数和偏好推断旅行节奏。"""

    # preferences：用户偏好列表。
    preferences = parsed_request["preferences"]

    if parsed_request["days"] >= 6 or any(item in preferences for item in ["自然", "咖啡", "温泉", "海边"]):
        return "松弛慢旅行"
    if parsed_request["days"] <= 3 and any(item in preferences for item in ["购物", "夜景", "动漫"]):
        return "高效城市探索"
    return "舒适均衡节奏"


def infer_audience(parsed_request: dict) -> str:
    """infer_audience：根据偏好推断适合人群。"""

    # preferences：用户偏好列表。
    preferences = parsed_request["preferences"]

    if "亲子" in preferences:
        return "家庭与亲子出行"
    if any(item in preferences for item in ["动漫", "购物", "夜景"]):
        return "城市玩家与潮流爱好者"
    if any(item in preferences for item in ["自然", "徒步", "海边"]):
        return "自然风景和慢旅行人群"
    return "第一次到访和自由行用户"


def build_summary_metrics(parsed_request: dict) -> dict:
    """build_summary_metrics：生成攻略摘要区需要展示的指标。"""

    # preferences_count：偏好数量，用于生成推荐强度。
    preferences_count = len(parsed_request["preferences"])

    # recommendation_score：推荐强度评分。
    recommendation_score = "4.9 / 5" if preferences_count >= 3 else "4.6 / 5"

    return {
        "推荐强度": recommendation_score,
        "旅行节奏": infer_trip_pace(parsed_request),
        "适合人群": infer_audience(parsed_request),
        "预计总花费": estimate_total_cost(parsed_request),
    }


def extract_slot_text(day_text: str, slot_label: str) -> str:
    """extract_slot_text：从某一天行程中提取上午、中午、下午或晚上的内容。"""

    # slot_pattern：匹配指定时间段的 Markdown 行。
    slot_pattern = rf"(?:^|\n)\s*[-*]?\s*(?:\*\*)?{slot_label}(?:\*\*)?\s*[：:]\s*(.+)"
    match = re.search(slot_pattern, day_text)
    if match:
        return clean_markdown_text(match.group(1))
    return ""


def split_place_and_description(slot_text: str) -> tuple[str, str]:
    """split_place_and_description：把时间段内容拆成地点和说明。"""

    # parts：按中文或英文竖线拆开的地点和说明。
    parts = [part.strip() for part in re.split(r"[｜|]", slot_text, maxsplit=1)]
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]

    return "", slot_text


def build_timeline_days(section_map: dict, parsed_request: dict) -> tuple[list[dict], list[str]]:
    """build_timeline_days：把每日行程 Markdown 转成时间线数据。"""

    # itinerary_text：每日行程 Markdown 内容。
    itinerary_text = section_map.get("每日行程", "").strip()
    if not itinerary_text:
        return [], ["未找到“每日行程”区域。"]

    # timeline_markdown：补回二级标题，复用统一的 Day 块解析函数。
    timeline_markdown = f"## 每日行程\n{itinerary_text}"

    # day_blocks：从 Markdown 中解析出的每日行程块。
    day_blocks = parse_itinerary_day_blocks(timeline_markdown)

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    # validation_errors：时间线渲染前的结构化校验问题。
    validation_errors = []

    if len(day_blocks) != expected_days:
        validation_errors.append(f"模型返回 {len(day_blocks)} 天行程，系统识别用户需要 {expected_days} 天。")

    # day_map：按 Day 编号索引每日行程，方便检查缺失天数。
    day_map = {day_block["day"]: day_block for day_block in day_blocks}
    missing_days = [day for day in range(1, expected_days + 1) if day not in day_map]
    if missing_days:
        validation_errors.append(f"缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    # slot_config：时间线四个固定时段。
    slot_config = [
        {"label": "上午", "time": "09:00 - 11:30", "icon": "AM"},
        {"label": "中午", "time": "12:00 - 13:30", "icon": "NO"},
        {"label": "下午", "time": "14:00 - 17:30", "icon": "PM"},
        {"label": "晚上", "time": "18:30 - 21:30", "icon": "EV"},
    ]

    # generic_terms：不允许出现在时间线里的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # seen_places：已出现地点集合，用于避免同一地点重复安排。
    seen_places = set()

    # seen_themes：已出现主题集合，用于避免每天主题重复。
    seen_themes = set()

    # timeline_days：最终时间线数据。
    timeline_days = []
    for day_number in range(1, expected_days + 1):
        day_block = day_map.get(day_number)
        if not day_block:
            continue

        # day_theme：当前 Day 的主题文本。
        day_theme = clean_markdown_text(day_block["theme"])
        if day_theme in seen_themes:
            validation_errors.append(f"Day {day_number} 主题重复：{day_theme}。")
        seen_themes.add(day_theme)

        # slot_list：单日四个时间段的数据。
        slot_list = []
        for slot in slot_config:
            slot_text = extract_slot_text(day_block["body"], slot["label"])
            if not slot_text:
                validation_errors.append(f"Day {day_number} 缺少{slot['label']}安排。")
                continue

            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_number} {slot['label']}仍包含空泛模板词：{slot_text[:40]}")

            # slot_parts：时间段内容必须包含地点、推荐理由、预计耗时、交通或预约提醒。
            slot_parts = [part.strip() for part in re.split(r"[｜|]", slot_text) if part.strip()]
            if len(slot_parts) < 4:
                validation_errors.append(
                    f"Day {day_number} {slot['label']}未完整包含地点、推荐理由、预计耗时、交通或预约提醒。"
                )
                continue

            place, description = split_place_and_description(slot_text)
            if not place:
                validation_errors.append(f"Day {day_number} {slot['label']}未按“具体地点｜推荐理由｜预计耗时｜交通或预约提醒”格式输出。")
                continue

            if place in seen_places:
                validation_errors.append(f"重复安排地点：{place}。")
            seen_places.add(place)

            slot_list.append(
                {
                    "label": slot["label"],
                    "time": slot["time"],
                    "icon": slot["icon"],
                    "place": place,
                    "description": description,
                }
            )

        timeline_days.append({"title": f"Day {day_number}：{day_block['theme']}", "slots": slot_list})

    if validation_errors:
        return [], validation_errors

    return timeline_days, []


def build_timeline_days_from_json(travel_json: dict | None, parsed_request: dict) -> tuple[list[dict], list[str]]:
    """build_timeline_days_from_json：把结构化 JSON 转成时间线卡片数据。"""

    # validation_errors：结构化 JSON 的校验错误。
    validation_errors = validate_structured_travel_json(travel_json, parsed_request)
    if validation_errors:
        return [], validation_errors

    # slot_config：英文 JSON 字段和页面展示标签的对应关系。
    slot_config = [
        {"key": "morning", "label": "上午", "icon": "AM"},
        {"key": "noon", "label": "中午", "icon": "NO"},
        {"key": "afternoon", "label": "下午", "icon": "PM"},
        {"key": "evening", "label": "晚上", "icon": "EV"},
    ]

    # timeline_days：最终时间线数据。
    timeline_days = []
    for day_item in travel_json.get("daily_itinerary", []):
        # segment_destination：多目的地时展示当前日期所属目的地。
        segment_destination = str(day_item.get("segment_destination", "")).strip()

        # title_prefix：多目的地时间线标题前缀。
        title_prefix = f"{segment_destination}｜" if segment_destination else ""

        # slot_list：单日四个时间段的数据。
        slot_list = []
        for slot in slot_config:
            # slot_data：结构化 JSON 中的时间段对象。
            slot_data = day_item[slot["key"]]
            description = (
                f"{slot_data['original_name']}｜{slot_data['reason']}｜{slot_data['duration']}｜"
                f"{slot_data['transport']}；{slot_data['booking_note']}"
            )
            slot_list.append(
                {
                    "label": slot["label"],
                    "time": slot_data["time"],
                    "icon": slot["icon"],
                    "place": slot_data["place"],
                    "description": description,
                    "original_name": slot_data["original_name"],
                    "reason": slot_data["reason"],
                    "duration": slot_data["duration"],
                    "transport": slot_data["transport"],
                    "booking_note": slot_data["booking_note"],
                }
            )

        timeline_days.append({"title": f"Day {day_item['day']}：{title_prefix}{day_item['theme']}", "slots": slot_list})

    return timeline_days, []


def build_food_cards(section_map: dict, parsed_request: dict) -> list[dict]:
    """build_food_cards：把美食推荐 Markdown 转成美食卡片数据。"""

    # food_items：美食推荐列表。
    food_items = extract_bullet_items(section_map.get("美食推荐", ""), max_items=6)

    if not food_items:
        food_items = [
            "本地代表料理：优先选择路线附近的高评分店｜人均 80-180 元｜适合第一顿正式餐",
            "街区小吃：适合放在下午或夜间，边走边吃更轻松｜人均 30-80 元｜适合探索街区",
            "甜品或咖啡：作为下午休息点，也适合拍照｜人均 40-100 元｜适合慢旅行",
        ]

    # budget_level：预算档位，用于美食卡片的人均预算兜底。
    budget_level = parsed_request.get("budget_level", parsed_request["budget"])

    # default_budget：美食卡片的人均预算兜底。
    default_budget = "人均 80-180 人民币(CNY)" if budget_level != "经济预算" else "人均 30-90 人民币(CNY)"

    # food_cards：最终美食卡片数据。
    food_cards = []
    for item in food_items:
        title = item
        detail = "结合行程路线选择，减少排队和跨区移动。"
        if "：" in item:
            title, detail = item.split("：", 1)
        elif ":" in item:
            title, detail = item.split(":", 1)

        # detail_parts：按竖线拆出的理由、预算和场景。
        detail_parts = [part.strip() for part in re.split(r"[｜|]", detail) if part.strip()]
        reason = detail_parts[0] if detail_parts else detail
        budget = detail_parts[1] if len(detail_parts) > 1 else default_budget
        scene = detail_parts[2] if len(detail_parts) > 2 else "适合穿插在当日行程中"

        food_cards.append(
            {
                "title": clean_markdown_text(title)[:34],
                "reason": clean_markdown_text(reason),
                "budget": clean_markdown_text(budget),
                "scene": clean_markdown_text(scene),
                "location": "位置：建议结合当日行程区域确认",
                "nearby_spot": "当日行程附近",
                "booking_note": "热门时段建议提前确认或预约",
                "map_keyword": clean_markdown_text(title)[:34],
            }
        )

    return food_cards


def build_food_cards_from_json(travel_json: dict | None) -> list[dict]:
    """build_food_cards_from_json：把结构化 JSON 中的美食推荐转成卡片数据。"""

    if not isinstance(travel_json, dict):
        return []

    # food_recommendations：结构化 JSON 中的美食推荐列表。
    food_recommendations = travel_json.get("food_recommendations", [])
    if not isinstance(food_recommendations, list):
        return []

    # food_cards：最终美食卡片数据。
    food_cards = []
    for food_item in food_recommendations[:6]:
        if not isinstance(food_item, dict):
            continue

        # name_cn/name_original：店铺中文名与英文/当地原名。
        name_cn = clean_markdown_text(str(food_item.get("name_cn", "")))
        name_original = clean_markdown_text(str(food_item.get("name_original", "")))
        title = f"{name_cn}（{name_original}）" if name_original and name_original != name_cn else name_cn

        food_cards.append(
            {
                "title": title or "待确认美食点",
                "location": clean_markdown_text(str(food_item.get("location", ""))) or "建议以地图搜索原名确认",
                "nearby_spot": clean_markdown_text(str(food_item.get("nearby_spot", ""))) or "适合穿插在当日行程中",
                "reason": clean_markdown_text(str(food_item.get("reason", ""))) or "结合行程路线选择，减少跨区移动。",
                "budget": clean_markdown_text(str(food_item.get("budget", ""))) or "人均预算待确认",
                "scene": clean_markdown_text(str(food_item.get("scene", ""))) or "适合穿插在当日行程中",
                "booking_note": clean_markdown_text(str(food_item.get("booking_note", ""))) or "热门时段建议提前确认或预约",
                "map_keyword": clean_markdown_text(str(food_item.get("map_keyword", ""))) or title,
            }
        )

    return food_cards


def build_advice_cards(section_text: str, fallback_items: list[str], max_items: int = 4) -> list[dict]:
    """build_advice_cards：把交通建议或避坑提醒转成卡片数据。"""

    # advice_items：从 Markdown 中提取出的建议列表。
    advice_items = extract_bullet_items(section_text, max_items=max_items) or fallback_items

    # advice_cards：最终建议卡片数据。
    advice_cards = []
    for index, item in enumerate(advice_items[:max_items], start=1):
        title = f"建议 {index}"
        description = item
        if "：" in item and len(item.split("：", 1)[0]) <= 16:
            title, description = item.split("：", 1)
        elif ":" in item and len(item.split(":", 1)[0]) <= 16:
            title, description = item.split(":", 1)
        else:
            title = clean_markdown_text(item)[:14]

        advice_cards.append({"title": clean_markdown_text(title), "description": clean_markdown_text(description)})

    return advice_cards


def render_hero() -> None:
    """render_hero：渲染页面顶部的产品标题区。"""

    st.markdown(
        """
        <nav class="top-nav">
            <div class="nav-brand"><span class="brand-mark">T</span><span>TripAgent</span></div>
            <div class="nav-links">
                <span>AI旅行规划</span>
                <span>示例</span>
                <span>反馈</span>
            </div>
        </nav>
        <section class="hero product-hero">
            <div class="hero-layout">
                <div>
                    <div class="eyebrow">AI Private Travel Advisor · Magazine Edition</div>
                    <h1>一句话生成你的专属旅行路线</h1>
                    <p>输入目的地、天数和偏好，AI 为你规划每日行程、美食、预算、交通、天气与避坑提醒。</p>
                    <div class="hero-proof">
                        <span>多目的地连续规划</span>
                        <span>天气与出行提醒</span>
                        <span>Markdown 一键带走</span>
                    </div>
                </div>
                <aside class="hero-panel product-preview">
                    <div class="preview-cover">
                        <div class="preview-cover-label">
                            <span>AI Travel Magazine</span>
                            <strong>Nanjing × Jiangxi<br>7 Days Journey</strong>
                        </div>
                    </div>
                    <div class="preview-steps">
                        <div class="preview-step">
                            <b>01</b>
                            <div><span>Route</span><p>自动拆分多目的地，每天主题不同。</p></div>
                        </div>
                        <div class="preview-step">
                            <b>02</b>
                            <div><span>Food & Weather</span><p>美食、天气、预算和避坑提醒一起整理。</p></div>
                        </div>
                        <div class="preview-step">
                            <b>03</b>
                            <div><span>Export</span><p>生成可复制、可下载的 Markdown 攻略。</p></div>
                        </div>
                    </div>
                </aside>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_box() -> tuple[bool, str]:
    """render_input_box：渲染醒目的自然语言输入框。"""

    with st.container(border=True):
        st.markdown('<p class="input-kicker">Start with one sentence</p>', unsafe_allow_html=True)
        st.markdown('<p class="input-title">告诉我你想怎么旅行</p>', unsafe_allow_html=True)
        st.markdown('<p class="sample-title">选择一个示例，或直接输入你的旅行需求</p>', unsafe_allow_html=True)

        # sample_columns：用于横向排列示例标签按钮。
        sample_columns = st.columns(4)
        for index, sample_prompt in enumerate(SAMPLE_PROMPTS):
            if sample_columns[index].button(sample_prompt["label"], key=f"sample_prompt_{index}"):
                st.session_state["travel_request_input"] = sample_prompt["prompt"]

        with st.form("travel_request_form"):
            # user_input：用户输入的一句话旅行需求。
            user_input = st.text_area(
                label="旅行需求",
                label_visibility="collapsed",
                placeholder="例如：我想去南京游玩3天，然后再去江西游玩，喜欢历史文化、美食和夜景，预算8000",
                key="travel_request_input",
            )

            # submitted：用户是否点击了生成按钮。
            submitted = st.form_submit_button("生成专属旅行方案")

        st.markdown('<p class="hint">不用填复杂表单，一句话就够。没写天数默认 3 天 2 晚；预算数字没写单位时默认人民币 CNY。</p>', unsafe_allow_html=True)

    return submitted, user_input


def render_cover(parsed_request: dict, cover_image_url: str) -> None:
    """render_cover：渲染大封面图区域。"""

    # cover_destination_text：封面专用目的地文本，多目的地用“×”营造旅行杂志标题感。
    cover_destination_text = " × ".join(parsed_request.get("destinations", [])) or parsed_request["destination"]

    # safe_destination：转义后的目的地文本，避免 HTML 注入。
    safe_destination = html.escape(cover_destination_text)

    # safe_preferences：转义后的偏好文本。
    safe_preferences = html.escape("、".join(parsed_request["preferences"]))

    # safe_cover_image_url：转义后的封面图片地址。
    safe_cover_image_url = html.escape(cover_image_url, quote=True)

    # badge_items：封面上展示的旅行关键信息标签。
    badge_items = [
        f"{parsed_request['days']} 天 {parsed_request['nights']} 晚",
        parsed_request["budget"],
        *parsed_request["preferences"][:4],
    ]

    # badge_html：封面标签 HTML。
    badge_html = "".join(f'<span class="cover-badge">{html.escape(item)}</span>' for item in badge_items)

    st.markdown(
        f"""
        <section class="cover-card" style='background-image: url("{safe_cover_image_url}");'>
            <div class="cover-content">
                <div class="label">AI TRAVEL MAGAZINE</div>
                <h2>{safe_destination}</h2>
                <div class="cover-dayline">{parsed_request["days"]} Days Journey · {html.escape(parsed_request["budget"])}</div>
                <p>{safe_preferences} · 由 AI 生成的旅行封面与城市探索计划</p>
                <div class="cover-badges">{badge_html}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_summary_bento(parsed_request: dict) -> None:
    """render_summary_bento：用 bento grid 展示攻略摘要信息。"""

    # preferences_text：用于展示的偏好文本。
    preferences_text = "、".join(parsed_request["preferences"])

    # metric_map：推荐强度、旅行节奏、适合人群和预计总花费。
    metric_map = build_summary_metrics(parsed_request)

    # budget_note：预算卡片中的说明文本。
    budget_note = parsed_request.get("budget_exchange_hint") or "价格为区间估算，出发前需再次确认。"

    st.markdown(
        f"""
        <h2 class="section-heading">攻略摘要</h2>
        <p class="section-subtitle">系统从你的自然语言输入中提取旅行关键参数，并补充可执行的规划指标。</p>
        <div class="bento-grid">
            <div class="bento-card large warm"><span>目的地</span><strong>{html.escape(parsed_request["destination"])}</strong><p>本次攻略围绕城市动线、主题偏好和轻量避坑提醒展开。</p></div>
            <div class="bento-card"><span>旅行天数</span><strong>{parsed_request["days"]} 天 {parsed_request["nights"]} 晚</strong><p>按每日四段式节奏规划。</p></div>
            <div class="bento-card"><span>预算</span><strong>{html.escape(parsed_request["budget"])}</strong><p>{html.escape(budget_note)}</p></div>
            <div class="bento-card large"><span>偏好标签</span><strong>{html.escape(preferences_text)}</strong><p>用于安排主题街区、美食和拍照点。</p></div>
            <div class="bento-card"><span>推荐强度</span><strong>{html.escape(metric_map["推荐强度"])}</strong><p>基于偏好匹配度估算。</p></div>
            <div class="bento-card"><span>旅行节奏</span><strong>{html.escape(metric_map["旅行节奏"])}</strong><p>兼顾体验密度和休息时间。</p></div>
            <div class="bento-card"><span>适合人群</span><strong>{html.escape(metric_map["适合人群"])}</strong><p>可按同行人群继续微调。</p></div>
            <div class="bento-card warm"><span>预计总花费</span><strong>{html.escape(metric_map["预计总花费"])}</strong><p>不含跨城机票或长途交通。</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trip_segments_overview(parsed_request: dict) -> None:
    """render_trip_segments_overview：展示多目的地或推断天数的分段总览。"""

    # trip_segments：系统识别出的目的地分段。
    trip_segments = parsed_request.get("trip_segments", [])
    if not trip_segments:
        return

    # should_show_segments：多目的地或存在默认推断说明时展示分段总览。
    should_show_segments = parsed_request.get("trip_type") == "multi_destination" or bool(parsed_request.get("trip_notes"))
    if not should_show_segments:
        return

    # segment_cards：每个目的地分段的卡片 HTML。
    segment_cards = []
    current_day_start = 1
    for segment in trip_segments:
        # destination：分段目的地。
        destination = str(segment.get("destination", "")).strip()

        # segment_days：分段天数。
        segment_days = int(segment.get("days", DEFAULT_TRAVEL_DAYS))

        # day_range：页面展示的连续 Day 范围。
        day_range = f"Day {current_day_start}-Day {current_day_start + segment_days - 1}"
        current_day_start += segment_days

        # inferred_label：未写天数时明确标注默认推断。
        inferred_label = "默认 " if segment.get("days_inferred") else ""

        # note：分段说明，省份/大区域或默认天数提示。
        note = str(segment.get("note", "")).strip() or "按用户输入的目的地和偏好生成分段路线。"

        segment_cards.append(
            f"""
            <article class="segment-card">
                <span>{html.escape(day_range)}</span>
                <strong>{html.escape(destination)} · {inferred_label}{segment_days} 天 {max(0, segment_days - 1)} 晚</strong>
                <p>{html.escape(note)}</p>
            </article>
            """
        )

    st.markdown(
        f"""
        <h2 class="section-heading">行程分段总览</h2>
        <p class="section-subtitle">多目的地会按连续日期拆分；未说明天数的目的地会明确标注默认规划。</p>
        <div class="segment-overview-grid">{"".join(segment_cards)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_card(section_map: dict) -> None:
    """render_overview_card：展示详细攻略的简要说明卡片。"""

    # overview_text：详细旅游攻略内容。
    overview_text = section_map.get("详细旅游攻略", "")
    if not overview_text:
        return

    with st.container(border=True):
        st.markdown("### 旅行编辑摘要")
        st.markdown(overview_text)


def render_timeline(
    section_map: dict,
    parsed_request: dict,
    travel_json: dict | None = None,
    json_errors: list[str] | None = None,
    json_raw: str | None = None,
) -> None:
    """render_timeline：用时间线样式展示每日行程。"""

    # timeline_days：优先从结构化 JSON 构建每日行程时间线数据。
    if travel_json:
        timeline_days, timeline_errors = build_timeline_days_from_json(travel_json, parsed_request)
    else:
        timeline_days = []
        timeline_errors = json_errors or ["未生成可用于页面渲染的结构化 JSON。"]

    if timeline_errors:
        st.markdown(
            """
            <h2 class="section-heading">每日行程时间线</h2>
            <p class="section-subtitle">每日行程 JSON 没有通过结构化校验，因此没有使用默认模板补齐。</p>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.error("每日行程解析失败或天数不足。请重新生成，系统不会用重复模板自动补齐。")
            for timeline_error in timeline_errors[:8]:
                st.markdown(f"- {timeline_error}")
            if len(timeline_errors) > 8:
                st.markdown(f"- 还有 {len(timeline_errors) - 8} 条校验问题未展示。")
            st.markdown("- 请点击页面上方的“生成专属旅行攻略”重新生成。")
        with st.expander("查看模型返回原文", expanded=False):
            if json_raw:
                st.code(json_raw, language="json")
            else:
                st.markdown("未获取到结构化 JSON 原文。")
        return

    # day_html_list：每天独立卡片 HTML。
    day_html_list = []
    for day in timeline_days:
        slot_html_list = []
        for slot in day["slots"]:
            # original_name/reason/duration/transport/booking_note：结构化 JSON 中的时间段详情。
            original_name = html.escape(slot.get("original_name", ""))
            reason = html.escape(slot.get("reason", slot.get("description", "")))
            duration = html.escape(slot.get("duration", ""))
            transport = html.escape(slot.get("transport", ""))
            booking_note = html.escape(slot.get("booking_note", ""))

            # slot_detail_html：分层展示的时间段信息，避免大段文字堆叠。
            slot_detail_html = (
                f'<div class="slot-original">{original_name}</div>'
                f'<p class="slot-desc">{reason}</p>'
                '<div class="slot-meta-grid">'
                f'<div class="slot-meta-item"><strong>耗时</strong><br>{duration}</div>'
                f'<div class="slot-meta-item"><strong>交通</strong><br>{transport}</div>'
                f'<div class="slot-meta-item"><strong>预约/注意</strong><br>{booking_note}</div>'
                "</div>"
            )
            slot_html_list.append(
                '<div class="timeline-slot">'
                f'<div class="slot-icon">{html.escape(slot["icon"])}</div>'
                "<div>"
                f'<div class="slot-time">{html.escape(slot["label"])} · {html.escape(slot["time"])}</div>'
                f'<div class="slot-place">{html.escape(slot["place"])}</div>'
                f"{slot_detail_html}"
                "</div>"
                "</div>"
            )

        day_html_list.append(
            '<article class="timeline-day">'
            f'<h3>{html.escape(day["title"])}</h3>'
            f'{"".join(slot_html_list)}'
            "</article>"
        )

    # timeline_html：无缩进 HTML，避免 Markdown 把 HTML 识别为代码块。
    timeline_html = (
        '<h2 class="section-heading">每日行程时间线</h2>'
        '<p class="section-subtitle">每天拆成上午、中午、下午和晚上四个时间段，便于实际执行。</p>'
        f'<div class="timeline-grid">{"".join(day_html_list)}</div>'
    )

    st.markdown(timeline_html, unsafe_allow_html=True)


def render_food_cards(section_map: dict, parsed_request: dict, travel_json: dict | None = None) -> None:
    """render_food_cards：用卡片展示美食推荐。"""

    # food_cards：美食卡片数据。
    food_cards = build_food_cards_from_json(travel_json) or build_food_cards(section_map, parsed_request)

    # food_html：美食卡片 HTML。
    food_html = ""
    for food in food_cards:
        food_html += f"""
        <article class="food-card">
            <h3>{html.escape(food["title"])}</h3>
            <div class="food-location">📍 位置：{html.escape(food["location"])}，靠近 {html.escape(food["nearby_spot"])}</div>
            <p>{html.escape(food["reason"])}</p>
            <div class="food-map-keyword">地图搜索：{html.escape(food["map_keyword"])}</div>
            <div class="food-meta">
                <span>{html.escape(food["budget"])}</span>
                <span>{html.escape(food["scene"])}</span>
                <span>{html.escape(food["booking_note"])}</span>
            </div>
        </article>
        """

    st.markdown(
        f"""
        <h2 class="section-heading">美食推荐</h2>
        <p class="section-subtitle">把餐饮当成行程体验的一部分，而不是临时补位。</p>
        <div class="food-grid">{food_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_advice_sections(section_map: dict) -> None:
    """render_advice_sections：展示交通建议和避坑提醒。"""

    # transport_fallback：交通建议兜底内容。
    transport_fallback = [
        "城市内优先使用地铁、公交或官方交通卡，减少频繁打车。",
        "每天尽量围绕一个区域规划，避免跨城式来回移动。",
        "机场或车站到酒店先查官方线路，再对比打车价格。",
        "最后一天优先选择寄存点或酒店寄存，减少拖行李时间。",
    ]

    # warning_fallback：避坑提醒兜底内容。
    warning_fallback = [
        "不要把热门景点、热门餐厅和远距离交通挤在同一天。",
        "出发前确认营业时间、预约方式和交通路线。",
        "夜景点受天气影响明显，建议保留备选方案。",
        "购物和伴手礼尽量放在后半程，避免一路背负行李。",
    ]

    # transport_cards：交通建议卡片数据。
    transport_cards = build_advice_cards(section_map.get("交通建议", ""), transport_fallback, max_items=4)

    # warning_cards：避坑提醒卡片数据。
    warning_cards = build_advice_cards(section_map.get("避坑提醒", ""), warning_fallback, max_items=4)

    # budget_items：预算估算列表。
    budget_items = extract_bullet_items(section_map.get("预算估算", ""), max_items=4)

    transport_html = "".join(
        f"""
        <article class="info-card">
            <div class="card-title-row">
                <div class="info-icon">i</div>
                <h3>{html.escape(card["title"])}</h3>
            </div>
            <p>{html.escape(card["description"])}</p>
        </article>
        """
        for card in transport_cards
    )

    warning_html = "".join(
        f"""
        <article class="warning-card">
            <div class="card-title-row">
                <div class="warning-icon">!</div>
                <h3>{html.escape(card["title"])}</h3>
            </div>
            <p>{html.escape(card["description"])}</p>
        </article>
        """
        for card in warning_cards
    )

    st.markdown(
        f"""
        <h2 class="section-heading">交通建议</h2>
        <p class="section-subtitle">优先减少无效移动，把时间留给真正的体验。</p>
        <div class="info-grid">{transport_html}</div>
        """,
        unsafe_allow_html=True,
    )

    if budget_items:
        # budget_html：预算估算卡片 HTML。
        budget_html = "".join(
            f"""
            <article class="budget-card">
                <span>Budget Note {index}</span>
                <p>{html.escape(budget_item)}</p>
            </article>
            """
            for index, budget_item in enumerate(budget_items, start=1)
        )
        st.markdown(
            f"""
            <h2 class="section-heading">预算估算</h2>
            <p class="section-subtitle">把花费拆成交通、住宿、餐饮和机动预算，方便你出发前调整。</p>
            <div class="budget-grid">{budget_html}</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <h2 class="section-heading">避坑提醒</h2>
        <p class="section-subtitle">提前规避高概率踩坑点，让行程更稳定。</p>
        <div class="warning-grid">{warning_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_source_info(section_map: dict, generated_at: str | None = None) -> None:
    """render_source_info：展示信息来源与更新时间区域。"""

    # source_text：模型生成的来源与更新时间内容，普通用户界面不直接展示技术状态。
    source_text = section_map.get("信息来源与更新时间", "")

    with st.container(border=True):
        st.markdown('<div class="source-card-title">信息与更新时间</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <p class="source-card-text">本攻略由 AI 根据你的输入和当前可用信息整理生成。</p>
            <p class="source-card-text">更新时间：{html.escape(generated_at or datetime.now().strftime('%Y-%m-%d %H:%M'))}</p>
            <p class="source-card-text">门票、预约、开放时间、交通政策和天气情况可能变化，请出行前以官方渠道和天气 App 为准。</p>
            """,
            unsafe_allow_html=True,
        )
        if source_text:
            with st.expander("开发者调试信息", expanded=False):
                st.markdown(source_text)


def render_weather_section(weather_cards: list[dict] | None) -> None:
    """render_weather_section：渲染天气与出行提醒卡片。"""

    st.markdown('<h2 class="section-heading">天气与出行提醒 🌦️</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">根据最近几天的天气情况整理携带建议，出发前请再用天气 App 复核一次。</p>',
        unsafe_allow_html=True,
    )

    if not weather_cards:
        st.markdown(
            """
            <div class="weather-fallback">天气信息暂时无法获取，请出行前查看天气 App。</div>
            """,
            unsafe_allow_html=True,
        )
        return

    # weather_card_html_list：所有目的地天气卡片 HTML。
    weather_card_html_list = []
    for weather_card in weather_cards:
        # destination：天气卡片展示的目的地。
        destination = html.escape(str(weather_card.get("destination", "目的地")))

        if weather_card.get("error"):
            weather_card_html_list.append(
                f"""
                <article class="weather-card">
                    <h3>{destination}｜未来 {WEATHER_FORECAST_DAYS} 天天气</h3>
                    <p class="weather-advice">{html.escape(str(weather_card["error"]))}</p>
                </article>
                """
            )
            continue

        # day_html_list：单个目的地内的每日天气 HTML。
        day_html_list = []
        for day_weather in weather_card.get("days", []):
            # will_rain_text：是否可能下雨的展示文本。
            will_rain_text = "可能下雨" if day_weather.get("will_rain") else "降雨风险较低"

            day_html_list.append(
                f"""
                <div class="weather-day">
                    <div class="weather-icon">{html.escape(str(day_weather.get("weather_icon", "🌦️")))}</div>
                    <div>
                        <div class="weather-date">{html.escape(str(day_weather.get("date", "")))}</div>
                        <div class="weather-main">天气：{html.escape(str(day_weather.get("weather_text", "天气待确认")))}</div>
                        <div class="weather-meta">
                            <span>温度：{html.escape(format_weather_value(day_weather.get("temperature_min"), "°C"))} - {html.escape(format_weather_value(day_weather.get("temperature_max"), "°C"))}</span>
                            <span>湿度：{html.escape(format_weather_value(day_weather.get("humidity"), "%"))}</span>
                            <span>降水概率：{html.escape(format_weather_value(day_weather.get("precipitation_probability"), "%"))}</span>
                            <span>{html.escape(will_rain_text)}</span>
                        </div>
                        <p class="weather-advice">建议：{html.escape(str(day_weather.get("advice", "天气信息仅供参考，请出行前查看天气 App。")))}</p>
                    </div>
                </div>
                """
            )

        weather_card_html_list.append(
            f"""
            <article class="weather-card">
                <h3>{destination}｜未来 {WEATHER_FORECAST_DAYS} 天天气</h3>
                {"".join(day_html_list)}
            </article>
            """
        )

    st.markdown(
        f"""
        <div class="weather-grid">{"".join(weather_card_html_list)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_travel_blessing() -> None:
    """render_travel_blessing：在攻略最后展示温和的旅行祝福语。"""

    st.markdown(
        """
        <div class="blessing-card">
            祝你这次旅行顺利又开心。记得提前确认天气、门票和交通安排，慢慢走、好好看，把喜欢的风景都装进记忆里。祝你旅途愉快呀～ 🌿✨🧳
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_visual_guide(
    markdown_text: str,
    parsed_request: dict,
    travel_json: dict | None = None,
    json_errors: list[str] | None = None,
    json_raw: str | None = None,
    weather_cards: list[dict] | None = None,
    generated_at: str | None = None,
) -> None:
    """render_visual_guide：把 Markdown 攻略渲染成高级卡片式视觉结果。"""

    # section_map：按标题拆分后的攻略内容。
    section_map = split_markdown_sections(markdown_text)

    if not section_map:
        with st.container(border=True):
            st.markdown(markdown_text)
        render_timeline({}, parsed_request, travel_json, json_errors, json_raw)
        render_weather_section(weather_cards)
        render_source_info({}, generated_at)
        render_travel_blessing()
        return

    render_overview_card(section_map)
    render_timeline(section_map, parsed_request, travel_json, json_errors, json_raw)
    render_food_cards(section_map, parsed_request, travel_json)
    render_advice_sections(section_map)
    render_weather_section(weather_cards)
    render_source_info(section_map, generated_at)
    render_travel_blessing()


def render_copy_button(markdown_text: str) -> None:
    """render_copy_button：渲染可复制 Markdown 攻略的按钮。"""

    # markdown_json：安全注入到 JavaScript 的攻略文本。
    markdown_json = json.dumps(markdown_text, ensure_ascii=False)

    st.html(
        f"""
        <style>
        .copy-widget {{
            font-family: Arial, sans-serif;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            padding: 8px 0;
            width: 100%;
        }}
        .copy-widget button {{
            border: 1px solid rgba(246, 199, 111, 0.32);
            border-radius: 999px;
            padding: 12px 18px;
            background: linear-gradient(135deg, #f6c76f, #fb923c);
            color: #17120a;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 12px 28px rgba(251, 146, 60, 0.18);
            width: 100%;
        }}
        .copy-widget span {{
            color: #cbd5e1;
            font-size: 14px;
        }}
        </style>
        <div class="copy-widget">
            <button id="copy-markdown-button">复制攻略</button>
            <span id="copy-markdown-status">一键复制 Markdown 全文。</span>
        </div>
        <script>
        const markdownText = {markdown_json};
        const copyButton = document.getElementById("copy-markdown-button");
        const copyStatus = document.getElementById("copy-markdown-status");
        copyButton.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText(markdownText);
                copyStatus.textContent = "已复制到剪贴板。";
            }} catch (error) {{
                copyStatus.textContent = "复制失败，请手动复制下方原文。";
            }}
        }});
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_result_actions(markdown_text: str) -> None:
    """render_result_actions：在结果底部展示复制、下载和重新生成操作。"""

    with st.container(border=True):
        st.markdown('<div class="result-actions-title">结果操作</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="source-card-text">复制给同行人、下载成 Markdown，或基于当前输入重新生成一版。</p>',
            unsafe_allow_html=True,
        )

        # action_columns：结果操作按钮区域。
        action_columns = st.columns([1.2, 0.8, 0.8])
        with action_columns[0]:
            render_copy_button(markdown_text)
        with action_columns[1]:
            st.download_button(
                label="下载 Markdown",
                data=markdown_text,
                file_name="ai-travel-guide.md",
                mime="text/markdown",
                key="download_markdown_action",
            )
        with action_columns[2]:
            if st.button("重新生成", key="regenerate_result_button"):
                st.session_state["force_regenerate"] = True
                st.rerun()


def render_markdown_source(markdown_text: str) -> None:
    """render_markdown_source：用折叠区域展示 Markdown 原文，并提供复制和下载。"""

    st.markdown('<h2 class="section-heading">Markdown 原文</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">默认收起，适合最后复制到笔记、公众号或行程文档中。</p>', unsafe_allow_html=True)

    with st.expander("查看可复制的 Markdown 攻略", expanded=False):
        # action_columns：复制和下载按钮区域。
        action_columns = st.columns([1, 1])
        with action_columns[0]:
            render_copy_button(markdown_text)
        with action_columns[1]:
            st.download_button(
                label="下载 Markdown 文件",
                data=markdown_text,
                file_name="ai-travel-guide.md",
                mime="text/markdown",
            )

        st.code(markdown_text, language="markdown")


def render_debug_panel(parsed_request: dict) -> None:
    """render_debug_panel：默认折叠展示系统识别出的结构化参数。"""

    # debug_data：开发调试用的结构化解析结果。
    debug_data = {
        "trip_type": parsed_request.get("trip_type"),
        "destination": parsed_request["destination"],
        "destinations": parsed_request.get("destinations", []),
        "trip_segments": parsed_request.get("trip_segments", []),
        "trip_notes": parsed_request.get("trip_notes", []),
        "days": parsed_request["days"],
        "nights": parsed_request["nights"],
        "budget_amount": parsed_request.get("budget_amount"),
        "currency": parsed_request.get("budget_currency") or "未指定",
        "style": parsed_request.get("style") or parsed_request.get("budget_level"),
        "preferences": "、".join(parsed_request["preferences"]),
    }

    with st.expander("Debug：查看系统识别参数", expanded=False):
        st.json(debug_data)


def format_public_search_status(search_message: str | None) -> str:
    """format_public_search_status：把内部搜索状态转换为普通用户可理解的提示。"""

    if not search_message:
        return "实时信息未启用，请出行前二次确认。"

    # raw_message：内部搜索状态文案。
    raw_message = str(search_message)
    if "已启用" in raw_message:
        return "已参考当前可用公开信息。"
    if "缓存" in raw_message:
        return "已参考近期可用信息。"
    if "未启用" in raw_message or "未配置" in raw_message:
        return "实时信息未启用，请出行前二次确认。"
    if "额度" in raw_message or "失败" in raw_message or "受限" in raw_message:
        return "实时信息暂时不可用，请出行前二次确认。"

    return "请出行前再次确认门票、预约、开放时间和交通政策。"


def render_search_status(search_message: str | None) -> None:
    """render_search_status：用小标签展示 Tavily 联网搜索状态。"""

    # safe_message：转义后的状态文案，避免 HTML 注入。
    safe_message = html.escape(format_public_search_status(search_message))
    st.markdown(
        f"""
        <div class="search-status-pill">
            <span class="search-status-dot"></span>
            <span>{safe_message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trust_strip(search_message: str | None, generated_at: str | None) -> None:
    """render_trust_strip：展示轻量信任说明，提醒用户核对实时信息。"""

    # search_status_text：联网搜索状态文案。
    search_status_text = format_public_search_status(search_message)

    # generated_time_text：攻略生成时间。
    generated_time_text = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")

    st.markdown(
        f"""
        <div class="trust-strip">
            <article class="trust-card">
                <span>Search Status</span>
                <strong>{html.escape(search_status_text)}</strong>
                <p>如未联网，实时门票、预约和开放时间请出行前再次核对。</p>
            </article>
            <article class="trust-card">
                <span>AI Generated</span>
                <strong>攻略由 AI 生成，仅供参考</strong>
                <p>路线、预算和餐饮建议适合作为初步规划，不替代官方信息。</p>
            </article>
            <article class="trust-card">
                <span>Updated</span>
                <strong>{html.escape(generated_time_text)}</strong>
                <p>门票、预约、开放时间和交通政策请以官方渠道为准。</p>
            </article>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_generation_count() -> int:
    """get_generation_count：读取当前浏览器 session 已生成攻略次数。"""

    return int(st.session_state.get("generation_count", 0))


def render_generation_quota() -> None:
    """render_generation_quota：展示 Beta 测试版当前 session 剩余生成次数。"""

    # remaining_count：当前 session 剩余生成次数。
    remaining_count = max(0, MAX_GENERATIONS_PER_SESSION - get_generation_count())
    st.markdown(
        f"""
        <p class="generation-quota">Beta 测试额度：本会话剩余 {remaining_count}/{MAX_GENERATIONS_PER_SESSION} 次生成。</p>
        """,
        unsafe_allow_html=True,
    )


def build_result_data(user_input: str) -> dict:
    """build_result_data：根据用户输入生成结果数据，但不把完整用户输入保存到 session。"""

    # generated_at：本次攻略生成时间。
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # parsed_request：自然语言解析结果。
    parsed_request = parse_travel_request(user_input)

    # cover_image_url：封面图地址，第一版是本地 SVG 占位。
    cover_image_url = generate_cover_image_url(parsed_request)

    with st.status("正在理解你的旅行需求...", expanded=True) as loading_status:
        time.sleep(0.2)
        if get_bool_config("USE_TAVILY", True) and get_tavily_api_key():
            loading_status.update(label="正在检索目的地最新信息...", state="running")
        else:
            loading_status.update(label="当前未启用联网搜索，正在切换普通生成模式...", state="running")

        # facts_context：联网搜索整理出的事实校验上下文。
        facts_context, source_records, search_message = build_facts_context(parsed_request)
        loading_status.update(label="正在规划每日路线...", state="running")
        # markdown_text/travel_json：最终展示的 Markdown 和页面时间线使用的结构化 JSON。
        markdown_text, api_message, travel_json, json_raw, json_errors = generate_travel_content(
            user_input,
            parsed_request,
            facts_context,
        )
        loading_status.update(label="正在整理美食与交通建议...", state="running")
        time.sleep(0.2)
        # weather_cards：Open-Meteo 免费天气数据，失败不影响攻略生成。
        try:
            weather_cards = build_weather_cards(parsed_request, travel_json)
        except Exception:
            weather_cards = []
        markdown_text = append_weather_and_blessing_to_markdown(markdown_text, weather_cards, generated_at)
        loading_status.update(label="正在生成专属旅行方案...", state="running")
        time.sleep(0.2)
        loading_status.update(label="专属旅行方案已生成", state="complete", expanded=False)

    return {
        "parsed_request": parsed_request,
        "cover_image_url": cover_image_url,
        "markdown_text": markdown_text,
        "api_message": api_message,
        "travel_json": travel_json,
        "json_raw": json_raw,
        "json_errors": json_errors,
        "weather_cards": weather_cards,
        "search_message": search_message,
        "generated_at": generated_at,
    }


def render_result_data(result_data: dict) -> None:
    """render_result_data：渲染已经生成并缓存在当前 session 中的攻略结果。"""

    # parsed_request：结构化旅行参数，不包含任何 API Key。
    parsed_request = result_data["parsed_request"]

    # markdown_text：最终展示和复制的 Markdown 攻略。
    markdown_text = result_data["markdown_text"]

    render_search_status(result_data.get("search_message"))
    render_trust_strip(result_data.get("search_message"), result_data.get("generated_at"))

    if result_data.get("api_message"):
        st.info(result_data["api_message"])

    render_debug_panel(parsed_request)
    render_cover(parsed_request, result_data["cover_image_url"])
    render_summary_bento(parsed_request)
    render_trip_segments_overview(parsed_request)
    render_visual_guide(
        markdown_text,
        parsed_request,
        result_data.get("travel_json"),
        result_data.get("json_errors"),
        result_data.get("json_raw"),
        result_data.get("weather_cards"),
        result_data.get("generated_at"),
    )
    render_result_actions(markdown_text)
    render_markdown_source(markdown_text)


def render_result(user_input: str) -> None:
    """render_result：兼容旧调用方式，立即生成并渲染完整旅行攻略。"""

    render_result_data(build_result_data(user_input))


def render_beta_notice() -> None:
    """render_beta_notice：在页面底部展示 Beta 测试版和隐私安全提醒。"""

    st.markdown(
        f"""
        <div class="beta-notice">{html.escape(BETA_NOTICE_TEXT)}</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """main：应用入口函数。"""

    setup_page()
    render_hero()

    # submitted：是否点击生成按钮。
    submitted, user_input = render_input_box()
    render_generation_quota()

    # force_regenerate：结果页“重新生成”按钮触发的再生成动作。
    force_regenerate = bool(st.session_state.pop("force_regenerate", False))

    if submitted or force_regenerate:
        if not user_input.strip():
            st.warning("请先输入一句旅行需求。")
        elif get_generation_count() >= MAX_GENERATIONS_PER_SESSION:
            st.warning(
                f"当前 Beta 测试版每个浏览器会话最多生成 {MAX_GENERATIONS_PER_SESSION} 次攻略。"
                "请刷新浏览器会话或稍后再试。"
            )
        else:
            # generation_count：点击生成才增加次数，页面重绘不会重复消耗 API。
            st.session_state["generation_count"] = get_generation_count() + 1

            # last_result_data：只保存生成后的旅行参数和攻略结果，不保存完整用户输入或任何密钥。
            st.session_state["last_result_data"] = build_result_data(user_input.strip())

    if "last_result_data" in st.session_state:
        render_result_data(st.session_state["last_result_data"])
    else:
        st.markdown(
            """
            <p class="hint">示例输入：我想去东京旅游，喜欢动漫、美食和夜景，预算5000，想玩 3 天 2 晚。</p>
            """,
            unsafe_allow_html=True,
        )

    render_beta_notice()


if __name__ == "__main__":
    main()
import html
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None


# DEFAULT_TRAVEL_DAYS：用户没有写旅行天数时，默认按 3 天处理。
DEFAULT_TRAVEL_DAYS = 3

# DEFAULT_TRAVEL_NIGHTS：用户没有写住宿晚数时，默认按 2 晚处理。
DEFAULT_TRAVEL_NIGHTS = 2

# DEFAULT_BUDGET_LEVEL：用户没有写预算时，默认使用普通预算。
DEFAULT_BUDGET_LEVEL = "普通预算"

# DEFAULT_BUDGET_CURRENCY：用户输入预算数字但没有写货币单位时，默认按人民币处理。
DEFAULT_BUDGET_CURRENCY = "CNY"

# DEFAULT_DESTINATION：用户没有写明确目的地时，用于演示的默认目的地。
DEFAULT_DESTINATION = "东京"

# DEEPSEEK_BASE_URL：DeepSeek API 的基础地址，OpenAI SDK 会通过这个地址请求 DeepSeek。
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DEFAULT_DEEPSEEK_MODEL：用户没有在 .env 配置模型时，默认使用的 DeepSeek 模型。
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

# DEFAULT_SEARCH_MAX_RESULTS：每个搜索查询最多保留的结果数量。
DEFAULT_SEARCH_MAX_RESULTS = 3

# DEFAULT_TAVILY_SEARCH_DEPTH：Tavily 默认使用 basic 搜索，控制搜索额度消耗。
DEFAULT_TAVILY_SEARCH_DEPTH = "basic"

# DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE：每份攻略默认最多调用 Tavily 的次数。
DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE = 1

# TAVILY_CACHE_FILE：Tavily 搜索结果本地缓存文件，避免 24 小时内重复消耗额度。
TAVILY_CACHE_FILE = "tavily_cache.json"

# TAVILY_CACHE_TTL_SECONDS：Tavily 缓存有效期，默认 24 小时。
TAVILY_CACHE_TTL_SECONDS = 24 * 60 * 60

# TAVILY_CACHE_PATH：Tavily 缓存文件的绝对路径。
TAVILY_CACHE_PATH = Path(__file__).with_name(TAVILY_CACHE_FILE)

# MAX_GENERATIONS_PER_SESSION：Beta 测试版每个浏览器会话最多生成攻略次数，避免 API 被滥用。
MAX_GENERATIONS_PER_SESSION = 3

# BETA_NOTICE_TEXT：上线前页面底部展示的 Beta 和隐私安全提醒。
BETA_NOTICE_TEXT = "当前为 Beta 测试版。AI 生成内容仅供参考，门票、预约、开放时间、交通政策等信息请以官方渠道为准。请勿输入身份证号、手机号、住址、护照号等敏感个人信息。"

# SAMPLE_PROMPTS：Hero 输入区的示例旅行需求，点击后会自动填入对话框。
SAMPLE_PROMPTS = [
    {"label": "东京3天动漫美食游", "prompt": "我想去东京旅游，喜欢动漫、美食和夜景，预算5000，3 天 2 晚"},
    {"label": "大阪京都5日自由行", "prompt": "我想去大阪京都自由行，喜欢历史、美食、购物和拍照，普通预算，5 天 4 晚"},
    {"label": "云南7日慢旅行", "prompt": "我想去云南慢旅行，喜欢自然、咖啡、拍照和少数民族文化，普通预算，7 天 6 晚"},
    {"label": "首尔4日拍照美食游", "prompt": "我想去首尔旅游，喜欢咖啡、购物、夜景、拍照和美食，普通预算，4 天 3 晚"},
]


if load_dotenv:
    # load_dotenv：读取本地 .env 文件，方便初学者不用每次手动设置环境变量。
    load_dotenv()


def get_config_value(config_name: str, default_value: str = "") -> str:
    """get_config_value：优先从 .env/环境变量读取配置，其次兼容 Streamlit secrets。"""

    # env_value：load_dotenv 后从系统环境变量中读取到的配置值。
    env_value = os.getenv(config_name)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()

    try:
        # secret_value：Streamlit Cloud 部署时可从 st.secrets 读取的配置值。
        secret_value = st.secrets.get(config_name)
        if secret_value is not None and str(secret_value).strip():
            return str(secret_value).strip()
    except Exception:
        return default_value

    return default_value


def get_bool_config(config_name: str, default_value: bool = False) -> bool:
    """get_bool_config：把环境变量或 secrets 中的开关配置转换成布尔值。"""

    # raw_value：配置原始字符串。
    raw_value = get_config_value(config_name, str(default_value)).strip().lower()
    return raw_value in {"1", "true", "yes", "y", "on", "启用", "是"}


def get_int_config(config_name: str, default_value: int) -> int:
    """get_int_config：读取整数配置，非法值自动使用默认值。"""

    # raw_value：配置原始字符串。
    raw_value = get_config_value(config_name, str(default_value)).strip()
    try:
        return int(raw_value)
    except ValueError:
        return default_value


def setup_page() -> None:
    """setup_page：设置 Streamlit 页面基础信息和自定义样式。"""

    st.set_page_config(
        page_title="AI 旅游攻略 Agent",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # custom_css：控制页面视觉风格，让 Streamlit 默认界面更接近高级旅行杂志和 AI 工具。
    custom_css = """
    <style>
    :root {
        --bg-deep: #05070f;
        --panel: rgba(15, 23, 42, 0.58);
        --panel-strong: rgba(15, 23, 42, 0.82);
        --panel-warm: rgba(120, 75, 32, 0.18);
        --line: rgba(255, 255, 255, 0.14);
        --line-soft: rgba(255, 255, 255, 0.08);
        --text-soft: #cbd5e1;
        --text-muted: #94a3b8;
        --cyan: #38bdf8;
        --mint: #34d399;
        --rose: #fb7185;
        --gold: #f6c76f;
        --orange: #fb923c;
        --shadow: 0 28px 90px rgba(0, 0, 0, 0.34);
        --glass-blur: blur(20px);
    }

    .stApp {
        color: #f8fafc;
        background:
            linear-gradient(115deg, rgba(251, 146, 60, 0.14) 0%, transparent 28%),
            linear-gradient(235deg, rgba(56, 189, 248, 0.15) 0%, transparent 32%),
            linear-gradient(180deg, rgba(255, 255, 255, 0.035), transparent 22%),
            linear-gradient(145deg, #05070f 0%, #0d1324 38%, #111827 68%, #05070f 100%);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.028) 1px, transparent 1px);
        background-size: 72px 72px;
        mask-image: linear-gradient(180deg, rgba(0,0,0,0.72), transparent 72%);
        opacity: 0.38;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.35rem;
        padding-bottom: 5rem;
    }

    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 3.1rem;
        padding: 0.8rem 1rem;
        border: 1px solid var(--line-soft);
        border-radius: 999px;
        background: rgba(8, 12, 24, 0.58);
        box-shadow: 0 16px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.02rem;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: inline-grid;
        place-items: center;
        color: #111827;
        background: linear-gradient(135deg, var(--gold), #fff7d6 48%, var(--orange));
        box-shadow: 0 0 0 1px rgba(255,255,255,0.32), 0 12px 28px rgba(251, 146, 60, 0.24);
    }

    .nav-links {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: var(--text-soft);
        font-size: 0.92rem;
    }

    .nav-links span {
        padding: 0.45rem 0.72rem;
        border-radius: 999px;
        color: #dbeafe;
    }

    .hero {
        margin-bottom: 2rem;
    }

    .hero-layout {
        display: grid;
        grid-template-columns: minmax(0, 1.16fr) minmax(280px, 0.84fr);
        gap: 1.35rem;
        align-items: stretch;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 0.8rem;
        border: 1px solid rgba(246, 199, 111, 0.34);
        border-radius: 999px;
        color: #fde68a;
        background: rgba(120, 75, 32, 0.22);
        font-size: 0.84rem;
        margin-bottom: 1.15rem;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2.55rem, 5.6vw, 5.65rem);
        line-height: 0.98;
        letter-spacing: 0;
        max-width: 930px;
    }

    .hero p {
        margin: 1.15rem 0 0;
        max-width: 720px;
        color: #d1d5db;
        font-size: 1.08rem;
        line-height: 1.85;
    }

    .hero-panel {
        min-height: 100%;
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 1.25rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.035)),
            linear-gradient(145deg, rgba(251, 146, 60, 0.10), rgba(56, 189, 248, 0.08));
        box-shadow: var(--shadow);
        backdrop-filter: var(--glass-blur);
    }

    .mini-card {
        border: 1px solid var(--line-soft);
        border-radius: 20px;
        padding: 1rem;
        background: rgba(3, 7, 18, 0.42);
        margin-bottom: 0.85rem;
    }

    .mini-card span {
        display: block;
        color: var(--gold);
        font-size: 0.78rem;
        margin-bottom: 0.45rem;
    }

    .mini-card strong {
        display: block;
        font-size: 1.2rem;
        margin-bottom: 0.35rem;
    }

    .mini-card p {
        margin: 0;
        color: var(--text-muted);
        line-height: 1.55;
        font-size: 0.92rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--line) !important;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.105), rgba(255, 255, 255, 0.035)),
            rgba(10, 15, 30, 0.58) !important;
        border-radius: 26px !important;
        box-shadow: var(--shadow);
        backdrop-filter: var(--glass-blur);
    }

    [data-testid="stVerticalBlockBorderWrapper"] h3 {
        color: #fff7ed;
        letter-spacing: 0;
    }

    .input-title {
        font-size: 1.02rem;
        color: #fef3c7;
        margin: 0 0 0.65rem;
        font-weight: 700;
    }

    .sample-title {
        color: var(--text-muted);
        font-size: 0.88rem;
        margin: 0.85rem 0 0.45rem;
    }

    .stTextArea textarea {
        min-height: 136px !important;
        border-radius: 22px !important;
        border: 1px solid rgba(246, 199, 111, 0.24) !important;
        background:
            linear-gradient(145deg, rgba(2, 6, 23, 0.78), rgba(15, 23, 42, 0.62)) !important;
        color: #f8fafc !important;
        font-size: 1.03rem !important;
        line-height: 1.65 !important;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
    }

    .stTextArea textarea:focus {
        border-color: rgba(246, 199, 111, 0.88) !important;
        box-shadow: 0 0 0 4px rgba(246, 199, 111, 0.14) !important;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        width: 100%;
        border: 1px solid rgba(246, 199, 111, 0.26);
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(246, 199, 111, 0.95), rgba(251, 146, 60, 0.92));
        color: #17120a;
        font-weight: 800;
        padding: 0.78rem 1rem;
        box-shadow: 0 14px 34px rgba(251, 146, 60, 0.18);
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        color: #17120a;
        filter: brightness(1.05);
        border-color: rgba(255, 247, 237, 0.55);
    }

    .cover-card {
        aspect-ratio: 16 / 9;
        min-height: 420px;
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background-size: cover;
        background-position: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 34px 110px rgba(0, 0, 0, 0.45);
        margin: 2.15rem 0 1.45rem;
    }

    .cover-card::after {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(90deg, rgba(2, 6, 23, 0.76), rgba(2, 6, 23, 0.22) 58%, rgba(2, 6, 23, 0.64)),
            linear-gradient(180deg, rgba(2, 6, 23, 0.05) 20%, rgba(2, 6, 23, 0.82));
    }

    .cover-content {
        position: absolute;
        inset: auto clamp(1.25rem, 4vw, 3.2rem) clamp(1.25rem, 4vw, 3.2rem) clamp(1.25rem, 4vw, 3.2rem);
        z-index: 1;
    }

    .cover-content .label {
        color: #fde68a;
        font-size: 0.88rem;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .cover-content h2 {
        margin: 0;
        font-size: clamp(2.55rem, 6.2vw, 5.4rem);
        line-height: 0.98;
        letter-spacing: 0;
    }

    .cover-content p {
        margin: 0.9rem 0 0;
        max-width: 760px;
        color: #f8fafc;
        font-size: 1rem;
        line-height: 1.75;
    }

    .cover-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1rem;
    }

    .cover-badge {
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        color: #fff7ed;
        background: rgba(15, 23, 42, 0.46);
        backdrop-filter: blur(10px);
        font-size: 0.86rem;
    }

    .bento-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        grid-auto-rows: minmax(112px, auto);
        gap: 0.9rem;
        margin: 1rem 0 2rem;
    }

    .bento-card {
        border: 1px solid var(--line-soft);
        border-radius: 24px;
        padding: 1.05rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.52);
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        backdrop-filter: var(--glass-blur);
        min-height: 112px;
    }

    .bento-card.large {
        grid-column: span 2;
    }

    .bento-card.warm {
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.19), rgba(251, 146, 60, 0.065)),
            rgba(15, 23, 42, 0.52);
    }

    .bento-card span {
        display: block;
        color: #fcd34d;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }

    .bento-card strong {
        display: block;
        color: #f8fafc;
        font-size: clamp(1.05rem, 2.2vw, 1.45rem);
        line-height: 1.18;
        letter-spacing: 0;
    }

    .bento-card p {
        color: var(--text-muted);
        line-height: 1.55;
        margin: 0.55rem 0 0;
        font-size: 0.92rem;
    }

    .section-heading {
        margin: 2.25rem 0 0.35rem;
        color: #fff7ed;
        font-size: clamp(1.55rem, 3vw, 2.1rem);
        letter-spacing: 0;
    }

    .section-subtitle {
        margin: 0 0 1.1rem;
        color: var(--text-muted);
        line-height: 1.65;
    }

    .timeline-grid,
    .food-grid,
    .info-grid,
    .warning-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.6rem;
    }

    .timeline-day,
    .food-card,
    .info-card,
    .warning-card {
        border: 1px solid var(--line-soft);
        border-radius: 26px;
        padding: 1.15rem;
        background:
            linear-gradient(145deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.25);
        backdrop-filter: var(--glass-blur);
    }

    .timeline-day h3,
    .food-card h3,
    .info-card h3,
    .warning-card h3 {
        margin: 0 0 0.85rem;
        color: #fff7ed;
        letter-spacing: 0;
    }

    .card-title-row {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin-bottom: 0.82rem;
    }

    .card-title-row h3 {
        margin: 0;
    }

    .info-icon,
    .warning-icon {
        width: 36px;
        height: 36px;
        border-radius: 13px;
        display: grid;
        place-items: center;
        font-weight: 900;
        flex: 0 0 auto;
    }

    .info-icon {
        color: #082f49;
        background: linear-gradient(135deg, #7dd3fc, #38bdf8);
    }

    .warning-icon {
        color: #17120a;
        background: linear-gradient(135deg, #f6c76f, #fb923c);
    }

    .timeline-slot {
        display: grid;
        grid-template-columns: 44px minmax(0, 1fr);
        gap: 0.85rem;
        padding: 0.82rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .timeline-slot:first-of-type {
        border-top: 0;
        padding-top: 0;
    }

    .slot-icon {
        width: 40px;
        height: 40px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        color: #17120a;
        background: linear-gradient(135deg, #f6c76f, #fb923c);
        font-weight: 900;
        box-shadow: 0 12px 26px rgba(251, 146, 60, 0.18);
    }

    .slot-time {
        color: #fcd34d;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .slot-place {
        color: #f8fafc;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .slot-desc,
    .food-card p,
    .info-card p,
    .warning-card p {
        color: #cbd5e1;
        line-height: 1.62;
        margin: 0;
        font-size: 0.93rem;
    }

    .food-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.9rem;
    }

    .food-meta span {
        border: 1px solid rgba(246, 199, 111, 0.24);
        border-radius: 999px;
        padding: 0.32rem 0.58rem;
        color: #fde68a;
        background: rgba(120, 75, 32, 0.18);
        font-size: 0.8rem;
    }

    .food-location {
        color: #9ca3af;
        font-size: 0.82rem;
        line-height: 1.55;
        margin: -0.42rem 0 0.76rem;
    }

    .food-map-keyword {
        color: #fcd34d;
        font-size: 0.78rem;
        line-height: 1.45;
        margin-top: 0.65rem;
        opacity: 0.9;
    }

    .info-card {
        background:
            linear-gradient(145deg, rgba(56, 189, 248, 0.13), rgba(255, 255, 255, 0.032)),
            rgba(15, 23, 42, 0.54);
    }

    .warning-card {
        border-color: rgba(251, 146, 60, 0.22);
        background:
            linear-gradient(145deg, rgba(251, 146, 60, 0.18), rgba(127, 29, 29, 0.10)),
            rgba(15, 23, 42, 0.54);
    }

    .markdown-actions {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 220px;
        gap: 0.8rem;
        align-items: center;
        margin-bottom: 0.7rem;
    }

    .hint {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.65;
    }

    .search-status-pill {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        max-width: 100%;
        gap: 0.48rem;
        margin: 0.3rem 0 1.1rem;
        padding: 0.5rem 0.72rem;
        border-radius: 999px;
        border: 1px solid rgba(246, 199, 111, 0.22);
        background: rgba(15, 23, 42, 0.48);
        color: #fde68a;
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
        backdrop-filter: var(--glass-blur);
        font-size: 0.86rem;
        line-height: 1.4;
    }

    .search-status-dot {
        width: 0.46rem;
        height: 0.46rem;
        flex: 0 0 auto;
        border-radius: 999px;
        background: var(--gold);
        box-shadow: 0 0 18px rgba(246, 199, 111, 0.52);
    }

    .generation-quota {
        color: #fde68a;
        font-size: 0.86rem;
        margin-top: 0.55rem;
        opacity: 0.92;
    }

    .beta-notice {
        margin-top: 2.2rem;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(246, 199, 111, 0.22);
        border-radius: 20px;
        background:
            linear-gradient(145deg, rgba(246, 199, 111, 0.12), rgba(255, 255, 255, 0.035)),
            rgba(15, 23, 42, 0.58);
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.9rem;
        backdrop-filter: var(--glass-blur);
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--line-soft);
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.50);
        backdrop-filter: var(--glass-blur);
    }

    @media (max-width: 760px) {
        .block-container {
            padding-top: 0.9rem;
        }

        .top-nav {
            border-radius: 22px;
            align-items: flex-start;
        }

        .nav-links {
            display: none;
        }

        .hero-layout,
        .timeline-grid,
        .food-grid,
        .info-grid,
        .warning-grid,
        .markdown-actions {
            grid-template-columns: 1fr;
        }

        .bento-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .bento-card.large {
            grid-column: span 2;
        }

        .cover-card {
            min-height: 340px;
            border-radius: 24px;
        }
    }

    @media (max-width: 520px) {
        .bento-grid {
            grid-template-columns: 1fr;
        }

        .bento-card.large {
            grid-column: span 1;
        }
    }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)


def parse_chinese_number(number_text: str) -> int:
    """parse_chinese_number：把常见中文数字转换成整数。"""

    # chinese_number_map：保存中文数字到阿拉伯数字的对应关系。
    chinese_number_map = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    if number_text.isdigit():
        return int(number_text)

    if number_text == "十":
        return 10

    if number_text.startswith("十"):
        return 10 + chinese_number_map.get(number_text[-1], 0)

    if "十" in number_text:
        # parts：中文数字按“十”拆分后的十位和个位。
        parts = number_text.split("十")
        tens = chinese_number_map.get(parts[0], 1)
        ones = chinese_number_map.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones

    return chinese_number_map.get(number_text, DEFAULT_TRAVEL_DAYS)


def parse_chinese_amount(amount_text: str) -> float | None:
    """parse_chinese_amount：把中文预算金额转换成数字，例如“一万”转成 10000。"""

    # cleaned_amount：清理空格后的中文金额文本。
    cleaned_amount = amount_text.strip()
    if not cleaned_amount:
        return None

    if re.fullmatch(r"[0-9][0-9,]*(?:\.\d+)?", cleaned_amount):
        return float(cleaned_amount.replace(",", ""))

    # chinese_digit_map：中文数字字符和数值的对应关系。
    chinese_digit_map = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    if cleaned_amount.endswith("万"):
        # base_text：中文金额中“万”前面的数字部分。
        base_text = cleaned_amount[:-1]
        if not base_text:
            return 10000.0
        base_value = parse_chinese_number(base_text)
        return float(base_value * 10000)

    if cleaned_amount in chinese_digit_map:
        return float(chinese_digit_map[cleaned_amount])

    if "千" in cleaned_amount:
        # thousand_parts：中文金额按“千”拆分后的千位和余数。
        thousand_parts = cleaned_amount.split("千", 1)
        thousands = parse_chinese_number(thousand_parts[0] or "一") * 1000
        rest = parse_chinese_amount(thousand_parts[1]) if thousand_parts[1] else 0
        return float(thousands + (rest or 0))

    return None


def normalize_currency_unit(currency_text: str | None) -> str:
    """normalize_currency_unit：把用户输入的货币单位统一成标准货币代码。"""

    if not currency_text:
        return DEFAULT_BUDGET_CURRENCY

    # normalized_unit：统一大小写并去除空格后的货币单位。
    normalized_unit = currency_text.strip().upper()

    # currency_alias_map：常见货币表达和标准货币代码的对应关系。
    currency_alias_map = {
        "人民币": "CNY",
        "RMB": "CNY",
        "CNY": "CNY",
        "元": "CNY",
        "块": "CNY",
        "日元": "JPY",
        "日币": "JPY",
        "日圓": "JPY",
        "JPY": "JPY",
        "美元": "USD",
        "美金": "USD",
        "USD": "USD",
        "欧元": "EUR",
        "欧": "EUR",
        "EUR": "EUR",
        "韩元": "KRW",
        "韩币": "KRW",
        "KRW": "KRW",
    }

    return currency_alias_map.get(normalized_unit, DEFAULT_BUDGET_CURRENCY)


def get_currency_name(currency_code: str) -> str:
    """get_currency_name：把标准货币代码转换成中文显示名称。"""

    # currency_name_map：标准货币代码和中文名称的对应关系。
    currency_name_map = {
        "CNY": "人民币",
        "JPY": "日元",
        "USD": "美元",
        "EUR": "欧元",
        "KRW": "韩元",
    }

    return currency_name_map.get(currency_code, currency_code)


def format_budget_amount(amount: float) -> str:
    """format_budget_amount：把预算金额格式化为适合页面展示的文本。"""

    # numeric_amount：兼容 int 和 float 的预算金额数值。
    numeric_amount = float(amount)

    if numeric_amount.is_integer():
        return f"{int(numeric_amount):,}"

    return f"{numeric_amount:,.2f}".rstrip("0").rstrip(".")


def parse_budget_info(cleaned_input: str, budget_level: str) -> dict:
    """parse_budget_info：识别用户输入中的预算金额和货币单位。"""

    # budget_pattern：匹配“预算5000”“预算一万”“预算10万日元”“预算800 USD”等表达。
    budget_pattern = re.compile(
        r"(?:预算|总预算|花费|费用)\s*"
        r"(?:约|大概|大约|控制在|不超过|不超|以内|左右|是|为|:|：)?\s*"
        r"([0-9][0-9,]*(?:\.\d+)?|[零一二两三四五六七八九十百千万]+)\s*"
        r"(万)?\s*"
        r"(人民币|日元|日币|日圓|美元|美金|欧元|欧|韩元|韩币|USD|EUR|JPY|KRW|RMB|CNY|元|块)?",
        re.IGNORECASE,
    )

    # budget_match：预算金额匹配结果。
    budget_match = budget_pattern.search(cleaned_input)
    if not budget_match:
        return {
            "amount": None,
            "currency": None,
            "currency_name": None,
            "display": budget_level,
            "level": budget_level,
            "has_explicit_amount": False,
        }

    # amount_text：预算金额文本。
    amount_text = budget_match.group(1)

    # amount_value：预算金额数值。
    amount_value = parse_chinese_amount(amount_text)
    if amount_value is None:
        amount_value = float(amount_text.replace(",", ""))

    if budget_match.group(2) and "万" not in amount_text:
        amount_value *= 10000

    # currency_code：标准货币代码；用户没写单位时默认 CNY。
    currency_code = normalize_currency_unit(budget_match.group(3))

    # currency_name：中文货币名称。
    currency_name = get_currency_name(currency_code)

    # budget_display：页面和提示词展示的预算文本。
    budget_display = f"{format_budget_amount(amount_value)} {currency_name} ({currency_code})"

    # normalized_amount：整数金额保存为 int，便于 Debug 区显示 10000 而不是 10000.0。
    normalized_amount = int(amount_value) if float(amount_value).is_integer() else amount_value

    return {
        "amount": normalized_amount,
        "currency": currency_code,
        "currency_name": currency_name,
        "display": budget_display,
        "level": budget_level,
        "has_explicit_amount": True,
    }


def infer_destination_currency(destination: str) -> str | None:
    """infer_destination_currency：根据目的地粗略推断当地常用货币。"""

    # destination_currency_keywords：国外目的地关键词和当地货币代码。
    destination_currency_keywords = {
        "JPY": ["日本", "东京", "大阪", "京都", "北海道", "冲绳", "奈良", "福冈", "名古屋", "札幌", "箱根"],
        "KRW": ["韩国", "首尔", "釜山", "济州"],
        "EUR": [
            "欧洲",
            "法国",
            "巴黎",
            "意大利",
            "罗马",
            "米兰",
            "德国",
            "柏林",
            "西班牙",
            "巴塞罗那",
            "荷兰",
            "阿姆斯特丹",
            "葡萄牙",
            "希腊",
            "瑞士",
        ],
        "USD": ["美国", "纽约", "洛杉矶", "旧金山", "西雅图", "夏威夷"],
    }

    for currency_code, keyword_list in destination_currency_keywords.items():
        if any(keyword in destination for keyword in keyword_list):
            return currency_code

    return None


def build_exchange_hint(parsed_request: dict) -> str | None:
    """build_exchange_hint：为国外目的地生成粗略换算提示。"""

    # budget_amount：用户输入的预算金额。
    budget_amount = parsed_request.get("budget_amount")
    if not budget_amount:
        return None

    # destination_currency：根据目的地推断出的当地货币。
    destination_currency = infer_destination_currency(parsed_request["destination"])
    if not destination_currency:
        return None

    # source_currency：用户输入预算的货币代码。
    source_currency = parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY

    # cny_to_currency_rate：人民币到其他货币的粗略换算比例。
    cny_to_currency_rate = {
        "JPY": 20.0,
        "KRW": 190.0,
        "EUR": 0.13,
        "USD": 0.14,
    }

    # currency_to_cny_rate：其他货币到人民币的粗略换算比例。
    currency_to_cny_rate = {
        "JPY": 0.05,
        "KRW": 0.0053,
        "EUR": 7.8,
        "USD": 7.2,
        "CNY": 1.0,
    }

    if source_currency == DEFAULT_BUDGET_CURRENCY and destination_currency in cny_to_currency_rate:
        # converted_amount：人民币预算换算成目的地当地货币的粗略金额。
        converted_amount = budget_amount * cny_to_currency_rate[destination_currency]
        return (
            f"粗略换算：{format_budget_amount(budget_amount)} 人民币约 "
            f"{format_budget_amount(converted_amount)} {get_currency_name(destination_currency)}"
            "，汇率仅供参考，请以出行前实际汇率为准。"
        )

    if source_currency != DEFAULT_BUDGET_CURRENCY and source_currency in currency_to_cny_rate:
        # converted_amount：外币预算换算成人民币的粗略金额。
        converted_amount = budget_amount * currency_to_cny_rate[source_currency]
        return (
            f"粗略换算：{format_budget_amount(budget_amount)} {get_currency_name(source_currency)}约 "
            f"{format_budget_amount(converted_amount)} 人民币"
            "，汇率仅供参考，请以出行前实际汇率为准。"
        )

    return None


def extract_destination(cleaned_input: str) -> str:
    """extract_destination：从用户输入中优先提取明确目的地。"""

    # destination_patterns：从强到弱排列的目的地匹配规则。
    destination_patterns = [
        r"(?:^|[，。,\s])([一-龥A-Za-z]{2,20})\s*(?:[0-9一二两三四五六七八九十]+)\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行)",
        r"想去(?!看看|看一看|看|尝|尝一尝|吃|逛)([一-龥A-Za-z]{2,20}?)(?:旅游|旅行|自由行|游|度假|玩|看|赏|吃|逛|，|。|,|\s|$)",
        r"去(?!看看|看一看|看|尝|尝一尝|吃|逛)([一-龥A-Za-z]{2,20}?)(?:旅游|旅行|自由行|游|度假|玩|看|赏|吃|逛|，|。|,|\s|$)",
        r"(?:^|[，。,\s])([一-龥A-Za-z]{2,20}?)\s*(?:旅游|旅行|自由行|度假|游)",
        r"目的地[:：]\s*([一-龥A-Za-z]{2,20})",
    ]

    # invalid_destination_words：不应被当成目的地的动作词或泛词。
    invalid_destination_words = {
        "看看",
        "看一看",
        "看",
        "尝",
        "尝一尝",
        "美食",
        "夜景",
        "其他景点",
        "景点",
    }

    for pattern in destination_patterns:
        match = re.search(pattern, cleaned_input)
        if not match:
            continue

        # destination：当前规则识别出的目的地。
        destination = match.group(1).strip()
        destination = re.sub(r"^(?:我|我们|本人)?(?:想去|要去|计划去|打算去|去)", "", destination).strip()
        if destination and destination not in invalid_destination_words:
            return destination

    return DEFAULT_DESTINATION


def extract_trip_days(cleaned_input: str) -> tuple[int, int]:
    """extract_trip_days：识别“7日游”“7天”“七日”等明确天数。"""

    # days_patterns：可识别的天数表达。
    days_patterns = [
        r"([0-9一二两三四五六七八九十]+)\s*(?:日游|日旅行|日自由行|天游|天旅行|天自由行)",
        r"([0-9一二两三四五六七八九十]+)\s*(?:天|日)(?!元|币)",
    ]

    for pattern in days_patterns:
        match = re.search(pattern, cleaned_input)
        if match:
            days = max(1, parse_chinese_number(match.group(1)))
            return days, max(0, days - 1)

    return DEFAULT_TRAVEL_DAYS, DEFAULT_TRAVEL_NIGHTS


def infer_budget_level(cleaned_input: str) -> str:
    """infer_budget_level：识别用户输入中的预算风格档位。"""

    if re.search(r"穷游|省钱|低预算|便宜|学生党", cleaned_input):
        return "经济预算"

    if re.search(r"舒适一点|舒适|舒服|品质|高端|豪华|不差钱|预算充足", cleaned_input):
        return "舒适预算"

    return DEFAULT_BUDGET_LEVEL


def extract_preferences(cleaned_input: str) -> list[str]:
    """extract_preferences：从用户输入中提取旅行偏好、景点和体验主题。"""

    # preference_keywords：可识别的旅行偏好关键词。
    preference_keywords = [
        "西湖",
        "灵隐寺",
        "美食",
        "夜景",
        "自然",
        "动漫",
        "购物",
        "历史",
        "拍照",
        "文化",
        "博物馆",
        "亲子",
        "海边",
        "徒步",
        "温泉",
        "咖啡",
        "艺术",
    ]

    # preferences：从用户输入中识别出的偏好列表。
    preferences = []
    for keyword in preference_keywords:
        if keyword in cleaned_input and keyword not in preferences:
            preferences.append(keyword)

    if not preferences:
        preferences = ["美食", "拍照"]

    return preferences


def extract_fact_check_spots(parsed_request: dict) -> list[str]:
    """extract_fact_check_spots：提取需要联网校验门票、预约和开放时间的景点。"""

    # generic_preferences：不适合作为具体景点搜索的泛偏好。
    generic_preferences = {
        "美食",
        "夜景",
        "自然",
        "动漫",
        "购物",
        "历史",
        "拍照",
        "文化",
        "博物馆",
        "亲子",
        "海边",
        "徒步",
        "温泉",
        "咖啡",
        "艺术",
    }

    # spot_list：从偏好里筛出的具体景点。
    spot_list = [item for item in parsed_request["preferences"] if item not in generic_preferences]

    if not spot_list:
        spot_list = ["主要景点"]

    return spot_list[:4]


def build_fact_search_queries(parsed_request: dict) -> list[str]:
    """build_fact_search_queries：生成省额度的合并搜索查询，避免每个景点单独搜索。"""

    # destination：目的地名称。
    destination = parsed_request["destination"]

    # spot_list：需要查询的景点列表。
    spot_list = extract_fact_check_spots(parsed_request)

    # important_spots：用户明确提到的重点景点，最多放入 4 个，避免 query 过长。
    important_spots = [spot for spot in spot_list if spot != "主要景点"][:4]

    # spot_text：重点景点文本，没有明确景点时使用热门景点兜底。
    spot_text = " ".join(important_spots) if important_spots else "热门景点"

    # query_list：省额度模式下的合并查询列表；默认只会执行第一条。
    query_list = [
        f"{destination} {spot_text} 旅游 景点 门票 预约 开放时间 最新 交通 政策",
        f"{destination} 官方 旅游 景区规则 门票 预约 开放时间 最新政策",
    ]

    return query_list


def get_tavily_api_key() -> str | None:
    """get_tavily_api_key：读取 Tavily API Key，并忽略示例占位值。"""

    # tavily_api_key：从 .env、环境变量或 Streamlit secrets 读取的 Tavily API Key。
    tavily_api_key = get_config_value("TAVILY_API_KEY", "").strip()
    if not tavily_api_key or tavily_api_key.startswith("tvly-your"):
        return None

    return tavily_api_key


def get_deepseek_api_key() -> str | None:
    """get_deepseek_api_key：读取 DeepSeek API Key，并忽略示例占位值。"""

    # deepseek_api_key：从 .env、环境变量或 Streamlit secrets 读取的 DeepSeek API Key。
    deepseek_api_key = get_config_value("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_api_key or deepseek_api_key.startswith("sk-your"):
        return None

    return deepseek_api_key


def get_deepseek_model_name() -> str:
    """get_deepseek_model_name：读取 DeepSeek 模型名，未配置时使用默认模型。"""

    return get_config_value("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def get_tavily_search_depth() -> str:
    """get_tavily_search_depth：读取 Tavily 搜索深度，并强制使用 basic 省额度模式。"""

    # configured_depth：用户配置的搜索深度；为省额度，任何非 basic 配置都会被降级为 basic。
    configured_depth = get_config_value("TAVILY_SEARCH_DEPTH", DEFAULT_TAVILY_SEARCH_DEPTH).strip().lower()
    return DEFAULT_TAVILY_SEARCH_DEPTH if configured_depth != DEFAULT_TAVILY_SEARCH_DEPTH else configured_depth


def get_tavily_max_searches_per_guide() -> int:
    """get_tavily_max_searches_per_guide：读取每份攻略最多搜索次数，并默认限制为 1 次。"""

    # configured_limit：用户配置的每份攻略最大 Tavily 调用次数。
    configured_limit = get_int_config("TAVILY_MAX_SEARCHES_PER_GUIDE", DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE)
    return max(0, min(configured_limit, DEFAULT_TAVILY_MAX_SEARCHES_PER_GUIDE))


def normalize_tavily_query(destination: str, query: str) -> str:
    """normalize_tavily_query：把目的地和 query 标准化，用于判断相似搜索并命中缓存。"""

    # token_list：从 query 中提取的中英文关键词。
    token_list = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", query.lower())

    # normalized_tokens：去重排序后的关键词，使词序轻微变化时仍可复用缓存。
    normalized_tokens = sorted(set(token_list))

    return f"{destination.strip().lower()}|{'|'.join(normalized_tokens)}"


def build_tavily_cache_key(destination: str, query: str) -> str:
    """build_tavily_cache_key：为目的地和相似 query 生成稳定缓存 key。"""

    # normalized_query：标准化后的 query 文本。
    normalized_query = normalize_tavily_query(destination, query)
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


def load_tavily_cache() -> dict:
    """load_tavily_cache：读取本地 Tavily 缓存文件。"""

    if not TAVILY_CACHE_PATH.exists():
        return {}

    try:
        with TAVILY_CACHE_PATH.open("r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    except Exception:
        return {}


def save_tavily_cache(cache_data: dict) -> None:
    """save_tavily_cache：把 Tavily 搜索结果写入本地缓存文件。"""

    with TAVILY_CACHE_PATH.open("w", encoding="utf-8") as cache_file:
        json.dump(cache_data, cache_file, ensure_ascii=False, indent=2)


def get_cached_tavily_results(destination: str, query: str) -> list[dict] | None:
    """get_cached_tavily_results：读取 24 小时内的 Tavily 缓存结果。"""

    # cache_data：本地缓存文件中的全部数据。
    cache_data = load_tavily_cache()

    # cache_key：当前目的地和 query 对应的缓存 key。
    cache_key = build_tavily_cache_key(destination, query)

    # cached_item：缓存中的单条搜索记录。
    cached_item = cache_data.get(cache_key)
    if not cached_item:
        return None

    # cached_at：缓存写入时间戳。
    cached_at = float(cached_item.get("cached_at", 0))
    if time.time() - cached_at > TAVILY_CACHE_TTL_SECONDS:
        return None

    return cached_item.get("results", [])


def set_cached_tavily_results(destination: str, query: str, results: list[dict]) -> None:
    """set_cached_tavily_results：缓存 Tavily 搜索结果，减少重复搜索消耗。"""

    # cache_data：本地缓存文件中的全部数据。
    cache_data = load_tavily_cache()

    # cache_key：当前目的地和 query 对应的缓存 key。
    cache_key = build_tavily_cache_key(destination, query)
    cache_data[cache_key] = {
        "destination": destination,
        "query": query,
        "cached_at": time.time(),
        "cached_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "results": results,
    }
    save_tavily_cache(cache_data)


def is_tavily_limit_error(error: Exception) -> bool:
    """is_tavily_limit_error：判断 Tavily 错误是否属于额度不足或请求受限。"""

    # error_text：错误文本，兼容 SDK 返回的不同异常格式。
    error_text = str(error).lower()
    limit_keywords = ["429", "quota", "rate limit", "ratelimit", "credits", "credit", "insufficient"]
    return any(keyword in error_text for keyword in limit_keywords)


def call_tavily_search(query: str, parsed_request: dict) -> tuple[list[dict], bool]:
    """call_tavily_search：调用 Tavily SDK 搜索，返回结果和是否命中缓存。"""

    # destination：目的地名称，用于缓存 key。
    destination = parsed_request["destination"]

    # cached_results：24 小时内缓存命中的搜索结果。
    cached_results = get_cached_tavily_results(destination, query)
    if cached_results is not None:
        return cached_results, True

    # tavily_api_key：Tavily API Key，从 .env、环境变量或 Streamlit secrets 读取。
    tavily_api_key = get_tavily_api_key()
    if not tavily_api_key or TavilyClient is None:
        return [], False

    # tavily_client：Tavily Python SDK 客户端。
    tavily_client = TavilyClient(api_key=tavily_api_key)

    # response_data：Tavily SDK 搜索返回结果；不启用 answer/raw/images/auto_parameters，控制额度消耗。
    response_data = tavily_client.search(
        query=query,
        search_depth=get_tavily_search_depth(),
        max_results=DEFAULT_SEARCH_MAX_RESULTS,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
        auto_parameters=False,
        timeout=12,
    )

    # results：Tavily 搜索结果列表。
    results = response_data.get("results", [])
    set_cached_tavily_results(destination, query, results)
    return results, False


def build_facts_context(parsed_request: dict) -> tuple[str, list[dict], str | None]:
    """build_facts_context：联网搜索并整理 facts_context，供 DeepSeek 生成攻略时引用。"""

    # searched_at：事实校验执行时间。
    searched_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not get_bool_config("USE_TAVILY", True):
        facts_context = f"""
联网事实校验状态：未启用
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：USE_TAVILY=false
更新时间：{searched_at}
页面提示：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
""".strip()
        return facts_context, [], "未启用联网搜索"

    if TavilyClient is None:
        facts_context = f"""
联网事实校验状态：执行失败
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：未安装 tavily-python
更新时间：{searched_at}
页面提示：联网搜索失败，已切换普通模式。
""".strip()
        return facts_context, [], "联网搜索失败，已切换普通模式"

    # tavily_api_key：Tavily API Key，用于判断是否启用搜索。
    tavily_api_key = get_tavily_api_key()
    if not tavily_api_key:
        facts_context = f"""
联网事实校验状态：未配置
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：未配置 TAVILY_API_KEY
更新时间：{searched_at}
页面提示：未配置 Tavily，当前为普通生成模式。
""".strip()
        return facts_context, [], "未配置 Tavily，当前为普通生成模式"

    # max_searches：每份攻略最多 Tavily 调用次数，默认限制为 1 次。
    max_searches = get_tavily_max_searches_per_guide()
    if max_searches <= 0:
        facts_context = f"""
联网事实校验状态：未启用
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
原因：TAVILY_MAX_SEARCHES_PER_GUIDE=0
更新时间：{searched_at}
页面提示：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
""".strip()
        return facts_context, [], "未启用联网搜索"

    # query_list：本次事实校验需要执行的搜索查询。
    query_list = build_fact_search_queries(parsed_request)[:max_searches]

    # source_records：用于展示和传给模型的搜索结果。
    source_records = []

    # used_cache：本次搜索是否命中过本地缓存。
    used_cache = False

    # context_blocks：facts_context 中的文本块。
    context_blocks = [
        "联网事实校验状态：已执行",
        f"更新时间：{searched_at}",
        f"搜索模式：Tavily basic，省额度模式，每份攻略最多 {max_searches} 次搜索。",
        "使用范围：门票、预约规则、开放时间、交通政策等易变化信息只能基于以下搜索结果整理。",
        "注意：根据搜索结果整理，仍需出行前二次确认。",
    ]

    try:
        for query in query_list:
            # results：单个 query 的网页搜索结果。
            results, cache_hit = call_tavily_search(query, parsed_request)
            used_cache = used_cache or cache_hit
            context_blocks.append(f"\n### 搜索查询：{query}")
            context_blocks.append(f"- 结果来源：{'本地 24 小时缓存' if cache_hit else 'Tavily basic 搜索'}")

            if not results:
                context_blocks.append("- 未查到可用结果。")
                continue

            for result in results:
                # title：搜索结果标题。
                title = result.get("title", "未命名来源")

                # url：搜索结果链接。
                url = result.get("url", "")

                # content：搜索结果摘要内容。
                content = result.get("content", "") or result.get("snippet", "")
                content = re.sub(r"\s+", " ", content).strip()

                source_records.append({"query": query, "title": title, "url": url, "content": content})
                context_blocks.append(f"- 标题：{title}\n  链接：{url}\n  摘要：{content[:360]}")
    except Exception as error:
        if is_tavily_limit_error(error):
            facts_context = f"""
联网事实校验状态：额度不足
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
更新时间：{searched_at}
错误：{error}
要求：门票、预约、开放时间、交通政策等易变化信息不可编造；请写“建议出行前二次确认”。
""".strip()
            return facts_context, source_records, "Tavily 额度不足，已切换普通模式"

        facts_context = f"""
联网事实校验状态：执行失败
生成模式：普通生成模式，仅使用 DeepSeek 生成攻略
更新时间：{searched_at}
错误：{error}
要求：门票、预约、开放时间、交通政策等易变化信息不可编造；如果 facts_context 没有查到，请写“建议出行前再次核对”。
""".strip()
        return facts_context, source_records, "联网搜索失败，已切换普通模式"

    if not source_records:
        context_blocks.append("\n结论：未查到足够搜索结果。不要编造门票、预约、开放时间，请提示建议出行前再次核对。")

    if used_cache:
        context_blocks[0] = "联网事实校验状态：缓存命中"
        return "\n".join(context_blocks), source_records, "使用缓存搜索结果"

    return "\n".join(context_blocks), source_records, "已启用联网搜索"


def parse_travel_request(user_input: str) -> dict:
    """parse_travel_request：从用户的一句话里提取目的地、天数、预算和偏好。"""

    # cleaned_input：去掉多余空格后的用户输入。
    cleaned_input = user_input.strip()

    # destination：识别到的目的地，识别不到时使用默认目的地。
    destination = extract_destination(cleaned_input)

    # days：识别到的旅行天数。
    days, nights = extract_trip_days(cleaned_input)

    # nights：识别到的住宿晚数。
    # nights_match：匹配“2晚”“两晚”等住宿晚数表达。
    nights_match = re.search(r"([0-9一二两三四五六七八九十]+)\s*晚", cleaned_input)
    if nights_match:
        nights = max(0, parse_chinese_number(nights_match.group(1)))

    # budget_level：识别到的预算档位。
    budget_level = infer_budget_level(cleaned_input)

    # budget_info：识别到的预算金额、货币单位和展示文本。
    budget_info = parse_budget_info(cleaned_input, budget_level)

    # preferences：从用户输入中识别出的偏好列表。
    preferences = extract_preferences(cleaned_input)

    # parsed_request：最终返回给页面和大模型的结构化旅行需求。
    parsed_request = {
        "destination": destination,
        "days": days,
        "nights": nights,
        "budget": budget_info["display"],
        "budget_level": budget_info["level"],
        "style": budget_info["level"].replace("预算", ""),
        "budget_amount": budget_info["amount"],
        "budget_currency": budget_info["currency"],
        "budget_currency_name": budget_info["currency_name"],
        "budget_has_explicit_amount": budget_info["has_explicit_amount"],
        "preferences": preferences,
    }

    # budget_exchange_hint：国外目的地的粗略换算提示。
    parsed_request["budget_exchange_hint"] = build_exchange_hint(parsed_request)

    return parsed_request


def build_ai_prompt(user_input: str, parsed_request: dict, facts_context: str) -> str:
    """build_ai_prompt：把用户输入、解析结果和联网事实上下文整理成给大模型的提示词。"""

    # preferences_text：把偏好列表合并成适合模型阅读的字符串。
    preferences_text = "、".join(parsed_request["preferences"])

    # exchange_hint_text：国外目的地预算粗略换算提示。
    exchange_hint_text = parsed_request.get("budget_exchange_hint") or "无"

    # budget_currency_text：预算货币单位说明。
    budget_currency_text = (
        f"{parsed_request['budget_currency_name']} ({parsed_request['budget_currency']})"
        if parsed_request.get("budget_currency")
        else "未指定具体金额"
    )

    # search_enabled：是否拿到了可用于事实校验的联网或缓存结果。
    search_enabled = "联网事实校验状态：已执行" in facts_context or "联网事实校验状态：缓存命中" in facts_context

    if search_enabled:
        # fact_rules：启用联网搜索时，对易变化信息使用 facts_context 的强约束规则。
        fact_rules = """
16. 门票、预约规则、开放时间、景区政策、交通政策等易变化信息，必须优先依据 facts_context 写。
17. 如果 facts_context 没有明确说明对应信息，不能编造，请写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。
18. 对门票、预约、开放时间这类信息，必须标注“根据搜索结果整理，仍需出行前二次确认”。
19. 必须增加“信息来源与更新时间”区域，列出来源标题、链接和更新时间；如果搜索结果没有覆盖某项信息，也要说明未查到。
20. 避坑提醒要明确、实用。
21. 必须包含以下二级标题，并保持标题文字完全一致：
""".strip()
    else:
        # fact_rules：普通生成模式下不使用联网结果，但提醒用户二次确认易变化信息。
        fact_rules = """
16. 当前未启用联网搜索，请按普通生成模式输出攻略。
17. DeepSeek 不能编造最新门票、预约规则、开放时间、景区政策或交通政策。
18. 对所有可能变化的信息，必须写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。
19. 必须增加“信息来源与更新时间”区域，并写明：当前未启用联网搜索，门票、预约、开放时间等信息请出行前二次确认。
20. 避坑提醒要明确、实用。
21. 必须包含以下二级标题，并保持标题文字完全一致：
""".strip()

    return f"""
用户原始需求：
{user_input}

联网事实校验 facts_context：
{facts_context}

系统已识别：
- 目的地：{parsed_request["destination"]}
- 旅行天数：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 预算单位：{budget_currency_text}
- 预算换算提示：{exchange_hint_text}
- 偏好：{preferences_text}

请生成一份中文旅行攻略，要求：
1. 使用 Markdown。
2. 内容具体、可执行，不要泛泛而谈。
3. 每日行程必须严格生成 {parsed_request["days"]} 天，从 Day 1 到 Day {parsed_request["days"]}，不能少生成，也不能只生成 3 天。
4. 每一天必须有不同主题，标题格式必须是“### Day 1：主题名”，例如“### Day 1：西湖经典路线”。
5. 用户明确提到的景点和偏好必须优先安排：{preferences_text}。
6. 如果用户提到的景点不足以填满全部天数，请根据目的地、偏好、预算和 facts_context 补充适合景点；搜索结果不足时请写“不确定，请出行前核对”，不要编造实时事实。
7. 每天必须包含“上午”“中午”“下午”“晚上”四个时间段。
8. 每个时间段必须写成：- 上午：具体地点｜推荐理由｜预计耗时｜交通或预约提醒。
9. 不要反复使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
10. 同一景点、同一餐厅、同一区域不要重复出现，除非用户明确要求。
11. 美食推荐建议写成：- 名称：推荐理由｜人均预算｜适合场景。
12. 预算估算要分交通、住宿、餐饮、门票体验、机动费用，并明确预算单位。
13. 如果用户输入了预算数字但没有货币单位，必须按人民币 CNY 理解，不要按目的地当地货币理解。
14. 如果用户明确写了美元、日元、欧元、韩元或 USD/EUR/JPY/KRW，必须尊重用户输入的货币单位。
15. 如果存在“预算换算提示”，请在预算估算中补充这条提示，并说明汇率仅供参考。
{fact_rules}
## 旅行封面文案
## 详细旅游攻略
## 每日行程
## 美食推荐
## 交通建议
## 预算估算
## 避坑提醒
## 信息来源与更新时间
""".strip()


def call_deepseek_chat(prompt: str, instructions: str) -> tuple[str | None, str | None]:
    """call_deepseek_chat：执行一次 DeepSeek Chat Completions 调用并返回文本。"""

    if OpenAI is None:
        return None, "没有安装 openai 依赖，已使用本地演示攻略。"

    # api_key：DeepSeek API Key，从 .env、环境变量或 Streamlit secrets 读取。
    api_key = get_deepseek_api_key()
    if not api_key:
        return None, "未配置 DEEPSEEK_API_KEY，已使用本地演示攻略。"

    # model_name：当前使用的 DeepSeek 模型名称，可通过 DEEPSEEK_MODEL 修改。
    model_name = get_deepseek_model_name()

    try:
        # client：OpenAI SDK 客户端，通过 base_url 指向 DeepSeek 服务。
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

        # response：DeepSeek Chat Completions API 返回的大模型结果。
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        )

        # response_text：从模型回复中取出的文本。
        response_text = response.choices[0].message.content
        if not response_text:
            return None, "DeepSeek 返回内容为空，已使用本地演示攻略。"

        return response_text.strip(), None
    except Exception as error:
        return None, f"DeepSeek API 调用失败，已使用本地演示攻略。错误：{error}"


def build_structured_json_prompt(user_input: str, parsed_request: dict, facts_context: str) -> str:
    """build_structured_json_prompt：生成结构化旅行 JSON 的 DeepSeek 提示词。"""

    # preferences_json：用户偏好 JSON 文本，确保用户输入优先。
    preferences_json = json.dumps(parsed_request["preferences"], ensure_ascii=False)

    # search_enabled：是否有可用于事实校验的 Tavily 搜索结果。
    search_enabled = "联网事实校验状态：已执行" in facts_context or "联网事实校验状态：缓存命中" in facts_context

    # fact_rule_text：联网事实约束文本。
    fact_rule_text = (
        "门票、预约、开放时间、景区政策、交通政策必须优先依据 facts_context；如果 facts_context 没有明确说明，不要编造，写“建议出行前二次确认”。"
        if search_enabled
        else "当前未启用联网搜索，不能编造最新门票、预约、开放时间、景区政策或交通政策；必须写“具体信息请出行前以官方渠道为准”或“建议出行前二次确认”。"
    )

    return f"""
用户原始需求：
{user_input}

联网事实校验 facts_context：
{facts_context}

系统已识别参数，必须严格使用，不能被模型猜测或默认值覆盖：
- destination: {parsed_request["destination"]}
- days: {parsed_request["days"]}
- nights: {parsed_request["nights"]}
- budget_amount: {parsed_request.get("budget_amount")}
- currency: {parsed_request.get("budget_currency") or "CNY"}
- budget_level: {parsed_request.get("style") or parsed_request.get("budget_level")}
- preferences: {preferences_json}

请只返回合法 JSON，不要输出 Markdown，不要解释，不要使用代码块。
JSON 顶层必须包含：
destination, days, nights, budget, preferences, daily_itinerary, food_recommendations

budget 必须包含：
amount, currency, level

daily_itinerary 必须 exactly {parsed_request["days"]} 天，从 day 1 到 day {parsed_request["days"]}。
每天必须包含：
day, theme, morning, noon, afternoon, evening

morning/noon/afternoon/evening 每个对象必须包含以下非空字段：
time, place, original_name, reason, duration, transport, booking_note

food_recommendations 必须包含 4-6 家店或小吃点。
每个美食推荐对象必须包含以下非空字段：
name_cn, name_original, location, nearby_spot, reason, budget, scene, booking_note, map_keyword

写作规则：
1. 用户明确提到的景点和偏好必须优先安排：{preferences_json}。
2. 每一天主题必须不同，不能重复。
3. 同一景点、同一餐厅、同一区域不要重复安排。
4. 不要使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
5. place 必须是具体地点，original_name 必须包含中文名和英文/原名；没有英文名时写中文原名。
6. reason、transport、booking_note 必须具体，不能空泛。
7. 美食推荐如果是国外目的地，name_original 必须尽量保留英文名、当地语言原名或常用地图搜索名。
8. 如果无法确认具体地址，不要编造门牌号；location 可以写“市中心区域”“靠近某某景点”“建议以 Google Maps 搜索原名确认”。
9. map_keyword 必须适合复制到 Google Maps / Apple Maps / 百度地图 / 高德地图搜索。
10. {fact_rule_text}
""".strip()


def extract_json_text(model_output: str) -> str:
    """extract_json_text：从模型输出中提取 JSON 文本，兼容代码块和前后解释。"""

    # fenced_match：匹配 ```json 代码块中的 JSON。
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", model_output, flags=re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    # start_index/end_index：提取第一个对象到最后一个对象之间的内容。
    start_index = model_output.find("{")
    end_index = model_output.rfind("}")
    if start_index >= 0 and end_index > start_index:
        return model_output[start_index : end_index + 1].strip()

    return model_output.strip()


def parse_structured_json_output(model_output: str | None) -> tuple[dict | None, list[str]]:
    """parse_structured_json_output：把模型原始输出解析为 JSON 对象。"""

    if not model_output:
        return None, ["模型没有返回 JSON 内容。"]

    # json_text：提取后的 JSON 文本。
    json_text = extract_json_text(model_output)
    try:
        # parsed_json：解析后的 JSON 对象。
        parsed_json = json.loads(json_text)
    except json.JSONDecodeError as error:
        return None, [f"JSON 解析失败：{error}"]

    if not isinstance(parsed_json, dict):
        return None, ["JSON 顶层必须是对象。"]

    return parsed_json, []


def validate_structured_travel_json(travel_json: dict | None, parsed_request: dict) -> list[str]:
    """validate_structured_travel_json：校验每日行程 JSON 是否完整、准确、不重复。"""

    if not isinstance(travel_json, dict):
        return ["结构化结果不是 JSON 对象。"]

    # validation_errors：JSON 校验错误列表。
    validation_errors = []

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    if travel_json.get("destination") != parsed_request["destination"]:
        validation_errors.append(f"destination 不一致，应为 {parsed_request['destination']}。")

    if travel_json.get("days") != expected_days:
        validation_errors.append(f"days 不一致，应为 {expected_days}。")

    if travel_json.get("nights") != parsed_request["nights"]:
        validation_errors.append(f"nights 不一致，应为 {parsed_request['nights']}。")

    # budget_json：模型返回的预算对象。
    budget_json = travel_json.get("budget")
    if not isinstance(budget_json, dict):
        validation_errors.append("budget 必须是对象。")
    else:
        if parsed_request.get("budget_has_explicit_amount") and budget_json.get("amount") != parsed_request.get("budget_amount"):
            validation_errors.append(f"budget.amount 不一致，应为 {parsed_request.get('budget_amount')}。")
        if parsed_request.get("budget_currency") and budget_json.get("currency") != parsed_request.get("budget_currency"):
            validation_errors.append(f"budget.currency 不一致，应为 {parsed_request.get('budget_currency')}。")
        if not str(budget_json.get("level", "")).strip():
            validation_errors.append("budget.level 不能为空。")

    # preferences_json：模型返回的偏好列表。
    preferences_json = travel_json.get("preferences")
    if not isinstance(preferences_json, list):
        validation_errors.append("preferences 必须是数组。")
    else:
        missing_preferences = [preference for preference in parsed_request["preferences"] if preference not in preferences_json]
        if missing_preferences:
            validation_errors.append(f"preferences 缺少用户明确偏好：{'、'.join(missing_preferences)}。")

    # daily_itinerary：模型返回的每日行程数组。
    daily_itinerary = travel_json.get("daily_itinerary")
    if not isinstance(daily_itinerary, list):
        return validation_errors + ["daily_itinerary 必须是数组。"]

    if len(daily_itinerary) != expected_days:
        validation_errors.append(f"daily_itinerary 必须 exactly {expected_days} 天，当前为 {len(daily_itinerary)} 天。")

    # required_slots：每天必须包含的四个时间段。
    required_slots = ["morning", "noon", "afternoon", "evening"]

    # required_slot_fields：每个时间段对象必须包含的字段。
    required_slot_fields = ["time", "place", "original_name", "reason", "duration", "transport", "booking_note"]

    # generic_terms：不允许出现的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # seen_days/themes/places：用于检查编号、主题和地点重复。
    seen_days = set()
    seen_themes = set()
    seen_places = set()

    for day_index, day_item in enumerate(daily_itinerary, start=1):
        if not isinstance(day_item, dict):
            validation_errors.append(f"Day {day_index} 必须是对象。")
            continue

        day_number = day_item.get("day")
        if day_number != day_index:
            validation_errors.append(f"Day {day_index} 的 day 字段应为 {day_index}，当前为 {day_number}。")
        if day_number in seen_days:
            validation_errors.append(f"Day 编号重复：{day_number}。")
        seen_days.add(day_number)

        theme = str(day_item.get("theme", "")).strip()
        if not theme:
            validation_errors.append(f"Day {day_index} theme 不能为空。")
        elif theme in seen_themes:
            validation_errors.append(f"Day {day_index} theme 重复：{theme}。")
        seen_themes.add(theme)

        for slot_name in required_slots:
            # slot_data：单个时间段对象。
            slot_data = day_item.get(slot_name)
            if not isinstance(slot_data, dict):
                validation_errors.append(f"Day {day_index} 缺少 {slot_name} 对象。")
                continue

            for field_name in required_slot_fields:
                field_value = slot_data.get(field_name)
                if field_value is None or not str(field_value).strip():
                    validation_errors.append(f"Day {day_index} {slot_name}.{field_name} 不能为空。")

            slot_text = " ".join(str(slot_data.get(field_name, "")) for field_name in required_slot_fields)
            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_index} {slot_name} 包含空泛模板词。")

            place = str(slot_data.get("place", "")).strip()
            if place:
                if place in seen_places:
                    validation_errors.append(f"重复安排地点：{place}。")
                seen_places.add(place)

    missing_days = [day for day in range(1, expected_days + 1) if day not in seen_days]
    if missing_days:
        validation_errors.append(f"daily_itinerary 缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    # food_recommendations：模型返回的美食推荐数组。
    food_recommendations = travel_json.get("food_recommendations")
    if not isinstance(food_recommendations, list):
        validation_errors.append("food_recommendations 必须是数组。")
    elif not food_recommendations:
        validation_errors.append("food_recommendations 不能为空。")
    else:
        # required_food_fields：每个美食推荐对象必须包含的字段。
        required_food_fields = [
            "name_cn",
            "name_original",
            "location",
            "nearby_spot",
            "reason",
            "budget",
            "scene",
            "booking_note",
            "map_keyword",
        ]

        # seen_food_names：用于检查店铺名称是否重复。
        seen_food_names = set()
        for food_index, food_item in enumerate(food_recommendations, start=1):
            if not isinstance(food_item, dict):
                validation_errors.append(f"food_recommendations 第 {food_index} 项必须是对象。")
                continue

            for field_name in required_food_fields:
                field_value = food_item.get(field_name)
                if field_value is None or not str(field_value).strip():
                    validation_errors.append(f"food_recommendations 第 {food_index} 项 {field_name} 不能为空。")

            food_name_key = f"{food_item.get('name_cn', '')}|{food_item.get('name_original', '')}".strip()
            if food_name_key in seen_food_names:
                validation_errors.append(f"重复推荐店铺：{food_item.get('name_cn', '')}。")
            seen_food_names.add(food_name_key)

    return validation_errors


def normalize_structured_travel_json(travel_json: dict, parsed_request: dict) -> dict:
    """normalize_structured_travel_json：用系统识别参数覆盖 JSON 顶层关键字段，保证用户输入优先。"""

    # normalized_json：复制后的结构化结果。
    normalized_json = dict(travel_json)
    normalized_json["destination"] = parsed_request["destination"]
    normalized_json["days"] = parsed_request["days"]
    normalized_json["nights"] = parsed_request["nights"]
    normalized_json["preferences"] = parsed_request["preferences"]
    normalized_json["budget"] = {
        "amount": parsed_request.get("budget_amount"),
        "currency": parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY,
        "level": parsed_request.get("style") or parsed_request.get("budget_level"),
    }
    return normalized_json


def build_json_repair_prompt(
    original_output: str,
    validation_errors: list[str],
    parsed_request: dict,
    facts_context: str,
) -> str:
    """build_json_repair_prompt：根据校验错误要求 DeepSeek 只修复 JSON。"""

    # error_text：校验错误说明。
    error_text = "\n".join(f"- {error}" for error in validation_errors)

    return f"""
你刚才返回的每日行程 JSON 没有通过校验，错误如下：
{error_text}

请基于原始内容修复 JSON。
必须返回合法 JSON。
必须包含 exactly {parsed_request["days"]} 天。
每天必须包含 morning/noon/afternoon/evening。
每个时间段必须包含 time/place/original_name/reason/duration/transport/booking_note。
food_recommendations 必须包含 4-6 项。
每个美食推荐必须包含 name_cn/name_original/location/nearby_spot/reason/budget/scene/booking_note/map_keyword。
必须保留系统识别参数：
- destination: {parsed_request["destination"]}
- days: {parsed_request["days"]}
- nights: {parsed_request["nights"]}
- budget_amount: {parsed_request.get("budget_amount")}
- currency: {parsed_request.get("budget_currency") or DEFAULT_BUDGET_CURRENCY}
- budget_level: {parsed_request.get("style") or parsed_request.get("budget_level")}
- preferences: {json.dumps(parsed_request["preferences"], ensure_ascii=False)}

联网事实校验 facts_context：
{facts_context}

原始输出：
{original_output}

不要输出 Markdown，不要解释，只返回 JSON。
""".strip()


def call_deepseek_structured_json_api(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
) -> tuple[dict | None, str | None, list[str], str | None]:
    """call_deepseek_structured_json_api：生成并校验结构化旅行 JSON，失败时自动修复一次。"""

    # instructions：要求模型只返回 JSON 的系统提示。
    instructions = """
你是旅行规划结构化数据生成器。只返回合法 JSON，不输出 Markdown，不解释。
所有用户明确输入的目的地、天数、预算、货币和偏好优先级最高。
不要编造门票、预约、开放时间、景区政策等实时信息。
""".strip()

    # json_prompt：第一次生成结构化 JSON 的提示词。
    json_prompt = build_structured_json_prompt(user_input, parsed_request, facts_context)

    # raw_output：第一次模型原始输出。
    raw_output, api_message = call_deepseek_chat(json_prompt, instructions)
    if not raw_output:
        return None, raw_output, [api_message or "DeepSeek 没有返回结构化 JSON。"], api_message

    # travel_json：第一次解析出的 JSON。
    travel_json, parse_errors = parse_structured_json_output(raw_output)
    validation_errors = parse_errors or validate_structured_travel_json(travel_json, parsed_request)
    if not validation_errors and travel_json:
        return normalize_structured_travel_json(travel_json, parsed_request), raw_output, [], None

    # repair_prompt：JSON 校验失败后的修复提示。
    repair_prompt = build_json_repair_prompt(raw_output, validation_errors, parsed_request, facts_context)

    # repaired_output：修复后的模型原始输出。
    repaired_output, repair_message = call_deepseek_chat(repair_prompt, instructions)
    if not repaired_output:
        return None, raw_output, validation_errors + [repair_message or "DeepSeek JSON 修复没有返回内容。"], repair_message

    # repaired_json：修复后解析出的 JSON。
    repaired_json, repair_parse_errors = parse_structured_json_output(repaired_output)
    repair_errors = repair_parse_errors or validate_structured_travel_json(repaired_json, parsed_request)
    if repair_errors:
        return None, repaired_output, repair_errors, "每日行程 JSON 修复后仍未通过校验。"

    return normalize_structured_travel_json(repaired_json, parsed_request), repaired_output, [], "每日行程 JSON 第一次未通过校验，已自动修复。"


def build_markdown_from_json_prompt(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
    travel_json: dict,
) -> str:
    """build_markdown_from_json_prompt：基于结构化 JSON 生成 Markdown 攻略提示词。"""

    # structured_json_text：结构化旅行 JSON 文本。
    structured_json_text = json.dumps(travel_json, ensure_ascii=False, indent=2)

    return f"""
用户原始需求：
{user_input}

系统识别参数：
- 目的地：{parsed_request["destination"]}
- 旅行天数：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 偏好：{"、".join(parsed_request["preferences"])}

联网事实校验 facts_context：
{facts_context}

结构化 JSON：
{structured_json_text}

请基于上面的结构化 JSON 生成中文 Markdown 攻略。
要求：
1. 不要改变 JSON 中的目的地、天数、预算、偏好和 daily_itinerary。
2. 每日行程必须按 JSON 中的 daily_itinerary 写，不要自由新增重复路线。
3. 门票、预约、开放时间、景区政策必须优先依据 facts_context；没有明确搜索结果时写“建议出行前二次确认”。
4. 内容具体、可执行，保留中文名 + 英文/原名。
5. “美食推荐”必须基于 JSON 中的 food_recommendations，每条必须写店名中文名、英文/当地原名、位置、附近景点/区域、人均预算、适合场景、预约提示和地图搜索关键词。
6. 如果无法确认具体地址，不要编造门牌号；写“市中心区域”“靠近某某景点”或“建议以 Google Maps 搜索原名确认”。
7. 必须包含以下二级标题，并保持标题文字完全一致：
## 旅行封面文案
## 详细旅游攻略
## 每日行程
## 美食推荐
## 交通建议
## 预算估算
## 避坑提醒
## 信息来源与更新时间
""".strip()


def build_markdown_from_structured_json(travel_json: dict, parsed_request: dict, facts_context: str) -> str:
    """build_markdown_from_structured_json：当 Markdown 二次生成失败时，用合格 JSON 生成可复制攻略。"""

    # source_updated_at：攻略信息更新时间。
    source_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # itinerary_lines：从结构化 JSON 生成的每日行程 Markdown。
    itinerary_lines = []
    for day_item in travel_json.get("daily_itinerary", []):
        itinerary_lines.append(f"### Day {day_item['day']}：{day_item['theme']}")
        for slot_key, slot_label in [("morning", "上午"), ("noon", "中午"), ("afternoon", "下午"), ("evening", "晚上")]:
            slot = day_item[slot_key]
            itinerary_lines.append(
                f"- {slot_label}：{slot['place']}｜{slot['reason']}｜{slot['duration']}｜{slot['transport']}；{slot['booking_note']}"
            )

    # food_lines：从结构化 JSON 生成的美食推荐 Markdown。
    food_lines = []
    for food_item in travel_json.get("food_recommendations", []):
        food_title = food_item["name_cn"]
        if food_item.get("name_original") and food_item["name_original"] != food_title:
            food_title = f"{food_title}（{food_item['name_original']}）"
        food_lines.append(
            f"- {food_title}：位置 {food_item['location']}，靠近 {food_item['nearby_spot']}｜"
            f"{food_item['reason']}｜{food_item['budget']}｜{food_item['scene']}｜"
            f"{food_item['booking_note']}｜地图搜索：{food_item['map_keyword']}"
        )

    # preferences_text：用户旅行偏好。
    preferences_text = "、".join(parsed_request["preferences"])

    return f"""
## 旅行封面文案
{parsed_request["destination"]} {parsed_request["days"]} 天 {parsed_request["nights"]} 晚旅行计划：围绕 {preferences_text} 安排路线。

## 详细旅游攻略
- 目的地：{parsed_request["destination"]}
- 行程长度：{parsed_request["days"]} 天 {parsed_request["nights"]} 晚
- 预算：{parsed_request["budget"]}
- 旅行风格：{preferences_text}
- 说明：本 Markdown 根据已通过校验的结构化 JSON 生成。

## 每日行程
{chr(10).join(itinerary_lines)}

## 美食推荐
{chr(10).join(food_lines) if food_lines else f"- 请结合每日路线选择附近餐厅，热门餐厅建议提前预约或取号｜人均预算按 {parsed_request['budget']} 控制｜适合午餐和晚餐"}

## 交通建议
- 每天优先围绕同一区域规划，减少跨区往返。
- 景区门票、预约和开放时间建议出行前二次确认。

## 预算估算
- 用户预算：{parsed_request["budget"]}
- 交通、住宿、餐饮、门票体验和机动费用建议按实际日期二次核算。

## 避坑提醒
- 不要把热门景点、热门餐厅和远距离交通挤在同一天。
- 对所有可能变化的信息，建议出行前二次确认。

## 信息来源与更新时间
- 更新时间：{source_updated_at}
- 来源说明：结构化 JSON 已通过程序校验；门票、预约、开放时间等仍需出行前二次确认。
- 联网事实状态：{facts_context.splitlines()[0] if facts_context else "未启用联网搜索"}
""".strip()


def extract_markdown_section_text(markdown_text: str, heading: str) -> str:
    """extract_markdown_section_text：从 Markdown 中提取指定二级标题下的正文。"""

    # normalized_markdown：保证开头有换行，便于正则匹配二级标题。
    normalized_markdown = "\n" + markdown_text.strip()

    # section_pattern：匹配指定二级标题到下一个二级标题之间的内容。
    section_pattern = rf"\n##\s+{re.escape(heading)}\s*\n([\s\S]*?)(?=\n##\s+|\Z)"
    match = re.search(section_pattern, normalized_markdown)
    return match.group(1).strip() if match else ""


def parse_itinerary_day_blocks(markdown_text: str) -> list[dict]:
    """parse_itinerary_day_blocks：解析 Markdown 每日行程区中的 Day 块。"""

    # itinerary_text：每日行程区域 Markdown。
    itinerary_text = extract_markdown_section_text(markdown_text, "每日行程")
    if not itinerary_text:
        return []

    # day_matches：匹配 Day 标题。
    day_matches = list(re.finditer(r"(?m)^###\s*Day\s*([0-9一二两三四五六七八九十]+)\s*[：:]?\s*(.*?)\s*$", itinerary_text))

    # day_blocks：解析后的 Day 数据。
    day_blocks = []
    for index, match in enumerate(day_matches):
        start_index = match.end()
        end_index = day_matches[index + 1].start() if index + 1 < len(day_matches) else len(itinerary_text)
        day_number = parse_chinese_number(match.group(1))
        theme = match.group(2).strip() or f"Day {day_number}"
        body = itinerary_text[start_index:end_index].strip()
        day_blocks.append({"day": day_number, "theme": theme, "body": body})

    return day_blocks


def validate_itinerary_markdown(markdown_text: str, parsed_request: dict) -> list[str]:
    """validate_itinerary_markdown：校验模型生成的每日行程是否满足不重复和天数要求。"""

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    # day_blocks：模型输出中的 Day 块。
    day_blocks = parse_itinerary_day_blocks(markdown_text)

    # validation_errors：行程校验错误列表。
    validation_errors = []

    if len(day_blocks) < expected_days:
        validation_errors.append(f"每日行程只生成了 {len(day_blocks)} 天，用户需要 {expected_days} 天。")
    elif len(day_blocks) > expected_days:
        validation_errors.append(f"每日行程生成了 {len(day_blocks)} 天，用户只需要 {expected_days} 天。")

    # day_number_set：实际出现的 Day 编号集合。
    day_number_set = {day_block["day"] for day_block in day_blocks}
    missing_days = [day for day in range(1, expected_days + 1) if day not in day_number_set]
    if missing_days:
        validation_errors.append(f"缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    # required_slots：每天必须包含的四个时间段。
    required_slots = ["上午", "中午", "下午", "晚上"]

    # generic_terms：不允许反复出现的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # signature_set：用于检查整天内容是否重复。
    signature_set = set()

    # theme_set：用于检查每天主题是否重复。
    theme_set = set()

    # seen_places：用于检查具体地点是否重复安排。
    seen_places = set()

    for day_block in day_blocks[:expected_days]:
        # day_theme：单日主题，必须和其他天不同。
        day_theme = clean_markdown_text(day_block["theme"])
        if day_theme in theme_set:
            validation_errors.append(f"Day {day_block['day']} 主题重复：{day_theme}。")
        theme_set.add(day_theme)

        day_signature_parts = []
        for slot_label in required_slots:
            slot_text = extract_slot_text(day_block["body"], slot_label)
            if not slot_text:
                validation_errors.append(f"Day {day_block['day']} 缺少{slot_label}安排。")
                continue

            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_block['day']} {slot_label}使用了空泛模板词：{slot_text[:40]}")

            # slot_parts：时间段内容应拆成地点、推荐理由、预计耗时、交通或预约提醒。
            slot_parts = [part.strip() for part in re.split(r"[｜|]", slot_text) if part.strip()]
            if len(slot_parts) < 4:
                validation_errors.append(
                    f"Day {day_block['day']} {slot_label}格式不完整，需要“具体地点｜推荐理由｜预计耗时｜交通或预约提醒”。"
                )

            # place：时间段内容中竖线前面的具体地点。
            place = slot_parts[0] if slot_parts else ""
            if place:
                if place in seen_places:
                    validation_errors.append(f"重复安排地点：{place}。")
                seen_places.add(place)

            day_signature_parts.append(slot_text)

        # day_signature：单日四段安排合并后的指纹。
        day_signature = "||".join(day_signature_parts)
        if day_signature and day_signature in signature_set:
            validation_errors.append(f"Day {day_block['day']} 与其他日期的行程内容高度重复。")
        signature_set.add(day_signature)

    return validation_errors


def build_itinerary_retry_prompt(base_prompt: str, validation_errors: list[str], parsed_request: dict) -> str:
    """build_itinerary_retry_prompt：根据校验错误生成二次生成提示词。"""

    # error_text：校验错误说明。
    error_text = "\n".join(f"- {error}" for error in validation_errors)

    return f"""
{base_prompt}

上一次输出的每日行程不合格，必须重新生成完整攻略，重点修复：
{error_text}

强制要求：
1. 必须生成 Day 1 到 Day {parsed_request["days"]}，一天都不能少。
2. 每一天主题必须不同。
3. 不要使用“核心街区”“本地风味餐厅”“主题体验”“夜景与晚餐区域”等空泛词。
4. 每个时间段必须写具体地点原名、推荐理由、预计耗时、交通或预约提醒。
5. 不要重复同一景点、同一餐厅、同一区域。
""".strip()


def call_deepseek_api(user_input: str, parsed_request: dict, facts_context: str) -> tuple[str | None, str | None]:
    """call_deepseek_api：使用 OpenAI Python SDK 调用 DeepSeek API 生成攻略文本。"""

    if OpenAI is None:
        return None, "没有安装 openai 依赖，已使用本地演示攻略。"

    # api_key：DeepSeek API Key，从 .env、环境变量或 Streamlit secrets 读取。
    api_key = get_deepseek_api_key()
    if not api_key:
        return None, "未配置 DEEPSEEK_API_KEY，已使用本地演示攻略。"

    # model_name：当前使用的 DeepSeek 模型名称，可通过 DEEPSEEK_MODEL 修改。
    model_name = get_deepseek_model_name()

    # client：OpenAI SDK 客户端，通过 base_url 指向 DeepSeek 服务。
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    # instructions：给模型的角色和输出风格要求。
    instructions = """
你是一名资深旅行编辑和行程规划师，擅长把用户的一句话需求整理成清晰、真实、好执行的旅行攻略。
请用中文输出，语气像旅行杂志编辑，但结构要像实用攻略工具。
不要编造实时价格或实时营业状态；涉及价格时用区间估算，并提醒以出行前查询为准。
""".strip()

    # prompt：最终发送给模型的完整提示词。
    prompt = build_ai_prompt(user_input, parsed_request, facts_context)

    try:
        def create_markdown(prompt_text: str) -> str | None:
            """create_markdown：执行一次 DeepSeek Chat Completions 调用并返回 Markdown 文本。"""

            # response：DeepSeek Chat Completions API 返回的大模型结果。
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt_text},
                ],
            )

            # markdown_content：从模型回复中取出的 Markdown 攻略文本。
            markdown_content = response.choices[0].message.content
            return markdown_content.strip() if markdown_content else None

        # markdown_text：从模型回复中取出的 Markdown 攻略文本。
        markdown_text = create_markdown(prompt)
        if not markdown_text:
            return None, "DeepSeek 返回内容为空，已使用本地演示攻略。"

        # validation_errors：第一次生成后的每日行程校验问题。
        validation_errors = validate_itinerary_markdown(markdown_text, parsed_request)
        if not validation_errors:
            return markdown_text, None

        # retry_prompt：校验失败时给模型的二次生成提示词。
        retry_prompt = build_itinerary_retry_prompt(prompt, validation_errors, parsed_request)

        # retry_markdown：二次生成的 Markdown 攻略文本。
        retry_markdown = create_markdown(retry_prompt)
        if not retry_markdown:
            error_summary = "；".join(validation_errors[:4])
            return markdown_text, f"DeepSeek 二次生成返回内容为空，已展示第一次结果；每日行程可能不完整：{error_summary}"

        # retry_errors：二次生成后再次校验每日行程。
        retry_errors = validate_itinerary_markdown(retry_markdown, parsed_request)
        if retry_errors:
            error_summary = "；".join(retry_errors[:4])
            return retry_markdown, f"每日行程校验未完全通过：{error_summary}。页面会显示解析问题，请重新生成或调整输入。"

        return retry_markdown, "检测到第一次每日行程不完整，已自动重新生成并通过校验。"
    except Exception as error:
        return None, f"DeepSeek API 调用失败，已使用本地演示攻略。错误：{error}"


def build_demo_markdown(parsed_request: dict, facts_context: str = "") -> str:
    """build_demo_markdown：没有 API Key 或 API 失败时生成本地演示攻略。"""

    # destination：攻略目的地。
    destination = parsed_request["destination"]

    # days：旅行天数。
    days = parsed_request["days"]

    # nights：住宿晚数。
    nights = parsed_request["nights"]

    # budget：预算档位。
    budget = parsed_request["budget"]

    # budget_exchange_hint：国外目的地预算粗略换算提示。
    budget_exchange_hint = parsed_request.get("budget_exchange_hint")

    # preferences_text：用户旅行偏好。
    preferences_text = "、".join(parsed_request["preferences"])

    # source_updated_at：本地演示攻略的信息更新时间。
    source_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # source_note：本地演示攻略的信息来源说明。
    source_note = (
        "未配置搜索 API 或搜索失败，本地演示攻略没有使用实时联网来源；门票、预约、开放时间建议出行前再次核对。"
        if "联网事实校验状态：已执行" not in facts_context and "联网事实校验状态：缓存命中" not in facts_context
        else "已传入联网搜索 facts_context；具体门票、预约和开放时间请以搜索来源及出行前二次确认为准。"
    )

    return f"""
## 旅行封面文案
{destination} {days} 天 {nights} 晚旅行计划：把 {preferences_text} 放进行程主线，用轻松但不松散的节奏完成一次有记忆点的城市探索。

## 详细旅游攻略
- 目的地：{destination}
- 行程长度：{days} 天 {nights} 晚
- 预算：{budget}
- 旅行风格：{preferences_text}
- 规划思路：第一天熟悉城市动线，第二天深入主题体验，最后一天安排轻量活动和购物补漏。
{f"- 换算提示：{budget_exchange_hint}" if budget_exchange_hint else ""}

## 每日行程
本地演示模式不会生成每日行程模板。请配置 DEEPSEEK_API_KEY 后由 DeepSeek 按目的地、天数、偏好和联网事实生成完整不重复行程。

## 美食推荐
- 本地代表料理：优先选择评分稳定、翻台快、位置靠近行程路线的店｜人均 80-180 人民币(CNY)｜适合第一顿正式餐
- 街区小吃：适合放在下午或夜间，不要把所有排队店集中到同一天｜人均 30-80 人民币(CNY)｜适合边逛边吃
- 甜品或咖啡：适合安排在步行较多的下午，作为休息点｜人均 40-100 人民币(CNY)｜适合拍照和休息
- 预约型餐厅：如果是热门目的地，建议提前 3 到 7 天确认｜人均 180-400 人民币(CNY)｜适合纪念日晚餐

## 交通建议
- 城市内优先使用地铁、公交或官方交通卡，减少频繁打车。
- 每天尽量围绕一个区域规划，避免跨城式来回移动。
- 机场或车站到酒店先查官方线路，再对比打车价格。
- 如果有大件行李，最后一天优先选择寄存点或酒店寄存。

## 预算估算
- 用户预算：{budget}
{f"- {budget_exchange_hint}" if budget_exchange_hint else ""}
- 交通：经济预算约 150-300 人民币(CNY)/人，普通预算约 300-600 人民币(CNY)/人，高预算按实际打车和跨城交通增加。
- 住宿：经济预算约 300-600 人民币(CNY)/晚，普通预算约 600-1200 人民币(CNY)/晚，高预算约 1200 人民币(CNY)/晚以上。
- 餐饮：约 150-350 人民币(CNY)/人/天，热门餐厅和预约餐厅另算。
- 门票体验：约 100-500 人民币(CNY)/人，主题展、乐园、演出费用可能更高。
- 机动费用：建议预留总预算的 10%-20%。

## 避坑提醒
- 不要把热门景点、热门餐厅和远距离交通挤在同一天。
- 不要只看社交平台种草，出发前确认营业时间、预约方式和交通路线。
- 夜景点通常受天气影响明显，建议保留备选方案。
- 购物和伴手礼尽量放在后半程，避免一路背负行李。
- 本攻略为第一版演示内容，真实出行前请再次确认价格、营业时间和交通信息。

## 信息来源与更新时间
- 更新时间：{source_updated_at}
- 来源说明：{source_note}
- 门票、预约、开放时间：根据搜索结果整理，仍需出行前二次确认；如果没有联网结果，请勿将本地演示内容视为实时信息。
""".strip()


def generate_travel_markdown(user_input: str, parsed_request: dict, facts_context: str) -> tuple[str, str | None]:
    """generate_travel_markdown：优先用大模型生成攻略，失败时回退到本地演示攻略。"""

    # ai_markdown：大模型生成的 Markdown 文本。
    ai_markdown, api_message = call_deepseek_api(user_input, parsed_request, facts_context)
    if ai_markdown:
        return ai_markdown, api_message

    # demo_markdown：本地演示 Markdown 文本。
    demo_markdown = build_demo_markdown(parsed_request, facts_context)
    return demo_markdown, api_message


def call_deepseek_markdown_from_json_api(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
    travel_json: dict,
) -> tuple[str | None, str | None]:
    """call_deepseek_markdown_from_json_api：基于合格 JSON 生成 Markdown 攻略。"""

    # instructions：给模型的 Markdown 写作角色要求。
    instructions = """
你是一名资深旅行编辑和行程规划师。请基于给定 JSON 写中文 Markdown 攻略。
不能改变 JSON 中的行程天数、地点、预算和偏好；不要编造实时营业状态。
""".strip()

    # markdown_prompt：基于结构化 JSON 生成 Markdown 的提示词。
    markdown_prompt = build_markdown_from_json_prompt(user_input, parsed_request, facts_context, travel_json)
    return call_deepseek_chat(markdown_prompt, instructions)


def generate_travel_content(
    user_input: str,
    parsed_request: dict,
    facts_context: str,
) -> tuple[str, str | None, dict | None, str | None, list[str]]:
    """generate_travel_content：先生成结构化 JSON，再基于 JSON 生成 Markdown 攻略。"""

    # travel_json：用于页面时间线渲染的结构化旅行数据。
    travel_json, json_raw, json_errors, json_message = call_deepseek_structured_json_api(
        user_input,
        parsed_request,
        facts_context,
    )

    if not travel_json:
        # demo_markdown：结构化 JSON 失败时仍保留页面其他区域，不使用假行程补齐。
        demo_markdown = build_demo_markdown(parsed_request, facts_context)
        error_summary = "；".join(json_errors[:4]) if json_errors else "结构化 JSON 未生成。"
        api_message = f"每日行程 JSON 未通过校验：{error_summary}"
        if json_message and json_message not in api_message:
            api_message = f"{api_message}；{json_message}"
        return demo_markdown, api_message, None, json_raw, json_errors

    # markdown_text：基于合格 JSON 生成的 Markdown 攻略。
    markdown_text, markdown_message = call_deepseek_markdown_from_json_api(
        user_input,
        parsed_request,
        facts_context,
        travel_json,
    )

    # api_messages：需要展示给用户的生成状态说明。
    api_messages = []
    if json_message:
        api_messages.append(json_message)

    if markdown_text:
        if markdown_message:
            api_messages.append(markdown_message)
        return markdown_text, "；".join(api_messages) or None, travel_json, json_raw, []

    # Markdown 二次生成失败时，用已通过校验的 JSON 生成可复制攻略，不生成假行程。
    fallback_markdown = build_markdown_from_structured_json(travel_json, parsed_request, facts_context)
    if markdown_message:
        api_messages.append(f"Markdown 生成失败，已根据合格 JSON 生成可复制攻略：{markdown_message}")
    else:
        api_messages.append("Markdown 生成失败，已根据合格 JSON 生成可复制攻略。")

    return fallback_markdown, "；".join(api_messages), travel_json, json_raw, []


def generate_cover_image_url(parsed_request: dict) -> str:
    """generate_cover_image_url：生成封面图地址，后续可替换为图片生成 API。"""

    # destination：封面图上显示的目的地。
    destination = parsed_request["destination"]

    # preferences_text：封面图上显示的旅行偏好。
    preferences_text = " / ".join(parsed_request["preferences"][:4])

    # cover_svg：使用旅行杂志感 SVG 占位图，保证没有图片 API 时也能显示大封面。
    cover_svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
      <defs>
        <linearGradient id="sky" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#111827"/>
          <stop offset="28%" stop-color="#26324f"/>
          <stop offset="62%" stop-color="#92400e"/>
          <stop offset="100%" stop-color="#020617"/>
        </linearGradient>
        <linearGradient id="sunset" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#fef3c7" stop-opacity="0.86"/>
          <stop offset="42%" stop-color="#fb923c" stop-opacity="0.38"/>
          <stop offset="100%" stop-color="#020617" stop-opacity="0"/>
        </linearGradient>
        <linearGradient id="water" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="#0ea5e9" stop-opacity="0.72"/>
          <stop offset="52%" stop-color="#14b8a6" stop-opacity="0.42"/>
          <stop offset="100%" stop-color="#f97316" stop-opacity="0.48"/>
        </linearGradient>
        <filter id="grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch"/>
          <feColorMatrix type="saturate" values="0"/>
          <feComponentTransfer>
            <feFuncA type="table" tableValues="0 0.18"/>
          </feComponentTransfer>
        </filter>
        <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="24" stdDeviation="24" flood-color="#020617" flood-opacity="0.42"/>
        </filter>
        <clipPath id="photoClip">
          <rect x="760" y="115" width="470" height="560" rx="32"/>
        </clipPath>
      </defs>
      <rect width="1600" height="900" fill="url(#sky)"/>
      <rect width="1600" height="900" fill="url(#sunset)" opacity="0.62"/>
      <rect width="1600" height="900" filter="url(#grain)" opacity="0.36"/>
      <path d="M0 565 C165 470 305 515 440 430 C570 350 690 392 820 315 C1010 205 1190 295 1600 185 L1600 900 L0 900 Z" fill="#0f172a" opacity="0.76"/>
      <path d="M0 625 C230 508 365 610 545 515 C710 428 865 560 1020 468 C1188 368 1365 438 1600 315 L1600 900 L0 900 Z" fill="#1e293b" opacity="0.72"/>
      <path d="M0 690 C260 590 430 710 675 625 C910 544 1060 690 1308 568 C1435 506 1510 520 1600 475 L1600 900 L0 900 Z" fill="url(#water)" opacity="0.78"/>
      <path d="M0 742 C210 700 390 785 640 735 C880 688 1030 790 1285 708 C1430 662 1510 675 1600 642 L1600 900 L0 900 Z" fill="#020617" opacity="0.70"/>
      <g filter="url(#softShadow)" opacity="0.95">
        <rect x="760" y="115" width="470" height="560" rx="32" fill="#f8fafc" opacity="0.92"/>
        <g clip-path="url(#photoClip)">
          <rect x="760" y="115" width="470" height="560" fill="#0f172a"/>
          <rect x="760" y="115" width="470" height="560" fill="url(#sky)" opacity="0.52"/>
          <circle cx="1110" cy="230" r="72" fill="#fde68a" opacity="0.86"/>
          <path d="M760 430 C845 360 910 380 975 320 C1055 245 1115 360 1230 275 L1230 675 L760 675 Z" fill="#334155"/>
          <path d="M760 520 C900 455 960 550 1080 488 C1145 455 1188 470 1230 430 L1230 675 L760 675 Z" fill="#0f766e" opacity="0.7"/>
          <path d="M760 575 C860 540 940 615 1055 558 C1120 524 1170 540 1230 512 L1230 675 L760 675 Z" fill="#0ea5e9" opacity="0.55"/>
          <path d="M860 675 L1018 444 L1135 675 Z" fill="#f8fafc" opacity="0.82"/>
          <path d="M924 675 L1018 500 L1080 675 Z" fill="#f59e0b" opacity="0.55"/>
        </g>
      </g>
      <path d="M1320 170 C1390 210 1425 268 1450 350" stroke="#fde68a" stroke-width="3" stroke-dasharray="12 16" fill="none" opacity="0.55"/>
      <path d="M1450 350 l28 -12 l-20 31 z" fill="#fde68a" opacity="0.75"/>
      <g opacity="0.78">
        <rect x="120" y="640" width="420" height="3" fill="#fde68a"/>
        <rect x="120" y="665" width="315" height="3" fill="#f8fafc" opacity="0.52"/>
        <rect x="120" y="690" width="250" height="3" fill="#f8fafc" opacity="0.34"/>
      </g>
      <text x="126" y="145" fill="#fde68a" font-size="32" font-family="Arial, sans-serif" letter-spacing="6">AI TRAVEL MAGAZINE</text>
      <text x="126" y="232" fill="#ffffff" font-size="72" font-family="Arial, sans-serif" font-weight="700">{html.escape(destination)}</text>
      <text x="130" y="292" fill="#e5e7eb" font-size="30" font-family="Arial, sans-serif">{html.escape(preferences_text)}</text>
    </svg>
    """

    return "data:image/svg+xml;charset=utf-8," + quote(cover_svg)


def split_markdown_sections(markdown_text: str) -> dict:
    """split_markdown_sections：把 Markdown 按二级标题拆成多个展示卡片。"""

    # section_map：保存标题和正文的对应关系。
    section_map = {}

    # normalized_markdown：保证文本开头有换行，方便正则切分。
    normalized_markdown = "\n" + markdown_text.strip()

    # matches：匹配所有“## 标题”和标题后正文。
    matches = re.finditer(r"\n##\s+(.+?)\n([\s\S]*?)(?=\n##\s+|\Z)", normalized_markdown)
    for match in matches:
        title = match.group(1).strip()
        content = match.group(2).strip()
        section_map[title] = content

    return section_map


def clean_markdown_text(markdown_text: str) -> str:
    """clean_markdown_text：清理 Markdown 符号，方便放进自定义 HTML 卡片。"""

    # cleaned_text：去掉列表符号、粗体和多余空格后的文本。
    cleaned_text = re.sub(r"^[\-\*\d\.\s]+", "", markdown_text.strip())
    cleaned_text = re.sub(r"[*`#]+", "", cleaned_text)
    return cleaned_text.strip()


def extract_bullet_items(section_text: str, max_items: int = 6) -> list[str]:
    """extract_bullet_items：从 Markdown 段落中提取列表项。"""

    # item_list：从 Markdown 中提取出的列表内容。
    item_list = []
    for line in section_text.splitlines():
        stripped_line = line.strip()
        if re.match(r"^[-*]\s+", stripped_line) or re.match(r"^\d+[.、]\s+", stripped_line):
            item = clean_markdown_text(stripped_line)
            if item:
                item_list.append(item)

    if not item_list and section_text.strip():
        # fallback_lines：当模型没有使用列表时，按非空行兜底提取。
        fallback_lines = [clean_markdown_text(line) for line in section_text.splitlines() if clean_markdown_text(line)]
        item_list = fallback_lines

    return item_list[:max_items]


def estimate_total_cost(parsed_request: dict) -> str:
    """estimate_total_cost：根据天数和预算档位估算不含大交通的人均总花费。"""

    if parsed_request.get("budget_has_explicit_amount"):
        return f"按 {parsed_request['budget']} 控制"

    # budget_level：用户预算档位。
    budget_level = parsed_request.get("budget_level", parsed_request["budget"])

    # days：旅行天数。
    days = parsed_request["days"]

    # nights：住宿晚数。
    nights = parsed_request["nights"]

    if budget_level == "经济预算":
        day_cost, night_cost = 260, 320
    elif budget_level == "高预算":
        day_cost, night_cost = 980, 1600
    else:
        day_cost, night_cost = 480, 720

    # low_cost：较低估算值。
    low_cost = days * day_cost + nights * night_cost

    # high_cost：较高估算值。
    high_cost = int(low_cost * 1.35)

    return f"约 {low_cost:,}-{high_cost:,} 人民币(CNY)/人"


def infer_trip_pace(parsed_request: dict) -> str:
    """infer_trip_pace：根据天数和偏好推断旅行节奏。"""

    # preferences：用户偏好列表。
    preferences = parsed_request["preferences"]

    if parsed_request["days"] >= 6 or any(item in preferences for item in ["自然", "咖啡", "温泉", "海边"]):
        return "松弛慢旅行"
    if parsed_request["days"] <= 3 and any(item in preferences for item in ["购物", "夜景", "动漫"]):
        return "高效城市探索"
    return "舒适均衡节奏"


def infer_audience(parsed_request: dict) -> str:
    """infer_audience：根据偏好推断适合人群。"""

    # preferences：用户偏好列表。
    preferences = parsed_request["preferences"]

    if "亲子" in preferences:
        return "家庭与亲子出行"
    if any(item in preferences for item in ["动漫", "购物", "夜景"]):
        return "城市玩家与潮流爱好者"
    if any(item in preferences for item in ["自然", "徒步", "海边"]):
        return "自然风景和慢旅行人群"
    return "第一次到访和自由行用户"


def build_summary_metrics(parsed_request: dict) -> dict:
    """build_summary_metrics：生成攻略摘要区需要展示的指标。"""

    # preferences_count：偏好数量，用于生成推荐强度。
    preferences_count = len(parsed_request["preferences"])

    # recommendation_score：推荐强度评分。
    recommendation_score = "4.9 / 5" if preferences_count >= 3 else "4.6 / 5"

    return {
        "推荐强度": recommendation_score,
        "旅行节奏": infer_trip_pace(parsed_request),
        "适合人群": infer_audience(parsed_request),
        "预计总花费": estimate_total_cost(parsed_request),
    }


def extract_slot_text(day_text: str, slot_label: str) -> str:
    """extract_slot_text：从某一天行程中提取上午、中午、下午或晚上的内容。"""

    # slot_pattern：匹配指定时间段的 Markdown 行。
    slot_pattern = rf"(?:^|\n)\s*[-*]?\s*(?:\*\*)?{slot_label}(?:\*\*)?\s*[：:]\s*(.+)"
    match = re.search(slot_pattern, day_text)
    if match:
        return clean_markdown_text(match.group(1))
    return ""


def split_place_and_description(slot_text: str) -> tuple[str, str]:
    """split_place_and_description：把时间段内容拆成地点和说明。"""

    # parts：按中文或英文竖线拆开的地点和说明。
    parts = [part.strip() for part in re.split(r"[｜|]", slot_text, maxsplit=1)]
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]

    return "", slot_text


def build_timeline_days(section_map: dict, parsed_request: dict) -> tuple[list[dict], list[str]]:
    """build_timeline_days：把每日行程 Markdown 转成时间线数据。"""

    # itinerary_text：每日行程 Markdown 内容。
    itinerary_text = section_map.get("每日行程", "").strip()
    if not itinerary_text:
        return [], ["未找到“每日行程”区域。"]

    # timeline_markdown：补回二级标题，复用统一的 Day 块解析函数。
    timeline_markdown = f"## 每日行程\n{itinerary_text}"

    # day_blocks：从 Markdown 中解析出的每日行程块。
    day_blocks = parse_itinerary_day_blocks(timeline_markdown)

    # expected_days：用户明确要求或系统识别出的旅行天数。
    expected_days = parsed_request["days"]

    # validation_errors：时间线渲染前的结构化校验问题。
    validation_errors = []

    if len(day_blocks) != expected_days:
        validation_errors.append(f"模型返回 {len(day_blocks)} 天行程，系统识别用户需要 {expected_days} 天。")

    # day_map：按 Day 编号索引每日行程，方便检查缺失天数。
    day_map = {day_block["day"]: day_block for day_block in day_blocks}
    missing_days = [day for day in range(1, expected_days + 1) if day not in day_map]
    if missing_days:
        validation_errors.append(f"缺少 Day {', Day '.join(str(day) for day in missing_days)}。")

    # slot_config：时间线四个固定时段。
    slot_config = [
        {"label": "上午", "time": "09:00 - 11:30", "icon": "AM"},
        {"label": "中午", "time": "12:00 - 13:30", "icon": "NO"},
        {"label": "下午", "time": "14:00 - 17:30", "icon": "PM"},
        {"label": "晚上", "time": "18:30 - 21:30", "icon": "EV"},
    ]

    # generic_terms：不允许出现在时间线里的空泛模板词。
    generic_terms = ["核心街区", "本地风味餐厅", "主题体验", "夜景与晚餐区域"]

    # seen_places：已出现地点集合，用于避免同一地点重复安排。
    seen_places = set()

    # seen_themes：已出现主题集合，用于避免每天主题重复。
    seen_themes = set()

    # timeline_days：最终时间线数据。
    timeline_days = []
    for day_number in range(1, expected_days + 1):
        day_block = day_map.get(day_number)
        if not day_block:
            continue

        # day_theme：当前 Day 的主题文本。
        day_theme = clean_markdown_text(day_block["theme"])
        if day_theme in seen_themes:
            validation_errors.append(f"Day {day_number} 主题重复：{day_theme}。")
        seen_themes.add(day_theme)

        # slot_list：单日四个时间段的数据。
        slot_list = []
        for slot in slot_config:
            slot_text = extract_slot_text(day_block["body"], slot["label"])
            if not slot_text:
                validation_errors.append(f"Day {day_number} 缺少{slot['label']}安排。")
                continue

            if any(term in slot_text for term in generic_terms):
                validation_errors.append(f"Day {day_number} {slot['label']}仍包含空泛模板词：{slot_text[:40]}")

            # slot_parts：时间段内容必须包含地点、推荐理由、预计耗时、交通或预约提醒。
            slot_parts = [part.strip() for part in re.split(r"[｜|]", slot_text) if part.strip()]
            if len(slot_parts) < 4:
                validation_errors.append(
                    f"Day {day_number} {slot['label']}未完整包含地点、推荐理由、预计耗时、交通或预约提醒。"
                )
                continue

            place, description = split_place_and_description(slot_text)
            if not place:
                validation_errors.append(f"Day {day_number} {slot['label']}未按“具体地点｜推荐理由｜预计耗时｜交通或预约提醒”格式输出。")
                continue

            if place in seen_places:
                validation_errors.append(f"重复安排地点：{place}。")
            seen_places.add(place)

            slot_list.append(
                {
                    "label": slot["label"],
                    "time": slot["time"],
                    "icon": slot["icon"],
                    "place": place,
                    "description": description,
                }
            )

        timeline_days.append({"title": f"Day {day_number}：{day_block['theme']}", "slots": slot_list})

    if validation_errors:
        return [], validation_errors

    return timeline_days, []


def build_timeline_days_from_json(travel_json: dict | None, parsed_request: dict) -> tuple[list[dict], list[str]]:
    """build_timeline_days_from_json：把结构化 JSON 转成时间线卡片数据。"""

    # validation_errors：结构化 JSON 的校验错误。
    validation_errors = validate_structured_travel_json(travel_json, parsed_request)
    if validation_errors:
        return [], validation_errors

    # slot_config：英文 JSON 字段和页面展示标签的对应关系。
    slot_config = [
        {"key": "morning", "label": "上午", "icon": "AM"},
        {"key": "noon", "label": "中午", "icon": "NO"},
        {"key": "afternoon", "label": "下午", "icon": "PM"},
        {"key": "evening", "label": "晚上", "icon": "EV"},
    ]

    # timeline_days：最终时间线数据。
    timeline_days = []
    for day_item in travel_json.get("daily_itinerary", []):
        # slot_list：单日四个时间段的数据。
        slot_list = []
        for slot in slot_config:
            # slot_data：结构化 JSON 中的时间段对象。
            slot_data = day_item[slot["key"]]
            description = (
                f"{slot_data['original_name']}｜{slot_data['reason']}｜{slot_data['duration']}｜"
                f"{slot_data['transport']}；{slot_data['booking_note']}"
            )
            slot_list.append(
                {
                    "label": slot["label"],
                    "time": slot_data["time"],
                    "icon": slot["icon"],
                    "place": slot_data["place"],
                    "description": description,
                }
            )

        timeline_days.append({"title": f"Day {day_item['day']}：{day_item['theme']}", "slots": slot_list})

    return timeline_days, []


def build_food_cards(section_map: dict, parsed_request: dict) -> list[dict]:
    """build_food_cards：把美食推荐 Markdown 转成美食卡片数据。"""

    # food_items：美食推荐列表。
    food_items = extract_bullet_items(section_map.get("美食推荐", ""), max_items=6)

    if not food_items:
        food_items = [
            "本地代表料理：优先选择路线附近的高评分店｜人均 80-180 元｜适合第一顿正式餐",
            "街区小吃：适合放在下午或夜间，边走边吃更轻松｜人均 30-80 元｜适合探索街区",
            "甜品或咖啡：作为下午休息点，也适合拍照｜人均 40-100 元｜适合慢旅行",
        ]

    # budget_level：预算档位，用于美食卡片的人均预算兜底。
    budget_level = parsed_request.get("budget_level", parsed_request["budget"])

    # default_budget：美食卡片的人均预算兜底。
    default_budget = "人均 80-180 人民币(CNY)" if budget_level != "经济预算" else "人均 30-90 人民币(CNY)"

    # food_cards：最终美食卡片数据。
    food_cards = []
    for item in food_items:
        title = item
        detail = "结合行程路线选择，减少排队和跨区移动。"
        if "：" in item:
            title, detail = item.split("：", 1)
        elif ":" in item:
            title, detail = item.split(":", 1)

        # detail_parts：按竖线拆出的理由、预算和场景。
        detail_parts = [part.strip() for part in re.split(r"[｜|]", detail) if part.strip()]
        reason = detail_parts[0] if detail_parts else detail
        budget = detail_parts[1] if len(detail_parts) > 1 else default_budget
        scene = detail_parts[2] if len(detail_parts) > 2 else "适合穿插在当日行程中"

        food_cards.append(
            {
                "title": clean_markdown_text(title)[:34],
                "reason": clean_markdown_text(reason),
                "budget": clean_markdown_text(budget),
                "scene": clean_markdown_text(scene),
                "location": "位置：建议结合当日行程区域确认",
                "nearby_spot": "当日行程附近",
                "booking_note": "热门时段建议提前确认或预约",
                "map_keyword": clean_markdown_text(title)[:34],
            }
        )

    return food_cards


def build_food_cards_from_json(travel_json: dict | None) -> list[dict]:
    """build_food_cards_from_json：把结构化 JSON 中的美食推荐转成卡片数据。"""

    if not isinstance(travel_json, dict):
        return []

    # food_recommendations：结构化 JSON 中的美食推荐列表。
    food_recommendations = travel_json.get("food_recommendations", [])
    if not isinstance(food_recommendations, list):
        return []

    # food_cards：最终美食卡片数据。
    food_cards = []
    for food_item in food_recommendations[:6]:
        if not isinstance(food_item, dict):
            continue

        # name_cn/name_original：店铺中文名与英文/当地原名。
        name_cn = clean_markdown_text(str(food_item.get("name_cn", "")))
        name_original = clean_markdown_text(str(food_item.get("name_original", "")))
        title = f"{name_cn}（{name_original}）" if name_original and name_original != name_cn else name_cn

        food_cards.append(
            {
                "title": title or "待确认美食点",
                "location": clean_markdown_text(str(food_item.get("location", ""))) or "建议以地图搜索原名确认",
                "nearby_spot": clean_markdown_text(str(food_item.get("nearby_spot", ""))) or "适合穿插在当日行程中",
                "reason": clean_markdown_text(str(food_item.get("reason", ""))) or "结合行程路线选择，减少跨区移动。",
                "budget": clean_markdown_text(str(food_item.get("budget", ""))) or "人均预算待确认",
                "scene": clean_markdown_text(str(food_item.get("scene", ""))) or "适合穿插在当日行程中",
                "booking_note": clean_markdown_text(str(food_item.get("booking_note", ""))) or "热门时段建议提前确认或预约",
                "map_keyword": clean_markdown_text(str(food_item.get("map_keyword", ""))) or title,
            }
        )

    return food_cards


def build_advice_cards(section_text: str, fallback_items: list[str], max_items: int = 4) -> list[dict]:
    """build_advice_cards：把交通建议或避坑提醒转成卡片数据。"""

    # advice_items：从 Markdown 中提取出的建议列表。
    advice_items = extract_bullet_items(section_text, max_items=max_items) or fallback_items

    # advice_cards：最终建议卡片数据。
    advice_cards = []
    for index, item in enumerate(advice_items[:max_items], start=1):
        title = f"建议 {index}"
        description = item
        if "：" in item and len(item.split("：", 1)[0]) <= 16:
            title, description = item.split("：", 1)
        elif ":" in item and len(item.split(":", 1)[0]) <= 16:
            title, description = item.split(":", 1)
        else:
            title = clean_markdown_text(item)[:14]

        advice_cards.append({"title": clean_markdown_text(title), "description": clean_markdown_text(description)})

    return advice_cards


def render_hero() -> None:
    """render_hero：渲染页面顶部的产品标题区。"""

    st.markdown(
        """
        <nav class="top-nav">
            <div class="nav-brand"><span class="brand-mark">T</span><span>TripAgent</span></div>
            <div class="nav-links">
                <span>AI旅行规划器</span>
                <span>示例</span>
                <span>反馈</span>
            </div>
        </nav>
        <section class="hero">
            <div class="hero-layout">
                <div>
                    <div class="eyebrow">AI Travel Planner · Magazine Edition</div>
                    <h1>把一句旅行灵感变成完整攻略</h1>
                    <p>像和 AI 旅行编辑聊天一样输入需求，TripAgent 会自动识别目的地、天数、预算和偏好，并生成封面、路线、餐饮、交通、预算和避坑提醒。</p>
                </div>
                <aside class="hero-panel">
                    <div class="mini-card">
                        <span>Planning Depth</span>
                        <strong>从灵感到日程</strong>
                        <p>自动拆解目的地、旅行节奏和主题偏好，输出可复制的 Markdown 攻略。</p>
                    </div>
                    <div class="mini-card">
                        <span>Visual Guide</span>
                        <strong>杂志式结果页</strong>
                        <p>大封面、bento 摘要、每日时间线和重点提醒，适合快速浏览。</p>
                    </div>
                </aside>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_input_box() -> tuple[bool, str]:
    """render_input_box：渲染醒目的自然语言输入框。"""

    with st.container(border=True):
        st.markdown('<p class="input-title">告诉我你想怎么旅行</p>', unsafe_allow_html=True)
        st.markdown('<p class="sample-title">选择一个示例，或直接输入你的旅行需求</p>', unsafe_allow_html=True)

        # sample_columns：用于横向排列示例标签按钮。
        sample_columns = st.columns(4)
        for index, sample_prompt in enumerate(SAMPLE_PROMPTS):
            if sample_columns[index].button(sample_prompt["label"], key=f"sample_prompt_{index}"):
                st.session_state["travel_request_input"] = sample_prompt["prompt"]

        with st.form("travel_request_form"):
            # user_input：用户输入的一句话旅行需求。
            user_input = st.text_area(
                label="旅行需求",
                label_visibility="collapsed",
                placeholder="例如：我想去东京旅游，喜欢动漫、美食和夜景，预算5000，3 天 2 晚",
                key="travel_request_input",
            )

            # submitted：用户是否点击了生成按钮。
            submitted = st.form_submit_button("生成专属旅行攻略")

        st.markdown('<p class="hint">不用填复杂表单，一句话就够。没写天数默认 3 天 2 晚；预算数字没写单位时默认人民币 CNY。</p>', unsafe_allow_html=True)

    return submitted, user_input


def render_cover(parsed_request: dict, cover_image_url: str) -> None:
    """render_cover：渲染大封面图区域。"""

    # safe_destination：转义后的目的地文本，避免 HTML 注入。
    safe_destination = html.escape(parsed_request["destination"])

    # safe_preferences：转义后的偏好文本。
    safe_preferences = html.escape("、".join(parsed_request["preferences"]))

    # safe_cover_image_url：转义后的封面图片地址。
    safe_cover_image_url = html.escape(cover_image_url, quote=True)

    # badge_items：封面上展示的旅行关键信息标签。
    badge_items = [
        f"{parsed_request['days']} 天 {parsed_request['nights']} 晚",
        parsed_request["budget"],
        *parsed_request["preferences"][:4],
    ]

    # badge_html：封面标签 HTML。
    badge_html = "".join(f'<span class="cover-badge">{html.escape(item)}</span>' for item in badge_items)

    st.markdown(
        f"""
        <section class="cover-card" style='background-image: url("{safe_cover_image_url}");'>
            <div class="cover-content">
                <div class="label">AI TRAVEL MAGAZINE</div>
                <h2>{safe_destination}</h2>
                <p>{safe_preferences} · 由 AI 生成的旅行封面与城市探索计划</p>
                <div class="cover-badges">{badge_html}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_summary_bento(parsed_request: dict) -> None:
    """render_summary_bento：用 bento grid 展示攻略摘要信息。"""

    # preferences_text：用于展示的偏好文本。
    preferences_text = "、".join(parsed_request["preferences"])

    # metric_map：推荐强度、旅行节奏、适合人群和预计总花费。
    metric_map = build_summary_metrics(parsed_request)

    # budget_note：预算卡片中的说明文本。
    budget_note = parsed_request.get("budget_exchange_hint") or "价格为区间估算，出发前需再次确认。"

    st.markdown(
        f"""
        <h2 class="section-heading">攻略摘要</h2>
        <p class="section-subtitle">系统从你的自然语言输入中提取旅行关键参数，并补充可执行的规划指标。</p>
        <div class="bento-grid">
            <div class="bento-card large warm"><span>目的地</span><strong>{html.escape(parsed_request["destination"])}</strong><p>本次攻略围绕城市动线、主题偏好和轻量避坑提醒展开。</p></div>
            <div class="bento-card"><span>旅行天数</span><strong>{parsed_request["days"]} 天 {parsed_request["nights"]} 晚</strong><p>按每日四段式节奏规划。</p></div>
            <div class="bento-card"><span>预算</span><strong>{html.escape(parsed_request["budget"])}</strong><p>{html.escape(budget_note)}</p></div>
            <div class="bento-card large"><span>偏好标签</span><strong>{html.escape(preferences_text)}</strong><p>用于安排主题街区、美食和拍照点。</p></div>
            <div class="bento-card"><span>推荐强度</span><strong>{html.escape(metric_map["推荐强度"])}</strong><p>基于偏好匹配度估算。</p></div>
            <div class="bento-card"><span>旅行节奏</span><strong>{html.escape(metric_map["旅行节奏"])}</strong><p>兼顾体验密度和休息时间。</p></div>
            <div class="bento-card"><span>适合人群</span><strong>{html.escape(metric_map["适合人群"])}</strong><p>可按同行人群继续微调。</p></div>
            <div class="bento-card warm"><span>预计总花费</span><strong>{html.escape(metric_map["预计总花费"])}</strong><p>不含跨城机票或长途交通。</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_card(section_map: dict) -> None:
    """render_overview_card：展示详细攻略的简要说明卡片。"""

    # overview_text：详细旅游攻略内容。
    overview_text = section_map.get("详细旅游攻略", "")
    if not overview_text:
        return

    with st.container(border=True):
        st.markdown("### 旅行编辑摘要")
        st.markdown(overview_text)


def render_timeline(
    section_map: dict,
    parsed_request: dict,
    travel_json: dict | None = None,
    json_errors: list[str] | None = None,
    json_raw: str | None = None,
) -> None:
    """render_timeline：用时间线样式展示每日行程。"""

    # timeline_days：优先从结构化 JSON 构建每日行程时间线数据。
    if travel_json:
        timeline_days, timeline_errors = build_timeline_days_from_json(travel_json, parsed_request)
    else:
        timeline_days = []
        timeline_errors = json_errors or ["未生成可用于页面渲染的结构化 JSON。"]

    if timeline_errors:
        st.markdown(
            """
            <h2 class="section-heading">每日行程时间线</h2>
            <p class="section-subtitle">每日行程 JSON 没有通过结构化校验，因此没有使用默认模板补齐。</p>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.error("每日行程解析失败或天数不足。请重新生成，系统不会用重复模板自动补齐。")
            for timeline_error in timeline_errors[:8]:
                st.markdown(f"- {timeline_error}")
            if len(timeline_errors) > 8:
                st.markdown(f"- 还有 {len(timeline_errors) - 8} 条校验问题未展示。")
            st.markdown("- 请点击页面上方的“生成专属旅行攻略”重新生成。")
        with st.expander("查看模型返回原文", expanded=False):
            if json_raw:
                st.code(json_raw, language="json")
            else:
                st.markdown("未获取到结构化 JSON 原文。")
        return

    # day_html_list：每天独立卡片 HTML。
    day_html_list = []
    for day in timeline_days:
        slot_html_list = []
        for slot in day["slots"]:
            slot_html_list.append(
                '<div class="timeline-slot">'
                f'<div class="slot-icon">{html.escape(slot["icon"])}</div>'
                "<div>"
                f'<div class="slot-time">{html.escape(slot["label"])} · {html.escape(slot["time"])}</div>'
                f'<div class="slot-place">{html.escape(slot["place"])}</div>'
                f'<p class="slot-desc">{html.escape(slot["description"])}</p>'
                "</div>"
                "</div>"
            )

        day_html_list.append(
            '<article class="timeline-day">'
            f'<h3>{html.escape(day["title"])}</h3>'
            f'{"".join(slot_html_list)}'
            "</article>"
        )

    # timeline_html：无缩进 HTML，避免 Markdown 把 HTML 识别为代码块。
    timeline_html = (
        '<h2 class="section-heading">每日行程时间线</h2>'
        '<p class="section-subtitle">每天拆成上午、中午、下午和晚上四个时间段，便于实际执行。</p>'
        f'<div class="timeline-grid">{"".join(day_html_list)}</div>'
    )

    st.markdown(timeline_html, unsafe_allow_html=True)


def render_food_cards(section_map: dict, parsed_request: dict, travel_json: dict | None = None) -> None:
    """render_food_cards：用卡片展示美食推荐。"""

    # food_cards：美食卡片数据。
    food_cards = build_food_cards_from_json(travel_json) or build_food_cards(section_map, parsed_request)

    # food_html：美食卡片 HTML。
    food_html = ""
    for food in food_cards:
        food_html += f"""
        <article class="food-card">
            <h3>{html.escape(food["title"])}</h3>
            <div class="food-location">📍 位置：{html.escape(food["location"])}，靠近 {html.escape(food["nearby_spot"])}</div>
            <p>{html.escape(food["reason"])}</p>
            <div class="food-map-keyword">地图搜索：{html.escape(food["map_keyword"])}</div>
            <div class="food-meta">
                <span>{html.escape(food["budget"])}</span>
                <span>{html.escape(food["scene"])}</span>
                <span>{html.escape(food["booking_note"])}</span>
            </div>
        </article>
        """

    st.markdown(
        f"""
        <h2 class="section-heading">美食推荐</h2>
        <p class="section-subtitle">把餐饮当成行程体验的一部分，而不是临时补位。</p>
        <div class="food-grid">{food_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_advice_sections(section_map: dict) -> None:
    """render_advice_sections：展示交通建议和避坑提醒。"""

    # transport_fallback：交通建议兜底内容。
    transport_fallback = [
        "城市内优先使用地铁、公交或官方交通卡，减少频繁打车。",
        "每天尽量围绕一个区域规划，避免跨城式来回移动。",
        "机场或车站到酒店先查官方线路，再对比打车价格。",
        "最后一天优先选择寄存点或酒店寄存，减少拖行李时间。",
    ]

    # warning_fallback：避坑提醒兜底内容。
    warning_fallback = [
        "不要把热门景点、热门餐厅和远距离交通挤在同一天。",
        "出发前确认营业时间、预约方式和交通路线。",
        "夜景点受天气影响明显，建议保留备选方案。",
        "购物和伴手礼尽量放在后半程，避免一路背负行李。",
    ]

    # transport_cards：交通建议卡片数据。
    transport_cards = build_advice_cards(section_map.get("交通建议", ""), transport_fallback, max_items=4)

    # warning_cards：避坑提醒卡片数据。
    warning_cards = build_advice_cards(section_map.get("避坑提醒", ""), warning_fallback, max_items=4)

    # budget_items：预算估算列表。
    budget_items = extract_bullet_items(section_map.get("预算估算", ""), max_items=4)

    transport_html = "".join(
        f"""
        <article class="info-card">
            <div class="card-title-row">
                <div class="info-icon">i</div>
                <h3>{html.escape(card["title"])}</h3>
            </div>
            <p>{html.escape(card["description"])}</p>
        </article>
        """
        for card in transport_cards
    )

    warning_html = "".join(
        f"""
        <article class="warning-card">
            <div class="card-title-row">
                <div class="warning-icon">!</div>
                <h3>{html.escape(card["title"])}</h3>
            </div>
            <p>{html.escape(card["description"])}</p>
        </article>
        """
        for card in warning_cards
    )

    st.markdown(
        f"""
        <h2 class="section-heading">交通建议</h2>
        <p class="section-subtitle">优先减少无效移动，把时间留给真正的体验。</p>
        <div class="info-grid">{transport_html}</div>
        """,
        unsafe_allow_html=True,
    )

    if budget_items:
        with st.container(border=True):
            st.markdown("### 预算估算")
            for budget_item in budget_items:
                st.markdown(f"- {budget_item}")

    st.markdown(
        f"""
        <h2 class="section-heading">避坑提醒</h2>
        <p class="section-subtitle">提前规避高概率踩坑点，让行程更稳定。</p>
        <div class="warning-grid">{warning_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_source_info(section_map: dict) -> None:
    """render_source_info：展示信息来源与更新时间区域。"""

    # source_text：模型生成的来源与更新时间内容。
    source_text = section_map.get("信息来源与更新时间", "")
    if not source_text:
        return

    with st.container(border=True):
        st.markdown("### 信息来源与更新时间")
        st.markdown(source_text)


def render_visual_guide(
    markdown_text: str,
    parsed_request: dict,
    travel_json: dict | None = None,
    json_errors: list[str] | None = None,
    json_raw: str | None = None,
) -> None:
    """render_visual_guide：把 Markdown 攻略渲染成高级卡片式视觉结果。"""

    # section_map：按标题拆分后的攻略内容。
    section_map = split_markdown_sections(markdown_text)

    if not section_map:
        with st.container(border=True):
            st.markdown(markdown_text)
        render_timeline({}, parsed_request, travel_json, json_errors, json_raw)
        return

    render_overview_card(section_map)
    render_timeline(section_map, parsed_request, travel_json, json_errors, json_raw)
    render_food_cards(section_map, parsed_request, travel_json)
    render_advice_sections(section_map)
    render_source_info(section_map)


def render_copy_button(markdown_text: str) -> None:
    """render_copy_button：渲染可复制 Markdown 攻略的按钮。"""

    # markdown_json：安全注入到 JavaScript 的攻略文本。
    markdown_json = json.dumps(markdown_text, ensure_ascii=False)

    st.html(
        f"""
        <style>
        .copy-widget {{
            font-family: Arial, sans-serif;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            padding: 8px 0;
        }}
        .copy-widget button {{
            border: 1px solid rgba(246, 199, 111, 0.32);
            border-radius: 999px;
            padding: 12px 18px;
            background: linear-gradient(135deg, #f6c76f, #fb923c);
            color: #17120a;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 12px 28px rgba(251, 146, 60, 0.18);
        }}
        .copy-widget span {{
            color: #cbd5e1;
            font-size: 14px;
        }}
        </style>
        <div class="copy-widget">
            <button id="copy-markdown-button">复制 Markdown 攻略</button>
            <span id="copy-markdown-status">也可以直接复制下方原文。</span>
        </div>
        <script>
        const markdownText = {markdown_json};
        const copyButton = document.getElementById("copy-markdown-button");
        const copyStatus = document.getElementById("copy-markdown-status");
        copyButton.addEventListener("click", async () => {{
            try {{
                await navigator.clipboard.writeText(markdownText);
                copyStatus.textContent = "已复制到剪贴板。";
            }} catch (error) {{
                copyStatus.textContent = "复制失败，请手动复制下方原文。";
            }}
        }});
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_markdown_source(markdown_text: str) -> None:
    """render_markdown_source：用折叠区域展示 Markdown 原文，并提供复制和下载。"""

    st.markdown('<h2 class="section-heading">Markdown 原文</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">默认收起，适合最后复制到笔记、公众号或行程文档中。</p>', unsafe_allow_html=True)

    with st.expander("查看可复制的 Markdown 攻略", expanded=False):
        # action_columns：复制和下载按钮区域。
        action_columns = st.columns([1, 1])
        with action_columns[0]:
            render_copy_button(markdown_text)
        with action_columns[1]:
            st.download_button(
                label="下载 Markdown 文件",
                data=markdown_text,
                file_name="ai-travel-guide.md",
                mime="text/markdown",
            )

        st.code(markdown_text, language="markdown")


def render_debug_panel(parsed_request: dict) -> None:
    """render_debug_panel：默认折叠展示系统识别出的结构化参数。"""

    # debug_data：开发调试用的结构化解析结果。
    debug_data = {
        "destination": parsed_request["destination"],
        "days": parsed_request["days"],
        "nights": parsed_request["nights"],
        "budget_amount": parsed_request.get("budget_amount"),
        "currency": parsed_request.get("budget_currency") or "未指定",
        "style": parsed_request.get("style") or parsed_request.get("budget_level"),
        "preferences": "、".join(parsed_request["preferences"]),
    }

    with st.expander("Debug：查看系统识别参数", expanded=False):
        st.json(debug_data)


def render_search_status(search_message: str | None) -> None:
    """render_search_status：用小标签展示 Tavily 联网搜索状态。"""

    if not search_message:
        return

    # safe_message：转义后的状态文案，避免 HTML 注入。
    safe_message = html.escape(search_message)
    st.markdown(
        f"""
        <div class="search-status-pill">
            <span class="search-status-dot"></span>
            <span>{safe_message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_generation_count() -> int:
    """get_generation_count：读取当前浏览器 session 已生成攻略次数。"""

    return int(st.session_state.get("generation_count", 0))


def render_generation_quota() -> None:
    """render_generation_quota：展示 Beta 测试版当前 session 剩余生成次数。"""

    # remaining_count：当前 session 剩余生成次数。
    remaining_count = max(0, MAX_GENERATIONS_PER_SESSION - get_generation_count())
    st.markdown(
        f"""
        <p class="generation-quota">Beta 测试额度：本会话剩余 {remaining_count}/{MAX_GENERATIONS_PER_SESSION} 次生成。</p>
        """,
        unsafe_allow_html=True,
    )


def build_result_data(user_input: str) -> dict:
    """build_result_data：根据用户输入生成结果数据，但不把完整用户输入保存到 session。"""

    # parsed_request：自然语言解析结果。
    parsed_request = parse_travel_request(user_input)

    # cover_image_url：封面图地址，第一版是本地 SVG 占位。
    cover_image_url = generate_cover_image_url(parsed_request)

    with st.status("AI 正在分析你的目的地", expanded=True) as loading_status:
        time.sleep(0.2)
        if get_bool_config("USE_TAVILY", True) and get_tavily_api_key():
            loading_status.update(label="正在联网校验门票、预约和开放时间", state="running")
        else:
            loading_status.update(label="当前未启用联网搜索，正在切换普通生成模式", state="running")

        # facts_context：联网搜索整理出的事实校验上下文。
        facts_context, source_records, search_message = build_facts_context(parsed_request)
        loading_status.update(label="正在生成结构化行程 JSON", state="running")
        # markdown_text/travel_json：最终展示的 Markdown 和页面时间线使用的结构化 JSON。
        markdown_text, api_message, travel_json, json_raw, json_errors = generate_travel_content(
            user_input,
            parsed_request,
            facts_context,
        )
        loading_status.update(label="正在基于 JSON 整理 Markdown 攻略", state="running")
        time.sleep(0.2)
        loading_status.update(label="攻略已生成", state="complete", expanded=False)

    return {
        "parsed_request": parsed_request,
        "cover_image_url": cover_image_url,
        "markdown_text": markdown_text,
        "api_message": api_message,
        "travel_json": travel_json,
        "json_raw": json_raw,
        "json_errors": json_errors,
        "search_message": search_message,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def render_result_data(result_data: dict) -> None:
    """render_result_data：渲染已经生成并缓存在当前 session 中的攻略结果。"""

    # parsed_request：结构化旅行参数，不包含任何 API Key。
    parsed_request = result_data["parsed_request"]

    # markdown_text：最终展示和复制的 Markdown 攻略。
    markdown_text = result_data["markdown_text"]

    render_search_status(result_data.get("search_message"))

    if result_data.get("api_message"):
        st.info(result_data["api_message"])

    render_debug_panel(parsed_request)
    render_cover(parsed_request, result_data["cover_image_url"])
    render_summary_bento(parsed_request)
    render_visual_guide(
        markdown_text,
        parsed_request,
        result_data.get("travel_json"),
        result_data.get("json_errors"),
        result_data.get("json_raw"),
    )
    render_markdown_source(markdown_text)


def render_result(user_input: str) -> None:
    """render_result：兼容旧调用方式，立即生成并渲染完整旅行攻略。"""

    render_result_data(build_result_data(user_input))


def render_beta_notice() -> None:
    """render_beta_notice：在页面底部展示 Beta 测试版和隐私安全提醒。"""

    st.markdown(
        f"""
        <div class="beta-notice">{html.escape(BETA_NOTICE_TEXT)}</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """main：应用入口函数。"""

    setup_page()
    render_hero()

    # submitted：是否点击生成按钮。
    submitted, user_input = render_input_box()
    render_generation_quota()

    if submitted:
        if not user_input.strip():
            st.warning("请先输入一句旅行需求。")
        elif get_generation_count() >= MAX_GENERATIONS_PER_SESSION:
            st.warning(
                f"当前 Beta 测试版每个浏览器会话最多生成 {MAX_GENERATIONS_PER_SESSION} 次攻略。"
                "请刷新浏览器会话或稍后再试。"
            )
        else:
            # generation_count：点击生成才增加次数，页面重绘不会重复消耗 API。
            st.session_state["generation_count"] = get_generation_count() + 1

            # last_result_data：只保存生成后的旅行参数和攻略结果，不保存完整用户输入或任何密钥。
            st.session_state["last_result_data"] = build_result_data(user_input.strip())

    if "last_result_data" in st.session_state:
        render_result_data(st.session_state["last_result_data"])
    else:
        st.markdown(
            """
            <p class="hint">示例输入：我想去东京旅游，喜欢动漫、美食和夜景，预算5000，想玩 3 天 2 晚。</p>
            """,
            unsafe_allow_html=True,
        )

    render_beta_notice()


if __name__ == "__main__":
    main()
