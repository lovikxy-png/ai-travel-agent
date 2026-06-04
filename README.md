# AI 旅游攻略 Agent

这是一个最小可运行版本的 Python + Streamlit 项目。用户输入一句自然语言旅行需求，系统会识别目的地、天数、预算和偏好，并通过 DeepSeek API 生成旅行攻略 Markdown。

## 项目文件

- `app.py`：主程序，包含页面、需求解析、大模型调用和结果展示。
- `requirements.txt`：项目依赖。
- `.env.example`：环境变量示例。
- `.streamlit/config.toml`：Streamlit 暗色主题配置。

## 本地运行

进入项目目录：

```powershell
cd "D:\claude code\ai-travel-agent"
```

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install -r requirements.txt
```

配置 DeepSeek API Key。推荐先复制环境变量示例文件：

```powershell
copy .env.example .env
notepad .env
```

把 `.env` 改成：

```env
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_MODEL=deepseek-v4-flash

TAVILY_API_KEY=你的 Tavily API Key
USE_TAVILY=true
TAVILY_SEARCH_DEPTH=basic
TAVILY_MAX_SEARCHES_PER_GUIDE=1
```

也可以使用临时环境变量，只对当前 PowerShell 窗口有效：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
$env:TAVILY_API_KEY="你的 Tavily API Key"
$env:USE_TAVILY="true"
$env:TAVILY_SEARCH_DEPTH="basic"
$env:TAVILY_MAX_SEARCHES_PER_GUIDE="1"
```

如果你暂时没有 DeepSeek API Key，也可以直接运行，页面会使用本地演示攻略。如果没有 `TAVILY_API_KEY`，系统不会编造门票、预约、开放时间，会提示出行前再次核对。

启动应用：

```powershell
streamlit run app.py
```

打开终端输出里的本地地址，通常是：

```text
http://localhost:8501
```

## 免费 Beta 部署说明

推荐使用 Streamlit Community Cloud 的免费方案部署测试版。不要把 `.env`、`.streamlit/secrets.toml`、`tavily_cache.json`、日志文件或任何真实 API Key 提交到 GitHub。

部署到 Streamlit Cloud 时，请只在平台的 Secrets 页面填写密钥，例如：

```toml
DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
DEEPSEEK_MODEL = "deepseek-v4-flash"

TAVILY_API_KEY = "你的 Tavily API Key"
USE_TAVILY = "true"
TAVILY_SEARCH_DEPTH = "basic"
TAVILY_MAX_SEARCHES_PER_GUIDE = "1"
```

这些内容只用于部署平台 Secrets，不要提交到代码仓库。

当前项目包含 Beta 额度保护：每个浏览器 session 最多生成 3 次攻略；Tavily 默认 basic 搜索，每份攻略最多搜索 1 次；Tavily 失败、额度不足或未配置 Key 时会自动切换普通生成模式。

## 后续可扩展

- 在 `generate_cover_image_url()` 中接入图片生成 API。
- 在 `call_deepseek_api()` 中替换成其他大模型服务。
- 增加地图、景点链接、天气和实时价格查询。
