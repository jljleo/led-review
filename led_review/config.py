"""全局配置：集中管理原本散落各处的常量。

所有可调参数收在这里，改行为不用翻业务代码。
"""

import json
import os
import sys

from dotenv import load_dotenv

# --- 项目路径 ---
# 工作对象根：启动时所在的项目目录（cwd）。agent 操作的是「用户项目」而不是
# mini-agent 自己的源码目录——pipx 安装后两者分离，按 __file__ 定位会指向
# site-packages。bench/测试通过 monkeypatch（tools.PROJECT_ROOT）替换到沙箱。
PROJECT_ROOT = os.getcwd()
# .env 两处查找：默认向上查找（仓库开发场景，config.py 在仓库根）+
# cwd（pipx 场景的唯一可靠位置——用户项目的 .env）
load_dotenv()
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
HISTORY_FILE = os.path.join(PROJECT_ROOT, ".chat_history")  # prompt_toolkit 历史（跨会话）
SESSION_FILE = os.path.join(PROJECT_ROOT, ".session.json")  # 会话存档（/resume 恢复用）

# --- 模型（多模型档案）---
#   - models.default.json：内置默认档案，按 config.py 的 __file__ 定位（仓库根或
#     site-packages）——不随 PROJECT_ROOT(cwd) 走，pipx 打包时随模块装入 wheel
#   - models.json：用户本地覆盖/新增档案（cwd，gitignored）
# 格式：顶层对象，键为档案名，值为 {"model", "base_url", "api_key_env", "context_tokens"}。
_DEFAULT_MODELS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.default.json")
_USER_MODELS_JSON = os.path.join(PROJECT_ROOT, "models.json")


def _load_profiles_from_file(path: str) -> dict:
    """加载 JSON 模型档案文件；失败时返回空字典并警告。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"警告：{path} 加载失败: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"警告：{path} 顶层必须是对象", file=sys.stderr)
        return {}
    return data


def refresh_default_profiles() -> None:
    """重新加载默认模型档案（models.default.json）。"""
    global MODEL_PROFILES
    MODEL_PROFILES = _load_profiles_from_file(_DEFAULT_MODELS_JSON)
    if not MODEL_PROFILES:
        raise SystemExit(
            f"默认模型档案文件 {_DEFAULT_MODELS_JSON} 缺失或为空，"
            "请从 models.json.example 复制或恢复仓库文件"
        )


def refresh_user_profiles(path: str | None = None) -> None:
    """重新加载用户模型档案；path 缺省使用项目根 models.json。"""
    global USER_PROFILES
    USER_PROFILES = _load_profiles_from_file(path or _USER_MODELS_JSON)


def get_profile(name: str) -> dict:
    """解析并归一化单个模型档案（用户档案覆盖默认档案）。

    返回字典包含：model, base_url, api_key_env, context_tokens,
    truncate_high_tokens, truncate_low_tokens。
    """
    if name in USER_PROFILES:
        raw = USER_PROFILES[name]
    elif name in MODEL_PROFILES:
        raw = MODEL_PROFILES[name]
    else:
        raise KeyError(name)
    if not isinstance(raw, dict):
        raise KeyError(f"档案 {name!r} 格式错误")
    for required in ("model", "base_url"):
        if required not in raw:
            raise KeyError(f"档案 {name!r} 缺少必填字段 {required!r}")
    context_tokens = raw.get("context_tokens", 128_000)
    # 高水位余量 = max(5% 窗口, 16K)：固定值的 28K 在小窗口档案(256K/128K)上
    # 会吃掉 11%/22% 窗口（见 E 19YY 水位讨论）；比例+下限保证各窗口同安全
    # 余量：1M→52K（宽裕）、262K→16K（下限）、128K→16K（下限）。
    high = context_tokens - max(16_000, int(context_tokens * 0.05))
    return {
        "name": name,
        "model": raw["model"],
        "base_url": raw["base_url"],
        "api_key_env": raw.get("api_key_env", "MOONSHOT_API_KEY"),
        "context_tokens": context_tokens,
        "truncate_high_tokens": high,
        "truncate_low_tokens": int(high * 0.6),
    }


def list_profiles() -> list[str]:
    """返回所有可用档案名（默认 + 用户覆盖/新增）。"""
    return list({**MODEL_PROFILES, **USER_PROFILES})


def apply_profile(name: str) -> None:
    """把全局常量切换到指定档案；失败时抛出 KeyError。"""
    global MODEL_PROFILE, MODEL, BASE_URL, API_KEY_ENV, CONTEXT_TOKENS
    global TRUNCATE_HIGH_TOKENS, TRUNCATE_LOW_TOKENS
    profile = get_profile(name)
    MODEL_PROFILE = name
    MODEL = profile["model"]
    BASE_URL = profile["base_url"]
    API_KEY_ENV = profile["api_key_env"]
    CONTEXT_TOKENS = profile["context_tokens"]
    TRUNCATE_HIGH_TOKENS = profile["truncate_high_tokens"]
    TRUNCATE_LOW_TOKENS = profile["truncate_low_tokens"]


MODEL_PROFILES: dict = {}
USER_PROFILES: dict = {}
refresh_default_profiles()
refresh_user_profiles()

MODEL_PROFILE = os.environ.get("MINI_AGENT_MODEL", "kimi-code")
try:
    apply_profile(MODEL_PROFILE)
except KeyError as exc:
    available = ", ".join(list_profiles())
    raise SystemExit(
        f"未知模型档案 {MODEL_PROFILE!r}（MINI_AGENT_MODEL），"
        f"可选：{available}；新模型请在 models.default.json 或 models.json 添加"
    ) from exc


def format_context_tokens(n: int | None = None) -> str:
    """上下文窗口的紧凑显示：128K / 1.0M（状态栏与横幅共用，pi 同款比例样式）。"""
    if n is None:
        n = CONTEXT_TOKENS
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n // 1000}K"


def format_tokens(n: int) -> str:
    """token 数的紧凑显示：0 / 999 / 1.5K / 15K / 1.5M。

    规则：
    - <1K 显示精确值
    - 1K~10K 保留一位小数（如 1.5K）
    - ≥10K 显示整数 K
    - ≥1M 显示一位小数 M
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n // 1_000}K"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)

# --- agent 循环 ---
# 无硬性轮次上限：交互场景人在看（业界交互模式均不设上限），失控防线是下面的
# 行为保险丝。bench 等无人值守场景的上限应加在调用侧，不污染交互循环。
# 死循环保险丝（行为识别，替代单纯的轮次计数）：同一 (工具名, 参数) 连续出现
# 此次数即判定死循环，强制结束本轮。正常任务每次调用参数不同，不会误伤。
MAX_SAME_TOOL_CALLS = 3

# --- 写后验证闭环（R0 未勾项，2026-09-11 落地） ---
# edit_file / write_file 改完文件后自动执行的校验命令模板列表（顺序执行，首个失败
# 即停，stderr 带行号回餵工具结果——模型下一轮看到自己改坏的东西当场自愈）。
# {path} 替换为改动文件的项目相对路径。默认空 = 仅保留 tree-sitter 语法自检
# （零回归、零额外开销）；配置示例：
#   POST_WRITE_CHECK_COMMANDS = ["ruff check {path}", "python -m py_compile {path}"]
# 校验失败不算工具失败（不抛异常）：验证信号优于执行结果（AGENT_DESIGN 49 条）。
POST_WRITE_CHECK_COMMANDS: list[str] = []

# --- 输出/上下文保护 ---
MAX_OUTPUT_LEN = 10_000  # 工具结果 / 命令输出的截断阈值：防大输出灌爆上下文
TOOL_RESULT_PREVIEW_LEN = 100  # 终端里工具结果的预览长度
MAX_TIMEOUT = 120  # bash 超时上限（秒）：由代码钳制，不信任模型传入的值

# --- 子 agent ---
# 类型化子 agent（pi 的 agents 目录 / opencode 的 subagent_type 同款思想）：
# "派谁"这个决策本身携带约束——类型决定工具边界与命令策略。
# 主 agent 派生时必须想清楚"这个任务需要什么能力才能完成"，而不是甩一个通用 agent。
# NOTE（2026-09）：沙箱机制已移除。子 agent 直接操作真实项目，隔离完全由
# SUBAGENT_TYPES 工具表 + command_policy + 用户审批保证。
SUBAGENT_TYPES = {
    "researcher": {
        # 只读调研：无写工具；bash 给上（grep/find/cat 走 allow 直通，非白名单命令硬拒）
        "tools": ("read_file", "search_tools", "run_bash"),
        "command_policy": "read_only",  # 确定性白名单：allow 直通，其余硬拒（零 LLM 成本）
        "description": "只读调研：读代码、搜历史，产出分析结论；不能改文件",
    },
    "coder": {
        # 改代码：可写可跑（bash 越界路径硬拒 + ambiguous 冒泡给人工审批），改动落真实项目靠 git 兜底
        "tools": ("read_file", "write_file", "edit_file", "run_bash", "search_tools"),
        "command_policy": "human",  # allow 也降级为 ask，deny 硬拒，ask → 冒泡给人工审批（opencode 式：规则 + 人）
        "description": "改代码：读写文件、跑命令；改动落真实项目（git 可恢复）",
    },
}
SUBAGENT_HIDDEN_TOOLS = ("spawn_subagent", "todo_write", "todo_read")
MAX_SUBAGENT_DEPTH = 1  # 嵌套限深：只允许主 agent 派一级子 agent
# spawn_researchers 并发上限：模型无依据选择并发度，收敛为配置常量（不对模型暴露）
SUBAGENT_MAX_PARALLEL = 3
# 子 agent 拒绝熔断：连续被拒此次数即 abort 本轮（codex GuardianRejectionCircuitBreaker
# 思路）——反复试探授权 = 边界划错了，掐死止血，结论带回主 agent 自我修正。
SUBAGENT_DENIAL_LIMIT = 3

# --- 斜杠命令 ---
# 退出词表（非斜杠命令，主循环直接识别；/quit 也走这里统一退出）
QUIT_COMMANDS = ("exit", "quit", ":q", "/quit")

# --- L3 工具结果瘦身（compact.py）---
TOOL_RESULT_KEEP_RECENT = 5  # 保护窗口：最近 N 条 tool 消息不瘦身（churn 防线，勿设 0）
TOOL_RESULT_MIN_SLIM_LEN = 500  # 原文短于此长度不瘦身：占位符本身 ~80 字符，太短是负收益
TOOL_ARG_ECHO_LEN = 60  # 占位符中参数回显的截断长度（防占位符自身膨胀）
# 触发阈值：历史总字符数低于此值完全不动作（保护 prompt cache）。
# 粗估 1 token ≈ 2 字符（中英混合语料），80K 字符 ≈ 40K tokens。
SLIM_TRIGGER_CHARS = 80_000
# 收益门槛：本轮瘦身能省下的字符总量低于此值就不动——省几百字符却顶掉几千 tokens 的
# 缓存前缀是净亏损（典型场景：reasoning 占大头、tool 结果很小的会话）
SLIM_MIN_SAVINGS_CHARS = 2_000

# --- L1 历史截断（compact.py，兜底防爆）---
# 触发用估算 token（chars//2）：达到高水位才截，一刀切到低水位。
# 双水位滞后：防“刚好切到阈值下、下轮又超”导致每轮都截、每轮缓存全失效。
# （按 token 而非消息条数：条数与上下文占用无量纲关系，一条大文件结果可顶几十条闲聊）
# 水位随所选模型的上下文窗口走：kimi 128K 窗口 → 100K/60K，与历史调参一致。
# 低水位按比例（而非固定减量）取：小窗口模型上固定减量会把滞后带扣成负数。
# 高水位余量 = max(5% 窗口, 16K)（同 get_profile 的公式，两处必须一致）：
# 固定 28K 对 1M 窗口只占 2.7% （宽裕），但对 256K/128K 档案吃掉 11%/22%
# （紧张）——按比例缩放后各窗口余量语义一致；下限 16K 防小窗口档案
# 余量被压缩到低于实测撑住的 12K（E13 实拍：120k 档 12K 余量撑过 97K 任务）。
TRUNCATE_HIGH_TOKENS = CONTEXT_TOKENS - max(16_000, int(CONTEXT_TOKENS * 0.05))
TRUNCATE_LOW_TOKENS = int(TRUNCATE_HIGH_TOKENS * 0.6)  # 截断目标：切完留 40% 滞后带增长

# --- 单条体积上限（compact.py，投影级）---
# 分层防御的“单条体积”层：MAX_OUTPUT_LEN 只管工具产出，user 输入无天然上限——
# 首条 user 超大会让保留区自身爆窗（会话永久报废），末条 user 超大会让 L1 兜底失效。
# 60K 字符 ≈ 30K tokens：即使首条+末条同时超大，两者合计仍远低于 128K 窗口。
SINGLE_MSG_CAP_CHARS = 60_000

# --- L2 摘要（compact.py，L1 的保值版）---
# 触发与 L1 同高水位：到线后先尝试让模型压缩中段，失败再回退硬切。
SUMMARIZE_MAX_CHARS = 150_000  # 摘要输入上限：中段超长时只取靠后部分（更贴近当前任务）

# --- 代码库感知（repo_map.py）---
# repo map 注入 system prompt 的字符预算（约 3000 字符 ≈ 1.5K tokens），
# 超预算截断，头部保留最高重要性文件/符号；明细靠 search_symbols 惰性取。
REPO_MAP_MAX_CHARS = 3000

# --- system 提示词 ---
SYSTEM_MESSAGES = [
    {
        "role": "system",
        "content": (
            "你是 Kimi，由 Moonshot AI 提供的人工智能助手，专注于软件工程与编程任务。"
            "你会优先使用已声明的工具（read_file、write_file、edit_file、run_bash、search_tools 等）"
            "来完成任务；需要联网获取实时信息时，使用 run_bash 执行 curl（执行前向用户说明要访问的地址）。"
            "现有工具不够用时，再用 search_tools 查找额外工具。"
            "你会为用户提供安全、有帮助、准确的回答，并拒绝涉及恐怖主义、种族歧视、黄色暴力等问题的请求。"
        ),
    },
    {
        "role": "system",
        "content": "文件读写、run_bash 等基础工具已直接声明可用；todo 清单、search_history 历史检索等工具用 search_tools 检索后即可调用。",
    },
    {
        "role": "system",
        "content": (
            "任务规划规则（todo_write/todo_read 需先经 search_tools 检索发现后调用）：\n"
            "1. 满足以下任一条件，先用 todo_write 建清单再动手：步骤 ≥3、涉及多个文件、需求模糊需要拆解。\n"
            "2. 清单全量覆盖：每次 todo_write 传入完整列表，逐项更新状态（pending → in_progress → completed），"
            "开始某步前标 in_progress，完成立即标 completed。\n"
            "3. 单轮问答、一步能完成的任务不要使用 todo，直接完成。\n"
            "4. 不确定当前进度时，先 todo_read 查看清单再继续。"
        ),
    },
    {
        "role": "system",
        "content": (
            "并发调研优化：当你需要同时调研多个独立的文件、目录或问题时，"
            "优先使用 spawn_researchers 工具并行派生 researcher 子 agent，"
            "而不是连续多次调用 read_file 或 run_bash。spawn_researchers 只用于只读调研，"
            "不能用于写文件或执行会修改项目的命令；需要写操作时仍使用单个 spawn_subagent（coder）顺序执行。"
        ),
    },
]
