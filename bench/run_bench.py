"""benchmark 驱动：批量任务 → 沙箱执行 → 分层判分 → 轨迹导出 → 回归对比。

用法（在项目根目录）：
    python bench/run_bench.py                 # 跑全部任务
    python bench/run_bench.py fix_fizzbuzz    # 只跑指定任务
    python bench/run_bench.py --compare       # 跑完后与 baseline 对比回归

任务结构（bench/tasks/<name>/）：
    workspace/   agent 的工作区，会被复制到独立临时目录（chat 任务）
    PROMPT.md    发给 agent 的任务描述（chat 任务）
    META.json    任务元数据：judge 类型（deterministic/graded/llm-judge）、rubric 等；
                 type=review 的任务无 PROMPT.md，改为 base/ + bug.patch + ground truth
    verify.py    判分脚本（deterministic/graded 任务；llm-judge 任务不需要）

review 任务结构（type=review，review 模式的地面真值评测）：
    base/        干净代码（提交 1）
    bug.patch    注入的 bug（提交 2，git apply；人可审计的单文件 bug 视图）
    META.json    type=review + bugs ground truth 清单（id/path/lines/description）
    评测语义：agent 不知道 bug 存在（与 fix_* 的「被告知去修」相反）——检出率/误报率

评分三层：
    deterministic：verify.py exit 0 = pass
    graded：verify.py 输出 score=0.8 之类，解析为 0~1 分
    llm-judge：无确定性判据时，judge.py 用 LLM + rubric 打分

可观测：TraceRecorder 挂在事件流上，每任务产出 .trace.jsonl 轨迹。
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BENCH_DIR = PROJECT_ROOT / "bench"
TASKS_DIR = BENCH_DIR / "tasks"
RESULTS_DIR = BENCH_DIR / "results"

import led_review.repo_map as repo_map  # noqa: E402
import led_review.tools.builtin as tools  # noqa: E402
import led_review.ui.renderer as ui  # noqa: E402
from bench.scoring import (  # noqa: E402
    build_summary,
    compare_summaries,
    load_manifest,
    load_meta,
    parse_verify_score,
    score_review,
)
from led_review import review as review_pipeline  # noqa: E402
from led_review.eval.judge import judge, make_client  # noqa: E402
from led_review.eval.trace import TraceRecorder  # noqa: E402
from led_review.kernel.agent import ChatSession  # noqa: E402


def discover_tasks(only: str | None = None) -> list[Path]:
    """发现任务目录：含 PROMPT.md（chat 任务）或 META type=review（review 任务）。

    verify.py 按需存在：deterministic / graded 任务必须有，llm-judge 任务
    （META.json 里 judge=llm-judge）没有 verify.py——产出是对话正文，判分走 judge.py。
    """
    tasks = sorted(
        d for d in TASKS_DIR.iterdir()
        if d.is_dir() and ((d / "PROMPT.md").exists() or load_meta(d).get("type") == "review")
    )
    if only:
        tasks = [d for d in tasks if d.name == only]
        if not tasks:
            sys.exit(f"找不到任务: {only}（可用: {[d.name for d in discover_tasks()]}）")
    return tasks


def last_assistant_text(session) -> str:
    """取会话最后一条 assistant 正文，作为 llm-judge 的产出输入。"""
    if not session:
        return ""
    for m in reversed(session.messages):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return ""


def score_task(meta: dict, task_dir: Path, sandbox: Path, session) -> dict:
    """按 META.judge 分层判分，返回 {passed, score, method, ...}。"""
    judge_type = meta.get("judge", "deterministic")

    if judge_type == "llm-judge":
        result = judge(make_client(), last_assistant_text(session), meta.get("rubric", ""))
        return {
            "passed": result["score"] >= 0.5,
            "score": result["score"],
            "method": "llm-judge",
            "judge_reason": result["reason"],
        }

    verify = task_dir / "verify.py"
    if not verify.exists():
        return {
            "passed": False,
            "score": 0.0,
            "method": judge_type,
            "error": f"{judge_type} 任务缺 verify.py（llm-judge 任务才不需要）",
        }

    proc = subprocess.run(
        [sys.executable, str(verify)],
        cwd=sandbox, capture_output=True, text=True, timeout=60,
    )
    verify_stdout, verify_stderr = proc.stdout[-2000:], proc.stderr[-2000:]

    if judge_type == "graded":
        score = parse_verify_score(proc.stdout)
        if score is not None:
            return {
                "passed": proc.returncode == 0 or score >= 0.5,
                "score": score,
                "method": "graded",
                "verify_stdout": verify_stdout,
                "verify_stderr": verify_stderr,
            }

    return {
        "passed": proc.returncode == 0,
        "score": 1.0 if proc.returncode == 0 else 0.0,
        "method": "deterministic",
        "verify_stdout": verify_stdout,
        "verify_stderr": verify_stderr,
    }


def apply_group(group: str) -> None:
    """按实验组切换代码库感知能力（A/B 对照用）。

    map    = 实验组：注入 repo map + search_symbols 可用
    nomap  = 对照组：不注入地图，工具面移除 search_symbols（含常驻声明与
              search_tools 发现入口，模型完全不知道它的存在）——只能 read_file/grep 盲探。
    """
    import led_review.kernel.agent as agent
    import led_review.tools.registry as tool_registry
    if group == "nomap":
        agent.build_repo_map_cached = lambda **kw: ""          # noqa: E731 不注入地图
        tool_registry.TOOLS.pop("search_symbols", None)        # 删除执行体
        # 常驻名单与 schema 面同步移除（一处残留即 KeyError 或 schema 泄漏）
        tool_registry.RESIDENT_TOOL_NAMES = tuple(
            n for n in tool_registry.RESIDENT_TOOL_NAMES if n != "search_symbols"
        )
        agent.BASE_TOOLS = agent.get_resident_tool_schemas()    # 常驻声明重算
    # map 组 = 默认行为，无需改动


def apply_model(model: str | None) -> None:
    """--model= 覆盖评测模型（默认 config.MODEL）。bench 结果带 model 标记。"""
    global MODEL_IN_USE
    if model:
        import led_review.config as config
        profile = config.MODEL_PROFILE
        config.MODEL_PROFILES[profile]["model"] = model
        config.apply_profile(profile)
        print(f"[bench] 评测模型: {model}")
    MODEL_IN_USE = model or _default_model()


def _default_model() -> str:
    import led_review.config as config
    return config.MODEL


MODEL_IN_USE: str = _default_model()


def apply_edit_mode(mode: str) -> None:
    """编辑容错 A/B：lenient = 策略链（L1 精确→L2 行级宽容→L3 指引）；
    strict = 模拟旧精确时代（容错层失效，L2 永不命中直接进 L3 报错）。
    """
    global EDIT_MODE
    import led_review.tools.builtin as tools
    if mode == "strict":
        tools._lenient_replace = lambda *a, **k: None  # noqa: E731 容错链失效
    EDIT_MODE = mode


EDIT_MODE: str = "lenient"

# review 通道模式：single（基线）/ parallel（E11 实验组），--review-mode= 切换
REVIEW_MODE: str = "single"


# ---- 旋钮注入（--knob=name:value，A/B 档位扫描；与 --review-mode 同族机制）----

KNOB: str = "none"


def apply_knob(knob: str) -> None:
    """实验旋钮注入：不碰产品代码，bench 侧覆盖生效点。

    支持档位：
        compress:baseline  触发线 = CTX-28K（现状；kimi-code 1M 窗口下评测任务不触发）
        compress:60k       触发线 = 60K（收紧到任务规模量级，强制 L3/L2/L1 真实触发）
        compress:120k      触发线 = 120K（接近 packaging 级大任务的次上限）
        repomap:<int>      repo map 字符预算（agent.REPO_MAP_MAX_CHARS，
                           与 apply_group 的同类 patch 手法）
    """
    global KNOB
    import led_review.config as config
    import led_review.kernel.agent as agent
    if knob.startswith("compress:"):
        mode = knob.split(":", 1)[1]
        # 1M 窗口下按 CTX 比例的档位对 ≤120K 评测任务永不触发（E 旋钮实验观测），
        # 给绝对触发线才有杠杆：60K/120K 是「人为收紧窗口」的压力测试
        # 基线档必须与产品默认同公式（config 高水位余量 = max(5% 窗口, 16K)），
        # 否则 --knob=compress:baseline 会在配置改后悄悄偏离真实默认
        _margin = max(16_000, int(config.CONTEXT_TOKENS * 0.05))
        levels = {"baseline": config.CONTEXT_TOKENS - _margin, "60k": 60_000, "120k": 120_000}
        if mode not in levels:
            sys.exit(f"--knob=compress: 取值 baseline|60k|120k，收到: {mode}")
        high = levels[mode]
        overrides = {"truncate_high_tokens": high, "truncate_low_tokens": int(high * 0.6)}
        orig = agent.get_profile
        agent.get_profile = lambda name, o=orig, ov=overrides: {**o(name), **ov}
    elif knob.startswith("repomap:"):
        try:
            chars = int(knob.split(":", 1)[1])
        except ValueError:
            sys.exit(f"--knob=repomap: 需要整数，收到: {knob}")
        agent.REPO_MAP_MAX_CHARS = chars
    elif knob == "prompt:strong":
        # E15 变体：输出规格强制化（定位纪律 + 复现/置信行），见 review.build_prompt
        import led_review.review as rv
        _orig = rv.build_prompt
        rv.build_prompt = lambda diff, **kw: _orig(diff, strong=True)  # noqa: E731
    else:
        sys.exit(f"--knob= 支持 compress:<baseline|60k|120k> | repomap:<int> | prompt:strong，收到: {knob}")
    KNOB = knob


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=bench@bench", "-c", "user.name=bench", *args],
        cwd=root, check=True, capture_output=True,
    )


def _build_review_sandbox(task_dir: Path, meta: dict) -> Path:
    """review 任务沙箱。两种来源：
    - local：base/（干净代码）+ bug.patch（git apply 注入）
    - remote（META.remote）：浅克隆真实仓库 @ F（修复提交），revert F 回注真实 bug——
      真实代码 + 真实缺陷，作者不发明 bug 只标注（E10）
    """
    sandbox = Path(tempfile.mkdtemp(prefix=f"bench_{task_dir.name}_"))
    remote = meta.get("remote")
    if remote:
        _git(sandbox, "init", "-q")
        _git(sandbox, "remote", "add", "origin", remote["repo"])
        _git(sandbox, "fetch", "-q", "--depth", "2", "origin", remote["commit"])
        _git(sandbox, "checkout", "-q", "FETCH_HEAD")  # F（修复后）状态
        _git(sandbox, "revert", "--no-commit", remote["commit"])  # 回注 bug
        _git(sandbox, "commit", "-qm", "feature")
        return sandbox
    shutil.copytree(task_dir / "base", sandbox, dirs_exist_ok=True)
    _git(sandbox, "init", "-q")
    _git(sandbox, "add", ".")
    _git(sandbox, "commit", "-qm", "base")
    _git(sandbox, "apply", str(task_dir / "bug.patch"))
    _git(sandbox, "add", ".")
    _git(sandbox, "commit", "-qm", "feature")
    return sandbox


def run_review_task(task_dir: Path, meta: dict) -> dict:
    """review 任务：沙箱重建（local/remote）→ review → 对照 ground truth。

    与 chat 任务的差异：无 PROMPT 无对话轮——review 管道自带 prompt；判分不靠
    verify.py 而靠 META 的 bugs 清单（确定性：path 匹配 + 行号区间）。
    """
    sandbox = _build_review_sandbox(task_dir, meta)

    saved_root = tools.PROJECT_ROOT
    saved_confirm = tools.confirm
    saved_repo_root = repo_map._IGNORED_ROOT
    tools.PROJECT_ROOT = str(sandbox)
    tools.confirm = lambda *args, **kwargs: True
    repo_map._IGNORED_ROOT = str(sandbox)
    sessions = []
    try:
        raw, findings, sessions = review_pipeline.run_review(
            meta.get("spec", "HEAD"), root=str(sandbox), mode=REVIEW_MODE)
    except Exception as e:
        print(f"[bench] review 异常中断: {type(e).__name__}: {e}", file=sys.stderr)
        raw, findings = "", []
    finally:
        tools.PROJECT_ROOT = saved_root
        tools.confirm = saved_confirm
        repo_map._IGNORED_ROOT = saved_repo_root

    findings_dicts = [f.__dict__ for f in findings]
    scoring = score_review(findings_dicts, meta.get("bugs", []), neutral=meta.get("neutral", []))
    return {
        "task": task_dir.name,
        **scoring,
        "findings": findings_dicts,
        "raw": raw,
        "prompt_tokens": sum(s.total_prompt_tokens for s in sessions),
        "completion_tokens": sum(s.total_completion_tokens for s in sessions),
        "cached_tokens": sum(s.total_cached_tokens for s in sessions),
        "sandbox": str(sandbox),
        "review_mode": REVIEW_MODE,
        "knob": KNOB,
        "messages": [],  # review 会话是只读短流程，不落消息体（结果文件体积控制）
    }


def run_task(task_dir: Path, meta: dict) -> tuple[dict, TraceRecorder]:
    """单任务全流程：复制工作区 → 沙箱内跑 agent（带 trace）→ 分层判分 → 返回记录。"""
    sandbox = Path(tempfile.mkdtemp(prefix=f"bench_{task_dir.name}_"))
    shutil.copytree(task_dir / "workspace", sandbox, dirs_exist_ok=True)
    prompt = (task_dir / "PROMPT.md").read_text(encoding="utf-8").strip()

    saved_root = tools.PROJECT_ROOT
    saved_confirm = tools.confirm
    saved_repo_root = repo_map._IGNORED_ROOT
    tools.PROJECT_ROOT = str(sandbox)
    tools.confirm = lambda *args, **kwargs: True
    repo_map._IGNORED_ROOT = str(sandbox)  # 注入的 repo map 索引进沙箱（否则会指向真实仓库）
    recorder = TraceRecorder(task_dir.name)
    session = None
    try:
        session = ChatSession()
        ui.consume_quiet(recorder.wrap(session.chat(prompt)))
    except Exception as e:
        print(f"[bench] agent 异常中断: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        tools.PROJECT_ROOT = saved_root
        tools.confirm = saved_confirm
        repo_map._IGNORED_ROOT = saved_repo_root

    scoring = score_task(meta, task_dir, sandbox, session)
    record = {
        "task": task_dir.name,
        **scoring,
        "prompt_tokens": session.total_prompt_tokens if session else 0,
        "completion_tokens": session.total_completion_tokens if session else 0,
        "cached_tokens": session.total_cached_tokens if session else 0,
        "sandbox": str(sandbox),
        "messages": session.messages if session else [],
    }
    return record, recorder


def main() -> None:
    compare = "--compare" in sys.argv[1:]
    only = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    group = next((a.split("=", 1)[1] for a in sys.argv[1:]
                  if a.startswith("--group=")), "map")
    if group not in ("map", "nomap"):
        sys.exit(f"--group= 取值 map|nomap，收到: {group}")
    model = next((a.split("=", 1)[1] for a in sys.argv[1:]
                  if a.startswith("--model=")), None)
    profile = next((a.split("=", 1)[1] for a in sys.argv[1:]
                    if a.startswith("--profile=")), None)
    edit_mode = next((a.split("=", 1)[1] for a in sys.argv[1:]
                      if a.startswith("--edit-mode=")), "lenient")
    if edit_mode not in ("lenient", "strict"):
        sys.exit(f"--edit-mode= 取值 lenient|strict，收到: {edit_mode}")
    global REVIEW_MODE
    REVIEW_MODE = next((a.split("=", 1)[1] for a in sys.argv[1:]
                        if a.startswith("--review-mode=")), "single")
    if REVIEW_MODE not in ("single", "parallel"):
        sys.exit(f"--review-mode= 取值 single|parallel，收到: {REVIEW_MODE}")

    knob = next((a.split("=", 1)[1] for a in sys.argv[1:]
                 if a.startswith("--knob=")), "none")

    apply_group(group)
    apply_model(model)
    if profile:
        global MODEL_IN_USE
        import led_review.config as config
        try:
            config.apply_profile(profile)
        except KeyError:
            sys.exit(f"--profile= 档案不存在: {profile}")
        MODEL_IN_USE = f"{profile}" if not model else model
        print(f"[bench] 模型档案: {profile} → {config.MODEL}")
    apply_edit_mode(edit_mode)
    apply_knob(knob)

    manifest = load_manifest(TASKS_DIR / "manifest.json")
    tasks = discover_tasks(only)
    RESULTS_DIR.mkdir(exist_ok=True)

    summary_path = RESULTS_DIR / "summary.json"
    prev_summary = None
    if compare and summary_path.exists():
        prev_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    print(f"[bench] {len(tasks)} 个任务待跑")
    records = []
    for task_dir in tasks:
        meta = load_meta(task_dir)
        judge_type = meta.get("judge", "deterministic")
        is_review = meta.get("type") == "review"
        print(f"[bench] ▶ {task_dir.name} [{judge_type}]")
        started = time.time()
        if is_review:
            record = run_review_task(task_dir, meta)
            recorder = None
        else:
            record, recorder = run_task(task_dir, meta)
        record["elapsed_s"] = round(time.time() - started, 1)
        record["group"] = group  # A/B 对照：map / nomap
        record["model"] = MODEL_IN_USE  # 评测模型（跨模型样本可区分）
        record["edit_mode"] = EDIT_MODE  # 编辑容错：lenient / strict
        records.append(record)

        ts = time.strftime("%Y%m%d-%H%M%S")
        out = RESULTS_DIR / f"{task_dir.name}-{ts}-{group}.json"
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        if recorder is not None:
            trace_out = RESULTS_DIR / f"{task_dir.name}-{ts}-{group}.trace.jsonl"
            trace_out.write_text(recorder.to_jsonl(), encoding="utf-8")

        mark = "✅ PASS" if record["passed"] else "❌ FAIL"
        tokens = record["prompt_tokens"] + record["completion_tokens"]
        print(f"[bench] {mark} · score={record['score']:.2f} · {tokens:,} tokens · {record['elapsed_s']}s")

    summary = build_summary(records, manifest.get("version", 1))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = sum(r["passed"] for r in records)
    total_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in records)
    print(f"\n[bench] 通过率 {passed}/{len(records)} · 均分 {summary['aggregate']['avg_score']} · 总消耗 {total_tokens:,} tokens")

    # MANDATE 提醒（非强制）：真实 API 评测必须留档到 EXPERIMENTS.md（见 AGENTS.md）
    print("▸ 记得按 MANDATE 把本次评测（含负结果）追加到 bench/EXPERIMENTS.md 并提交")

    if compare:
        print("\n[bench] 回归对比（vs baseline）：")
        for line in compare_summaries(prev_summary, summary):
            print(line)
        threshold = manifest.get("regression_threshold", 0.1)
        regressions = []
        prev_tasks = (prev_summary or {}).get("tasks", {})
        for name, t in summary["tasks"].items():
            p = prev_tasks.get(name)
            if p and (p["score"] - t["score"]) > threshold:
                regressions.append(f"{name}（{p['score']}→{t['score']}）")
        if regressions:
            print(f"\n[bench] ⚠ 回归告警：{', '.join(regressions)}")
        else:
            print("\n[bench] 无回归")


if __name__ == "__main__":
    main()
