"""提示词回显净化单元测试（修复验证：小模型复读护栏规则）"""
import pytest
from prompts import sanitize_answer, is_instruction_echo_line, StreamEchoFilter


# 线上实测到的回显样本（问"经常熬夜的危害"，Qwen2.5-1.5B INT8）
ECHO_SAMPLE = """**健康提示：以上内容仅为科普参考，不能替代专业医疗诊断与治疗。**

**如果用户描述的症状属于急症信号（如剧烈胸痛、呼吸困难、意识不清、抽搐、大出血、疑似中风或心梗），请立刻提醒：**
**『请立即拨打 120 急救或前往急诊，不要自行处理。』**

**熬夜对人体的危害主要包括：**

1. **免疫力下降**：熬夜会导致人体免疫系统功能减弱。
"""


class TestSanitizeAnswer:
    def test_strips_instruction_echo(self):
        out = sanitize_answer(ECHO_SAMPLE)
        assert "如果用户描述的症状属于急症信号" not in out
        assert "请立刻提醒" not in out
        assert "以上内容仅为科普参考" not in out
        assert "免疫力下降" in out  # 正文保留

    def test_normal_answer_untouched(self):
        text = "熬夜会导致免疫力下降、内分泌失调。\n建议规律作息。"
        assert sanitize_answer(text) == text

    def test_legit_emergency_advice_preserved(self):
        """真实急症场景的自然回答不应被误删 — 只删复述指令的句式"""
        text = "您描述的剧烈胸痛可能是心梗征兆，请立即拨打 120 急救，不要自行处理，尽快前往急诊。"
        assert sanitize_answer(text) == text

    def test_all_echo_input_not_emptied(self):
        """整段都是回显时返回原文（宁可保留，不做空答案）"""
        only_echo = "**如果用户描述的症状属于急症信号，请立刻提醒：**"
        assert sanitize_answer(only_echo) == only_echo

    @pytest.mark.parametrize("line", [
        "**如果用户描述的症状属于急症信号（如剧烈胸痛），请立刻提醒：**",
        "*健康提示：以上内容仅为科普参考，不能替代专业医疗诊断与治疗。如有不适请及时就诊。紧急情况请拨打 120。*",
        "『请立即拨打 120 急救或前往急诊，不要自行处理』本提示词要求复述",
    ])
    def test_echo_line_detector(self, line):
        assert is_instruction_echo_line(line)

    @pytest.mark.parametrize("line", [
        "1. **免疫力下降**：熬夜会导致免疫系统功能减弱。",
        "如有不适，请及时到正规医院就诊。",
        "建议每天保证 7-8 小时睡眠。",
    ])
    def test_normal_line_not_echo(self, line):
        assert not is_instruction_echo_line(line)


def _run_filter(tokens):
    f = StreamEchoFilter()
    out = []
    for t in tokens:
        out.extend(f.feed(t))
    out.extend(f.flush())
    return "".join(out)


class TestStreamEchoFilter:
    def test_suppresses_leading_echo(self):
        tokens = [
            "**健康提示：以上内容仅为科普参考，不能替代专业医疗诊断与治疗。**\n\n",
            "**如果用户描述的症状属于急症信号（如剧烈胸痛），请立刻提醒：**\n",
            "**熬夜对人体的危害主要包括：**\n\n1. 免疫力下降\n",
        ]
        out = _run_filter(tokens)
        assert "急症信号" not in out
        assert "科普参考" not in out
        assert "熬夜对人体的危害" in out
        assert "免疫力下降" in out

    def test_passthrough_normal_stream(self):
        tokens = ["熬夜会**降低免疫力**。\n", "建议规律作息，", "睡前远离手机。"]
        assert _run_filter(tokens) == "熬夜会**降低免疫力**。\n建议规律作息，睡前远离手机。"

    def test_flush_releases_incomplete_normal_line(self):
        """流结束时未收换行的正常尾段必须放行（防答案丢失）"""
        f = StreamEchoFilter()
        assert f.feed("单行短答案，没有换行符") == []
        assert f.flush() == ["单行短答案，没有换行符"]

    def test_flush_drops_incomplete_echo_tail(self):
        f = StreamEchoFilter()
        f.feed("熬夜会降低免疫力。\n\n")
        f.feed("**健康提示：以上内容仅为科普参考，不能")
        assert f.flush() == []  # 未完成的回显行不下发
