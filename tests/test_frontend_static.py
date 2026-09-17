"""前端静态回归测试 — 锁定本轮安全/健壮性修复, 防止退化

不依赖浏览器, 直接对 templates/index.html 做结构断言:
XSS 防线(md 渲染管道/DOMPurify)、事件委托、无死代码、关键无障碍属性。
"""
import re
from pathlib import Path

import pytest

HTML = Path(__file__).parent.parent / "templates" / "index.html"


@pytest.fixture(scope="module")
def page() -> str:
    return HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script(page: str) -> str:
    m = re.search(r"<script>\n(.*?)</script>", page, re.S)
    assert m, "未找到内联主脚本"
    return m.group(1)


class TestXssDefense:
    def test_dompurify_loaded(self, page):
        assert "purify.min.js" in page, "必须引入 DOMPurify"

    def test_marked_output_goes_through_sanitizer(self, script):
        # marked.parse( 只允许出现在 DOMPurify.sanitize(marked.parse( 包装内
        calls = re.findall(r"marked\.parse\(", script)
        wrapped = re.findall(r"DOMPurify\.sanitize\(marked\.parse\(", script)
        assert calls, "应仍在使用 marked 渲染 Markdown"
        assert len(calls) == len(wrapped), "存在未经 md() 消毒的 marked.parse 调用"

    def test_no_dynamic_inline_onclick(self, script):
        # 动态内容(问题原文/id)不得再拼进 onclick 字符串
        assert 'onclick="askSample(' not in script
        assert "onclick=\"submitFeedback(this" not in script

    def test_sample_and_source_escaped(self, page):
        assert "escHtml(s.knowledge_point)" in page


class TestEventDelegation:
    def test_chip_data_attributes(self, script):
        for attr in ("data-qid", "data-idx", "data-sid", "data-del", "data-fb", "data-retry"):
            assert attr in script, f"缺少 {attr} 委托数据属性"

    def test_delegation_listeners(self, script):
        assert "addEventListener('click'" in script
        assert "closest('.source-chip')" in script


class TestRobustness:
    def test_answer_event_updates_state(self, script):
        # 兜底话术(answer 事件)必须写入 currentAnswerText, 否则被 finalize 清空
        case_block = script[script.index("case 'answer':"):script.index("case 'answer':") + 300]
        assert "currentAnswerText" in case_block

    def test_token_render_throttled(self, script):
        assert "requestAnimationFrame" in script
        assert "renderQueued" in script

    def test_empty_stop_fallback(self, script):
        assert "已停止生成" in script

    def test_session_restore_on_load(self, script):
        assert "qa-current-sid" in script
        assert "restoreConversation" in script

    def test_no_dead_code(self, script):
        for gone in ("function addRetrievalTag", "function addFeedbackButtons", "function escAttr", "function retryLast"):
            assert gone not in script, f"死代码未清理: {gone}"

    def test_no_orphan_css(self, page):
        # 悬空属性: 属性声明直接跟在上一条规则的 `}` 之后(而非任何选择器内)
        css = page[page.index("<style>"):page.index("</style>")]
        assert re.search(r"\}\s*\n\s*flex-shrink:\s*0;\s*\n\s*\}", css) is None


class TestAccessibility:
    def test_mode_pill_is_button(self, page):
        m = re.search(r"<(button|div)[^>]*id=\"modePill\"", page)
        assert m and m.group(1) == "button", "模式切换应为键盘可达的 button"
        assert 'aria-pressed' in page

    def test_textarea_labelled(self, page):
        m = re.search(r"<textarea[^>]*>", page)
        assert m and "aria-label" in m.group(0)

    def test_focus_styles(self, page):
        assert ":focus-visible" in page
