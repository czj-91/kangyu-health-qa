"""提示词模板单元测试"""
import pytest
from prompts import build_rag_prompt, RAG_SYSTEM_PROMPT, DISCLAIMER_SUFFIX, QUERY_REWRITE_PROMPT


class TestRAGPrompt:
    def test_build_rag_prompt_contains_question(self):
        docs = [
            {
                "question": "头痛怎么办？",
                "answer": "多休息，必要时就医。",
                "knowledge_point": "内科常见病",
                "similarity": 0.95,
            }
        ]
        prompt = build_rag_prompt("头痛怎么办？", docs)
        assert "头痛怎么办？" in prompt

    def test_build_rag_prompt_contains_system_prompt(self):
        docs = [
            {
                "question": "测试",
                "answer": "测试答案",
                "knowledge_point": "综合健康",
                "similarity": 0.90,
            }
        ]
        prompt = build_rag_prompt("测试", docs)
        assert "医疗健康" in prompt or "科普" in prompt

    def test_build_rag_prompt_includes_disclaimer(self):
        """修复验证：DISCLAIMER_SUFFIX 现在实际附加到 prompt 中"""
        docs = [
            {
                "question": "感冒怎么办？",
                "answer": "多喝水多休息。",
                "knowledge_point": "内科常见病",
                "similarity": 0.88,
            }
        ]
        prompt = build_rag_prompt("感冒怎么办？", docs)
        assert "120" in prompt, "免责声明中应包含 120 急救提示"

    def test_build_rag_prompt_with_multiple_docs(self):
        docs = [
            {"question": "Q1", "answer": "A1", "knowledge_point": "内科", "similarity": 0.9},
            {"question": "Q2", "answer": "A2", "knowledge_point": "外科", "similarity": 0.8},
            {"question": "Q3", "answer": "A3", "knowledge_point": "儿科", "similarity": 0.7},
        ]
        prompt = build_rag_prompt("测试问题", docs)
        assert "参考资料1" in prompt
        assert "参考资料2" in prompt
        assert "参考资料3" in prompt

    def test_build_rag_prompt_with_chat_history(self):
        docs = [
            {"question": "Q1", "answer": "A1", "knowledge_point": "内科", "similarity": 0.9},
        ]
        history = "用户: 我头痛\n助教: 建议你多休息\n"
        prompt = build_rag_prompt("那需要吃药吗？", docs, chat_history=history)
        assert "对话历史" in prompt
        assert "我头痛" in prompt

    def test_build_rag_prompt_with_similarity_format(self):
        docs = [
            {"question": "Q1", "answer": "A1", "knowledge_point": "内科", "similarity": 0.7532},
        ]
        prompt = build_rag_prompt("test", docs)
        assert "75.32%" in prompt  # 相似度格式化


class TestSystemPrompt:
    def test_system_prompt_contains_safety_rules(self):
        assert "120" in RAG_SYSTEM_PROMPT or "急救" in RAG_SYSTEM_PROMPT
        assert "诊断" in RAG_SYSTEM_PROMPT
        assert "科普" in RAG_SYSTEM_PROMPT

    def test_disclaimer_contains_warning(self):
        assert "120" in DISCLAIMER_SUFFIX
        assert "科普" in DISCLAIMER_SUFFIX
        assert "诊断" in DISCLAIMER_SUFFIX or "治疗" in DISCLAIMER_SUFFIX


class TestQueryRewrite:
    def test_rewrite_prompt_format(self):
        result = QUERY_REWRITE_PROMPT.format(
            chat_history="用户: 头痛\n助教: 建议休息\n",
            question="那该吃什么药？"
        )
        assert "头痛" in result
        assert "那该吃什么药？" in result
        assert "对话历史" in result
        assert "改写规则" in result

    def test_rewrite_prompt_contains_rules(self):
        result = QUERY_REWRITE_PROMPT.format(chat_history="无", question="test")
        assert "指代词" in result  # 规则：处理指代词
        assert "口语" in result or "术语" in result  # 规则：口语规范化
