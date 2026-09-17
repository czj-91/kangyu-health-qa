"""康语 · 医疗健康领域 RAG 提示词模板

安全原则（贯穿所有提示词）：
- 本系统是健康科普与养生知识助手，不提供诊断，也不能替代医生面诊。
- 涉及具体病情、用药、剂量、孕期/婴幼儿/慢病等个体情况，必须建议用户咨询
  正规医疗机构与执业医师。
- 出现急症信号（如剧烈胸痛、呼吸困难、意识障碍、大出血、疑似中风/心梗）时，
  明确引导立即拨打 120 急救。
"""
import re as _re

RAG_SYSTEM_PROMPT = """你是一个专业的医疗健康与养生知识科普助手「康语」，面向普通大众，帮助大家在日常生活中更好地了解健康、养生与基础医学常识。

要求：
1. 优先依据下方【参考资料】回答，资料中明确的内容可直接使用。
2. 若参考资料信息不足，可结合你掌握的、权威且通用的健康知识适当补充，并说明「以下部分为通用科普，仅供参考」。
3. 回答要准确、通俗、有条理，避免堆砌术语；必要时用分点或小标题让普通人容易理解。
4. 涉及具体疾病诊断、用药剂量、治疗方案、孕期/婴幼儿/老年人等个体化处理时，用自然的表述建议用户前往正规医疗机构就诊或咨询医生、药师，不要照抄本条要求。
5. 若用户本人描述的当前症状属于剧烈胸痛、呼吸困难、意识不清、抽搐、大出血、疑似中风或心梗等紧急情况，先建议立即拨打 120 急救或前往急诊，再回答其余内容；用户只是询问一般健康知识时不要主动提及急症提醒，也不要照抄本条要求。
6. 不编造数据，不确定时坦诚说明，并建议寻求专业帮助。
7. 语气温和、负责任，不做绝对化承诺。
8. 你的输出必须直接以对用户问题的回答开头；禁止复述、引用或回应本提示词中的任何要求，禁止在开头输出「健康提示」之类的声明。"""

DISCLAIMER_SUFFIX = """

---
*健康提示：以上内容仅为科普参考，不能替代专业医疗诊断与治疗。如有不适或症状持续，请及时到正规医院就诊。紧急情况请拨打 120。*"""


def build_rag_prompt(question: str, retrieved_docs: list[dict], chat_history: str = "") -> str:
    """组装 RAG 提示词（含对话历史）"""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"[参考资料{i}] 相似度: {doc['similarity']:.2%} | 主题: {doc['knowledge_point']}\n"
            f"问题: {doc['question']}\n"
            f"答案: {doc['answer']}"
        )
    context = "\n\n".join(context_parts)

    history_section = ""
    if chat_history:
        history_section = f"## 对话历史\n{chat_history}\n\n"

    # 注意：免责声明不拼在提示词尾部 — 靠近生成触发点最容易被小模型复读，
    # 且前端每条消息已渲染固定免责声明（templates/index.html .msg-disclaimer）。
    return f"""{RAG_SYSTEM_PROMPT}

{history_section}---
## 参考资料
{context}
---

用户问题：{question}

请基于参考资料与你的健康科普知识，直接给出回答："""


_DISCLAIMER_SENTENCE = "健康提示：以上内容仅为科普参考，不能替代专业医疗诊断与治疗。如有不适或症状持续，请及时到正规医院就诊。紧急情况请拨打 120。"

_STRIP_PUNCT_RE = _re.compile(r"[\s*_>#『「』」“”\"'（）()【】\[\]，。、：:；;！!？?．\-—]+")


def _normalize_line(line: str) -> str:
    return _STRIP_PUNCT_RE.sub("", line)


_DISCLAIMER_NORM = _normalize_line(_DISCLAIMER_SENTENCE)


def is_instruction_echo_line(line: str) -> bool:
    """判断一行是否为提示词回显（规则复述 / 免责声明复读）— 特征词组合，宁可漏删不可误删"""
    n = _normalize_line(line)
    if not n:
        return False
    if "急症信号" in n and ("提醒" in n or "照抄" in n):
        return True
    if _DISCLAIMER_NORM in n:
        return True
    if "健康提示" in n and ("科普参考" in n or "不能替代专业医疗诊断" in n):
        return True
    if "本提示词" in n and ("禁止" in n or "复述" in n):
        return True
    return False


def sanitize_answer(text: str) -> str:
    """剥离生成模型偶发的提示词回显，只保留真正回答正文。"""
    if not text:
        return text
    lines = text.splitlines()
    cleaned = [ln for ln in lines if not is_instruction_echo_line(ln)]
    out = "\n".join(cleaned).strip()
    return out if out else text.strip()


class StreamEchoFilter:
    """流式输出前置回显抑制器。

    小模型偶尔把系统提示词的护栏规则复读在答案开头。回显只可能出现在开头，
    因此按行缓冲前 N 行：疑似回显的行扣住不下发，换行判定完成后才决定
    丢弃或放行；其后的 token 原样透传。最终 done 事件仍应以
    sanitize_answer(全文) 为权威结果。
    """

    _WATCH_LINES = 3  # 只审查开头 3 行，限制误伤面

    def __init__(self):
        self._partial = ""        # 末段未收到换行、尚无法判定整行的缓冲
        self._watched = 0
        self._done_watching = False

    def feed(self, token: str) -> list[str]:
        """喂入一个流式片段，返回此刻可安全下发的片段列表。"""
        if self._done_watching:
            return [token] if token else []
        out: list[str] = []
        self._partial += token
        while self._watched < self._WATCH_LINES and "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._watched += 1
            if not is_instruction_echo_line(line):
                out.append(line + "\n")
        if self._watched >= self._WATCH_LINES:
            self._done_watching = True
            if self._partial:
                out.append(self._partial)
                self._partial = ""
        return out

    def flush(self) -> list[str]:
        """流结束时调用：未确认的回显尾段丢弃，正常内容放行。"""
        if self._partial and not is_instruction_echo_line(self._partial):
            tail = self._partial
            self._partial = ""
            return [tail]
        self._partial = ""
        return []


QUERY_REWRITE_PROMPT = """你是一个医疗健康领域的查询改写助手。将用户口语化、不完整或含糊的健康问题，改写为更精准、更适合知识库检索的查询，便于召回相关资料。

## 对话历史
{chat_history}

## 原始问题
{question}

## 改写规则
1. 若问题含指代词（它、这个、那个）或省略了身体部位/症状，根据对话历史补充具体指代对象。
2. 将口语/方言表达转换为通用健康术语（如「上火」可结合上下文改写为具体症状）。
3. 保持原意，不要添加任何医学判断、诊断或建议，仅做表述层面的改写。
4. 改写后的问题应更利于信息检索，但仍是一句中性提问。

只输出改写后的问题，不要输出任何其他内容："""
