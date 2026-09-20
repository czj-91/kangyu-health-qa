"""输入守卫单元测试 — 非健康/无意义输入识别 & 健康提问零误伤"""
import pytest
from input_guard import classify_input, has_health_signal, OFF_TOPIC_KINDS

HEALTH_SAMPLES = [
    "感冒了要多喝水吗",
    "经常熬夜有什么危害",
    "宝宝发烧38度怎么处理",
    "高血压吃什么药好",
    "嗓子疼吃什么好",          # 含"吃", 不能被饮食类误伤
    "今晚睡不着怎么办",
    "减肥可以吃香蕉吗",        # 含饮食词但属健康
    "体检报告上写窦性心律不齐严重吗",
    "我最近总是头晕是不是贫血",
    "python报错会影响健康吗",  # 含健康信号"健康", 放行(宁松勿严)
    "asdkjhasd感冒了怎么办",   # 混入乱码但有健康词 → 放行
    "帮我写一段关于健康的文案",  # 含"健康"信号 → 放行(保守策略)
]

OFF_TOPIC_SAMPLES = [
    "python快速排序怎么写",
    "java和golang哪个好用",
    "原神最新版本有什么角色",
    "王者荣耀怎么上王者",
    "帮我写一封请假条",
    "生成一篇关于秋天的作文",
    "stock 股票今天买什么",
    "写一个正则表达式匹配邮箱",
    "微积分期末重点有哪些",
]

GIBBERISH_SAMPLES = [
    "aaaa",
    "1111",
    "。。。",
    "as",   # 极短
    "啊",
    "??!!",
    "asdkjhasd",  # 长键盘乱码(实测漏网样本, 由 5 连辅音规则兜住)
]


class TestHealthNotBlocked:
    @pytest.mark.parametrize("q", HEALTH_SAMPLES)
    def test_health_questions_pass(self, q):
        assert classify_input(q) not in OFF_TOPIC_KINDS

    def test_health_signal_detector(self):
        assert has_health_signal("嗓子疼吃什么")
        assert not has_health_signal("原神抽卡")


class TestOffTopic:
    @pytest.mark.parametrize("q", OFF_TOPIC_SAMPLES)
    def test_off_domain_detected(self, q):
        assert classify_input(q) == "off_topic"


class TestGibberish:
    @pytest.mark.parametrize("q", GIBBERISH_SAMPLES)
    def test_gibberish_detected(self, q):
        assert classify_input(q) == "gibberish"


class TestUnclearPassThrough:
    @pytest.mark.parametrize("q", [
        "你是谁",            # 元问题, 未命中任何规则 → unclear → 放行给检索/LLM
        "今天天气怎么样",
        "那要注意什么",       # 典型追问(短句无强领域词) → unclear 放行
    ])
    def test_unclear_not_blocked(self, q):
        assert classify_input(q) == "unclear"

    def test_empty_is_gibberish(self):
        assert classify_input("") == "gibberish"
        assert classify_input("   ") == "gibberish"
