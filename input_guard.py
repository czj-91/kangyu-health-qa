"""输入相关性守卫 — 识别明显与健康/医疗无关或无意义的输入。

设计原则(基于实测):
- MacBERT 余弦相似度与 BGE rerank 分在本地小模型上**无法可靠区分** in/off topic
  (实测"感冒多喝水"in=0.025 vs "今晚吃什么"off=0.258 交叠), 故不用相似度阈值,
  避免误伤正常健康提问。
- 改用**规则**: 只对"明确属于其它领域"或"无意义字符"的输入返回引导语;
  任何含健康相关信号的输入一律放行(宁可放过, 不可错杀)。

纯函数、无第三方依赖, 便于单测。
"""
import re

# 明显非健康领域的强信号关键词(命中且无健康信号时判为 off-topic)
# 注: 不用 \b —— Python re 的 \w 含 CJK, "java和golang" 场景词边界失效, 改用 ASCII 环视
OFF_DOMAIN_PATTERNS = [
    # 编程 / 技术
    r"(?<![a-zA-Z])(python|java|javascript|golang|rust|sql|html|css|docker|kubernetes)(?![a-zA-Z])",
    r"(代码|编程|函数|报错|异常|debug|编译|算法|排序|爬虫|正则|接口|bug)",
    # 游戏 / 娱乐
    r"(原神|王者荣耀|英雄联盟|游戏|攻略|通关|抽卡|装备|副本)",
    r"(歌词|明星|八卦|综艺|追星)",
    # 写作 / 办公 / 学业
    r"(作文|论文|读后感|周报|日报|请假条|工作邮件)",
    r"(写.{0,4}(作文|论文|邮件|文案|代码|小说)|翻译|润色)",
    # 金融 / 购物 / 出行
    r"(股票|基金|涨停|开户|转账|支付|贷款|快递|物流|退货)",
    # 数学 / 理工科
    r"(微积分|方程|几何|物理题|化学式|量子|相对论)",
]

# 健康/医疗相关信号(命中任意一条则直接放行, 优先级最高)
HEALTH_SIGNAL_PATTERNS = [
    r"(感冒|发烧|发热|咳嗽|头疼|头痛|腹痛|肚子疼|胃疼|胃痛|腹泻|拉肚子|便秘|恶心|呕吐)",
    r"(嗓子|咽喉|喉咙|哑)",
    r"(血压|血糖|血脂|心率|脉搏|体温|过敏|发炎|感染|肿瘤|癌|结节|结石)",
    r"(症状|病|疾|疼痛|不适|难受|药|处方|就医|就诊|医院|诊所|医生|大夫|护士)",
    r"(体检|化验|检查|诊断|治疗|康复|手术|输液|疫苗)",
    r"(睡眠|失眠|熬夜|睡不着|多梦|疲劳|乏力|头晕|眩晕)",
    r"(减肥|瘦身|增肌|营养|膳食|饮食|忌口|维生素|蛋白|补钙|上火|养生|保健)",
    r"(怀孕|孕期|备孕|月经|生理期|宫颈|产检|哺乳|婴儿|宝宝|儿童|幼儿)",
    r"(心理|情绪|焦虑|抑郁|压力|心情)",
    r"(皮肤|长痘|痤疮|湿疹|脱发|美白|防晒|晒伤|烫伤|割伤|出血)",
    r"(近视|散光|牙疼|牙齿|蛀牙|口腔溃疡|鼻炎|咽炎|扁桃体|耳鸣)",
    r"(身体|器官|心脏|肝脏|肾|肺|肠胃|肠道|骨骼|关节|脊椎|颈椎|肌肉)",
    r"(急救|中毒|中暑|冻伤|抽搐|昏迷|窒息|中风|心梗|脑梗)",
    r"(中医|中药|针灸|拔罐|刮痧|艾灸|穴位|经络)",
    r"(运动|锻炼|跑步|健身|拉伸|劳损|扭伤)",
    r"(戒烟|戒酒|宿醉|解酒|喝酒|饮酒|熬夜)",
    r"(健康|医疗|医学科普|身体不适)",
]

# 无意义输入: 单字符重复 / 纯符号数字 / 极短乱码 / 长辅音连串
# 辅音连串规则: 连续 5+ 辅音字母在英文单词与汉语拼音中均不存在
# (strengths 最长连串 4; 拼音无辅音簇), 但能抓住 asdkjhasd 这类键盘乱码
GIBBERISH_PATTERNS = [
    r"^(.)\1{1,}$",            # aaaa / 1111 / 哈哈哈哈(交给健康词兜底)
    r"^[\W\d_]+$",             # 纯符号/数字/下划线, 无任何字母或汉字
    r"^[a-z]{1,4}$",           # 极短无空格英文串 asdf
    r"[bcdfghjklmnpqrstvwxz]{5,}",  # 键盘乱码 sdkjh / qwert
]

_HEALTH_COMPILED = [re.compile(p, re.I) for p in HEALTH_SIGNAL_PATTERNS]
_OFF_COMPILED = [re.compile(p, re.I) for p in OFF_DOMAIN_PATTERNS]
_GIBBERISH_COMPILED = [re.compile(p, re.I) for p in GIBBERISH_PATTERNS]

# 需要返回引导语的分类结果
OFF_TOPIC_KINDS = ("off_topic", "gibberish")


def has_health_signal(text: str) -> bool:
    """含任一健康/医疗相关信号 → True (应放行)"""
    return any(rx.search(text) for rx in _HEALTH_COMPILED)


def classify_input(text: str) -> str:
    """返回 'health' | 'off_topic' | 'gibberish' | 'unclear'.

    unclear = 既无健康信号也非明确他领域(如"你是谁""今天天气"), 交由上层
    按检索结果自行决定(通常走检索, 结果不相关时兜底话术自然生效)。
    """
    s = (text or "").strip()
    if len(s) < 2:
        return "gibberish"
    if has_health_signal(s):
        return "health"
    if any(rx.search(s) for rx in _GIBBERISH_COMPILED):
        return "gibberish"
    if any(rx.search(s) for rx in _OFF_COMPILED):
        return "off_topic"
    return "unclear"
