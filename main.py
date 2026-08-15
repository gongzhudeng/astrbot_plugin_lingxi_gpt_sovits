import base64
import html
import io
import random
import re
import uuid
import wave
from dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Plain, Record
from astrbot.core.platform import AstrMessageEvent

from .core.client import GSVApiClient, GSVRequestResult
from .core.config import PluginConfig
from .core.emotion import EmotionJudger
from .core.entry import EntryManager
from .core.local_data import LocalDataManager
from .core.profile_manager import ProfileManager
from .core.service import GPTSoVITSService

_PUNCT_RE = re.compile(r"([^，。！？,!?.…]+[，。！？,!?.…]?)")
_PSEUDO_TTS_BLOCK_RE = re.compile(
    r"(?P<block>(?:"
    r"<gsv_tts\b[^>]*>\s*(?:"
    r"<invoke\b(?=[^>]*\bname\s*=\s*(?:\"gsv_tts\"|'gsv_tts'))[^>]*>"
    r"[\s\S]*?</invoke>\s*"
    r")?</gsv_tts>"
    r"|"
    r"<invoke\b(?=[^>]*\bname\s*=\s*(?:\"gsv_tts\"|'gsv_tts'))[^>]*>"
    r"[\s\S]*?</invoke>"
    r"))\s*$",
    re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"<parameter\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</parameter>",
    re.IGNORECASE,
)
_MESSAGE_ATTR_RE = re.compile(
    r"\bmessage\s*=\s*(?P<quote>['\"])(?P<value>[\s\S]*?)(?P=quote)",
    re.IGNORECASE,
)
_NAME_MESSAGE_RE = re.compile(
    r"\bname\s*=\s*(['\"])message\1",
    re.IGNORECASE,
)
_DIRECT_DELIVERY_TEXT_EXTRA = "spark_direct_delivery_history_text"
_DIRECT_DELIVERY_KIND_EXTRA = "spark_direct_delivery_kind"
_INVOKE_BODY_RE = re.compile(
    r"<invoke\b[^>]*>(?P<body>[\s\S]*?)</invoke>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_PSEUDO_TTS_TEXT_KEY = "_gsv_pseudo_tts_text"
_PSEUDO_TTS_RECOVERY_KEY = "_gsv_pseudo_tts_recovery"


@dataclass(frozen=True)
class PseudoTTSCall:
    display_text: str
    tts_text: str
    matched: bool


def _plain_xml_text(value: str) -> str:
    return html.unescape(_TAG_RE.sub("", value)).strip()


def _extract_pseudo_tts_message(block: str) -> str:
    for match in _PARAMETER_RE.finditer(block):
        if _NAME_MESSAGE_RE.search(match.group("attrs")):
            return _plain_xml_text(match.group("body"))

    invoke_match = _INVOKE_BODY_RE.search(block)
    if not invoke_match:
        return ""

    invoke_body = invoke_match.group("body")
    attr_match = _MESSAGE_ATTR_RE.search(invoke_body)
    if attr_match:
        return html.unescape(attr_match.group("value")).strip()
    return _plain_xml_text(invoke_body)


def _parse_pseudo_tts_call(text: str) -> PseudoTTSCall:
    match = _PSEUDO_TTS_BLOCK_RE.search(text)
    if not match:
        return PseudoTTSCall(display_text=text, tts_text="", matched=False)

    display_text = text[: match.start()].rstrip()
    message = _extract_pseudo_tts_message(match.group("block"))
    return PseudoTTSCall(
        display_text=display_text,
        tts_text=message or display_text,
        matched=True,
    )


def _split_by_punctuation(text: str) -> list[str]:
    return [s.strip() for s in _PUNCT_RE.findall(text) if s.strip()]


def _split_by_sentence(text: str, group_size: int) -> list[str]:
    sentences = [s.strip() for s in re.split(r"[。.？?！!]", text) if s.strip()]
    return [
        "。".join(sentences[i : i + group_size])
        for i in range(0, len(sentences), group_size)
    ]


def _merge_wav_bytes(chunks: list[bytes]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out_wav:
        params_set = False
        for chunk in chunks:
            with wave.open(io.BytesIO(chunk), "rb") as in_wav:
                if not params_set:
                    out_wav.setparams(in_wav.getparams())
                    params_set = True
                out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
    return buf.getvalue()


def _resolve_busy_schedule_media_recorder(event, context):
    """Resolve the recorder from the tool event, then the shared plugin context."""
    callback = getattr(event, "_busy_schedule_record_media_success", None)
    if callable(callback):
        return callback
    callback = getattr(context, "_busy_schedule_record_media_success", None)
    return callback if callable(callback) else None


class GPTSoVITSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = PluginConfig(config, context)
        self.profile_mgr = ProfileManager(self.cfg.data_dir)
        self.local_data = LocalDataManager(self.cfg)
        self.entry_mgr = EntryManager(self.cfg)
        self.client = GSVApiClient(self.cfg)
        self.judger = EmotionJudger(self.cfg)
        self.service = GPTSoVITSService(self.cfg, self.client, self.local_data)

    async def initialize(self):
        if self.cfg.enabled:
            await self.service.load_model()

    async def terminate(self):
        await self.client.close()

    def _apply_profile(self, name: str) -> str | None:
        """加载角色并覆盖当前的模型配置、默认参数和情绪条目。失败返回错误信息，成功返回 None。"""
        profile = self.profile_mgr.get_profile(name)
        if not profile:
            return f"角色 '{name}' 不存在"

        model_data = profile.get("model", {})
        self.cfg._data["model"]["gpt_path"] = model_data.get("gpt_path", "")
        self.cfg._data["model"]["sovits_path"] = model_data.get("sovits_path", "")

        params_data = profile.get("default_params", {})
        for k, v in params_data.items():
            self.cfg._data["default_params"][k] = v

        self.cfg._data["entry_storage"] = profile.get("entry_storage", [])

        self.cfg.model.gpt_path = self.cfg.normalize_path(self.cfg.model.gpt_path)
        self.cfg.model.sovits_path = self.cfg.normalize_path(self.cfg.model.sovits_path)
        self.cfg.default_params["ref_audio_path"] = self.cfg.normalize_path(
            self.cfg.default_params["ref_audio_path"]
        )

        self.cfg.active_profile = name
        self.cfg.save_config()

        self.entry_mgr = EntryManager(self.cfg)
        self.service.update_config_sync(self.cfg)

        return None

    @staticmethod
    def _to_record(res: GSVRequestResult) -> Record:
        if res.file_path:
            try:
                return Record.fromFileSystem(res.file_path)
            except Exception:
                logger.warning(f"无法读取文件：{res.file_path}, 已忽略")
                pass

        if not res.data:
            raise ValueError("无法获取结果数据")

        b64 = base64.urlsafe_b64encode(res.data).decode()
        return Record.fromBase64(b64)

    async def _get_emotion_params(
        self, event: AstrMessageEvent, text: str
    ) -> dict | None:
        entry = None
        keyword_overrides_llm = bool(self.cfg.judge.keyword_overrides_llm)

        if keyword_overrides_llm:
            entry = self.entry_mgr.match_entry(text)
            if entry:
                logger.debug(f"关键词优先命中情绪: {entry.name}")

        if entry is None and self.cfg.judge.enabled_llm:
            labels = self.entry_mgr.get_names()
            emotion = await self.judger.judge_emotion(event, text=text, labels=labels)
            if emotion:
                entry = self.entry_mgr.get_entry(emotion)
                if entry:
                    logger.debug(f"LLM 命中情绪: {entry.name}")

        if entry is None and not keyword_overrides_llm:
            entry = self.entry_mgr.match_entry(text)
            if entry:
                logger.debug(f"关键词兜底命中情绪: {entry.name}")

        return entry.to_params() if entry else None

    async def _infer_with_emotion(
        self, event: AstrMessageEvent, text: str
    ) -> GSVRequestResult | None:
        """按情绪计算范围推理，返回结果；分段模式下合并 WAV，失败降级整段。"""
        scope = getattr(self.cfg.judge, "emotion_scope", "whole")
        media_type = self.cfg.default_params.get("media_type", "wav")

        if scope != "whole" and media_type == "wav":
            if scope == "punctuation":
                segments = _split_by_punctuation(text) or [text]
            else:
                group_size = getattr(self.cfg.judge, "sentence_group_size", 1)
                segments = _split_by_sentence(text, group_size) or [text]

            chunks: list[bytes] = []
            for seg in segments:
                event.set_extra("emotion", None)  # clear cache per segment
                params = await self._get_emotion_params(event, seg)
                res = await self.service.inference_raw(seg, extra_params=params)
                if not bool(res) or not res.data:
                    return res
                chunks.append(res.data)

            try:
                merged = _merge_wav_bytes(chunks)
                cache_params = self.service.prepare_params(text)
                cache_path = self.local_data.save_audio(merged, cache_params)
                return GSVRequestResult(
                    ok=True,
                    data=merged,
                    text=text,
                    file_path=str(cache_path) if cache_path else "",
                )
            except Exception as e:
                logger.warning(f"WAV 合并失败，降级整段推理: {e}")

        params = await self._get_emotion_params(event, text)
        return await self.service.inference(text, extra_params=params)

    @filter.on_llm_response(priority=100)
    async def recover_pseudo_tts_call(self, event: AstrMessageEvent, resp: LLMResponse):
        if not self.cfg.enabled or resp.role != "assistant" or resp.tools_call_name:
            return

        parsed = _parse_pseudo_tts_call(resp.completion_text or "")
        if not parsed.matched:
            return

        resp.completion_text = parsed.display_text
        event.set_extra(_PSEUDO_TTS_TEXT_KEY, parsed.tts_text)
        event.set_extra(_PSEUDO_TTS_RECOVERY_KEY, True)
        logger.info("检测到文本形式的 gsv_tts 调用，已清洗并恢复语音输出")

    @filter.on_decorating_result(priority=14)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """消息入口"""
        if not self.cfg.enabled:
            return
        cfg = self.cfg.auto
        recovery = bool(event.get_extra(_PSEUDO_TTS_RECOVERY_KEY, False))

        result = event.get_result()
        if not result:
            return
        chain = result.chain
        if not chain and not recovery:
            return
        if cfg.only_llm_result and not result.is_llm_result():
            return
        if not recovery and random.random() > cfg.tts_prob:
            return

        plain_texts = []
        for seg in chain:
            if isinstance(seg, Plain):
                plain_texts.append(seg.text)

        if len(plain_texts) != len(chain):
            return

        display_text = "\n".join(plain_texts).strip()
        tts_text = (
            str(event.get_extra(_PSEUDO_TTS_TEXT_KEY, "")).strip()
            if recovery
            else display_text
        )
        if not tts_text or len(tts_text) > cfg.max_msg_len:
            return

        res = await self._infer_with_emotion(event, tts_text)
        if not bool(res):
            return

        chain.clear()
        if recovery and display_text and display_text != tts_text:
            chain.append(Plain(display_text))
        chain.append(self._to_record(res))

    @filter.command("说", alias={"gsv", "GSV"})
    async def on_command(self, event: AstrMessageEvent):
        """说 [情绪] <内容>，直接调用GSV合成语音；可选情绪名强制指定情绪"""
        if not self.cfg.enabled:
            return

        body = event.message_str.partition(" ")[2].strip()
        if not body:
            yield event.plain_result("请输入要合成的文本")
            return

        first, _, rest = body.partition(" ")
        forced_entry = self.entry_mgr.get_entry(first) if first else None

        if forced_entry and rest.strip():
            text = rest.strip()
            res = await self.service.inference(
                text, extra_params=forced_entry.to_params()
            )
        else:
            text = body
            res = await self._infer_with_emotion(event, text)

        if not bool(res):
            yield event.plain_result(res.error)
            return

        yield event.chain_result([self._to_record(res)])

    @filter.command("语音概率", alias={"tts_probability"})
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def set_tts_probability(self, event: AstrMessageEvent):
        """查看或设置主动转语音发送的概率。"""
        if not self.cfg.enabled:
            return

        arg = event.message_str.partition(" ")[2].strip()
        if not arg:
            yield event.plain_result(
                f"当前主动转语音概率：{self.cfg.auto.tts_prob:g}。"
                "用法：语音概率 <0 到 1>"
            )
            return

        try:
            probability = float(arg)
        except ValueError:
            yield event.plain_result("概率必须是 0 到 1 之间的数字")
            return

        if not 0 <= probability <= 1:
            yield event.plain_result("概率必须在 0 到 1 之间")
            return

        self.cfg.auto.tts_prob = probability
        self.cfg.save_config()
        yield event.plain_result(f"已将主动转语音概率设置为：{probability:g}")

    @filter.command("重启GSV", alias={"重启gsv"})
    async def tts_control(self, event: AstrMessageEvent):
        """重启GPT_SoVITS"""
        if not self.cfg.enabled:
            return
        yield event.plain_result("重启TTS中...(报错信息请忽略，等待一会即可完成重启)")
        await self.service.restart()

    # ======================== 角色管理命令 ========================

    def _resolve_profile_name(self, arg: str) -> tuple[str | None, str | None]:
        """Resolve argument to profile name. Accepts index (1,2,3) or name.
        Returns (name, error_msg). On success error_msg is None.
        """
        names = self.profile_mgr.list_profiles()
        if not names:
            return None, "还没有保存的角色。使用「保存角色 名称」来创建一个。"

        # Try as index
        if arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(names):
                return names[idx], None
            return None, f"序号 {arg} 超出范围，当前共 {len(names)} 个角色。"

        # Try as name
        if self.profile_mgr.exists(arg):
            return arg, None

        return None, f"角色 '{arg}' 不存在。"

    @filter.command("保存角色", alias={"save_profile"})
    async def save_profile(self, event: AstrMessageEvent):
        """保存当前配置为角色"""
        if not self.cfg.enabled:
            return

        name = event.message_str.partition(" ")[2].strip()
        if not name:
            yield event.plain_result("请指定角色名称，例如：保存角色 我的角色")
            return

        self.profile_mgr.save_profile(
            name,
            model=self.cfg._data["model"],
            default_params=self.cfg._data["default_params"],
            entry_storage=self.cfg._data["entry_storage"],
        )
        yield event.plain_result(f"已保存当前配置为角色：{name}")

    @filter.command("语音角色", alias={"tts_role", "role"})
    async def tts_role(self, event: AstrMessageEvent):
        """查看/切换语音角色。不加参数列出所有角色，加序号或名称切换。"""
        if not self.cfg.enabled:
            return

        arg = event.message_str.partition(" ")[2].strip()

        # No argument: list all profiles
        if not arg:
            names = self.profile_mgr.list_profiles()
            if not names:
                yield event.plain_result(
                    "还没有保存的角色。使用「保存角色 名称」来创建一个。"
                )
                return

            lines = ["语音角色列表："]
            for i, name in enumerate(names, 1):
                active = " 👈" if name == self.cfg.active_profile else ""
                lines.append(f"{i}. {name}{active}")
            yield event.plain_result("\n".join(lines))
            return

        # Has argument: switch to profile
        name, err = self._resolve_profile_name(arg)
        if err:
            yield event.plain_result(err)
            return

        err = self._apply_profile(name)
        if err:
            yield event.plain_result(err)
            return

        await self.service.load_model()
        yield event.plain_result(f"已切换到角色：{name}")

    @filter.command("删除角色", alias={"delete_profile"})
    async def delete_profile(self, event: AstrMessageEvent):
        """删除指定角色（支持序号或名称）"""
        if not self.cfg.enabled:
            return

        arg = event.message_str.partition(" ")[2].strip()
        if not arg:
            yield event.plain_result(
                "请指定角色序号或名称，例如：删除角色 2 或 删除角色 我的角色"
            )
            return

        name, err = self._resolve_profile_name(arg)
        if err:
            yield event.plain_result(err)
            return

        if self.profile_mgr.delete_profile(name):
            if self.cfg.active_profile == name:
                self.cfg.active_profile = ""
                self.cfg.save_config()
            yield event.plain_result(f"已删除角色：{name}")
        else:
            yield event.plain_result(f"角色 '{name}' 不存在")

    # ======================== LLM 工具 ========================

    def _get_busy_schedule_media_recorder(self, event: AstrMessageEvent):
        return _resolve_busy_schedule_media_recorder(event, self.context)

    @filter.llm_tool()
    async def gsv_tts(self, event: AstrMessageEvent, message: str = ""):
        """
        用语音输出要讲的话。
        当用户明确希望听到你的声音、要求发送语音，或语音表达更符合需求时使用。
        message 仅填写需要合成的内容。
        Args:
            message(string): 要讲的话
        """
        if not isinstance(message, str) or not message.strip():
            return "语音文本不能为空"

        message = message.strip()
        try:
            params = await self._get_emotion_params(event, message)
            res = await self.service.inference(message, extra_params=params)
            if not bool(res):
                return res.error
            seg = self._to_record(res)
            await event.send(event.chain_result([seg]))
            event.set_extra(_DIRECT_DELIVERY_TEXT_EXTRA, message)
            event.set_extra(_DIRECT_DELIVERY_KIND_EXTRA, "voice")
            callback = self._get_busy_schedule_media_recorder(event)
            if callable(callback):
                try:
                    callback(
                        event.unified_msg_origin,
                        {"voice"},
                        operation_id=f"voice:{event.unified_msg_origin}:{uuid.uuid4().hex}",
                    )
                except Exception as exc:
                    logger.warning("记录忙碌执行记录失败: %s", exc)
            else:
                logger.debug("BusySchedule media recorder is unavailable")
        except Exception as e:
            return str(e)
