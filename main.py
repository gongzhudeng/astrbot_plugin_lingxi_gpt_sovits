import base64
import random

from astrbot.api import logger
from astrbot.api.event import filter
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
            if self.cfg.active_profile:
                self._apply_profile(self.cfg.active_profile)
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

        if self.cfg.judge.enabled_llm:
            labels = self.entry_mgr.get_names()
            emotion = await self.judger.judge_emotion(event, text=text, labels=labels)
            if emotion:
                entry = self.entry_mgr.get_entry(emotion)

        if entry is None:
            entry = self.entry_mgr.match_entry(text)

        return entry.to_params() if entry else None

    @filter.on_decorating_result(priority=14)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """消息入口"""
        if not self.cfg.enabled:
            return
        cfg = self.cfg.auto

        result = event.get_result()
        if not result:
            return
        chain = result.chain
        if not chain:
            return
        if cfg.only_llm_result and not result.is_llm_result():
            return
        if random.random() > cfg.tts_prob:
            return

        plain_texts = []
        for seg in chain:
            if isinstance(seg, Plain):
                plain_texts.append(seg.text)

        if len(plain_texts) != len(chain):
            return

        combined_text = "\n".join(plain_texts)

        if len(combined_text) > cfg.max_msg_len:
            return

        params = await self._get_emotion_params(event, combined_text)
        res = await self.service.inference(combined_text, extra_params=params)
        if not bool(res):
            return
        chain.clear()
        chain.append(self._to_record(res))

    @filter.command("说", alias={"gsv", "GSV"})
    async def on_command(self, event: AstrMessageEvent):
        """说 <内容>，直接调用GSV合成语音"""
        if not self.cfg.enabled:
            return

        text = event.message_str.partition(" ")[2]
        res = await self.service.inference(text)

        if not bool(res):
            yield event.plain_result(res.error)
            return

        yield event.chain_result([self._to_record(res)])

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
                yield event.plain_result("还没有保存的角色。使用「保存角色 名称」来创建一个。")
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
            yield event.plain_result("请指定角色序号或名称，例如：删除角色 2 或 删除角色 我的角色")
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

    @filter.llm_tool()
    async def gsv_tts(self, event: AstrMessageEvent, message: str = ""):
        """
        用语音输出要讲的话
        Args:
            message(string): 要讲的话
        """
        try:
            params = await self._get_emotion_params(event, message)
            res = await self.service.inference(message, extra_params=params)
            if not bool(res):
                return res.error
            seg = self._to_record(res)
            await event.send(event.chain_result([seg]))
        except Exception as e:
            return str(e)