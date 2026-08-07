from typing import Any

from astrbot.api import logger

from .client import GSVApiClient, GSVRequestResult
from .config import PluginConfig
from .local_data import LocalDataManager
from .text_split import split_text_with_symbols


class GPTSoVITSService:
    def __init__(
        self,
        config: PluginConfig,
        client: GSVApiClient,
        local_data: LocalDataManager,
    ):
        self.cfg = config.model
        self.default_params = config.default_params
        self.client = client
        self.local_data = local_data

    def update_config_sync(self, config: PluginConfig) -> None:
        """Update service configuration (sync)."""
        self.cfg = config.model
        self.default_params = config.default_params
        logger.info("服务配置已更新")

    async def load_model(self):
        if self.cfg.gpt_path:
            result = await self.client.set_gpt_weights(self.cfg.gpt_path)
            if result.ok:
                logger.info(f"GPT 模型已加载: {self.cfg.gpt_path}")
            else:
                logger.error(f"GPT 模型加载失败: {result.error}")

        if self.cfg.sovits_path:
            result = await self.client.set_sovits_weights(self.cfg.sovits_path)
            if result.ok:
                logger.info(f"SoVITS 模型已加载: {self.cfg.sovits_path}")
            else:
                logger.error(f"SoVITS 模型加载失败: {result.error}")

    def prepare_params(
        self,
        text: str,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("TTS 文本不能为空")

        params = self.default_params.copy()
        params["text"] = normalized_text
        if extra_params:
            params.update({k: v for k, v in extra_params.items() if k in params})

        custom_symbols = str(params.pop("custom_split_symbols", ""))
        if params.get("text_split_method") == "custom":
            params["text"] = split_text_with_symbols(
                str(params.get("text", "")), custom_symbols
            )
            params["text_split_method"] = "cut0"

        return params

    async def inference_raw(
        self,
        text: str,
        extra_params: dict[str, Any] | None = None,
    ) -> GSVRequestResult:
        """TTS inference without cache read/write (used for segment merging)."""
        params = self.prepare_params(text, extra_params)
        logger.debug(f"向 GSV 发起 TTS 请求（无缓存），参数: {params}")
        return await self.client.tts(params)

    async def inference(
        self,
        text: str,
        extra_params: dict[str, Any] | None = None,
    ) -> GSVRequestResult:
        """TTS 推理"""
        params = self.prepare_params(text, extra_params)

        if extra_params:
            filtered_params = {
                k: v for k, v in extra_params.items() if k in self.default_params
            }
            logger.debug(f"已更新已有参数: {filtered_params}")

        cached_audio = self.local_data.get_cached_audio(params)
        if cached_audio:
            cache_path, cached_data = cached_audio
            logger.debug("命中缓存，跳过 TTS 请求")
            return GSVRequestResult(
                ok=True,
                data=cached_data,
                text=str(params.get("text", "")),
                file_path=str(cache_path),
            )

        logger.debug(f"向 GSV 发起 TTS 请求，参数: {params}")
        result = await self.client.tts(params)

        if bool(result):
            cache_path = self.local_data.save_audio(result.data, params)
            if cache_path:
                result.file_path = str(cache_path)
        else:
            logger.error(f"TTS 推理失败: {result.error}")

        return result

    async def restart(self):
        result = await self.client.restart()
        if not result.ok:
            logger.error(f"重启失败: {result.error}")
