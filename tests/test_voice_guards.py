from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from data.plugins.astrbot_plugin_lingxi_gpt_sovits.core.client import GSVRequestResult
from data.plugins.astrbot_plugin_lingxi_gpt_sovits.core.service import GPTSoVITSService
from data.plugins.astrbot_plugin_lingxi_gpt_sovits.main import GPTSoVITSPlugin


def _make_service():
    config = SimpleNamespace(
        model=SimpleNamespace(),
        default_params={"text": "你要干嘛？", "media_type": "wav"},
    )
    return GPTSoVITSService(config, AsyncMock(), MagicMock())


@pytest.mark.parametrize("text", [None, "", "   ", "\n\t"])
def test_service_rejects_empty_text_before_default_params(text):
    service = _make_service()

    with pytest.raises(ValueError, match="TTS 文本不能为空"):
        service.prepare_params(text)


@pytest.mark.asyncio
async def test_service_does_not_call_cache_or_tts_for_empty_text():
    service = _make_service()

    with pytest.raises(ValueError, match="TTS 文本不能为空"):
        await service.inference(" ")

    service.local_data.get_cached_audio.assert_not_called()
    service.client.tts.assert_not_called()


@pytest.mark.asyncio
async def test_gsv_tool_rejects_empty_message_before_emotion_or_send():
    plugin = GPTSoVITSPlugin.__new__(GPTSoVITSPlugin)
    plugin.service = AsyncMock()
    plugin._get_emotion_params = AsyncMock()
    event = SimpleNamespace(send=AsyncMock())

    result = await plugin.gsv_tts(event, " \n ")

    assert result == "语音文本不能为空"
    plugin._get_emotion_params.assert_not_called()
    plugin.service.inference.assert_not_called()
    event.send.assert_not_called()


@pytest.mark.asyncio
async def test_gsv_tool_records_direct_voice_history_after_send():
    plugin = GPTSoVITSPlugin.__new__(GPTSoVITSPlugin)
    plugin.service = SimpleNamespace(
        inference=AsyncMock(
            return_value=GSVRequestResult(ok=True, data=b"wav", text="困死了，晚安。")
        )
    )
    plugin._get_emotion_params = AsyncMock(return_value={})
    plugin._to_record = MagicMock(return_value="record")
    plugin._get_busy_schedule_media_recorder = MagicMock(return_value=None)
    event = SimpleNamespace(
        send=AsyncMock(),
        chain_result=lambda chain: chain,
        set_extra=MagicMock(),
    )

    result = await plugin.gsv_tts(event, " 困死了，晚安。 ")

    assert result is None
    event.send.assert_awaited_once_with(["record"])
    assert event.set_extra.call_args_list == [
        call("spark_direct_delivery_history_text", "困死了，晚安。"),
        call("spark_direct_delivery_kind", "voice"),
    ]


@pytest.mark.asyncio
async def test_command_rejects_missing_text_before_inference():
    plugin = GPTSoVITSPlugin.__new__(GPTSoVITSPlugin)
    plugin.cfg = SimpleNamespace(enabled=True)
    plugin._infer_with_emotion = AsyncMock()
    event = SimpleNamespace(
        message_str="说",
        plain_result=lambda message: message,
    )

    result = [item async for item in plugin.on_command(event)]

    assert result == ["请输入要合成的文本"]
    plugin._infer_with_emotion.assert_not_called()


@pytest.mark.asyncio
async def test_service_keeps_explicit_text_and_normal_voice_request():
    service = _make_service()
    service.local_data.get_cached_audio.return_value = None
    service.client.tts.return_value = GSVRequestResult(
        ok=True,
        data=b"wav",
        text="你好",
    )

    params = service.prepare_params("  你好  ")
    result = await service.inference("  你好  ")

    assert params["text"] == "你好"
    assert result.ok is True
    service.client.tts.assert_awaited_once()
    assert service.client.tts.await_args.args[0]["text"] == "你好"
