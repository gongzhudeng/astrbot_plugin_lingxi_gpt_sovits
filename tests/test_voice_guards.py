from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from data.plugins.astrbot_plugin_lingxi_gpt_sovits.core.client import GSVRequestResult
from data.plugins.astrbot_plugin_lingxi_gpt_sovits.core.entry import (
    EmotionEntry,
    EntryManager,
)
from data.plugins.astrbot_plugin_lingxi_gpt_sovits.core.profile_manager import (
    ProfileManager,
)
from data.plugins.astrbot_plugin_lingxi_gpt_sovits.core.service import GPTSoVITSService
from data.plugins.astrbot_plugin_lingxi_gpt_sovits.main import GPTSoVITSPlugin


def _entry(name: str, enabled: bool | None, keywords: list[str]) -> EmotionEntry:
    data = {
        "name": name,
        "keywords": keywords,
        "ref_audio_path": "",
        "prompt_text": "",
        "prompt_lang": "zh",
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
    }
    if enabled is not None:
        data["enabled"] = enabled
    return EmotionEntry(data)


def test_emotion_entry_without_enabled_defaults_to_enabled():
    assert _entry("旧条目", None, []).is_enabled is True
    assert _entry("关闭条目", False, []).is_enabled is False


def test_entry_manager_ignores_disabled_entries_everywhere():
    manager = EntryManager.__new__(EntryManager)
    manager.entries = [
        _entry("已关闭", False, ["命中"]),
        _entry("已启用", True, ["命中"]),
    ]

    assert manager.get_names() == ["已启用"]
    assert manager.get_entry("已关闭") is None
    assert manager.get_entry("已启用").name == "已启用"
    assert manager.match_entry("需要命中").name == "已启用"


def test_profile_snapshot_preserves_emotion_enabled_switch(tmp_path: Path):
    manager = ProfileManager(tmp_path)
    entries = [{"name": "温柔", "enabled": False}]

    manager.save_profile("测试", {}, {}, entries)

    assert manager.get_profile("测试")["entry_storage"] == entries


@pytest.mark.asyncio
async def test_tts_probability_command_reads_and_updates_config():
    plugin = GPTSoVITSPlugin.__new__(GPTSoVITSPlugin)
    plugin.cfg = SimpleNamespace(
        enabled=True,
        auto=SimpleNamespace(tts_prob=0.15),
        save_config=MagicMock(),
    )
    event = SimpleNamespace(
        message_str="语音概率",
        plain_result=lambda message: message,
    )

    result = [item async for item in plugin.set_tts_probability(event)]

    assert result == ["当前主动转语音概率：0.15。用法：语音概率 <0 到 1>"]

    event.message_str = "语音概率 0.4"
    result = [item async for item in plugin.set_tts_probability(event)]

    assert result == ["已将主动转语音概率设置为：0.4"]
    assert plugin.cfg.auto.tts_prob == 0.4
    plugin.cfg.save_config.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["abc", "-0.1", "1.1", "nan", "inf"])
async def test_tts_probability_command_rejects_invalid_values(value):
    plugin = GPTSoVITSPlugin.__new__(GPTSoVITSPlugin)
    plugin.cfg = SimpleNamespace(
        enabled=True,
        auto=SimpleNamespace(tts_prob=0.15),
        save_config=MagicMock(),
    )
    event = SimpleNamespace(
        message_str=f"语音概率 {value}",
        plain_result=lambda message: message,
    )

    result = [item async for item in plugin.set_tts_probability(event)]

    assert len(result) == 1
    assert plugin.cfg.auto.tts_prob == 0.15
    plugin.cfg.save_config.assert_not_called()


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
