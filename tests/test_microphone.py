import numpy as np
import pytest

from voice_assistant.audio import AudioInputError, Microphone, StreamingResampler


@pytest.mark.parametrize("rate", [48000, 44100, 16000])
def test_resampling_is_continuous_and_independent_of_callback_boundaries(rate):
    wave = np.sin(2 * np.pi * 1000 * np.arange(rate) / rate).astype(np.float32)
    whole = StreamingResampler(rate).process(wave)
    resampler = StreamingResampler(rate)
    pieces = [resampler.process(wave[i:i + 713]) for i in range(0, len(wave), 713)]
    streamed = np.concatenate(pieces)
    assert len(streamed) == 16000
    np.testing.assert_allclose(streamed, whole, atol=1e-6)
    assert np.sqrt(np.mean(streamed[100:] ** 2)) > 0.6


def test_downsampling_suppresses_aliases():
    samples = np.sin(2 * np.pi * 12000 * np.arange(48000) / 48000).astype(np.float32)
    downsampled = StreamingResampler(48000).process(samples)
    assert np.sqrt(np.mean(downsampled[100:] ** 2)) < 0.01


@pytest.mark.parametrize("fail_during_open", [False, True])
async def test_microphone_uses_native_rate_and_emits_512_sample_frames(monkeypatch, fail_during_open):
    from voice_assistant import audio
    opened = []

    def check(**kwargs):
        if kwargs["samplerate"] == 16000 and not fail_during_open:
            raise audio.sd.PortAudioError("Invalid sample rate", -9997)

    class Stream:
        def __init__(self, **kwargs):
            if kwargs["samplerate"] == 16000:
                raise audio.sd.PortAudioError("Invalid sample rate", -9997)
            opened.append(kwargs)

        def start(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(audio.sd, "check_input_settings", check)
    monkeypatch.setattr(audio.sd, "query_devices", lambda *args: {"default_samplerate": 48000})
    monkeypatch.setattr(audio.sd, "InputStream", Stream)
    mic = Microphone(None)
    mic.start()
    assert opened[0]["samplerate"] == 48000
    assert opened[0]["blocksize"] == 1536
    mic._callback(np.zeros((1536, 1), dtype=np.float32), 1536, None, None)
    import asyncio
    frame = await asyncio.wait_for(mic.queue.get(), 0.2)
    assert frame.shape == (512,) and frame.dtype == np.float32
    mic.close()


async def test_device_failure_has_actionable_error(monkeypatch):
    from voice_assistant import audio

    def check(**kwargs):
        raise audio.sd.PortAudioError("Device unavailable", -9985)

    monkeypatch.setattr(audio.sd, "check_input_settings", check)
    mic = Microphone(None)
    with pytest.raises(AudioInputError, match="PipeWire"):
        mic.start()
