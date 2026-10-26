"""Unit tests for ollama_agent.core.audio module."""

from __future__ import annotations

import base64
import io
import unittest
import wave
from unittest.mock import MagicMock, patch

import numpy as np

from ollama_agent.core.audio import (
    DEFAULT_WAVEFORM_WIDTH,
    SAMPLE_RATE,
    WAVEFORM_BARS,
    AudioPrompt,
    AudioRecorder,
    encode_wav_bytes,
    format_audio_badge,
    is_silent,
    render_waveform,
    speech_frame_mask,
    trim_silence,
)


def _speech_signal(duration_s: float) -> np.ndarray:
    """Synthesize a harmonic voice-band signal (F0=150 Hz + harmonics)."""
    num_samples = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0.0, duration_s, num_samples, endpoint=False, dtype=np.float32)
    signal = sum(np.sin(2.0 * np.pi * 150.0 * k * t) / k for k in range(1, 8))
    normalized = (signal / np.max(np.abs(signal))).astype(np.float32)
    return (normalized * 0.2).reshape(-1, 1)


class TestAudioCore(unittest.TestCase):
    """Tests for VAD, silence trimming, WAV encoding, waveform rendering, and AudioRecorder."""

    def test_speech_frame_mask_rejects_short_buffers(self) -> None:
        self.assertEqual(speech_frame_mask(np.empty(0, dtype=np.float32), SAMPLE_RATE).size, 0)
        self.assertTrue(
            np.array_equal(
                speech_frame_mask(np.ones(320, dtype=np.float32), SAMPLE_RATE),
                np.array([False]),
            )
        )

    def test_is_silent_rejects_noise_clicks_and_out_of_band_signals(self) -> None:
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 1.0, SAMPLE_RATE, endpoint=False, dtype=np.float32)

        self.assertTrue(is_silent(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE))
        self.assertTrue(is_silent(rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1, SAMPLE_RATE))

        low_thump = (np.sin(2.0 * np.pi * 40.0 * t) * 0.4).astype(np.float32)
        high_tone = (np.sin(2.0 * np.pi * 5000.0 * t) * 0.4).astype(np.float32)
        self.assertTrue(is_silent(low_thump, SAMPLE_RATE))
        self.assertTrue(is_silent(high_tone, SAMPLE_RATE))

        click = np.zeros((SAMPLE_RATE, 1), dtype=np.float32)
        click[1600:1920] = _speech_signal(0.02)
        self.assertTrue(is_silent(click, SAMPLE_RATE))

        speech = _speech_signal(0.4)
        noisy_speech = speech + rng.standard_normal(speech.shape).astype(np.float32) * 0.02
        self.assertFalse(is_silent(speech, SAMPLE_RATE))
        self.assertFalse(is_silent(noisy_speech, SAMPLE_RATE))

    def test_trim_silence_removes_leading_and_trailing_quiet_and_clicks(self) -> None:
        speech = _speech_signal(0.3)
        quiet = np.zeros((1600, 1), dtype=np.float32)
        click = np.zeros((1600, 1), dtype=np.float32)
        click[:320] = _speech_signal(0.02)

        data = np.concatenate([click, quiet, speech, quiet, click])
        trimmed = trim_silence(data, SAMPLE_RATE)
        self.assertTrue(np.array_equal(trimmed, speech))

    def test_trim_silence_keeps_interior_pauses(self) -> None:
        part = _speech_signal(0.2)
        pause = np.zeros((1600, 1), dtype=np.float32)
        data = np.concatenate([part, pause, part])

        trimmed = trim_silence(data, SAMPLE_RATE)
        self.assertTrue(np.array_equal(trimmed, data))

    def test_trim_silence_returns_all_silent_audio_unchanged(self) -> None:
        quiet = np.zeros((1600, 1), dtype=np.float32)
        self.assertTrue(np.array_equal(trim_silence(quiet, SAMPLE_RATE), quiet))

    def test_encode_wav_bytes(self) -> None:
        self.assertEqual(encode_wav_bytes(np.empty(0, dtype=np.float32)), b"")
        speech = _speech_signal(0.25)
        wav_bytes = encode_wav_bytes(speech, SAMPLE_RATE, 1)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), SAMPLE_RATE)
            self.assertEqual(wf.getnframes(), len(speech))

    def test_render_waveform(self) -> None:
        self.assertEqual(render_waveform(np.empty(0, dtype=np.float32), width=8), "▁" * 8)
        self.assertEqual(render_waveform(np.zeros(100, dtype=np.float32), width=6), "▁" * 6)
        self.assertEqual(render_waveform(np.ones(10, dtype=np.float32), width=0), "▁")

        ramp = np.linspace(0.0, 1.0, 1200, dtype=np.float32).reshape(-1, 1)
        wf = render_waveform(ramp, width=DEFAULT_WAVEFORM_WIDTH)
        self.assertEqual(len(wf), DEFAULT_WAVEFORM_WIDTH)
        self.assertEqual(wf[0], WAVEFORM_BARS[0])
        self.assertEqual(wf[-1], WAVEFORM_BARS[-1])

    def test_format_audio_badge_and_audio_prompt(self) -> None:
        badge = format_audio_badge("▂▅▇█", 3.45)
        self.assertEqual(badge, "◉ Audio ▂▅▇█ (3.5s)")

        prompt_only_audio = AudioPrompt(text="", wav_base64="AAA=", duration=2.3, waveform="▂▅▇█")
        self.assertEqual(prompt_only_audio.badge, "◉ Audio ▂▅▇█ (2.3s)")
        self.assertEqual(prompt_only_audio.display_text, "◉ Audio ▂▅▇█ (2.3s)")
        self.assertEqual(str(prompt_only_audio), "◉ Audio ▂▅▇█ (2.3s)")

        prompt_with_text = AudioPrompt(text="Summarize this", wav_base64="AAA=", duration=4.0, waveform="▂▅▇█")
        self.assertEqual(prompt_with_text.display_text, "◉ Audio ▂▅▇█ (4.0s)\nSummarize this")
        self.assertEqual(str(prompt_with_text), "◉ Audio ▂▅▇█ (4.0s) Summarize this")

    @patch("ollama_agent.core.audio.sd.InputStream")
    def test_recorder_start_and_stop(self, mock_input_stream: MagicMock) -> None:
        stream_instance = MagicMock()
        mock_input_stream.return_value = stream_instance

        recorder = AudioRecorder(sample_rate=16000, channels=1)
        self.assertFalse(recorder.is_recording)

        recorder.start()
        mock_input_stream.assert_called_once()
        stream_instance.start.assert_called_once()
        self.assertTrue(recorder.is_recording)

        recorder.stop()
        stream_instance.stop.assert_called_once()
        stream_instance.close.assert_called_once()
        self.assertFalse(recorder.is_recording)

        recorder.stop()
        self.assertEqual(stream_instance.stop.call_count, 1)

    @patch("ollama_agent.core.audio.sd.InputStream")
    def test_recorder_start_failure_closes_stream_and_raises(self, mock_input_stream: MagicMock) -> None:
        stream_instance = MagicMock()
        stream_instance.start.side_effect = RuntimeError("Device unavailable")
        mock_input_stream.return_value = stream_instance

        recorder = AudioRecorder()
        with self.assertRaises(RuntimeError):
            recorder.start()

        stream_instance.close.assert_called_once()
        self.assertFalse(recorder.is_recording)

    def test_recorder_capture_and_build_prompt(self) -> None:
        recorder = AudioRecorder(sample_rate=SAMPLE_RATE, channels=1)
        self.assertTrue(recorder.is_silent)
        self.assertEqual(recorder.wav_bytes, b"")

        empty_prompt = recorder.build_prompt()
        self.assertEqual(empty_prompt.duration, 0.0)
        self.assertEqual(empty_prompt.wav_base64, "")

        quiet = np.zeros((1600, 1), dtype=np.float32)
        speech = _speech_signal(0.4)
        recorder._on_audio(quiet, len(quiet), None, None)
        recorder._on_audio(speech, len(speech), None, None)
        recorder._on_audio(quiet, len(quiet), None, None)

        self.assertFalse(recorder.is_silent)
        prompt = recorder.build_prompt("  Hello voice  ")
        self.assertEqual(prompt.text, "Hello voice")
        self.assertAlmostEqual(prompt.duration, 0.4, places=1)
        self.assertEqual(len(prompt.waveform), DEFAULT_WAVEFORM_WIDTH)

        decoded_wav = base64.b64decode(prompt.wav_base64)
        self.assertEqual(decoded_wav, recorder.wav_bytes)
        with wave.open(io.BytesIO(decoded_wav), "rb") as wf:
            self.assertEqual(wf.getnframes(), len(speech))


if __name__ == "__main__":
    unittest.main()
