"""Microphone audio capture, voice activity detection, and waveform rendering."""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
PCM16_MAX = 32767.0
PCM16_MIN_INT = -32768
PCM16_MAX_INT = 32767
BYTES_PER_SAMPLE_16BIT = 2

# Voice activity detection constants (20 ms framing at the configured sample rate).
FRAME_DURATION_S = 0.02
SPEECH_LOW_HZ = 85.0
SPEECH_HIGH_HZ = 3500.0
SPEECH_RMS_THRESHOLD = 0.012
SPEECH_SNR_FACTOR = 2.0
MIN_VOICE_BAND_RATIO = 0.5
MAX_SPEECH_FLATNESS = 0.25
MIN_SPEECH_FRAMES = 10
EPSILON = 1e-12

WAVEFORM_BARS = "▁▂▃▄▅▆▇█"
DEFAULT_WAVEFORM_WIDTH = 12
AUDIO_BADGE_PREFIX = "◉ Audio"


def speech_frame_mask(data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a boolean mask of 20 ms frames containing sustained human speech.

    Each frame is evaluated in the frequency domain and must simultaneously:
    1. Concentrate at least 50% of its AC energy in the 85-3500 Hz voice band
       (rejecting low-frequency mic bumps/pops and high-frequency clicks/hiss).
    2. Exhibit harmonic/formant structure via low spectral flatness <= 0.25
       (rejecting broadband white/pink noise, fan hum and breath noise).
    3. Exceed both the absolute RMS floor (-38 dBFS) and 2x the recording's
       adaptive broadband noise floor (+6 dB SNR).
    Isolated single-frame transients (< 40 ms) are discarded.
    """
    mono = data.mean(axis=1) if data.ndim > 1 else data
    frame_len = int(sample_rate * FRAME_DURATION_S)
    num_frames = len(mono) // frame_len
    if num_frames < 2:
        return np.zeros(num_frames, dtype=bool)

    frames = mono[: num_frames * frame_len].reshape(num_frames, frame_len)
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
    band_mask = (freqs >= SPEECH_LOW_HZ) & (freqs <= SPEECH_HIGH_HZ)

    weights = np.full(len(freqs), 2.0 / (frame_len**2))
    weights[0] = 0.0

    total_power = spectrum @ weights
    band_power = spectrum[:, band_mask] @ weights[band_mask]
    band_rms = np.sqrt(band_power)
    band_ratio = band_power / np.maximum(total_power, EPSILON)

    band_spec = spectrum[:, band_mask] + EPSILON
    flatness = np.exp(np.mean(np.log(band_spec), axis=1)) / np.mean(band_spec, axis=1)

    noise_estimate = band_rms * np.clip(flatness / MAX_SPEECH_FLATNESS, 0.0, 1.0)
    noise_floor = float(np.percentile(noise_estimate, 20))
    threshold = max(SPEECH_RMS_THRESHOLD, noise_floor * SPEECH_SNR_FACTOR)

    active = (
        (band_rms >= threshold)
        & (band_ratio >= MIN_VOICE_BAND_RATIO)
        & (flatness <= MAX_SPEECH_FLATNESS)
    )
    pair = active[:-1] & active[1:]
    sustained = np.zeros_like(active)
    sustained[:-1] |= pair
    sustained[1:] |= pair
    return sustained


def is_silent(data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
    """Report whether an audio buffer lacks sustained human speech."""
    return int(np.count_nonzero(speech_frame_mask(data, sample_rate))) < MIN_SPEECH_FRAMES


def _trim_frames(data: np.ndarray, sample_rate: int, active: np.ndarray) -> np.ndarray:
    if not np.any(active):
        return data

    frame_len = int(sample_rate * FRAME_DURATION_S)
    first = int(np.argmax(active))
    last = len(active) - int(np.argmax(active[::-1]))
    return data[first * frame_len : last * frame_len]


def trim_silence(data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Drop leading and trailing non-speech frames around the spoken segment."""
    return _trim_frames(data, sample_rate, speech_frame_mask(data, sample_rate))


def encode_wav_bytes(
    data: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> bytes:
    """Encode captured float32 audio samples into 16-bit PCM WAV bytes."""
    if data.size == 0:
        return b""

    pcm16 = (data * PCM16_MAX).clip(PCM16_MIN_INT, PCM16_MAX_INT).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(BYTES_PER_SAMPLE_16BIT)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def render_waveform(samples: np.ndarray, width: int = DEFAULT_WAVEFORM_WIDTH) -> str:
    """Render a compact Unicode bar waveform from audio samples."""
    if width <= 0 or samples.size == 0:
        return WAVEFORM_BARS[0] * max(1, width)

    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    rms_values = [
        float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)))) if seg.size > 0 else 0.0
        for seg in np.array_split(mono, width)
    ]
    peak = max(rms_values, default=0.0)
    if peak <= EPSILON:
        return WAVEFORM_BARS[0] * width

    max_idx = len(WAVEFORM_BARS) - 1
    return "".join(
        WAVEFORM_BARS[min(max_idx, max(0, int(round((val / peak) * max_idx))))]
        for val in rms_values
    )


def format_audio_badge(waveform: str, duration: float) -> str:
    """Format the visual audio badge shown in the conversation thread."""
    return f"{AUDIO_BADGE_PREFIX} {waveform} ({duration:.1f}s)"


@dataclass(frozen=True, slots=True)
class AudioPrompt:
    """Recorded voice prompt with optional companion text."""

    text: str
    wav_base64: str
    duration: float
    waveform: str

    @property
    def badge(self) -> str:
        """Return the formatted waveform badge with duration."""
        return format_audio_badge(self.waveform, self.duration)

    @property
    def display_text(self) -> str:
        """Return the badge followed by companion text if present."""
        return f"{self.badge}\n{self.text}" if self.text else self.badge

    def __str__(self) -> str:
        return f"{self.badge} {self.text}".strip()


class AudioRecorder:
    """Microphone recorder producing WAV bytes or an AudioPrompt once stopped."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._chunks: list[np.ndarray] = []
        self._cached_chunks = 0
        self._samples = np.empty(0, dtype=np.float32)
        self._active = np.empty(0, dtype=bool)
        self._stream: sd.InputStream | None = None

    @property
    def is_recording(self) -> bool:
        """Report whether the microphone input stream is currently active."""
        return self._stream is not None

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """sounddevice callback — enqueue an immutable copy of each chunk."""
        self._chunks.append(indata.copy())

    def start(self) -> None:
        """Open the input stream and begin recording."""
        self._chunks.clear()
        self._cached_chunks = 0
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._on_audio,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self._stream = stream

    def stop(self) -> None:
        """Stop and close the input stream."""
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        try:
            stream.stop()
        finally:
            stream.close()

    def _analyze(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._chunks:
            self._cached_chunks = 0
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)
        if self._cached_chunks != len(self._chunks):
            self._samples = np.concatenate(self._chunks)
            self._active = speech_frame_mask(self._samples, self.sample_rate)
            self._cached_chunks = len(self._chunks)
        return self._samples, self._active

    @property
    def wav_bytes(self) -> bytes:
        """Encode captured speech as 16-bit PCM WAV."""
        samples, active = self._analyze()
        trimmed = _trim_frames(samples, self.sample_rate, active)
        return encode_wav_bytes(trimmed, self.sample_rate, self.channels)

    @property
    def is_silent(self) -> bool:
        """Report whether the recording lacks sustained human speech."""
        _, active = self._analyze()
        return int(np.count_nonzero(active)) < MIN_SPEECH_FRAMES

    def build_prompt(self, text: str = "") -> AudioPrompt:
        """Build an AudioPrompt from the recorded speech and optional companion text."""
        samples, active = self._analyze()
        trimmed = _trim_frames(samples, self.sample_rate, active)
        duration = round(max(0.1, len(trimmed) / self.sample_rate), 1) if trimmed.size > 0 else 0.0
        waveform = render_waveform(trimmed)
        wav_data = encode_wav_bytes(trimmed, self.sample_rate, self.channels)
        wav_b64 = base64.b64encode(wav_data).decode("ascii")
        return AudioPrompt(
            text=text.strip(),
            wav_base64=wav_b64,
            duration=duration,
            waveform=waveform,
        )
