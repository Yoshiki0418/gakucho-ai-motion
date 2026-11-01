import base64
import io
import wave
from abc import ABC, abstractmethod


def wav_duration_ms(audio_b64: str) -> float:
    """Base64 の WAV から再生時間(ミリ秒)を求める。解析できない場合は 0。"""
    try:
        with wave.open(io.BytesIO(base64.b64decode(audio_b64)), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return (frames / rate) * 1000 if rate else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


class BaseTTS(ABC):
    """TTS の基底クラス"""

    _model_name: str

    @property
    def model_name(self) -> str:
        """カタログの tts_id に対応する固有キーを返す"""
        return self._model_name

    @abstractmethod
    def synthesize_to_base64(self, text: str) -> str:
        """テキストを音声に変換し、Base64 エンコードした文字列を返す"""
        ...

    def synthesize_with_duration(self, text: str) -> tuple[str, float]:
        """Base64 音声と、その再生時間(ミリ秒)を返す。

        モーションを発話に合わせて再生するには、フロントエンド側で
        「この発話がいつからいつまで鳴るか」が分かっている必要がある。
        既定実装は WAV を読み直して長さを求めるので、サブクラスが
        波形を持っている場合は上書きしたほうが速い。
        """
        b64 = self.synthesize_to_base64(text)
        return b64, wav_duration_ms(b64)
