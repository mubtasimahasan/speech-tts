"""
Created on Fri. Sept. 8 00:40:25 2023
@author: Dong Zhang
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union

import numpy as np
import torch
import torchaudio
from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames
import json
import os
import glob
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../speech-token-modified')))
from speechtokenizer import SpeechTokenizer


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Speechtokenizer:
    """SpeechTokenizer"""

    def __init__(
        self,
        ckpt_dir: str = "",
        device: Any = None,
    ) -> None:
        # config_path = glob.glob(os.path.join(os.path.dirname(ckpt_dir), "*.json"))[0]
        config_path = glob.glob(os.path.join(ckpt_dir, "*.json"))[0]
        ckpt_path = glob.glob(os.path.join(ckpt_dir, "*.pt"))[0]
        # load model
        model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        model.eval()

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device
        self.model = model.to(device)
        self.sample_rate = model.sample_rate
        self.channels = 1

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model.encode(wav.to(self.device))

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.model.decode(frames)


def sttokenize_audio(tokenizer: Speechtokenizer, audio_path: str):
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    # Print initial device information
    # print(f"convert_audio: wav tensor is on device {wav.device}")
    
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    
    # Move the Resample operation to the correct device
    device = wav.device
    resampler = torchaudio.transforms.Resample(sr, target_sr).to(device)
    
    wav = resampler(wav)
    
    # Print after resampling
    # print(f"Resampled wav tensor is on device {wav.device}")
    
    return wav


@dataclass
class STAudioTokenConfig:
    frame_shift: Seconds = 320.0 / 16000
    num_quantizers: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "STAudioTokenConfig":
        return STAudioTokenConfig(**data)


class STAudioTokenExtractor(FeatureExtractor):
    name = "speechtokenizer"
    config_type = STAudioTokenConfig

    def __init__(self, ckpt_dir: str, config: Optional[Any] = None):
        super(STAudioTokenExtractor, self).__init__(config)
        # Pass the model directory to the tokenizer
        self.tokenizer = Speechtokenizer(ckpt_dir)

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if sampling_rate != self.tokenizer.sample_rate:
            samples = convert_audio(
                samples,
                sampling_rate,
                self.tokenizer.sample_rate,
                self.tokenizer.channels,
            )
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(0)
        else:
            raise ValueError()

        device = self.tokenizer.device
        encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        codes = encoded_frames.permute(1,0,2)  # [B, n_q, T]
        if True:
            duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            assert abs(codes.shape[-1] - expected_num_frames) <= 1
            codes = codes[..., :expected_num_frames]
        return codes.cpu().squeeze(0).permute(1, 0).numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_quantizers

    def pad_tensor_list(self, tensor_list, device, padding_value=0):
        # 计算每个张量的长度
        lengths = [tensor.shape[0] for tensor in tensor_list]
        # 使用pad_sequence函数进行填充
        tensor_list = [torch.Tensor(t).to(device) for t in tensor_list]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )
        return padded_tensor, lengths

    def extract_batch(self, samples, sampling_rate, lengths) -> np.ndarray:
        samples = [wav.squeeze() for wav in samples]
        device = self.tokenizer.device
        samples, lengths = self.pad_tensor_list(samples, device)
        samples = samples.unsqueeze(1)

        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        if len(samples.shape) != 3:
            raise ValueError()
        if sampling_rate != self.tokenizer.sample_rate:
            samples = [
                convert_audio(
                    wav.to(device),
                    sampling_rate,
                    self.tokenizer.sample_rate,
                    self.tokenizer.channels,
                )
                for wav in samples
            ]
            samples = torch.stack(samples, 0) # convert samples from list to tensor
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        encoded_frames = encoded_frames.permute(1,0,2)  # [B, n_q, T]
        batch_codes = []
        for b, length in enumerate(lengths):
            codes = encoded_frames[b]
            duration = round(length / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            batch_codes.append(codes[..., :expected_num_frames])
        return [codes.cpu().permute(1, 0).numpy() for codes in batch_codes]



if __name__ == "__main__":

    audio_path = r'valle/examples/libritts/prompts/8455_210777_000067_000000.wav'
    speechtokenizer = Speechtokenizer()

    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        tokens = speechtokenizer.encode(wav)

    reconstructed = speechtokenizer.decode(tokens)

    
