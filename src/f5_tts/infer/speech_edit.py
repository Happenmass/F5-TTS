import os

os.environ["PYTOCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility

import torch
import torch.nn.functional as F
import torchaudio

from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder, save_spectrogram
from f5_tts.model import CFM, DiT, UNetT
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
from cached_path import cached_path
import re
import difflib

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# --------------------- Dataset Settings -------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'
target_rms = 0.1

tokenizer = "pinyin"
dataset_name = "Emilia_ZH_EN"
# ---------------------- infer setting ---------------------- #

seed = None  # int | None

exp_name = "F5TTS_Base"  # F5TTS_Base | E2TTS_Base
ckpt_step = 1200000

nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"  # euler | midpoint
sway_sampling_coef = -1.0
speed = 1.0
time_stamp_sample_rate = 16000

time_stamp_predictor = None

# --------------------- TimeStampPredictor Initialize -------------------- #
def load_timestamp_pipeline():
    return pipeline(
        task=Tasks.auto_speech_recognition,
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        model_revision="v2.0.4",
    )

def remove_punctuation(text):
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~，。！？：；“”‘’（）【】、～·"""
    return re.sub(f"[{punctuation}]+", "", text)

def compare_string(origin_chars, target_chars, speech_rate, timestamps):
    matcher = difflib.SequenceMatcher(a=origin_chars, b=target_chars)
    opcodes = matcher.get_opcodes()

    print("opcodes", opcodes)

    char_to_timestamp = {i: timestamps[i] for i in range(len(timestamps))}

    # 初始化修复时间段列表
    parts_to_edit = []
    fix_durations = []

    # 初始化上一个编辑操作的结束时间为 0
    previous_end_ts = 0.0

    for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == 'equal':
            continue  # 无需编辑
        elif tag in ['replace', 'delete', 'insert']:
            # 计算涉及的字符数
            origin_length = i2 - i1
            target_length = j2 - j1

            # 判断是否是第一个编辑操作且发生在第一个字符
            if idx == 0 and i1 == 0:
                start_ts = 0.0
            else:
                start_ts = previous_end_ts

            if tag == 'replace':
                duration_chars = j2 - j1 - (i2 - i1)
                fix_duration_refix = duration_chars / speech_rate  # 秒
                fix_duration_refix = round(fix_duration_refix, 2)

                # 使用原始时间戳的长度
                if origin_length == 0:
                    # 边界情况：替换为空
                    start_ts = char_to_timestamp[i1 - 1][1] / 1000.0 if i1 > 0 else 0.0
                    end_ts = start_ts
                else:
                    start_ts = char_to_timestamp[i1 - 1][1] / 1000.0
                    if idx == len(opcodes) - 1:
                        # 边界情况：替换最后一个字符
                        end_ts = char_to_timestamp[i2 - 1][1] / 1000.0
                    else:
                        end_ts = char_to_timestamp[i2][0] / 1000.0
                fix_duration = end_ts - start_ts + fix_duration_refix
                fix_duration = round(fix_duration, 2)

            elif tag == 'delete':
                # 删除按原始语速计算修复持续时间
                duration_chars = origin_length
                fix_duration = duration_chars / speech_rate  # 秒
                fix_duration = round(fix_duration, 2)

                if origin_length == 0:
                    start_ts = 0.0
                    end_ts = 0.0
                else:
                    start_ts = char_to_timestamp[i1 - 1][1] / 1000.0
                    if idx == len(opcodes) - 1:
                        # 边界情况：替换最后一个字符
                        end_ts = char_to_timestamp[i2 - 1][1] / 1000.0
                    else:
                        end_ts = char_to_timestamp[i2][0] / 1000.0
            elif tag == 'insert':
                # 插入按原始语速计算修复持续时间
                duration_chars = target_length
                fix_duration = duration_chars / speech_rate  # 秒
                fix_duration = round(fix_duration, 2)

                if i1 > 0:
                    start_ts = char_to_timestamp[i1 - 1][1] / 1000.0
                else:
                    start_ts = 0.0
                end_ts = start_ts + fix_duration  # 插入点的结束时间根据修复持续时间计算

            # 添加到修复列表
            parts_to_edit.append([round(start_ts, 2), round(end_ts, 2)])
            fix_durations.append(fix_duration)

            # 更新上一个编辑操作的结束时间
            previous_end_ts = end_ts

    return parts_to_edit, fix_durations

def prepare_edit(
    ref_audio,
    target_text,
    model,
    vocoder,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    parts_to_edit=[],
    fix_duration=[],
):
    global time_stamp_predictor
    if time_stamp_predictor is None:
        time_stamp_predictor = load_timestamp_pipeline()

    audio, sr = torchaudio.load(ref_audio)

    if sr != time_stamp_sample_rate:
        print("resample")
        resampler = torchaudio.transforms.Resample(sr, time_stamp_sample_rate)
        audio_time = resampler(audio)
    else:
        audio_time = audio

    res = time_stamp_predictor(audio_time[0].numpy(), time_stamp_sample_rate)[0]
    timestamps = res['timestamp']
    
    target_text_clean = remove_punctuation(target_text)

    origin_chars = list(remove_punctuation(res["text"]))
    target_chars = list(target_text_clean)

    # 计算原始音频的总时长（秒）
    total_duration_ms = timestamps[-1][1]  # 假设最后一个字符的结束时间为总时长
    total_duration_sec = total_duration_ms / 1000.0

    # 计算语速（字符每秒）
    num_chars = len(origin_chars)
    speech_rate = num_chars / total_duration_sec
    print(f"语速: {speech_rate:.2f} 字符/秒")
    if len(parts_to_edit) == 0 and len(fix_duration) == 0:
        parts_to_edit, fix_duration = compare_string(origin_chars, target_chars, speech_rate, timestamps)

    print(f"parts_to_edit: {parts_to_edit}")
    print(f"fix_duration: {fix_duration}")

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio_gen = resampler(audio)
    else:
        audio_gen = audio
    
    wav, mel = edit_infer(
        audio_gen,
        target_text,
        model,
        vocoder,
        mel_spec_type="vocos",
        target_rms=target_rms,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        parts_to_edit=parts_to_edit,
        fix_duration=fix_duration,
    )
    return wav.squeeze().cpu().numpy(), mel[0].cpu().numpy(), target_sample_rate, parts_to_edit, fix_duration

def edit_infer(
    ref_audio,
    target_text,
    model,
    vocoder,
    mel_spec_type,
    target_rms=0.1,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    parts_to_edit=[],
    fix_duration=[],
):
    fix_duration_cp = fix_duration.copy()
    if ref_audio.shape[0] > 1:
        ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
    if rms < target_rms:
        ref_audio = ref_audio * target_rms / rms
    offset = 0
    audio_ = torch.zeros(1, 0)
    edit_mask = torch.zeros(1, 0, dtype=torch.bool)
    for part in parts_to_edit:
        start, end = part
        part_dur = end - start if len(fix_duration_cp) == 0 else fix_duration_cp.pop(0)
        part_dur = part_dur * target_sample_rate
        start = start * target_sample_rate
        audio_ = torch.cat((audio_, ref_audio[:, round(offset) : round(start)], torch.zeros(1, round(part_dur))), dim=-1)
        edit_mask = torch.cat(
            (
                edit_mask,
                torch.ones(1, round((start - offset) / hop_length), dtype=torch.bool),
                torch.zeros(1, round(part_dur / hop_length), dtype=torch.bool),
            ),
            dim=-1,
        )
        offset = end * target_sample_rate
    # audio = torch.cat((audio_, audio[:, round(offset):]), dim = -1)
    edit_mask = F.pad(edit_mask, (0, ref_audio.shape[-1] // hop_length - edit_mask.shape[-1] + 1), value=True)
    ref_audio = ref_audio.to(device)
    edit_mask = edit_mask.to(device)

    # Text
    text_list = [target_text]
    if tokenizer == "pinyin":
        final_text_list = convert_char_to_pinyin(text_list)
    else:
        final_text_list = [text_list]
    print(f"text  : {text_list}")
    print(f"pinyin: {final_text_list}")

    # Duration
    ref_audio_len = 0
    duration = ref_audio.shape[-1] // hop_length

    # Inference
    with torch.inference_mode():
        generated, trajectory = model.sample(
            cond=ref_audio,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=seed,
            edit_mask=edit_mask,
        )

        # Final result
        generated = generated.to(torch.float32)
        generated = generated[:, ref_audio_len:, :]
        gen_mel_spec = generated.permute(0, 2, 1)
        if mel_spec_type == "vocos":
            generated_wave = vocoder.decode(gen_mel_spec).cpu()
        elif mel_spec_type == "bigvgan":
            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        return generated_wave, gen_mel_spec

if exp_name == "F5TTS_Base":
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

elif exp_name == "E2TTS_Base":
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
if __name__ == "__main__":
    repo_name = "F5-TTS"
    ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))
    output_dir = "tests"

    time_stamp_predictor = load_timestamp_pipeline()

    # [leverage https://github.com/MahmoudAshraf97/ctc-forced-aligner to get char level alignment]
    # pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
    # [write the origin_text into a file, e.g. tests/test_edit.txt]
    # ctc-forced-aligner --audio_path "src/f5_tts/infer/examples/basic/basic_ref_en.wav" --text_path "tests/test_edit.txt" --language "zho" --romanize --split_size "char"
    # [result will be saved at same path of audio file]
    # [--language "zho" for Chinese, "eng" for English]
    # [if local ckpt, set --alignment_model "../checkpoints/mms-300m-1130-forced-aligner"]

    # audio_to_edit = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    # origin_text = "Some call me nature, others call me mother nature."
    # target_text = "Some call me optimist, others call me realist."
    # parts_to_edit = [
    #     [1.42, 2.44],
    #     [4.04, 4.9],
    # ]  # stard_ends of "nature" & "mother nature", in seconds
    # fix_duration = [
    #     1.2,
    #     1,
    # ]  # fix duration for "optimist" & "realist", in seconds

    audio_to_edit = "src/f5_tts/infer/examples/basic/basic_ref_zh.wav"
    target_text = "对，"

    target_text_clean = remove_punctuation(target_text)

    # Audio
    audio, sr = torchaudio.load(audio_to_edit)
    if sr != time_stamp_sample_rate:
        print("resample")
        resampler = torchaudio.transforms.Resample(sr, time_stamp_sample_rate)
        audio_time = resampler(audio)

    res = time_stamp_predictor(audio_time[0].numpy(), time_stamp_sample_rate)[0]

    print(res)

    text = res["text"]
    timestamps = res['timestamp']

    origin_chars = list(remove_punctuation(res["text"]))
    target_chars = list(target_text_clean)
    predicted_chars = res["text"].split()


    # 计算原始音频的总时长（秒）
    total_duration_ms = timestamps[-1][1]  # 假设最后一个字符的结束时间为总时长
    total_duration_sec = total_duration_ms / 1000.0

    # 计算语速（字符每秒）
    num_chars = len(origin_chars)
    speech_rate = num_chars / total_duration_sec
    print(f"语速: {speech_rate:.2f} 字符/秒")

    parts_to_edit, fix_duration = compare_string(origin_chars, target_chars, speech_rate, timestamps)
    print(f"parts_to_edit: {parts_to_edit}")
    print(f"fix_duration: {fix_duration}")
    # -------------------------------------------------#

    use_ema = True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Vocoder model
    local = False
    if mel_spec_type == "vocos":
        vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Model
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    # Audio
    audio, sr = torchaudio.load(audio_to_edit)

    wav, mel = edit_infer(
        audio,
        target_text,
        model,
        vocoder,
        mel_spec_type,
        target_rms=target_rms,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        parts_to_edit=parts_to_edit,
        fix_duration=fix_duration,
    )

    save_spectrogram(mel[0].cpu().numpy(), f"{output_dir}/speech_edit_out.png")
    torchaudio.save(f"{output_dir}/speech_edit_out.wav", wav, target_sample_rate)