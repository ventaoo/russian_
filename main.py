import os
import time

from tqdm import tqdm
from transformers import pipeline, Pipeline
from pydub import AudioSegment

def video2voice():
    # TODO 解决从视频到音频文件的提取
    # https://ytmp3.ec/1/ 
    ...

def func_imp(func, **args):
    start = time.time()
    result = func(args['inputs'], **{key: value for key, value in args.items() if key != 'inputs'})
    end = time.time()
    print(f"time: {end - start}")

    return result

def audio_seg(path: str, feature_length: int=30000):
    base_name = os.path.splitext(path)[0]
    audio = AudioSegment.from_file(path)
    len_audio = len(audio)
    seg_times = int(len_audio / feature_length)
    for i in tqdm(range(seg_times), desc=f"path: {path} | len:{len_audio}"):
        slice_audio = audio[i * feature_length: (i + 1) * feature_length if ((i + 1) * feature_length < len_audio) else len_audio]
        slice_audio.export(f"{base_name}_{i+1}.mp3", format="mp3")

def voice2text(pipe: Pipeline, path: str):
    return func_imp(func=pipe, inputs=path)["text"]

def ru2zh(pipe: Pipeline, content: str, prefix = 'translate to zh') -> str:
    return func_imp(func=pipe, inputs=f"{prefix}: {content}")[0]['translation_text']

if __name__ == '__main__':
    # audio_seg(audio_path)
    voice2text_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
    ru2zh_piple = pipeline("translation", model="utrobinmv/t5_translate_en_ru_zh_small_1024")

    ru = voice2text(voice2text_pipe, './data/test_audio_1.mp3')
    zh = ru2zh(ru2zh_piple, ru)
    print(f"ru: {ru} | zh: {zh}")
