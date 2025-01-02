import os
import sys
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
from transformers import pipeline, Pipeline
from pydub import AudioSegment

def video2voice():
    # TODO 解决从视频到音频文件的提取
    # https://ytmp3.ec/1/ 
    ...

def audio_seg(ori_path: str, save_path: str, feature_length: int=30000, flag: bool=True):
    base_name = os.path.basename(ori_path).split('.')[0]
    audio = AudioSegment.from_file(ori_path)
    len_audio = len(audio)
    seg_times = int(len_audio / feature_length)
    for i in tqdm(range(seg_times), desc=f"audio_seg - path: {ori_path} -> {save_path} | len:{len_audio}"):
        export_path = os.path.join(save_path, base_name)
        if not os.path.exists(export_path): os.mkdir(export_path)
        export_name = f"{export_path}/{i+1}.mp3"
        if os.path.exists(export_name) and flag: continue
        slice_audio = audio[i * feature_length: (i + 1) * feature_length if ((i + 1) * feature_length < len_audio) else len_audio]
        slice_audio.export(export_name, format="mp3")

def voice2text(pipe: Pipeline, path: str):
    return pipe(path)['text']

def ru_convert(pipe: Pipeline, content: str, prefix = 'translate to zh') -> str:
    return pipe(f"{prefix}: {content}")[0]['translation_text']

if __name__ == '__main__':
    # 音频文件的路径 - TODO 后期需要自动化，从txt中读取对应的视频的链接，自动下载音频和视频
    ori_path = sys.argv[1]
    save_path = sys.argv[2]
    feature_length = int(sys.argv[3])
    

    voice2text_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
    ru2zh_piple = pipeline("translation", model="utrobinmv/t5_translate_en_ru_zh_small_1024")
    

    # 1. step - 分割音频
    audio_seg(ori_path, save_path, feature_length)
    # 2. step - 分割视频
    # TODO

    # 3. step - 循环处理所有文件
    file_continer = []
    for path, _ ,files in os.walk(save_path):
        for i in tqdm(range(len(files)), desc=f'path: {path}'):
            if files[i].split('.')[-1] != 'mp3': continue
            file = os.path.join(path, files[i])

            # 3.1 step - 音频转文本
            ru = voice2text(voice2text_pipe, file)
            # 3.2. step - 获取对应翻译 - 分析翻译内容 ｜ 分类 ｜ 删除没有文本的内容
            zh = ru_convert(ru2zh_piple, ru)
            en = ru_convert(ru2zh_piple, ru, 'translate to en')
            # 3.3 step - 过滤低信息密度数据
            file_continer.append({
                'ru': ru,
                'zh': zh,
                'en': en,
                'file': file,
                'ru_len': len(ru)
            })

        for i in tqdm(range(len(file_continer)), desc='file_continer'):
            f = file_continer[i]
            if f['ru_len'] < feature_length / 300: 
                print(f"ru: {f['ru']} | zh: {f['zh']} - len: {f['ru_len']}")
                os.remove(f['file'])
            else: 
                # 目前考虑所有数据存入一个二进制文件
                ...
        
            
