import os
import sys
import json
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

import ffmpeg
from tqdm import tqdm
from transformers import pipeline, Pipeline
from pydub import AudioSegment
from moviepy.video.io.VideoFileClip import VideoFileClip


def media_seg_ffmpeg(ori_path: str, save_path: str, segment_duration: int = 30, flag: bool=True):
    """
    使用 ffmpeg 分割视频,音频文件。
    
    Args:
        ori_path (str): 输入视频文件路径。
        save_path (str): 输出分割视频的保存路径。
        segment_duration (int): 每段视频的时长（秒）。
        flag (bool): 如果为 True，已存在的分割文件会被跳过。
        is_audio: 如果为true则是处理音频文件
    """
    base_name = os.path.basename(ori_path).split('.')[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    probe = ffmpeg.probe(ori_path, v="error", select_streams="v", show_entries="stream=duration")
    total_duration = float(probe['streams'][0]['duration'])

    
    video_metainfo = {
        "start": 0,
        "end": 0,
        "title": probe['format']['tags']['title'],
        "artist": probe['format']['tags']['artist'],
        "date": probe['format']['tags']['date'],
        "url": probe['format']['tags']['comment'],
        "description": probe['format']['tags']['comment']
    }

    # 计算分割的次数
    seg_times = int(total_duration // segment_duration)

    for i in tqdm(range(seg_times), desc=f"media_seg - path: {ori_path} -> {save_path} | duration:{total_duration}s"):
        export_path = os.path.join(save_path, base_name)
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        
        audio_export_name = os.path.join(export_path, f"{i+1}.mp3")
        video_export_name = os.path.join(export_path, f"{i+1}.mp4")
        info_export_name = os.path.join(export_path, f"{i+1}.json")
        
        # 如果文件已经存在并且 flag 为 True，则跳过
        if os.path.exists(video_export_name) and os.path.exists(audio_export_name) and flag:
            continue
        
        # 使用 ffmpeg 切割视频
        ffmpeg.input(ori_path, ss=i * segment_duration, t=segment_duration).output(audio_export_name, audio_bitrate='192k', acodec='libmp3lame').run()
        ffmpeg.input(ori_path, ss=i * segment_duration, t=segment_duration).output(video_export_name).run()
        
        video_metainfo['start'] = i * segment_duration
        video_metainfo['end'] = video_metainfo['start'] + segment_duration
        with open(info_export_name, 'w', encoding='utf-8') as f:
            json.dump(video_metainfo, f, ensure_ascii=False, indent=4)

    return video_metainfo

def voice2text(pipe: Pipeline, path: str):
    return pipe(path)['text']

def ru_convert(pipe: Pipeline, content: str, prefix = 'translate to zh') -> str:
    return pipe(f"{prefix}: {content}")[0]['translation_text']

if __name__ == '__main__':
    # 音频文件的路径 - TODO 后期需要自动化，从txt中读取对应的视频的链接，自动下载音频和视频
    media_path = sys.argv[1]
    save_path = sys.argv[2]
    segment_duration = int(sys.argv[3])
    

    # voice2text_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")
    # ru2zh_piple = pipeline("translation", model="utrobinmv/t5_translate_en_ru_zh_small_1024")
    

    # 1. step - 分割音频 | 视频 同时处理
    media_info = media_seg_ffmpeg(media_path, save_path, segment_duration) 
    print(f"media_info: {media_info}")

    # 2. step - 循环处理所有文件
    file_continer = []
    for path, _ ,files in os.walk(save_path):
        for i in tqdm(range(len(files)), desc=f'path: {path}'):
            if files[i].split('.')[-1] != 'mp3': continue
            file = os.path.join(path, files[i])

            # 2.1 step - 音频转文本
            ru = voice2text(voice2text_pipe, file)
            # 2.2. step - 获取对应翻译 - 分析翻译内容 ｜ 分类 ｜ 删除没有文本的内容
            zh = ru_convert(ru2zh_piple, ru)
            en = ru_convert(ru2zh_piple, ru, 'translate to en')
            # 2.3 step - 过滤低信息密度数据
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
        