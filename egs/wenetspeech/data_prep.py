"""prepare wenetspeech, transform it to aishell1 architecture
"""

import json
from tqdm import tqdm
import multiprocessing
import torchaudio.backend.sox_io_backend as sox
import torchaudio
import re
import os
import sys

def write_wav(audio):
    audio_path = audio["path"]
    aid = audio["aid"]
    segments = audio["segments"]
    # confirm if contains any target partition
    very_start = sys.maxsize
    last_end = -1
    filtered_segments = []
    for segment in segments:
        part = segment["subsets"]
        text = segment["text"]
        if len(set(["M", "DEV", "TEST_NET"]) & set(part)) >0 and len(re.sub("[\u4e00-\u9fa5]+", "", text)) == 0:
            start= segment["begin_time"]
            start= int(start * 48000)
            end = segment["end_time"]
            end = int(end * 48000)
            very_start = min(very_start, start)
            last_end = max(last_end, end)
            filtered_segments.append(segment)
    texts = []
    if last_end != -1:
        # waveforms, sample_rate = sox.load("/home/keshawnhsieh/data/wenetspeech/" + audio_path, normalize=False)
        waveforms, sample_rate = torchaudio.load("/home/keshawnhsieh/data/wenetspeech/" + audio_path, normalize=False, frame_offset=very_start, num_frames=last_end- very_start)

        for segment in filtered_segments:
            start = segment["begin_time"]
            end = segment["end_time"]
            sid = segment["sid"]
            # print("process", sid)
            text = segment["text"]

            start = int(start * sample_rate) -very_start
            end = int(end * sample_rate) - very_start
            audio = waveforms[:1, start:end]

            audio = torchaudio.transforms.Resample(
                sample_rate, 16000)(audio)

            part = segment["subsets"]
            if "DEV" in part:
                sub_dir = "dev"
            elif "TEST_NET" in part:# or "TEST_MEETING" in part:
                sub_dir= "test"
            elif "M" in part:
                sub_dir = "train"
            else:
                continue
            if not os.path.exists(f"/home/keshawnhsieh/data/wenetspeech_wav/data_aishell/wav/{sub_dir}/{aid}"):
                os.system(f"mkdir -p /home/keshawnhsieh/data/wenetspeech_wav/data_aishell/wav/{sub_dir}/{aid}")
            save_path = f"/home/keshawnhsieh/data/wenetspeech_wav/data_aishell/wav/{sub_dir}/{aid}/{sid}.wav"
            sox.save(save_path, audio, 16000, format="wav", bits_per_sample=16)
            text = sid + " " + text
            texts.append(text)
            # print(text)
    return texts

def write_wav2(audio):
    """
    在segment循环里面读opus的start：end段
    """
    audio_path = audio["path"]
    aid = audio["aid"]
    segments = audio["segments"]
    # confirm if contains any target partition
    filtered_segments = []
    for segment in segments:
        part = segment["subsets"]
        text = segment["text"]
        if len(set(["M", "DEV", "TEST_NET"]) & set(part)) >0 and len(re.sub("[\u4e00-\u9fa5]+", "", text)) == 0:
            filtered_segments.append(segment)
    if filtered_segments:
        # waveforms, sample_rate = sox.load("/home/keshawnhsieh/data/wenetspeech/" + audio_path, normalize=False)

        texts = []
        for segment in filtered_segments:
            start = segment["begin_time"]
            end = segment["end_time"]
            sid = segment["sid"]
            print("process", sid)
            text = segment["text"]
            waveforms, sample_rate = torchaudio.load("/home/keshawnhsieh/data/wenetspeech/" + audio_path, normalize=False,
                                                     frame_offset=int(start*48000), num_frames=int((end- start) * 48000))
            audio=waveforms
            audio = torchaudio.transforms.Resample(
                sample_rate, 16000)(audio)

            part = segment["subsets"]
            if "DEV" in part:
                sub_dir = "dev"
            elif "TEST_NET" in part:# or "TEST_MEETING" in part:
                sub_dir= "test"
            elif "M" in part:
                sub_dir = "train"
            else:
                continue
            if not os.path.exists(f"/home/keshawnhsieh/data/wenetspeech_wav/data_aishell/wav/{sub_dir}/{aid}"):
                os.system(f"mkdir -p /home/keshawnhsieh/data/wenetspeech_wav/data_aishell/wav/{sub_dir}/{aid}")
            save_path = f"/home/keshawnhsieh/data/wenetspeech_wav/data_aishell/wav/{sub_dir}/{aid}/{sid}.wav"
            sox.save(save_path, audio, 16000, format="wav", bits_per_sample=16)
            text = sid + " " + text
            texts.append(text)
            # print(text)
    return texts

if __name__ == "__main__":
    with open("/home/keshawnhsieh/data/wenetspeech/WenetSpeech.json", "r") as f:
        wenet_speech_js = json.load(f)
    # with open("/home/keshawnhsieh/data/wenetspeech/WenetSpeech.json1", "w") as f:
    #     wenet_speech_js["audios"] = wenet_speech_js["audios"][:1]
    #     json.dump(wenet_speech_js, f)

    audios = wenet_speech_js["audios"]
    print("all audios", len(audios))

    # search all audios and count the number of M & dev and test_net
    count =0
    for audio in tqdm(audios):
        segments = audio["segments"]
        for segment in segments:
            part= segment["subsets"]
            text = segment["text"]
            if len(set(["M", "DEV", "TEST_NET"]) & set(part)) > 0 and len(re.sub("[\u4e00-\u9fa5]+", "", text)) == 0:
                count +=1
                break
    print("M & dev & test_net", count)

    # print(audios)
    # results = write_wav(audios[0])
    pool = multiprocessing.Pool()
    import time
    st= time.time()
    results = pool.map(write_wav, audios)
    et = time.time()
    print("cost", et -st)
    pool.close()
    results2 = []
    for result in results:
        results2.extend(result)
    os.system(f"mkdir -p /home/keshawnhsieh/data/wenetspeech_wav/data_aishell/transcript")
    with open("/home/keshawnhsieh/data/wenetspeech_wav/data_aishell/transcript/aishell_transcript_v0.8.txt", "w") as f:
        for result in results2:
            f.write(result + "\n")

