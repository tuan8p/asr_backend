from scipy.io.wavfile import write
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time

def countdown_timer(duration):
    for remaining in range(duration, 0, -1):
        print(f"⏳ Còn lại: {remaining} giây", end='\r')
        time.sleep(1)
    print("⏳ Còn lại: 0 giây         ")

def is_valid_audio(file_path, min_rms=0.01, max_silence_ratio=0.8):
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono

    rms = np.sqrt(np.mean(audio**2))
    if rms < min_rms:
        print("⚠️ Âm lượng quá nhỏ.")
        return False

    frame_len = 2048
    hop_len = 512
    frame_count = 1 + (len(audio) - frame_len) // hop_len
    energies = np.array([
        np.sqrt(np.mean(audio[i*hop_len:i*hop_len+frame_len]**2))
        for i in range(frame_count)
    ])
    silence_ratio = np.mean(energies < min_rms)

    if silence_ratio > max_silence_ratio:
        print("⚠️ Quá nhiều khoảng im lặng.")
        return False

    return True

def preprocess_audio(file_path, processed_path="processed.wav", target_sr=16000):
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono

    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Chuẩn hóa
    audio = audio / np.max(np.abs(audio))

    sf.write(processed_path, audio, sr, subtype='PCM_16')
    print("✅ Đã tiền xử lý âm thanh.")
    return processed_path

def record_audio(filename="recorded.wav", duration=3, fs=16000):
    print(f"🎙️ Đang ghi âm trong {duration} giây...")

    timer_thread = threading.Thread(target=countdown_timer, args=(duration,))
    timer_thread.start()

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    timer_thread.join()

    write(filename, fs, recording)
    print("✅ Ghi âm xong.")
    # return filename

    start_time = time.time()

    # Kiểm tra chất lượng
    if is_valid_audio(filename):
        return preprocess_audio(filename), start_time
        # return filename
    else:
        print("❌ Âm thanh không đạt yêu cầu. Hãy thử lại.")
        return None