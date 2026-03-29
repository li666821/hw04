import whisper
import sounddevice as sd
import numpy as np
import queue

class RealTimeASR:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)
        self.q = queue.Queue()
        self.sample_rate = 16000
        
    def callback(self, indata, frames, time, status):
        """音频回调函数"""
        if status:
            print(status, flush=True)
        self.q.put(indata.copy())
    
    def recognize(self, duration=10):
        """实时识别"""
        print(f"开始录音，持续{duration}秒...")
        
        with sd.InputStream(callback=self.callback, channels=1, samplerate=self.sample_rate):
            audio_data = []
            for i in range(0, int(self.sample_rate * duration), 1024):
                if not self.q.empty():
                    audio_chunk = self.q.get()
                    audio_data.append(audio_chunk)
            
        audio = np.concatenate(audio_data, axis=0).flatten()
        audio = (audio * 32768).astype(np.int16)
        
        result = self.model.transcribe(audio)
        return result["text"]

if __name__ == "__main__":
    asr = RealTimeASR(model_name="base")
    text = asr.recognize(duration=10)
    print("识别结果:")
    print(text)
