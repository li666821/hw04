import whisper
import argparse

def recognize_audio(file_path, model_name="base"):
    """识别音频文件"""
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path)
    return result["text"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音频识别")
    parser.add_argument("audio_file", help="音频文件路径")
    parser.add_argument("--model", default="base", help="模型名称: tiny, base, small, medium, large")
    args = parser.parse_args()
    
    text = recognize_audio(args.audio_file, args.model)
    print("识别结果:")
    print(text)
