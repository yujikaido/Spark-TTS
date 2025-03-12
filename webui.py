import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
import yaml
import gc
import numpy as np
from datetime import datetime
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

# Path to settings file
SETTINGS_FILE_PATH = "Configs/sparktts_settings.yaml"

# Global variable for model
model = None

# Configure logging to print to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_settings(settings):
    """Save settings to a YAML file."""
    with open(SETTINGS_FILE_PATH, "w") as f:
        yaml.safe_dump(settings, f)

def load_settings():
    """Load settings from a YAML file."""
    try:
        with open(SETTINGS_FILE_PATH, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

def unload_model():
    """Unload model and clear CUDA cache."""
    global model
    if model is not None:
        del model
        model = None
        torch.cuda.empty_cache()
        gc.collect()
        logging.info("Model unloaded and GPU memory cleared.")

def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model, ensuring previous models are unloaded first."""
    global model
    unload_model()
    logging.info(f"Loading model from: {model_dir}")
    device = torch.device(f"cuda:{device}")
    model = SparkTTS(model_dir, device)
    return model

def split_text(text, max_length=200):
    """Split long text into manageable chunks."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
):
    """Perform TTS inference on text chunks and concatenate results."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    final_output_path = os.path.join(save_dir, f"{timestamp}.wav")
    
    chunks = split_text(text)
    logging.info(f"Processing {len(chunks)} text chunks.")
    
    audios = []
    for i, chunk in enumerate(chunks):
        logging.info(f"[Chunk {i+1}/{len(chunks)}] Generating: {chunk}")
        with torch.no_grad():
            wav = model.inference(
                chunk,
                prompt_speech,
                prompt_text,
                gender,
                pitch,
                speed,
            )
            if not isinstance(wav, torch.Tensor):
                wav = torch.tensor(wav, dtype=torch.float32)  # Ensure conversion to Tensor
            audios.append(wav)
    
    # Concatenate audio chunks
    final_audio = torch.cat(audios, dim=-1)
    sf.write(final_output_path, final_audio.cpu().numpy(), samplerate=16000)
    logging.info(f"Audio saved at: {final_output_path}")
    return final_output_path, final_output_path  # Returning both for Gradio compatibility

def build_ui(model_dir, device=0):
    global model
    model = initialize_model(model_dir, device)
    
    def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record):
        """TTS voice cloning with chunking."""
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text_clean = None if len(prompt_text) < 2 else prompt_text
        return run_tts(text, model, prompt_text_clean, prompt_speech)
    
    def voice_creation(text, gender, pitch, speed):
        """TTS voice creation with chunking."""
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]
        return run_tts(text, model, gender=gender, pitch=pitch_val, speed=speed_val)
    
    with gr.Blocks() as demo:
        gr.HTML('<h1 style="text-align: center;">Spark-TTS with Chunking</h1>')
        with gr.Tabs():
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    prompt_wav_upload = gr.Audio(sources="upload", type="filepath", label="Upload Reference Audio")
                    prompt_wav_record = gr.Audio(sources="microphone", type="filepath", label="Record Prompt Audio")
                with gr.Row():
                    text_input = gr.Textbox(label="Text", lines=3)
                    prompt_text_input = gr.Textbox(label="Prompt Speech Text", lines=3)
                audio_output = gr.Audio(label="Generated Audio", autoplay=True)
                generate_button = gr.Button("Generate")
                generate_button.click(
                    voice_clone,
                    inputs=[text_input, prompt_text_input, prompt_wav_upload, prompt_wav_record],
                    outputs=[audio_output],
                )
            
            with gr.TabItem("Voice Creation"):
                with gr.Row():
                    gender = gr.Radio(choices=["male", "female"], value="male", label="Gender")
                    pitch = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Pitch")
                    speed = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Speed")
                text_input_creation = gr.Textbox(label="Input Text", lines=3)
                create_button = gr.Button("Create Voice")
                audio_output_creation = gr.Audio(label="Generated Audio", autoplay=True)
                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed],
                    outputs=[audio_output_creation],
                )
    return demo

def parse_arguments():
    parser = argparse.ArgumentParser(description="Spark-TTS Gradio server with Chunking.")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Spark-TTS-0.5B", help="Model directory.")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID.")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server host/IP.")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    demo = build_ui(model_dir=args.model_dir, device=args.device)
    demo.launch(server_name=args.server_name, server_port=args.server_port)
