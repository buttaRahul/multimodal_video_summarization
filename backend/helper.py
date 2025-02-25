

import easyocr
import ffmpeg
import torch
from transformers import pipeline
from langchain.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import re
from keybert import KeyBERT
import stopwordsiso as stopwords
import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from download_video import download_video_from_url






TRANSLATION_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",  # Hindi to English
    "fr": "Helsinki-NLP/opus-mt-fr-en",  # French to English
    "es": "Helsinki-NLP/opus-mt-es-en",  # Spanish to English
    "de": "Helsinki-NLP/opus-mt-de-en",  # German to English
}


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def extract_audio(video_path,audio_path):
    ffmpeg.input(video_path).output(audio_path, format='mp3', acodec='mp3').run()
    print("Audio extracted successfully!")


def generate_transcript(audio_path, transcription_model):
    transcribe = pipeline(
        task="automatic-speech-recognition",
        model=transcription_model, 
        chunk_length_s=5, 
        device=device
    )

    transcript_data = transcribe(
        audio_path, 
        return_timestamps=True,
        generate_kwargs={"language": None, "task": "transcribe"}
    )

    detected_language = transcript_data.get("language", "en")

    if detected_language != "en" and detected_language in TRANSLATION_MODELS:
        translation_model = TRANSLATION_MODELS[detected_language]
        translator = pipeline("translation", model=translation_model, device=device)
        translated_text = translator(transcript_data["text"])[0]["translation_text"]
        transcript_data["translated_text"] = translated_text

    return transcript_data



def remove_repeated_words(text):
    words = text.split()
    cleaned_words = []
    prev_word = None
    
    for word in words:
        if word != prev_word:
            cleaned_words.append(word)
        prev_word = word
    
    return ' '.join(cleaned_words)



def key_timestamps(transcript_data, num_timestamps=50):
    kw_model = KeyBERT()  

    transcript_text = transcript_data["text"]

    # Split sentences based on common sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', transcript_text.strip())

    # Extract keywords
    keywords = kw_model.extract_keywords(
        transcript_text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english',  
        top_n=num_timestamps
    )

    # Extract key sentences by checking if they contain the keywords
    key_sentences = [s for s in sentences if any(kw in s for kw, _ in keywords)]

    key_sentence_timestamps = {}

    for chunk in transcript_data["chunks"]:
        chunk_text = chunk["text"]
        chunk_timestamp = chunk["timestamp"]
        
        for key_sentence in key_sentences:
            if key_sentence in chunk_text:
                if key_sentence not in key_sentence_timestamps:
                    key_sentence_timestamps[key_sentence] = []
                
                key_sentence_timestamps[key_sentence].append(chunk_timestamp)


    print("Key Sentence Timestamps:", key_sentence_timestamps)
    print(len(key_sentence_timestamps))
    return key_sentence_timestamps



def extract_keyframes(video_path, key_sentence_timestamps):
    cap = cv2.VideoCapture(video_path)
    
    key_timestamps = []
    for timestamp in key_sentence_timestamps.values():
        if isinstance(timestamp, list) and len(timestamp) > 0 and isinstance(timestamp[0], tuple):
            key_timestamps.append(int(timestamp[0][0] * 1000))  

    key_timestamps = [ts for ts in key_timestamps if ts > 0]

    frames = []

    for ts in key_timestamps:
        captured_frames = []

        for offset in [-1500, 0, 1500]:  
            target_time = ts + offset
            if target_time <= 0:  
                continue
            
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                captured_frames.append(img_str)
        
        frames.extend(captured_frames)

    cap.release()
    return frames


def generate_frame_descriptions(image_caption_model, frames):
    blip_processor = BlipProcessor.from_pretrained(image_caption_model)
    blip_model = BlipForConditionalGeneration.from_pretrained(image_caption_model).to(device)

    frame_descriptions = []

    for frame in frames:
        # Decode the base64 frame into an image
        frame_bytes = base64.b64decode(frame)
        frame_image = Image.open(BytesIO(frame_bytes))

        # Process the image with BlipProcessor
        blip_inputs = blip_processor(frame_image, return_tensors="pt").to(device)
        blip_output = blip_model.generate(**blip_inputs)
        blip_caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)

        # Use EasyOCR to extract text from the frame
        reader = easyocr.Reader(['en', 'hi'])
        ocr_results = reader.readtext(np.array(frame_image), detail=0)
        ocr_text = " ".join(ocr_results)  

        # Combine the caption and OCR text
        full_caption = f"{blip_caption}. OCR Text: {ocr_text}"
        frame_descriptions.append(full_caption)

    return frame_descriptions


def generate_video_summary(video_path, audio_path, transcription_model, image_caption_model, llm):
    print("IN GENERATE VIDEO SUMMARY")
    extract_audio(video_path, audio_path)
    transcript_data = generate_transcript(audio_path, transcription_model)
    key_sentence_timestamps = key_timestamps(transcript_data)
    frames = extract_keyframes(video_path, key_sentence_timestamps)
    frame_descriptions = generate_frame_descriptions(image_caption_model, frames)

    prompt_template = PromptTemplate(
        input_variables=["transcribed_text", "frame_descriptions"],
        template="""
        You have a transcribed text from a video and key visual descriptions of extracted frames.  
        Your task is to generate a seamless, holistic summary that merges insights from both the text and visuals without explicitly distinguishing them.  

        Transcribed Text:
        {transcribed_text}

        Frame Descriptions:
        {frame_descriptions}

        Generate a well-structured summary that naturally integrates the key details from both sources into a cohesive narrative.  
        Start the summary with: "The video..."
        """
    )

    chain = prompt_template | llm | StrOutputParser()
    summary = chain.invoke({
        "transcribed_text": transcript_data['text'],
        "frame_descriptions": "\n".join(frame_descriptions)
    })
    
    final_summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)


    try:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception as e:
        print(f"Error deleting files: {e}")


    return {"summary": final_summary, "frames": frames}


def generate_response(url):
    print("IN GENERATE RESPONSE")
    video_path = download_video_from_url(url)
    audio_path = os.path.splitext(video_path)[0] + ".mp3"

    
    llm = ChatOllama(
        model="deepseek-r1:8b",
        temperature=0,
    )

    transcription_model = "openai/whisper-small"
    image_caption_model = "Salesforce/blip-image-captioning-large"

    final_summary = generate_video_summary(video_path,audio_path,transcription_model,image_caption_model,llm)

    return final_summary





# def main():
#     response = generate_response("https://www.youtube.com/watch?v=Y8HIFRPU6pM")
#     print(response)

# if __name__ == "__main__":
#     main()