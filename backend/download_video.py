from yt_dlp import YoutubeDL
import os

def download_video_from_url(url):
    options = {
        'outtmpl': 'F:/multimodal_video_summarization/backend/%(title)s.%(ext)s',  
        'format': 'bestvideo+bestaudio/best',  
        'merge_output_format': 'mp4',  
    }   
    
    with YoutubeDL(options) as ydl:
        info_dict = ydl.extract_info(url, download=True)  
        file_path = ydl.prepare_filename(info_dict)  
    
    base, ext = os.path.splitext(file_path)
    if ext.lower() != '.mp4':  
        new_file_path = base + '.mp4'
        os.rename(file_path, new_file_path)
        return new_file_path

    return file_path


