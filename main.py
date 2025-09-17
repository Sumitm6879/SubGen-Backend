import os
import math
import warnings
from dotenv import load_dotenv
import numpy as np

# Core Libraries
import pysrt
import torch
import whisper
from moviepy import VideoFileClip
from pyannote.audio import Pipeline

# New and Updated Feature Libraries
from better_profanity import profanity
from google.cloud import translate_v2 as translate
import google.generativeai as genai # Import Gemini for analysis

# --- INITIAL SETUP & CONFIGURATION ---
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

# API Configuration
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- NEW: BIAS ANALYSIS FUNCTION ---
def analyze_transcript_for_bias(full_transcript, output_path):
    """
    Uses an LLM to analyze the full transcript for potential biases.
    """
    print("Analyzing transcript for potential bias...")
    if not GOOGLE_API_KEY:
        print("Google API Key not found. Skipping bias analysis.")
        return

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        prompt = f"""
        As an expert in media studies and societal impact, analyze the following transcript for potential biases.
        Consider the following aspects:
        1.  **Stereotypes:** Are there generalizations about groups of people based on race, gender, age, etc.?
        2.  **Unbalanced Viewpoints:** Is one side of an argument presented more favorably without acknowledging others?
        3.  **Emotionally Charged Language:** Is loaded language used to sway the audience's opinion?
        4.  **Assumptions:** Are there unstated assumptions that may favor a particular group or perspective?

        Provide a brief, neutral summary of your findings. If no significant biases are found, please state that.
        Do not express personal opinions; provide an objective analysis based on the text provided.

        Transcript to Analyze:
        ---
        {full_transcript}
        ---
        """

        response = model.generate_content(prompt)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("AI-Generated Bias and Content Analysis Report\n")
            f.write("="*40 + "\n\n")
            f.write(response.text)

        print(f"Bias analysis report saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred during bias analysis: {e}")


# --- All other functions (extract_audio, etc.) remain the same ---
def extract_audio(video_path, audio_output_path):
    print(f"Extracting audio from {video_path}...")
    with VideoFileClip(video_path) as video:
        video.audio.write_audiofile(audio_output_path, codec='pcm_s16le')
    print(f"Audio successfully extracted to: {audio_output_path}")

def transcribe_audio(audio_path):
    print("Transcribing audio with Whisper...")
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path, word_timestamps=True)
    print("Transcription complete.")
    return result.get('segments', [])

def diarize_audio(audio_path):
    print("Performing speaker diarization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_TOKEN)
    pipeline.to(device)
    diarization = pipeline(audio_path)
    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
    print("Diarization complete.")
    return speaker_turns

def translate_text_batch(texts, target_language):
    print(f"Translating a batch of {len(texts)} segments to {target_language}...")
    try:
        translate_client = translate.Client()
        results = translate_client.translate(texts, target_language=target_language)
        return [result['translatedText'] for result in results]
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return texts

def seconds_to_srt_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - math.floor(seconds)) * 1000)
    return pysrt.SubRipTime(hrs, mins, secs, millis)

def create_subtitle_segments(words, max_chars=80, max_words=10):
    segments = []
    if not words: return segments
    current_segment = {'text': words[0]['word'], 'start': words[0]['start'], 'end': words[0]['end'], 'speaker': words[0]['speaker'], 'logprobs': [words[0]['logprob']]}
    for i in range(1, len(words)):
        word = words[i]
        text_len = len(current_segment['text'])
        word_count = len(current_segment['text'].split())
        speaker_changed = word['speaker'] != current_segment['speaker']
        pause_is_long = (word['start'] - current_segment['end']) >= 1.0
        text_is_long = text_len >= max_chars
        too_many_words = word_count >= max_words
        if speaker_changed or pause_is_long or text_is_long or too_many_words:
            current_segment['avg_logprob'] = np.mean(current_segment['logprobs'])
            segments.append(current_segment)
            current_segment = {'text': word['word'], 'start': word['start'], 'end': word['end'], 'speaker': word['speaker'], 'logprobs': [word['logprob']]}
        else:
            current_segment['text'] += f" {word['word']}"
            current_segment['end'] = word['end']
            current_segment['logprobs'].append(word['logprob'])
    current_segment['avg_logprob'] = np.mean(current_segment['logprobs'])
    segments.append(current_segment)
    return segments

def generate_srt_file(output_path, segments, enable_diarization, enable_censoring, target_language=None, enable_confidence_colors=False):
    if target_language:
        original_texts = [seg['text'].strip() for seg in segments]
        translated_texts = translate_text_batch(original_texts, target_language)
        for i, seg in enumerate(segments):
            seg['translated_text'] = translated_texts[i]
    subs = pysrt.SubRipFile()
    for i, seg in enumerate(segments):
        text = seg.get('translated_text', seg['text'].strip())
        if enable_censoring:
            text = profanity.censor(text)
        if enable_diarization:
            text = f"[{seg['speaker']}]: {text}"
        if enable_confidence_colors:
            confidence_percent = math.exp(seg['avg_logprob']) * 100
            if confidence_percent > 85: color = "#00C853"
            elif confidence_percent > 60: color = "#FFAB00"
            else: color = "#D50000"
            text = f'<font color="{color}">{text}</font>'
        sub = pysrt.SubRipItem(index=i + 1, start=seconds_to_srt_time(seg['start']), end=seconds_to_srt_time(seg['end']), text=text)
        subs.append(sub)
    subs.save(output_path, encoding='utf-8')
    print(f"Subtitle file saved as: {output_path}")

# --- UPDATED: process_video ---
def process_video(video_path, options):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = f"{base_name}.wav"

    extract_audio(video_path, audio_path)
    transcription_segments = transcribe_audio(audio_path)
    speaker_turns = diarize_audio(audio_path)

    # --- Assemble full transcript for bias analysis ---
    full_transcript_text = " ".join([seg['text'].strip() for seg in transcription_segments])

    # --- NEW: Call bias analysis if option is enabled ---
    if options.get("analyze_bias"):
        report_path = f"{base_name}_bias_report.txt"
        analyze_transcript_for_bias(full_transcript_text, report_path)

    # --- (The rest of the function remains the same) ---
    all_words = []
    for seg in transcription_segments:
        avg_logprob = seg['avg_logprob']
        for word_info in seg.get('words', []):
            word_info['logprob'] = avg_logprob
            all_words.append(word_info)
    if not all_words:
        print("No words found in transcription. Exiting.")
        return
    for word in all_words:
        for turn in speaker_turns:
            if turn['start'] <= word['start'] < turn['end']:
                word['speaker'] = turn['speaker']
                break
        if 'speaker' not in word:
            word['speaker'] = "UNKNOWN"
    for i, word in enumerate(all_words):
        if word['speaker'] == "UNKNOWN":
            if i + 1 < len(all_words) and all_words[i+1]['speaker'] != "UNKNOWN":
                word['speaker'] = all_words[i+1]['speaker']
            elif i - 1 >= 0 and all_words[i-1]['speaker'] != "UNKNOWN":
                word['speaker'] = all_words[i-1]['speaker']
    final_segments = create_subtitle_segments(all_words)
    print("\n--- Generating SRT files based on options ---")

    std_path = f"{base_name}_standard_en_confidence.srt"
    generate_srt_file(std_path, final_segments, enable_diarization=False, enable_censoring=options['censor'], target_language=None, enable_confidence_colors=options['show_confidence'])

    diarized_path = f"{base_name}_diarized_en_confidence.srt"
    generate_srt_file(diarized_path, final_segments, enable_diarization=True, enable_censoring=options['censor'], target_language=None, enable_confidence_colors=options['show_confidence'])

    if options['translate_to']:
        lang_code = options['translate_to'].lower()[:2]
        translated_path = f"{base_name}_diarized_{lang_code}.srt"
        generate_srt_file(translated_path, final_segments, enable_diarization=True, enable_censoring=options['censor'], target_language=options['translate_to'], enable_confidence_colors=False)

    print("\nProcessing complete.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":

    # --- ⚙️ USER CONFIGURATION ---
    VIDEO_FILE = "sample4.mp4"

    PROCESSING_OPTIONS = {
        "censor": True,
        "translate_to": "en",
        "show_confidence": True,
        "analyze_bias": True # <-- New option to enable bias analysis
    }

    process_video(VIDEO_FILE, PROCESSING_OPTIONS)