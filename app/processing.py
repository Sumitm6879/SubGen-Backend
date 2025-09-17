import os
import math
import warnings
from dotenv import load_dotenv
import numpy as np
import json

# Core Libraries
import pysrt
import torch
import whisper
from moviepy import VideoFileClip
from pyannote.audio import Pipeline

# Feature Libraries
from better_profanity import profanity
from google.cloud import translate_v2 as translate
import google.generativeai as genai

# --- INITIAL SETUP & CONFIGURATION ---
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

# API Configuration
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# --- FEATURE FUNCTIONS ---

# In processing.py

def analyze_transcript_for_bias(full_transcript, output_path):
    """
    Uses an LLM to analyze the full transcript for potential biases AND returns the content.
    """
    print("Analyzing transcript for potential bias...")
    report_content = "Bias analysis was skipped because the Google API Key was not found."
    if not GOOGLE_API_KEY:
        print("Google API Key not found. Skipping bias analysis.")
        return report_content

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
        report_content = response.text # Capture the content

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("AI-Generated Bias and Content Analysis Report\n")
            f.write("="*40 + "\n\n")
            f.write(report_content)

        print(f"Bias analysis report saved to: {output_path}")
        return report_content # <-- ADD THIS LINE

    except Exception as e:
        error_message = f"An error occurred during bias analysis: {e}"
        print(error_message)
        return error_message

def translate_text_batch(texts, target_language):
    """
    Translates a list of texts to the target language using the Google Cloud Translation API.
    """
    print(f"Translating a batch of {len(texts)} segments to {target_language}...")
    try:
        translate_client = translate.Client()
        results = translate_client.translate(texts, target_language=target_language)
        return [result['translatedText'] for result in results]
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return texts


# --- CORE AI AND DATA PROCESSING FUNCTIONS ---

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

def create_subtitle_segments(words, max_chars=80, max_words=10):
    """
    Groups words into readable subtitle segments and calculates the average
    confidence score for each new segment.
    """
    segments = []
    if not words:
        return segments
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
    """
    Generates an SRT file with optional features, using the reliable Cloud Translation API.
    """
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

def seconds_to_srt_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - math.floor(seconds)) * 1000)
    return pysrt.SubRipTime(hrs, mins, secs, millis)


# --- THE MAIN CALLABLE ORCHESTRATION FUNCTION ---

def process_video_logic(video_path, options, output_dir, unique_id, base_url):
    """
    This function orchestrates the video processing pipeline AND yields
    JSON-formatted progress updates for Server-Sent Events.
    """
    base_name = unique_id
    audio_path = os.path.join(output_dir, f"{base_name}.wav")
    result_files = {}

    try:
        # --- Step 1: Audio Extraction ---
        yield json.dumps({"status": "Extracting audio..."})
        extract_audio(video_path, audio_path)

        # --- Step 2: Transcription ---
        yield json.dumps({"status": "Transcribing audio (this may take a moment)..."})
        transcription_segments = transcribe_audio(audio_path)
        if not transcription_segments:
            yield json.dumps({"status": "error", "message": "Transcription failed. The video may be silent or too short."})
            return

        # --- Step 3: Speaker Diarization (Conditional) ---
        speaker_turns = [] # <-- FIX: Initialize speaker_turns as an empty list
        if options.get("diarize_srt"):
            yield json.dumps({"status": "Identifying speakers..."})
            speaker_turns = diarize_audio(audio_path)

        # --- Step 4: Bias Analysis (Conditional) ---
        if options.get("analyze_bias"):
            yield json.dumps({"status": "Analyzing content for bias..."})
            full_transcript_text = " ".join([seg['text'].strip() for seg in transcription_segments])
            report_path = os.path.join(output_dir, f"{base_name}_bias_report.txt")
            report_content = analyze_transcript_for_bias(full_transcript_text, report_path)
            result_files['bias_report'] = {'path': report_path, 'content': report_content}

        # --- Step 5: Formatting Subtitles ---
        yield json.dumps({"status": "Formatting subtitles..."})
        all_words = []
        for seg in transcription_segments:
            avg_logprob = seg['avg_logprob']
            for word_info in seg.get('words', []):
                word_info['logprob'] = avg_logprob
                all_words.append(word_info)

        if not all_words:
            yield json.dumps({"status": "error", "message": "No words found in transcription."})
            return

        if options.get("diarize_srt"):
            for word in all_words:
                for turn in speaker_turns:
                    if turn['start'] <= word['start'] < turn['end']:
                        word['speaker'] = turn['speaker']
                        break
                if 'speaker' not in word:
                    word['speaker'] = "UNKNOWN"
            # Cleanup UNKNOWNs only if we are diarizing
            for i, word in enumerate(all_words):
                if word['speaker'] == "UNKNOWN":
                    if i + 1 < len(all_words) and all_words[i+1]['speaker'] != "UNKNOWN":
                        word['speaker'] = all_words[i+1]['speaker']
                    elif i - 1 >= 0 and all_words[i-1]['speaker'] != "UNKNOWN":
                        word['speaker'] = all_words[i-1]['speaker']
        else:
            # If not diarizing, assign a default speaker to all words
            for word in all_words:
                word['speaker'] = "SPEAKER_00"

        final_segments = create_subtitle_segments(all_words)

        # --- Step 6: Generating Final Files ---
        yield json.dumps({"status": "Generating final files..."})

        # Always generate the standard SRT file
        std_path = os.path.join(output_dir, f"{base_name}_standard_en.srt")
        generate_srt_file(std_path, final_segments, enable_diarization=False, enable_censoring=options['censor'], target_language=None, enable_confidence_colors=options['show_confidence'])
        result_files['standard_srt'] = std_path

        # Only generate the diarized SRT if requested
        if options.get("diarize_srt"):
            diarized_path = os.path.join(output_dir, f"{base_name}_diarized_en.srt")
            generate_srt_file(diarized_path, final_segments, enable_diarization=True, enable_censoring=options['censor'], target_language=None, enable_confidence_colors=options['show_confidence'])
            result_files['diarized_srt'] = diarized_path

        # Only generate the translated SRT if requested
        if options.get('translate_to'):
            lang_code = options['translate_to'].lower()[:2]
            translated_path = os.path.join(output_dir, f"{base_name}_diarized_{lang_code}.srt")
            # Translation implies diarization for context, so enable_diarization is true
            generate_srt_file(translated_path, final_segments, enable_diarization=True, enable_censoring=options['censor'], target_language=options['translate_to'], enable_confidence_colors=False)
            result_files['translated_srt'] = translated_path

        # --- Final Step: Prepare and yield the complete results ---
        final_response = {}
        for key, value in result_files.items():
            if key == 'bias_report' and isinstance(value, dict):
                final_response['bias_report_url'] = f"{base_url}/outputs/{os.path.basename(value['path'])}"
                final_response['bias_report_content'] = value['content']
            else:
                final_response[f'{key}_url'] = f"{base_url}/outputs/{os.path.basename(value)}"

        yield json.dumps({"status": "complete", "results": final_response})

    except Exception as e:
        print(f"Error during processing logic: {e}")
        yield json.dumps({"status": "error", "message": str(e)})
# def process_video_logic(video_path, options, output_dir, unique_id):
#     """
#     This function orchestrates the entire video processing pipeline
#     and returns a dictionary of output file paths.
#     """
#     base_name = unique_id
#     audio_path = os.path.join(output_dir, f"{base_name}.wav")

#     result_files = {}

#     # Step 1: Core Processing
#     extract_audio(video_path, audio_path)
#     transcription_segments = transcribe_audio(audio_path)

#     if not transcription_segments:
#         print("Transcription yielded no segments. Aborting.")
#         return result_files

#     speaker_turns = diarize_audio(audio_path)

#     # Step 2: Assemble Data and Run Analysis
#     full_transcript_text = " ".join([seg['text'].strip() for seg in transcription_segments])

#     if options.get("analyze_bias"):
#         report_path = os.path.join(output_dir, f"{base_name}_bias_report.txt")
#         # --- CHANGE: Capture the returned content ---
#         report_content = analyze_transcript_for_bias(full_transcript_text, report_path)
#         # --- CHANGE: Store both path and content ---
#         result_files['bias_report'] = {'path': report_path, 'content': report_content}

#     # Step 3: Refine Word-level Data
#     all_words = []
#     for seg in transcription_segments:
#         avg_logprob = seg['avg_logprob']
#         for word_info in seg.get('words', []):
#             word_info['logprob'] = avg_logprob
#             all_words.append(word_info)

#     if not all_words:
#         print("No words found in transcription. Exiting.")
#         return result_files

#     for word in all_words:
#         for turn in speaker_turns:
#             if turn['start'] <= word['start'] < turn['end']:
#                 word['speaker'] = turn['speaker']
#                 break
#         if 'speaker' not in word:
#             word['speaker'] = "UNKNOWN"

#     for i, word in enumerate(all_words):
#         if word['speaker'] == "UNKNOWN":
#             if i + 1 < len(all_words) and all_words[i+1]['speaker'] != "UNKNOWN":
#                 word['speaker'] = all_words[i+1]['speaker']
#             elif i - 1 >= 0 and all_words[i-1]['speaker'] != "UNKNOWN":
#                 word['speaker'] = all_words[i-1]['speaker']

#     final_segments = create_subtitle_segments(all_words)

#     # Step 4: Generate All Requested Output Files
#     print("\n--- Generating output files based on options ---")

#     std_path = os.path.join(output_dir, f"{base_name}_standard_en.srt")
#     generate_srt_file(std_path, final_segments, enable_diarization=False, enable_censoring=options['censor'], target_language=None, enable_confidence_colors=options['show_confidence'])
#     result_files['standard_srt'] = std_path

#     diarized_path = os.path.join(output_dir, f"{base_name}_diarized_en.srt")
#     generate_srt_file(diarized_path, final_segments, enable_diarization=True, enable_censoring=options['censor'], target_language=None, enable_confidence_colors=options['show_confidence'])
#     result_files['diarized_srt'] = diarized_path

#     if options['translate_to']:
#         lang_code = options['translate_to'].lower()[:2]
#         translated_path = os.path.join(output_dir, f"{base_name}_diarized_{lang_code}.srt")
#         generate_srt_file(translated_path, final_segments, enable_diarization=True, enable_censoring=options['censor'], target_language=options['translate_to'], enable_confidence_colors=False)
#         result_files['translated_srt'] = translated_path

#     print("\nFile generation complete.")
#     return result_files