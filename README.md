# AI Media Accessibility API

This project provides an API for processing video files to generate accessible media outputs such as subtitles, speaker diarization, bias analysis, and translation. It is built with FastAPI and integrates several AI models and cloud services.

## Features

- **Video Upload & Processing**: Upload video files and process them asynchronously.
- **Automatic Speech Recognition**: Uses Whisper for transcription.
- **Speaker Diarization**: Identifies speakers using pyannote.
- **Profanity Censoring**: Optionally censors profane words.
- **Bias Analysis**: Uses Gemini LLM to analyze transcript for bias (requires Google API key).
- **Subtitle Generation**: Generates SRT files with optional speaker labels and confidence coloring.
- **Translation**: Translates subtitles using Google Cloud Translation API.
- **Streaming Progress**: Real-time progress updates via Server-Sent Events (SSE).
- **CORS Support**: Ready for integration with React or other frontends.

## Setup

1. **Clone the repository**
   ```
   git clone https://github.com/Sumitm6879/SubGen-Backend.git
   cd SubGen-Backend
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   Create a `.env` file in the root directory and set the following (as needed):
   ```
   HUGGINGFACE_TOKEN=your_huggingface_token
   GOOGLE_API_KEY=your_google_api_key
   ```

4. **Google Cloud Setup**
   - For translation, set up Google Cloud credentials as described in [Google Cloud docs](https://cloud.google.com/translate/docs/setup).

## Running the API

```
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Endpoints

- `POST /upload`: Upload a video and options. Returns a job ID.
- `GET /stream/{job_id}`: Streams processing progress and results.
- `GET /outputs/{filename}`: Download generated files.
- `GET /`: Health check.

## Output Files

- Standard English SRT
- Diarized English SRT (if enabled)
- Translated SRT (if enabled)
- Bias analysis report (if enabled)

## Notes

- For production, consider using persistent storage and a job queue.
- Some features require GPU and cloud API keys.

## License

MIT License
