# OpenAI Real-time WebRTC Demo

A real-time audio chat application using OpenAI's realtime audio API with WebRTC. Ask about the weather in any location and get real-time responses using Open-Meteo API.

## Features

- Real-time audio streaming
- Live transcription
- Weather function integration
- WebRTC communication
- FastAPI backend to get a ephemeral session token and to get weather data

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: 
   - Windows: `.venv\Scripts\activate`
   - Unix/macOS: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.env` file with your OpenAI API key:   

```bash
OPENAI_API_KEY=your-key-here
```

## Running

1. Start server: `python app.py`
2. Open index.html in a browser (Tip: use live server extension for VSCode)
3. Click Start and allow microphone access
4. Try asking: "What's the weather like in Amsterdam?"

## Files

- app.py: FastAPI backend server
- index.html: Frontend interface
- requirements.txt: Python dependencies
- test.http: API endpoint tests
- .env: Environment variables (create this)

## Notes

- For development use only
- Never commit your .env file
- Requires OpenAI API key
- Uses Open-Meteo API for weather data