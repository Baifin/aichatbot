import os
import tempfile
import pygame
import speech_recognition as sr
from gtts import gTTS
from langdetect import detect, LangDetectException
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import threading
import whisper

# ------------------- Configuration -------------------

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'ta': 'Tamil',
    'ml': 'Malayalam',
    'hi': 'Hindi'
}

SPEECH_RECOGNITION_LANGS = {
    'en': 'en-US',
    'ta': 'ta-IN',
    'ml': 'ml-IN',
    'hi': 'hi-IN'
}

BOT_INTRO = {
    'en': "My name is Voice Mate, developed by Team Aura as a SaaS customer service bot for all types of businesses and organizations.",
    'ta': "என் பெயர் Voice Mate, இது Team Aura உருவாக்கியது, அனைத்து நிறுவனங்களுக்கும் வணிகங்களுக்கும் SaaS வாடிக்கையாளர் சேவை உதவியாளராக இருக்கிறது.",
    'hi': "मेरा नाम Voice Mate है, जिसे Team Aura द्वारा सभी प्रकार के व्यवसायों और संगठनों के लिए एक SaaS ग्राहक सेवा बॉट के रूप में विकसित किया गया है।",
    'ml': "എന്റെ പേര് Voice Mate ആണ്, ഇത് Team Aura വികസിപ്പിച്ചിരിക്കുന്നത്, എല്ലാ ബിസിനസ്സുകൾക്കും സ്ഥാപനങ്ങൾക്കും വേണ്ടി ഉള്ള ഒരു SaaS കസ്റ്റമർ സർവീസ് ബോട്ടാണ്."
}

pygame.mixer.init()
app = Flask(__name__)
CORS(app)

# Load Whisper model
whisper_model = whisper.load_model("base")
userdata = {}

# ------------------- Utilities -------------------

def speak_text(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path)
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        os.remove(temp_path)
    except Exception as e:
        print(f"[TTS Error]: {e}")

def generate_response(prompt, lang_code, name=None, issue=None):
    language_name = SUPPORTED_LANGUAGES.get(lang_code, 'English')
    personalization = ""
    if name:
        personalization += f"The user's name is {name}. "
    if issue:
        personalization += f"The user is dealing with {issue}. "

    # Ensure the response is generated in the language detected
    system_prompt = (
        f"You are a helpful educational assistant bot also teacher. "
        f"Support queries on attendance, results, GPA, assignments, fees, library, study materials, mental health, events, and exams. "
        f"Help students, teachers, and admins. "
        f"{personalization} "
        f"Please respond in hindi,english,tamil,malayalam only, matching the language of the user's input."
    )

    headers = {
        "Authorization": "Bearer d550d999cd8cdcf05ecbc03188d7326f51c369ae45fc7f45c5144c59a63ed046",  # Replace with your actual API key
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Llama-3-8b-chat-hf",  # Ensure this model supports multilingual responses
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"[HTTP Error]: {e}")
        return "I'm sorry, I couldn't process your request right now."

# ------------------- Routes -------------------

@app.route('/process_input', methods=['POST'])
def process_input():
    data = request.get_json()
    user_input = data.get("user_input")
    voice_enabled = data.get("VoiceEnabled", False)

    try:
        lang_code = detect(user_input)
    except LangDetectException:
        lang_code = 'en'

    if lang_code not in SUPPORTED_LANGUAGES:
        lang_code = 'en'

    # Name extraction
    extracted_name = None
    if "my name is" in user_input.lower():
        extracted_name = user_input.lower().split("my name is")[-1].strip().split()[0]
        userdata["name"] = extracted_name

    # Issue extraction
    keywords = [
        "i have", "i am dealing with", "i’m dealing with",
        "i have been diagnosed with", "suffering with",
        "i am feeling", "i'm feeling"
    ]
    for phrase in keywords:
        if phrase in user_input.lower():
            issue = user_input.lower().split(phrase)[-1].strip()
            userdata["issue"] = issue
            break

    response = generate_response(user_input, lang_code, userdata.get("name"), userdata.get("issue"))
    response_data = {"message": response}
    if voice_enabled:
        threading.Thread(target=speak_text, args=(response, lang_code)).start()

    return jsonify(response_data)

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_file.save(temp_audio.name)
        audio_path = temp_audio.name

    try:
        result = whisper_model.transcribe(audio_path)
        text = result["text"]
        lang_code = result.get("language", "en")
        print(f"[Whisper] Transcribed text: {text}")
        print(f"[Whisper] Detected language: {lang_code}")

        if lang_code not in SUPPORTED_LANGUAGES:
            lang_code = "en"

        response = generate_response(text, lang_code, userdata.get("name"), userdata.get("issue"))
        threading.Thread(target=speak_text, args=(response, lang_code)).start()

        return jsonify({
            "transcription": text,
            "message": response,
            "lang_code": lang_code
        })

    except Exception as e:
        print(f"[Whisper Error]: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(audio_path)

@app.route('/change_language', methods=['POST'])
def change_language():
    data = request.get_json()
    user_input = data.get("user_input", "")
    manual_lang_map = {
        "talk in tamil": "ta",
        "speak tamil": "ta",
        "தமிழ்ல பேசு": "ta",
        "talk in hindi": "hi",
        "speak hindi": "hi",
        "हिंदी में बोलो": "hi",
        "talk in malayalam": "ml",
        "speak malayalam": "ml",
        "മലയാളത്തിൽ conversar": "ml"
    }

    lang_code = "en"
    if user_input.lower() in manual_lang_map:
        lang_code = manual_lang_map[user_input.lower()]
        print(f"[Manual Switch] Language changed to {SUPPORTED_LANGUAGES.get(lang_code)}")
        return jsonify({
            "message": f"Language changed to {SUPPORTED_LANGUAGES.get(lang_code)}",
            "lang_code": lang_code
        })

    return jsonify({
        "message": "Language not changed. Command not recognized.",
        "lang_code": lang_code
    })

# ------------------- Entry Point -------------------

if __name__ == "__main__":
    app.run(debug=True)
