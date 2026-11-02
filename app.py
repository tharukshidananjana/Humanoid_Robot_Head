import streamlit as st
import torch
import time
from transformers import pipeline

# Page Configuration
st.set_page_config(page_title="Silicon Smooth Animated Robot Face", layout="centered")

# Load Hugging Face Pre-trained Emotion Detection Model
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

with st.spinner("Hugging Face AI ඊමෝෂන් මොඩල් එක ලෝඩ් වෙමින් පවතී... කරුණාකර රැඳී සිටින්න."):
    emotion_classifier = load_emotion_model()

# Custom CSS for Silicon-like Smooth Round Face & Curved Eyebrows
st.markdown("""
    <style>
    .main {
        background-color: #04060f;
        color: #ffffff;
    }
    .robot-head {
        width: 340px;
        height: 340px;
        background: radial-gradient(circle at 30% 30%, #22304a, #0b101d);
        border-radius: 50%;
        margin: 20px auto;
        border: 4px solid rgba(0, 255, 255, 0.6);
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.4), inset 0 0 30px rgba(0, 255, 255, 0.2), inset 0 20px 20px rgba(255, 255, 255, 0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-around;
        padding: 45px 20px;
        position: relative;
        transition: all 0.5s ease;
    }
    .eyebrows-row {
        display: flex;
        justify-content: space-between;
        width: 60%;
        transition: all 0.4s ease;
    }
    /* Curved Eyebrows */
    .eyebrow {
        width: 50px;
        height: 18px;
        border-top: 5px solid #00ffff;
        border-radius: 50% 50% 0 0;
        box-shadow: 0 -4px 10px rgba(0, 255, 255, 0.8);
        transition: all 0.4s ease;
    }
    .eyes-row {
        display: flex;
        justify-content: space-between;
        width: 65%;
    }
    .eye {
        width: 55px;
        height: 55px;
        background: radial-gradient(circle, rgba(0,255,255,0.4) 0%, rgba(0,100,100,0.8) 100%);
        border: 2px solid #00ffff;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.7);
        transition: all 0.4s ease;
    }
    .pupil {
        width: 20px;
        height: 20px;
        background: #ffffff;
        border-radius: 50%;
        box-shadow: 0 0 10px #ffffff;
        transition: all 0.4s ease;
    }
    .mouth {
        width: 90px;
        height: 20px;
        background: #ff0055;
        border-radius: 10px;
        box-shadow: 0 0 18px #ff0055;
        transition: all 0.4s ease;
    }
    .status-text {
        font-size: 22px;
        font-weight: bold;
        color: #00ffff;
        margin-top: 20px;
        text-align: center;
        font-family: sans-serif;
        text-shadow: 0 0 10px rgba(0,255,255,0.5);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Silicon-Style AI Animated Robot Face")
st.write("සිලිකන් ස්මූත් මෝස්තරයකින් සහ වක්‍ර ඇහිබැමි (Curved Eyebrows) සමඟ AI හැඟීම් ප්‍රදර්ශනය කිරීම.")

default_conversation = """Hello! Let us test how artificial intelligence detects my feelings.
I am so thrilled and excited to work on this advanced data science project!
Haha, that joke was genuinely hilarious and made me laugh so much.
Hmm, I need to analyze how this neural network processes complex data.
Wait, why does this unexpected error keep appearing on my screen?
Oh, I am feeling quite exhausted and sleepy after working all night.
I am so furious and angry about why this code failed again!
Oh no, I feel so sad and disappointed right now.
Ew, this messy code structure is totally disgusting and terrible.
Everything is wonderful, I am feeling so happy and peaceful today!"""

conversation_input = st.text_area("සංවාද පෙළ (AI Input Conversation):", value=default_conversation, height=130)
start_button = st.button("Start Silicon AI Flow")

def get_silicon_robot_styles(emotion):
    emotion = emotion.lower()
    if emotion in ["joy", "happiness"]:
        return "transform: rotate(0deg) scaleY(1);", "transform: scale(1); background: radial-gradient(circle, rgba(0,255,136,0.5) 0%, rgba(0,150,80,0.8) 100%); border-color: #00ff88;", "width: 100px; height: 28px; border-radius: 0 0 50px 50px; background: #00ff88; box-shadow: 0 0 22px #00ff88;", "transform: scale(1.1);", "😊 Joy / Happy"
    elif emotion == "excitement":
        return "transform: translateY(-8px) scaleY(1.2); border-top-color: #00ffff;", "transform: scale(1.2);", "width: 85px; height: 32px; border-radius: 50%; background: #00ffff; box-shadow: 0 0 25px #00ffff;", "transform: scale(1.3);", "🤩 Excited"
    elif emotion == "amusement":
        return "transform: rotate(-10deg);", "transform: scale(1.1);", "width: 95px; height: 24px; border-radius: 0 0 40px 40px; background: #ff00ff; box-shadow: 0 0 20px #ff00ff;", "transform: scale(1.2);", "😆 Laughing / Amusement"
    elif emotion == "surprise":
        return "transform: translateY(-12px) scaleY(1.4);", "transform: scale(1.3); background: radial-gradient(circle, rgba(255,255,0,0.5) 0%, rgba(150,150,0,0.8) 100%); border-color: #ffff00;", "width: 45px; height: 38px; border-radius: 50%; background: #ffff00; box-shadow: 0 0 25px #ffff00;", "transform: scale(1.4);", "😲 Surprised"
    elif emotion in ["realization", "curiosity"]:
        return "transform: translateY(-4px);", "transform: scale(0.9);", "width: 55px; height: 14px; border-radius: 7px; background: #00ffff;", "transform: scale(0.9);", "🤔 Thinking / Curious"
    elif emotion == "confusion":
        return "transform: rotate(15deg);", "transform: scale(1);", "width: 65px; height: 14px; border-radius: 10px; background: #ffaa00;", "transform: scale(1);", "😕 Confused"
    elif emotion == "sadness":
        return "transform: rotate(12deg) scaleY(0.8); border-top-color: #00aaff;", "transform: scale(0.85);", "width: 85px; height: 14px; border-radius: 30px 30px 0 0; background: #00aaff; box-shadow: 0 0 18px #00aaff;", "transform: scale(0.8);", "😢 Sad"
    elif emotion == "anger":
        return "transform: rotate(-22deg) scaleY(0.7); border-top-color: #ff0000;", "transform: scale(0.95); background: radial-gradient(circle, rgba(255,0,0,0.5) 0%, rgba(150,0,0,0.8) 100%); border-color: #ff0000;", "width: 85px; height: 12px; border-radius: 6px; background: #ff0000; box-shadow: 0 0 22px #ff0000;", "transform: scale(0.9);", "😡 Angry"
    elif emotion == "disgust":
        return "transform: rotate(6deg);", "transform: scale(0.9);", "width: 75px; height: 12px; border-radius: 6px; background: #aa00ff;", "transform: scale(0.9);", "🤢 Disgust"
    elif emotion in ["nervousness", "fatigue"]:
        return "transform: translateY(4px) scaleY(0.7);", "transform: scale(0.8);", "width: 70px; height: 12px; border-radius: 6px; background: #888888;", "transform: scale(0.7);", "🥱 Sleepy / Bored"
    else:
        return "transform: rotate(0deg);", "transform: scale(1);", "width: 90px; height: 16px; border-radius: 8px; background: #ff0055;", "transform: scale(1);", f"😐 Neutral ({emotion.capitalize()})"

status_placeholder = st.empty()
face_placeholder = st.empty()
sentence_placeholder = st.empty()

if start_button:
    sentences = [s.strip() for s in conversation_input.split('\n') if s.strip()]
    
    for sentence in sentences:
        predictions = emotion_classifier(sentence)
        top_pred = predictions[0][0] if isinstance(predictions[0], list) else predictions[0]
        
        detected_emotion = top_pred['label']
        confidence = top_pred['score']
        
        eb_style, eye_style, mouth_style, pupil_style, status_label = get_silicon_robot_styles(detected_emotion)
        
        sentence_placeholder.markdown(f"### 💬 සංවාදය: *\"{sentence}\"*")
        status_placeholder.markdown(f"<div class='status-text'>AI Detected: {status_label} (Confidence: {confidence:.2f})</div>", unsafe_allow_html=True)
        
        robot_html = f"""
        <div class="robot-head">
            <div class="eyebrows-row">
                <div class="eyebrow" style="{eb_style}"></div>
                <div class="eyebrow" style="{eb_style}"></div>
            </div>
            <div class="eyes-row">
                <div class="eye" style="{eye_style}"><div class="pupil" style="{pupil_style}"></div></div>
                <div class="eye" style="{eye_style}"><div class="pupil" style="{pupil_style}"></div></div>
            </div>
            <div class="mouth" style="{mouth_style}"></div>
        </div>
        """
        
        face_placeholder.markdown(robot_html, unsafe_allow_html=True)
        time.sleep(4.0)