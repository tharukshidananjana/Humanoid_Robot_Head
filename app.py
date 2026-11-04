import streamlit as st
import torch
import time
from transformers import pipeline

# ==========================================
# RASPBERRY PI GPIO & SMOOTH SERVO MOTOR SETUP
# ==========================================
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    
    # Pin definitions for Servo Motors
    SERVO_EYEBROW_PIN = 17
    SERVO_MOUTH_PIN = 18
    
    GPIO.setup(SERVO_EYEBROW_PIN, GPIO.OUT)
    GPIO.setup(SERVO_MOUTH_PIN, GPIO.OUT)
    
    # Initialize PWM for Servos (50Hz standard frequency)
    eyebrow_pwm = GPIO.PWM(SERVO_EYEBROW_PIN, 50)
    mouth_pwm = GPIO.PWM(SERVO_MOUTH_PIN, 50)
    
    # Start at Neutral position (~90 degrees -> 7.5% duty cycle)
    eyebrow_pwm.start(7.5)
    mouth_pwm.start(7.5)
    
    HARDWARE_MODE = True
except (ImportError, RuntimeError):
    HARDWARE_MODE = False  # Runs safely on Laptop/PC without GPIO errors

# Tracking current motor angles for smooth interpolation (Weighted transition)
current_eyebrow_angle = 90  
current_mouth_angle = 90    

def smooth_move_servo(pwm_channel, start_angle, target_angle, step_delay=0.02):
    """
    Moves the servo motor smoothly from start_angle to target_angle 
    using small incremental steps to prevent abrupt mechanical jumps.
    """
    if not HARDWARE_MODE:
        return
        
    step = 1 if start_angle < target_angle else -1
    for angle in range(start_angle, target_angle + step, step):
        # Convert angle (0 to 180 degrees) to PWM duty cycle (2.5% to 12.5%)
        duty_cycle = 2.5 + (angle / 18.0)
        pwm_channel.ChangeDutyCycle(duty_cycle)
        time.sleep(step_delay)

def move_physical_motors_smoothly(emotion):
    """
    Maps detected AI emotions to specific target angles and applies smooth weighted transitions.
    """
    global current_eyebrow_angle, current_mouth_angle
    
    if not HARDWARE_MODE:
        return
        
    emotion = emotion.lower()
    
    # Define target angles (0 to 180) for Eyebrows and Mouth across emotions (Supporting all 11 model classes)
    if emotion in ["joy", "happiness"]:
        target_eyebrow, target_mouth = 70, 120
    elif emotion == "excitement":
        target_eyebrow, target_mouth = 60, 130
    elif emotion == "amusement":
        target_eyebrow, target_mouth = 75, 115
    elif emotion == "surprise":
        target_eyebrow, target_mouth = 50, 140
    elif emotion == "fear":
        target_eyebrow, target_mouth = 45, 135
    elif emotion in ["realization", "disappointment"]:
        target_eyebrow, target_mouth = 80, 95
    elif emotion == "curiosity":
        target_eyebrow, target_mouth = 75, 100
    elif emotion == "confusion":
        target_eyebrow, target_mouth = 100, 90
    elif emotion == "sadness":
        target_eyebrow, target_mouth = 110, 60
    elif emotion == "anger":
        target_eyebrow, target_mouth = 40, 80
    elif emotion == "disgust":
        target_eyebrow, target_mouth = 95, 85
    else:  # Default / Neutral state
        target_eyebrow, target_mouth = 90, 90

    # Execute smooth incremental movement
    smooth_move_servo(eyebrow_pwm, current_eyebrow_angle, target_eyebrow, step_delay=0.02)
    smooth_move_servo(mouth_pwm, current_mouth_angle, target_mouth, step_delay=0.02)
    
    current_eyebrow_angle = target_eyebrow
    current_mouth_angle = target_mouth


# ==========================================
# STREAMLIT UI & AI MODEL CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Humanoid Robot Head - Fluid Realistic Face Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Hugging Face Pre-trained Emotion Detection Model (Supports all 11 distilroberta emotion categories)
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

with st.spinner("Loading Hugging Face AI Emotion Detection Model... Please wait."):
    emotion_classifier = load_emotion_model()

bg_gradient = "radial-gradient(circle at 30% 30%, #ffdfc4, #f5c29d)"
face_border = "#d69e2e"

st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
        color: #ffffff;
    }
    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #e2e8f0;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 15px;
        color: #a0aec0;
        margin-bottom: 25px;
    }
    .robot-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 10px;
    }
    .eyebrows-row {
        display: flex;
        justify-content: space-between;
        width: 140px;
        margin-bottom: 12px;
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .eyebrow {
        width: 45px;
        height: 12px;
        border-top: 4px solid #4a5568;
        border-radius: 50% 50% 0 0;
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .eyes-row {
        display: flex;
        justify-content: space-between;
        width: 150px;
        margin-bottom: 20px;
    }
    .eye {
        width: 50px;
        height: 50px;
        background: #ffffff;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .pupil {
        width: 22px;
        height: 22px;
        background: #2d3748;
        border-radius: 50%;
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .mouth {
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .status-badge {
        font-size: 16px;
        font-weight: 700;
        color: #68d391;
        background: rgba(104, 211, 145, 0.1);
        padding: 8px 16px;
        border-radius: 20px;
        border: 1px solid rgba(104, 211, 145, 0.3);
        text-align: center;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title updated: "Advanced Silicone" removed as requested
st.markdown("<div class='main-title'>🤖 Humanoid Robot Head - Fluid Realistic Face Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Designed specifically to test smooth and fluid emotional transitions in real-time across all 11 Hugging Face emotion classes.</div>", unsafe_allow_html=True)

col_input, col_preview = st.columns([1.2, 1])

with col_input:
    st.markdown("### 💬 Transition Test Conversation Input")
    
    
    # Conversation containing dialogue lines covering all 11 model emotions
    default_dialogue = default_dialogue = """1. I am feeling completely calm, peaceful, and neutral right now.
2. I wonder why this is happening, such intense curiosity.
3. Wait a second, I just realized how this works!
4. I am completely surprised and astonished by this sudden turn of events!
5. Oh wow, I am so extremely happy and joyful about this wonderful news!
6. This is so exciting, I cannot wait for what comes next!
7. Haha, that was genuinely hilarious and full of amusement!
8. Why did this happen? I am so confused and mixed up right now.
9. This is so disappointing and frustrating, I feel terrible.
10. Oh no, I feel so heartbroken, lonely, and deeply sad about what happened.
11. Oh no, I trembling and feeling so afraid of what might happen.
12. Why did you do that? I am extremely angry and furious right now!
13. Ugh, this is utterly disgusting and repulsive!"""

    conversation_input = st.text_area("Dialogue Lines:", value=default_dialogue, height=240, label_visibility="collapsed")
    start_button = st.button("🚀 Start Smooth Transition Test", use_container_width=True)

with col_preview:
    st.markdown("### 🌟 Humanoid Face (Live Preview)")
    face_slot = st.empty()
    status_slot = st.empty()
    
    face_slot.markdown(f"""
        <div class="robot-container">
            <div style="width: 290px; height: 290px; background: {bg_gradient}; border-radius: 50%; box-shadow: 0 10px 25px rgba(0,0,0,0.3), inset 0 2px 10px rgba(255,255,255,0.2); display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; border: 4px solid {face_border};">
                <div class="eyebrows-row">
                    <div class="eyebrow"></div>
                    <div class="eyebrow"></div>
                </div>
                <div class="eyes-row">
                    <div class="eye"><div class="pupil"></div></div>
                    <div class="eye"><div class="pupil"></div></div>
                </div>
                <div class="mouth" style="width: 50px; height: 10px; border-bottom: 3px solid #4a5568; border-radius: 0;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    status_slot.markdown("<div class='status-badge'>Fluid Expression State: Neutral (Ready)</div>", unsafe_allow_html=True)

live_footer = st.empty()

def get_fluid_robot_styles(emotion):
    emotion = emotion.lower()
    if emotion in ["joy", "happiness"]:
        return "transform: rotate(0deg); border-top-color: #38a169;", "transform: rotate(0deg); border-top-color: #38a169;", "transform: scale(1);", "width: 85px; height: 35px; border-bottom: 5px solid #38a169; border-radius: 0 0 45px 45px; background: rgba(56,161,105,0.15);", "transform: scale(1.05); background: #38a169;", "😊 Joy / Happy"
    elif emotion == "excitement":
        return "transform: translateY(-4px) scaleY(1.2); border-top-color: #319795;", "transform: translateY(-4px) scaleY(1.2); border-top-color: #319795;", "transform: scale(1.1);", "width: 75px; height: 38px; border: 4px solid #319795; border-radius: 50%; background: rgba(49,151,149,0.2);", "transform: scale(1.15); background: #319795;", "🤩 Excited"
    elif emotion == "amusement":
        return "transform: rotate(-6deg); border-top-color: #805ad5;", "transform: rotate(6deg); border-top-color: #805ad5;", "transform: scale(1.02);", "width: 80px; height: 30px; border-bottom: 4px solid #805ad5; border-radius: 0 0 40px 40px; background: rgba(128,90,213,0.15);", "transform: scale(1.05); background: #805ad5;", "😆 Laughing / Amusement"
    elif emotion == "surprise":
        return "transform: translateY(-6px) scaleY(1.3); border-top-color: #d69e2e;", "transform: translateY(-6px) scaleY(1.3); border-top-color: #d69e2e;", "transform: scale(1.15);", "width: 40px; height: 40px; border: 4px solid #d69e2e; border-radius: 50%; background: rgba(214,158,46,0.2);", "transform: scale(1.2); background: #d69e2e;", "😲 Surprised"
    elif emotion == "fear":
        return "transform: translateY(-8px) scaleY(1.3); border-top-color: #e53e3e;", "transform: translateY(-8px) scaleY(1.3); border-top-color: #e53e3e;", "transform: scale(1.2);", "width: 35px; height: 35px; border: 3px solid #e53e3e; border-radius: 50%; background: rgba(229,62,62,0.2);", "transform: scale(1.25); background: #e53e3e;", "😨 Fear"
    elif emotion == "realization":
        return "transform: translateY(-2px); border-top-color: #3182ce;", "transform: translateY(-2px); border-top-color: #3182ce;", "transform: scale(0.98);", "width: 55px; height: 16px; border-bottom: 4px solid #3182ce; border-radius: 10px;", "transform: scale(0.95); background: #3182ce;", "💡 Realization"
    elif emotion == "curiosity":
        return "transform: translateY(-4px) rotate(-6deg); border-top-color: #3182ce;", "transform: translateY(-4px) rotate(6deg); border-top-color: #3182ce;", "transform: scale(1.02);", "width: 45px; height: 20px; border: 3px solid #3182ce; border-radius: 50%;", "transform: scale(1); background: #3182ce;", "🤔 Curiosity"
    elif emotion == "confusion":
        return "transform: rotate(10deg); border-top-color: #dd6b20;", "transform: rotate(-5deg); border-top-color: #dd6b20;", "transform: scale(1);", "width: 60px; height: 16px; border-top: 4px solid #dd6b20; border-radius: 10px;", "transform: scale(1); background: #dd6b20;", "😕 Confused"
    elif emotion == "disappointment":
        return "transform: rotate(-10deg) translateY(2px); border-top: 4px solid #718096; border-radius: 0 50% 50% 0;", "transform: rotate(10deg) translateY(2px); border-top: 4px solid #718096; border-radius: 50% 0 0 50%;", "transform: scale(0.92);", "width: 65px; height: 14px; border-top: 4px solid #718096; border-radius: 20px;", "transform: scale(0.9); background: #718096;", "😞 Disappointment"
    elif emotion == "sadness":
        return "transform: rotate(-15deg) translateY(2px); border-top: 4px solid #3182ce; border-radius: 0 50% 50% 0;", "transform: rotate(15deg) translateY(2px); border-top: 4px solid #3182ce; border-radius: 50% 0 0 50%;", "transform: scale(0.9);", "width: 70px; height: 16px; border-top: 4px solid #3182ce; border-radius: 35px 35px 0 0;", "transform: scale(0.9); background: #3182ce;", "😢 Sad"
    elif emotion == "anger":
        return "transform: rotate(25deg) translateY(4px); border-top: 4px solid #e53e3e; border-radius: 50% 0 0 50%;", "transform: rotate(-25deg) translateY(4px); border-top: 4px solid #e53e3e; border-radius: 0 50% 50% 0;", "transform: scale(0.95);", "width: 70px; height: 10px; border-top: 4px solid #e53e3e; border-radius: 2px;", "transform: scale(0.95); background: #e53e3e;", "😡 Angry"
    elif emotion == "disgust":
        return "transform: translateY(-8px) rotate(-12deg); border-top: 4px solid #805ad5; border-radius: 50% 10% 0 0;", "transform: translateY(6px) rotate(12deg); border-top: 4px solid #805ad5; border-radius: 10% 50% 0 0;", "transform: scale(0.95);", "width: 65px; height: 16px; border-top: 4px solid #805ad5; border-radius: 20px 5px 20px 5px; background: rgba(128,90,213,0.1);", "transform: scale(0.95); background: #805ad5;", "🤢 Disgust"
    else:
        return "transform: rotate(0deg); border-top-color: #4a5568;", "transform: rotate(0deg); border-top-color: #4a5568;", "transform: scale(1);", "width: 50px; height: 10px; border-bottom: 3px solid #4a5568; border-radius: 0;", "transform: scale(1); background: #4a5568;", f"😐 Neutral ({emotion.capitalize()})"

if start_button:
    lines = [line.strip() for line in conversation_input.split('\n') if line.strip()]
    total_lines = len(lines)
    
    for idx, sentence in enumerate(lines, 1):
        predictions = emotion_classifier(sentence)
        top_pred = predictions[0][0] if isinstance(predictions[0], list) else predictions[0]
        
        detected_emotion = top_pred['label']
        confidence = top_pred['score']
        
        # Trigger smooth physical servo motor movement
        move_physical_motors_smoothly(detected_emotion)
        
        eb_left_style, eb_right_style, eye_style, mouth_style, pupil_style, status_label = get_fluid_robot_styles(detected_emotion)
        
        live_footer.markdown(f"""
            <div style='background: #131b2e; padding: 12px 20px; border-radius: 8px; border: 1px solid #1f293d; margin-top: 15px;'>
                🟢 <b>Transition Turn ({idx}/{total_lines}):</b> "{sentence}" | <b>Emotion:</b> {status_label} (Confidence: {confidence:.2f})
            </div>
        """, unsafe_allow_html=True)
        
        status_slot.markdown(f"<div class='status-badge'>Fluid Expression State: {status_label}</div>", unsafe_allow_html=True)
        
        face_html = f"""
        <div class="robot-container">
            <div style="width: 290px; height: 290px; background: {bg_gradient}; border-radius: 50%; box-shadow: 0 10px 25px rgba(0,0,0,0.3), inset 0 2px 10px rgba(255,255,255,0.2); display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; border: 4px solid {face_border};">
                <div class="eyebrows-row">
                    <div class="eyebrow" style="{eb_left_style}"></div>
                    <div class="eyebrow" style="{eb_right_style}"></div>
                </div>
                <div class="eyes-row">
                    <div class="eye" style="{eye_style}"><div class="pupil" style="{pupil_style}"></div></div>
                    <div class="eye" style="{eye_style}"><div class="pupil" style="{pupil_style}"></div></div>
                </div>
                <div class="mouth" style="{mouth_style}"></div>
            </div>
        </div>
        """
        
        face_slot.markdown(face_html, unsafe_allow_html=True)
        time.sleep(3.5)  # Adequate pause to witness the smooth transition

    # Return to Neutral state after testing completes
    move_physical_motors_smoothly("neutral")
    status_slot.markdown("<div class='status-badge'>Fluid Expression State: Neutral (Ready)</div>", unsafe_allow_html=True)