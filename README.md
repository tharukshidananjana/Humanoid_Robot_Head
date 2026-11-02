**** AI-Powered Emotion-Aware Humanoid Robot Head ****

An interactive IoT and AI project developed for the SLT Mobitel R&D internship. This system analyzes text input in real-time, detects emotions using a deep learning transformer model, and physically expresses those emotions through a dynamic digital web interface and smooth Raspberry Pi servo motor movements.

---

🌟 Key Features
- Real-Time Emotion AI: Utilizes Hugging Face's `j-hartmann/emotion-english-distilroberta-base` model to classify text into emotions (Joy, Sadness, Anger, Neutral, etc.).
- Dynamic Web Interface: Built with Streamlit for seamless user interaction and visual feedback.
- Hardware Servo Control: Interfaces with Raspberry Pi GPIO pins to execute smooth degree-by-degree physical movements (`step_delay = 0.035`).
- Single-File Architecture: Integrated UI, AI pipeline, and hardware logic cleanly contained within a single `app.py` script.

---
🛠️ Tech Stack
- Language: Python
- AI / Deep Learning: Hugging Face Transformers, PyTorch / TensorFlow
- Web Framework: Streamlit
- Hardware & IoT: Raspberry Pi, Servo Motors (`RPi.GPIO`)
- Version Control:GitHub

---

🚀 Quick Start Guide

1. Clone the Repository:
   
   git clone [https://github.com/tharukshidananjana/Humanoid_Robot_Head.git](https://github.com/tharukshidananjana/Humanoid_Robot_Head.git)
   cd Humanoid_Robot_Head
   
2.Activate Virtual Environment & Install Dependencies:

  source venv/bin/activate
  pip install streamlit transformers torch RPi.GPIO

3.Run the Application:

  streamlit run app.py
  
4.Access the UI:

  Open your browser and go to http://localhost:8501.

💡 Industrial Applications

Smart customer service kiosks and interactive digital interfaces for telecommunication and enterprise environments.

Research platform for Human-Robot Interaction (HRI) combining AI with physical IoT hardware.
