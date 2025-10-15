# 🤖 AI Room Guard Agent

An intelligent security agent developed for the EE782 course assignment. This Python-based system uses a laptop's webcam and microphone to monitor a room, recognize trusted individuals, and engage in a multi-level escalating dialogue with unrecognized persons to deter intrusion.



## Features

* **Voice Activated:** Starts and stops monitoring with spoken commands ("guard my room" / "stop guarding").
* **Real-time Face Recognition:** Employs the `DeepFace` library with `SFace` and `yolov8` for a fast and accurate distinction between enrolled "Trusted" individuals and "Unknown" persons.
* **Intelligent Intruder Dialogue:** Engages with unknown individuals using a multi-level escalation system powered by the Google Gemini LLM.
* **Intruder Photo Capture:** Automatically captures and saves a photo of an intruder if they fail the final security challenge.
* **Audible Alerts:** Provides spoken feedback for all state changes and issues an audible alarm during the final escalation level.
* **Event Logging:** Maintains a detailed `guard_log.txt` file with timestamps of all significant events.

---

## Tech Stack & Libraries

* **Language:** Python 3.8+
* **Core AI Libraries:**
    * **Computer Vision:** OpenCV, DeepFace (`SFace` model, `yolov8` detector)
    * **Speech Recognition (ASR):** SpeechRecognition (using Google's API)
    * **Text-to-Speech (TTS):** Google’s TTS
    * **Language Model (LLM):** Google Gemini (`gemini-2.5-flash`)
* **Audio:** Pygame (for alert sounds), PyAudio
* **Data Handling:** Pandas, NumPy, Pickle

---

## Getting Started

Follow these instructions to set up and run the AI Room Guard on your local machine.

### **1. Prerequisites**

* Python 3.8 or higher installed.
* A working webcam and microphone.

### **2. Clone the Repository**

Open your terminal and clone this repository:
```bash
git clone https://github.com/Soumyadeepj/AI-Agent.git
```

### **3. Set Up a Virtual Environment**

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### **4. Install Dependencies**

Install all the required Python libraries using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### **5. Configuration & Enrollment**

You must complete these steps for the agent to function correctly.

**A. Google Gemini API Key**
1.  Go to [Google AI Studio](https://makersuite.google.com/) to generate a free API key.
2.  Open the main Python script `main_guard.py` and paste your key into the designated variable:
    ```python
    genai.configure(api_key="YOUR_API_KEY_HERE")
    ```
**B. Enroll Trusted Faces**
1.  Create a folder named `enroll` in the project's root directory.
2.  Inside `enroll`, create a sub-folder for each trusted person (e.g., `enroll/person_name/`).
3.  Place several clear photos (JPEG or PNG) of that person in their sub-folder.
 ---

## Run the Agent

Once all dependencies are installed and the configuration is complete, run the main script from your terminal:

```bash
python main_guard.py
```

The program will initialize, load the face database, and wait for your command.

## Usage Commands

* **Activate:** Say **"guard my room"** clearly into your microphone.
* **Deactivate:** Say **"stop guarding"**.
* **Exit Program:** Make sure the webcam feed window is active and press the **'q'** key on your keyboard.

---

## Configuration

You can tune the agent's sensitivity by modifying these constants at the top of the script:

* `RECOGNITION_THRESHOLD`: Adjusts face recognition sensitivity. Higher means stricter. (Default: `0.4`)
* `PROCESS_EVERY_N_FRAMES`: Controls how often face recognition is run. Higher values improve performance but reduce detection speed. (Default: `1`)
---
