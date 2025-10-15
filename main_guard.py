import cv2
import numpy as np
import pandas as pd
import time
from deepface import DeepFace
import os
import pickle
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import google.generativeai as genai
import pyttsx3
import os
import logging

# ============================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================
MODEL_NAME = "ArcFace"
DETECTOR = "yolov8"  
ENROLL_DIR = "enroll"
DB_PATH = "enrolled_faces.pkl"
PROTOTYPES_PATH = "face_prototypes.pkl"
RECOGNITION_THRESHOLD = 0.4 
INTRUDER_FOLDER = "intruders"

# ============================================================
# 2. FACE RECOGNITION UTILITIES
# ============================================================
def get_embedding(img_or_path):
    """
    Safely detects a face, resizes it, and then gets the embedding.
    Returns None if no face is detected.
    """
    try:
        # Explicitly detect and extract face(s) from the image.
        extracted_faces = DeepFace.extract_faces(
            img_path=img_or_path,
            detector_backend=DETECTOR,
            enforce_detection=True # This will raise an error if no face is found
        )
        # The actual face image data is in the 'face' key of the first dictionary.
        face_img = extracted_faces[0]['face']
        
        # The extracted face is a normalized numpy array (0.0 to 1.0).
        # We need to scale it to a standard image format (0-255, 8-bit integer).
        face_img = (face_img * 255).astype(np.uint8)

        # ArcFace requires the input to be exactly 112x112 pixels.
        face_img = cv2.resize(face_img, (112, 112))

        # Generate the embedding from the resized face.
        embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend='skip' # Crucial for speed and correctness
        )
        
        return np.array(embedding[0]["embedding"], dtype=np.float32)

    except ValueError:
        # This error is correctly thrown by extract_faces if no face is detected.
        return None
    except Exception as e:
        # Catch any other unexpected errors during the process.
        print(f"An unexpected error occurred in get_embedding: {e}")
        return None

def build_enrollment_db(enroll_dir, out_path):
    """
    Builds the enrollment database from images inside subfolders of `enroll_dir`.
    Each subfolder should be named after the person.
    """
    rows = []
    if not os.path.exists(enroll_dir):
        os.makedirs(enroll_dir)
        print(f"Created enrollment directory: {enroll_dir}")
        print("Please add sub-folders with images of trusted people.")
        return None

    # Iterate over person folders
    persons = sorted([d for d in os.listdir(enroll_dir) if os.path.isdir(os.path.join(enroll_dir, d))])
    if not persons:
        print(f"No person folders found in {enroll_dir}. Cannot enroll.")
        return None

    for person in persons:
        pdir = os.path.join(enroll_dir, person)
        for img_path in [os.path.join(pdir, f) for f in os.listdir(pdir)]:
            try:
                emb = get_embedding(img_path)
                if emb is not None:
                    rows.append({"person": person, "embedding": emb})
                    print(f"[OK] Enrolled {person} from {os.path.basename(img_path)}")
            except Exception as e:
                print(f"[ERR] Could not process {img_path}: {e}")

    if not rows:
        print("Enrollment failed. No faces could be processed.")
        return None

    # Save all embeddings as a DataFrame
    df = pd.DataFrame(rows)
    with open(out_path, "wb") as f:
        pickle.dump(df, f)

    print(f"\nSaved DB with {len(df)} entries for {df['person'].nunique()} persons -> {out_path}")
    return df

def build_person_prototypes(df: pd.DataFrame) -> dict:
    """
    Averages embeddings per person to create a prototype vector.
    """
    protos = {}
    for person, group in df.groupby("person"):
        embs = np.stack(group["embedding"].to_list(), axis=0)
        proto = embs.mean(axis=0)
        proto /= np.linalg.norm(proto)  # Normalize
        protos[person] = proto.astype(np.float32)
    return protos

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-12)*(np.linalg.norm(b)+1e-12)))

def identify_face(embedding, person_protos, threshold):
    """Identifies the closest person by cosine similarity."""
    if embedding is None:
        return "No Face", 0.0

    best_p, best_s = "Unknown", -1.0
    for person, proto in person_protos.items():
        s = cosine_sim(embedding, proto)
        if s > best_s:
            best_s, best_p = s, person

    if best_s >= threshold:
        return best_p, best_s
    return "Unknown", best_s

def capture_intruder_photo(frame):
    """Saves the current frame as an image of the intruder."""
    try:
        # Create the intruder folder if it doesn't exist
        if not os.path.exists(INTRUDER_FOLDER):
            os.makedirs(INTRUDER_FOLDER)

        # Create a unique filename with a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(INTRUDER_FOLDER, f"intruder_{timestamp}.jpg")

        # Save the current frame as a JPEG file
        cv2.imwrite(filename, frame)
        print(f"Intruder photo saved: {filename}")
    except Exception as e:
        print(f"Error saving intruder photo: {e}")

# ============================================================
# 3. GEMINI + SPEECH SETUP
# ============================================================
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='guard_log.txt',
    filemode='a'
)

# Configure Gemini API
try:
    # api_key = os.getenv("GEMINI_API_KEY")
    # genai.configure(api_key="-----")
    llm = genai.GenerativeModel('gemini-2.5-flash')
    print("Gemini Pro model configured successfully.")
except Exception as e:
    llm = None
    print(f"Error configuring Gemini: {e}. LLM features will be disabled.")

# State variables
is_guard_active = False
recognizer = sr.Recognizer()

def speak(text):
    """Converts text to speech and plays it."""
    try:
        print(f"AI Guard says: {text}")
        tts = gTTS(text=text, lang='en', slow=False)
        audio_file = "response.mp3"
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)
    except Exception as e:
        print(f"Error in speak function: {e}")
       
def get_llm_judgment(level, intruder_response):
    """
    Sends intruder's speech to Gemini and returns judgment:
    TRUSTWORTHY / SUSPICIOUS / SPECIFIC / EVASIVE
    """
    if not llm:
        return "NO"

    try:
        if level == 1:
            # Level 1: Check if intruder gives clear name or reason
            prompt = f"""
            A person was asked "Who are you?". They responded: "{intruder_response}".
            Does this response clearly state a name or a reason for being there?
            Respond with only YES or NO.
            """
            response = llm.generate_content(prompt)
            return "TRUSTWORTHY" if response.text.strip().upper() == "YES" else "SUSPICIOUS"

        elif level == 2:
            # Level 2: Check if they mention your name
            prompt = f"""
            A person was asked for the resident's name. They responded: "{intruder_response}".
            Does this response contain Sahill or similar?
            Respond with only YES or NO.
            """
            response = llm.generate_content(prompt)
            return "SPECIFIC" if response.text.strip().upper() == "YES" else "EVASIVE"

    except Exception as e:
        print(f"LLM Error: {e}")
        return "SUSPICIOUS"

    return "SUSPICIOUS"

def listen_for_response(timeout=5):
    """Listens for short verbal responses from the intruder."""
    try:
        with sr.Microphone() as source:
            print("Listening for response...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout)
        response_text = recognizer.recognize_google(audio).lower()
        print(f"Heard response: '{response_text}'")
        return response_text
    except (sr.UnknownValueError, sr.WaitTimeoutError):
        print("No response heard.")
        return None


# ============================================================
# 5. FACE DATABASE INITIALIZATION
# ============================================================

if not os.path.exists(PROTOTYPES_PATH):
    print("Building face database...")
    df = build_enrollment_db(ENROLL_DIR, DB_PATH)
    person_protos = build_person_prototypes(df) if df is not None else {}
    if person_protos:
        with open(PROTOTYPES_PATH, "wb") as f: pickle.dump(person_protos, f)
else:
    with open(PROTOTYPES_PATH, "rb") as f: person_protos = pickle.load(f)
    print("Loaded existing face database.")

ACTIVATION_COMMAND = "guard my room"
DEACTIVATION_COMMAND = "stop guarding"
is_guard_active = False
recognizer = sr.Recognizer()

# ============================================================
# 6. COMMAND HANDLER (VOICE ACTIVATION)
# ============================================================
def handle_command(recognizer, audio):
    """Handles voice commands to activate/deactivate guard mode."""
    global is_guard_active
    try:
        command = recognizer.recognize_google(audio).lower()
        if not is_guard_active and ACTIVATION_COMMAND in command: 

            is_guard_active = True
            speak("Hi Soumyadeep, Guard mode activated. Monitoring your room.")
            print("✅ Guard mode ACTIVATED. Monitoring...")
      
        elif is_guard_active and DEACTIVATION_COMMAND in command:

            is_guard_active = False
            print("Welcome Back Soumyadeep, Guard mode deactivated.")
            speak("Guard mode deactivated.")     
    except sr.UnknownValueError:
        # print("Could not understand audio, please try again.")
        pass
    except sr.RequestError as e:
        # This error happens if there's an issue with the API request
        print(f"Could not request results from Google Speech Recognition service; {e}")
        

# ============================================================
# 7. MAIN LOOP - CAMERA MONITORING & ESCALATION LOGIC
# ============================================================

stop_listening = recognizer.listen_in_background(sr.Microphone(), handle_command)
print("AI Guard Agent Initialized. Waiting for activation command...")
speak("AI guard agent initialized. Waiting for activation command.")
cap = None
escalation_level = 0              #level of escalation
last_intruder_seen_time = 0       #time of intruder seen last
last_challenge_time = 0           #last time when escalation flag
INTRUDER_RESET_TIMEOUT = 25      #reset the setting after this time if no intruder seen in this period
CHALLENGE_COOLDOWN = 2           #wait to flag next escalation
frame_counter = 0                #count the number of frame 
PROCESS_EVERY_N_FRAMES = 1        
#basic initialization
last_known_name = " "
last_known_score = 0.0
level_count = 0

# --- Main loop ---
while True:
    if is_guard_active:
        # Initialize webcam
        if cap is None: cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Error, camera unavailable.")
            is_guard_active = False
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # Process frames periodically
        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            face_embedding = get_embedding(frame)
            name, score = identify_face(face_embedding, person_protos, RECOGNITION_THRESHOLD)
            last_known_name, last_known_score = name, score
        frame_counter += 1

        # Display info on video feed
        label = f"{last_known_name} ({last_known_score:.2f}) Level: {escalation_level}"
        color = (0, 0, 255) if last_known_name == "Unknown" else (0, 255, 0)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow('AI Guard', frame)

        # --- Intruder Detected ---
        if face_embedding is not None and last_known_name == "Unknown":
            last_intruder_seen_time = time.time()

            # Cooldown before next challenge
            if time.time() - last_challenge_time > CHALLENGE_COOLDOWN or last_challenge_time == 0:
                if escalation_level == 0:
                    level_count += 1
                    if level_count == 3:
                        speak("Hi, AI Agent here!")
                    if level_count > 3:
                        escalation_level = 1
                        level_count = 0
                    last_challenge_time = time.time()

                elif escalation_level == 1:
                    speak("Excuse me, I don't recognize you. Who are you?")
                    response = listen_for_response()
                    if response and get_llm_judgment(1, response) == "TRUSTWORTHY":
                        speak("Your presence is noted and will be informed to Soumyadeep. You can leave now.")
                        logging.info(f"Intruder L1 response: '{response}'")
                        escalation_level = 0
                    else:
                        escalation_level = 2
                    last_challenge_time = time.time()

                elif escalation_level == 2:
                    speak("For security, please state the name of the resident.")
                    response = listen_for_response()
                    if response and get_llm_judgment(2, response) == "SPECIFIC":
                        speak("Thank you. Your statement has been logged. Please leave now.")
                        escalation_level = 0
                    else:
                        escalation_level = 3
                    last_challenge_time = time.time()

                elif escalation_level == 3:
                    speak("Your response is not satisfactory. This is a final warning.")
                    capture_intruder_photo(frame)
                    speak("Countdown started.")
                    for i in range(3):
                        speak(f"{3-i}")
                        time.sleep(0.1)
                    playsound("./alarm.wav")
                    last_challenge_time = time.time()

        # Reset state if intruder disappears
        if time.time() - last_intruder_seen_time > INTRUDER_RESET_TIMEOUT and escalation_level > 0:
            print("Intruder gone. Resetting state.")
            escalation_level = 0
            last_known_name = " "

    elif not is_guard_active and cap is not None:
        # Stop camera feed when not guarding
        cap.release()
        cv2.destroyAllWindows()
        cap = None
        escalation_level = 0
        time.sleep(0.1)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean exit
stop_listening(wait_for_stop=False)
if cap:
    cap.release()
cv2.destroyAllWindows()
print("Program exited.")