import speech_recognition as sr
import cv2
from gtts import gTTS
from playsound import playsound
import os
def speak(text):
    """Converts text to speech and plays it."""
    try:
        print(f"AI Guard says: {text}")
        # Create a gTTS object
        tts = gTTS(text=text, lang='en')
        # Save the audio file
        audio_file = "response.mp3"
        tts.save(audio_file)
        # Play the audio file
        playsound(audio_file)
        # Remove the file after playing
        os.remove(audio_file)
    except Exception as e:
        print(f"Error in speak function: {e}")
# --- 1. Initialization ---

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Define the activation command
ACTIVATION_COMMAND = "guard my room"

# State management variable
is_guard_active = False

print("AI Guard Agent Initialized. Waiting for activation command...")
speak("AI guard agent initialized. Waiting for activation command.")

# --- 2. Main Application Loop ---
while True:
    try:
        # If guard mode is NOT active, listen for the command
        if not is_guard_active:
            # Use the microphone as the audio source
            with sr.Microphone() as source:
                print("Listening for 'Guard my room'...")
                # Adjust for ambient noise to improve accuracy
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen to the user's input
                audio = recognizer.listen(source)

            # Recognize the speech using Google Web Speech API
            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"Heard: '{command}'")

                # Check if the activation command is in the recognized speech
                if ACTIVATION_COMMAND in command:
                    is_guard_active = True
                    speak("Guard mode activated. Monitoring your room.")
                    print("✅ Guard mode ACTIVATED. Monitoring...")

            except sr.UnknownValueError:
                # This error means the library could not understand the audio
                print("Could not understand audio, please try again.")
            except sr.RequestError as e:
                # This error happens if there's an issue with the API request
                print(f"Could not request results from Google Speech Recognition service; {e}")

        # If guard mode IS active, run the monitoring logic
        else:
            # --- Webcam Logic ---
            # This part sets up for Milestone 2
            cap = cv2.VideoCapture(0) # 0 is the default camera

            if not cap.isOpened():
                print("Error: Cannot open camera.")
                break

            while is_guard_active:
                ret, frame = cap.read()
                if not ret:
                    break

                # Display the resulting frame
                cv2.imshow('AI Guard - Monitoring', frame)

                # For now, we don't have a deactivation command via voice.
                # Press 'q' on the keyboard to stop monitoring and exit the program.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Guard mode DEACTIVATED by user. Exiting.")
                    is_guard_active = False # Deactivate guard mode
                    # Break the inner while loop
                    break
            
            # Release camera resources and close windows
            cap.release()
            cv2.destroyAllWindows()
            
            # Since we broke the inner loop by pressing 'q', we should also break the outer one to exit.
            break

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
        break