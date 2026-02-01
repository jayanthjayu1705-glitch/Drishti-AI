import cv2
import easyocr
from ultralytics import YOLO
import pyttsx3
import threading
import time
import numpy as np
import queue
import speech_recognition as sr

# --- 1. CONFIGURATION ---
print("Initializing Drishti-AI (Voice Command Enhanced)...")

# Camera Setup - Try multiple camera sources
camera_sources = [
    'http://192.168.137.172:8080/video',  # First IP camera (Add /video)
    'http://192.168.137.100:8080/video',  # Second IP camera (Example)
    0                                     # Default webcam as fallback
]

# Logic to pick the first working camera
cap = None
for source in camera_sources:
    print(f"Trying camera source: {source}")
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        print(f"Connected to {source}")
        break
    else:
        cap.release()

if cap is None or not cap.isOpened():
    print("Error: Could not connect to any camera.")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for faster response

# Brains
model = YOLO("yolov8n.pt") 
reader = easyocr.Reader(['en']) 

# Speech Recognition Setup
recognizer = sr.Recognizer()

# Voice Setup (Offline & Fast)
temp_engine = pyttsx3.init()
voices = temp_engine.getProperty('voices')
SELECTED_VOICE_ID = voices[0].id 
for voice in voices:
    if "Zira" in voice.name or "female" in voice.name.lower():
        SELECTED_VOICE_ID = voice.id
        break
del temp_engine

# --- THREADING VARIABLES ---
speech_queue = queue.Queue()
frame_lock = threading.Lock()
latest_frame = None
latest_results = None
is_running = True
is_navigating = True
voice_command_active = False
last_command_time = 0
COMMAND_COOLDOWN = 2  # seconds between voice commands

# --- WORKER 1: VOICE THREAD ---
def speech_worker():
    eng = pyttsx3.init()
    eng.setProperty('voice', SELECTED_VOICE_ID)
    eng.setProperty('rate', 160)
    
    while True:
        text = speech_queue.get()
        if text is None: break
        try:
            eng.say(text)
            eng.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
        speech_queue.task_done()

t_speech = threading.Thread(target=speech_worker, daemon=True)
t_speech.start()

def speak(text, priority=False):
    print(f"[AI]: {text}")
    if priority:
        with speech_queue.mutex:
            speech_queue.queue.clear()
    speech_queue.put(text)

# --- WORKER 2: AI BRAIN THREAD ---
def ai_worker():
    global latest_results, latest_frame
    while is_running:
        if is_navigating and latest_frame is not None:
            with frame_lock:
                frame_to_process = latest_frame.copy()
            results = model(frame_to_process, conf=0.4, verbose=False)
            latest_results = results 
            time.sleep(0.01) 
        else:
            time.sleep(0.1)

t_ai = threading.Thread(target=ai_worker, daemon=True)
t_ai.start()

# --- WORKER 3: VOICE COMMAND LISTENER THREAD ---
def voice_listener_worker():
    global voice_command_active, last_command_time
    try:
        mic = sr.Microphone()
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Voice listener ready. Say 'Hey Drishti'...")
    except Exception as e:
        print(f"Microphone initialization error: {e}")
        print("Voice commands will not be available.")
        return
    
    while is_running:
        if voice_command_active:
            try:
                print("Listening for command...")
                with mic as source:
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                
                command = recognizer.recognize_google(audio).lower()
                print(f"Voice Command: {command}")
                
                current_time = time.time()
                if current_time - last_command_time > COMMAND_COOLDOWN:
                    process_voice_command(command)
                    last_command_time = current_time
                    
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except Exception as e:
                print(f"Voice recognition error: {e}")
        
        time.sleep(0.1)

def process_voice_command(command):
    """Process voice commands and trigger appropriate actions"""
    global is_navigating
    
    # Check for wake word
    if "hey drishti" in command or "hey dristhi" in command or "hey d" in command:
        speak("Yes? How can I help you?", priority=True)
        return
    
    # Check for money-related commands
    if any(keyword in command for keyword in ["how much money", "what currency", "check money", "scan money", "money amount"]):
        is_navigating = False
        speak("Checking currency...", priority=True)
        
        # Get the latest frame for scanning
        with frame_lock:
            if latest_frame is not None:
                frame_to_scan = latest_frame.copy()
            else:
                speak("No camera feed available.", priority=True)
                is_navigating = True
                return
        
        # Define money box area (center of frame)
        h, w, _ = frame_to_scan.shape
        box_size = 180
        x_start, y_start = w//2 - box_size, h//2 - box_size
        x_end, y_end = w//2 + box_size, h//2 + box_size
        
        # Scan for money
        amount = scan_money_in_roi(frame_to_scan, x_start, y_start, x_end, y_end)
        if amount:
            speak(f"This is {amount}.", priority=True)
        else:
            speak("I cannot detect any currency. Please place money in the center of the camera view.", priority=True)
        
        time.sleep(1)
        is_navigating = True
    
    # Check for scene description commands
    elif any(keyword in command for keyword in ["what do you see", "describe scene", "scan scene", "what's around"]):
        is_navigating = False
        speak("Let me look around...", priority=True)
        
        if latest_results:
            names = [model.names[int(b.cls[0])] for b in latest_results[0].boxes]
            if names:
                counts = {}
                for name in names:
                    if name in counts:
                        counts[name] += 1
                    else:
                        counts[name] = 1
                
                # Create natural language description
                items = []
                for obj, count in counts.items():
                    if count == 1:
                        items.append(f"a {obj}")
                    else:
                        items.append(f"{count} {obj}s")
                
                if len(items) == 1:
                    summary = f"I see {items[0]}."
                elif len(items) == 2:
                    summary = f"I see {items[0]} and {items[1]}."
                else:
                    summary = f"I see {', '.join(items[:-1])}, and {items[-1]}."
                
                speak(summary, priority=True)
            else:
                speak("Path is clear.", priority=True)
        else:
            speak("Still analyzing the scene...", priority=True)
        
        is_navigating = True
    
    # Check for color identification
    elif any(keyword in command for keyword in ["what color", "identify color", "color check"]):
        is_navigating = False
        speak("Checking color...", priority=True)
        
        with frame_lock:
            if latest_frame is not None:
                frame_to_check = latest_frame.copy()
                h, w, _ = frame_to_check.shape
                center_px = frame_to_check[h//2, w//2]
                
                hsv = cv2.cvtColor(np.uint8([[center_px]]), cv2.COLOR_BGR2HSV)[0][0]
                hue = hsv[0]
                
                col = "Unknown"
                if hue < 10 or hue > 170: col = "Red"
                elif hue < 25: col = "Orange"
                elif hue < 35: col = "Yellow"
                elif hue < 85: col = "Green"
                elif hue < 130: col = "Blue"
                elif hue < 170: col = "Purple"
                elif hsv[1] < 30: col = "White"
                if hsv[2] < 40: col = "Black"
                
                speak(f"The color in the center is {col}.", priority=True)
        
        time.sleep(0.5)
        is_navigating = True
    
    # Check for navigation commands
    elif any(keyword in command for keyword in ["start navigation", "begin guiding", "help me walk"]):
        speak("Navigation is already active. I will warn you about obstacles.", priority=True)
    
    elif any(keyword in command for keyword in ["stop navigation", "pause guiding"]):
        speak("Navigation paused. Say 'start navigation' to resume.", priority=True)
    
    # Check for greeting
    elif any(keyword in command for keyword in ["hello", "hi", "good morning", "good afternoon"]):
        speak("Hello! How can I assist you today?", priority=True)
    
    # Check for thanks
    elif any(keyword in command for keyword in ["thank you", "thanks"]):
        speak("You're welcome! I'm here to help.", priority=True)
    
    else:
        speak("Sorry, I didn't understand that command. You can ask me about money, colors, or what I see.", priority=True)

t_voice_listener = threading.Thread(target=voice_listener_worker, daemon=True)
t_voice_listener.start()

# --- HELPER FUNCTIONS ---
def get_accurate_position(center_x, frame_width):
    if center_x < frame_width * 0.375: return "Left"
    elif center_x > frame_width * 0.625: return "Right"
    else: return "Center"

def estimate_distance(box_height, frame_height=480):
    if box_height == 0: return 99
    
    focal_length = frame_height * 1.2
    estimated_distance = (1.7 * focal_length) / box_height
    
    if estimated_distance < 3.0:
        estimated_distance = estimated_distance * 0.8
    
    return round(max(estimated_distance, 0.5), 1)

def enhance_image_for_ocr(img):
    unflipped = cv2.flip(img, 1) 
    gray = cv2.cvtColor(unflipped, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) 
    return gray

def scan_money_in_roi(frame, x_start, y_start, x_end, y_end):
    """Scan money in the specified ROI and return detected amount"""
    roi = frame[y_start:y_end, x_start:x_end]
    clean_roi = enhance_image_for_ocr(roi)
    try:
        res = reader.readtext(clean_roi, allowlist='0123456789₹Rs$')
        if res:
            all_text = " ".join([r[1] for r in res])
            print(f"Detected text in money box: {all_text}")
            
            # Check for specific currency patterns
            if "500" in all_text or "₹500" in all_text:
                return "500 Rupees"
            elif "200" in all_text or "₹200" in all_text:
                return "200 Rupees"
            elif "100" in all_text or "₹100" in all_text:
                return "100 Rupees"
            elif "50" in all_text or "₹50" in all_text:
                return "50 Rupees"
            elif "20" in all_text or "₹20" in all_text:
                return "20 Rupees"
            elif "10" in all_text or "₹10" in all_text:
                return "10 Rupees"
            elif "5" in all_text or "₹5" in all_text:
                return "5 Rupees"
            elif "$" in all_text or "dollar" in all_text.lower():
                # Extract dollar amounts
                import re
                dollar_amounts = re.findall(r'\$\d+', all_text)
                if dollar_amounts:
                    return f"{dollar_amounts[0]} Dollars"
            else:
                # Try to extract any number sequence
                import re
                numbers = re.findall(r'\d+', all_text)
                if numbers:
                    largest = max(numbers, key=len)
                    if len(largest) >= 2:
                        return f"{largest} Rupees"
        return None
    except Exception as e:
        print(f"Money scan error: {e}")
        return None

# --- MAIN LOOP ---
speak("Drishti Online with Voice Commands. Say 'Hey Drishti' to begin.", priority=True)
last_nav_time = 0
last_person_warning_time = 0
PERSON_WARNING_INTERVAL = 2

frame_counter = 0
start_time = time.time()

while True:
    success, frame = cap.read()
    if not success: 
        print("Failed to read frame from camera. Retrying...")
        time.sleep(1)
        continue
    
    frame = cv2.flip(frame, 1) 
    h, w, _ = frame.shape
    
    with frame_lock:
        latest_frame = frame
    
    # --- VISUALS ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (int(w*0.375), h), (0,0,50), -1) 
    cv2.rectangle(overlay, (int(w*0.625), 0), (w, h), (0,0,50), -1)
    cv2.rectangle(overlay, (int(w*0.375), 0), (int(w*0.625), h), (0,50,0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Money Box (Green rectangle)
    box_size = 180
    x_start, y_start = w//2 - box_size, h//2 - box_size
    x_end, y_end = w//2 + box_size, h//2 + box_size
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.putText(frame, "MONEY SCAN AREA", (x_start, y_start-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Voice command status indicator
    voice_status_color = (0, 255, 0) if voice_command_active else (0, 0, 255)
    voice_status_text = "VOICE ACTIVE" if voice_command_active else "VOICE OFF"
    cv2.putText(frame, voice_status_text, (w-150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_status_color, 2)
    
    # FPS counter
    frame_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_counter / elapsed_time
        frame_counter = 0
        start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # --- AUTOMATIC NAVIGATION & DETECTION ---
    if latest_results:
        zone_blocked = {"Left": False, "Center": False, "Right": False}
        closest_obj = None
        max_height = 0
        person_in_center = None
        
        for box in latest_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h_box = y2 - y1
            
            cls_id = int(box.cls[0])
            obj_name = model.names[cls_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, obj_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 2)
            
            if h_box < h * 0.05: continue 
            
            dist = estimate_distance(h_box, h)
            obj_center = (x1 + x2) / 2
            obj_pos = get_accurate_position(obj_center, w)
            
            dist_text = f"{dist}m"
            cv2.putText(frame, dist_text, (x1, y2+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,100), 1)
            
            if dist < 2.5:
                zone_blocked[obj_pos] = True
            
            if obj_name == "person" and obj_pos == "Center":
                person_in_center = {
                    "distance": dist,
                    "box_height": h_box,
                    "center_x": obj_center
                }
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame, f"PERSON {dist}m", (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if h_box > max_height:
                max_height = h_box
                closest_obj = {
                    "name": obj_name,
                    "center": obj_center,
                    "dist": dist,
                    "pos": obj_pos
                }
        
        current_time = time.time()
        
        # --- PERSON IN CENTER WARNING SYSTEM ---
        if person_in_center:
            distance = person_in_center['distance']
            
            suggestion = ""
            if distance < 1.0:
                suggestion = "Person very close! Step back carefully."
            elif distance < 2.0:
                if not zone_blocked["Left"] and not zone_blocked["Right"]:
                    suggestion = f"Person {distance}m ahead. You can go left or right."
                elif not zone_blocked["Left"]:
                    suggestion = f"Person {distance}m ahead. Suggest moving left."
                elif not zone_blocked["Right"]:
                    suggestion = f"Person {distance}m ahead. Suggest moving right."
                else:
                    suggestion = f"Person {distance}m ahead. Path blocked. Stop."
            else:
                suggestion = f"Person {distance}m ahead. Path is clear."
            
            if current_time - last_person_warning_time > PERSON_WARNING_INTERVAL:
                speak(suggestion, priority=True)
                last_person_warning_time = current_time
            
            cv2.putText(frame, suggestion, (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # --- GENERAL NAVIGATION LOGIC ---
        if current_time - last_nav_time > 3.0:
            if closest_obj and closest_obj['dist'] < 2.5 and closest_obj['name'] != "person":
                pos = closest_obj['pos']
                
                path_advice = ""
                if pos == "Center":
                    if not zone_blocked["Right"]: 
                        path_advice = "Object center. Move Right."
                    elif not zone_blocked["Left"]: 
                        path_advice = "Object center. Move Left."
                    else: 
                        path_advice = "Path Blocked. Stop."
                        
                elif pos == "Left":
                    if not zone_blocked["Right"] and not zone_blocked["Center"]: 
                        path_advice = "Object left. Keep Right."
                    else: 
                        path_advice = "Caution Left."
                         
                elif pos == "Right":
                    if not zone_blocked["Left"] and not zone_blocked["Center"]: 
                        path_advice = "Object right. Keep Left."
                    else: 
                        path_advice = "Caution Right."

                speak(f"{closest_obj['name']} {pos}. {path_advice}", priority=True)
                last_nav_time = current_time

    # --- STATUS DISPLAY ---
    cv2.putText(frame, "DRISHTI AI - VOICE COMMAND ENABLED", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display control instructions
    instructions = [
        "Say 'HEY DRISHTI' to activate voice commands",
        "C: Check Color | SPACE: Read Text | M: Scan Money",
        "V: Toggle Voice | A: Auto Money Scan | Q: Quit"
    ]
    for i, instr in enumerate(instructions):
        cv2.putText(frame, instr, (10, h - 80 + i*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 100), 1)

    # --- MANUAL CONTROLS ---
    key = cv2.waitKey(1) & 0xFF

    # [C] COLOR
    if key == ord('c'):
        is_navigating = False
        speak("Checking color...", priority=True)
        center_px = frame[h//2, w//2]
        hsv = cv2.cvtColor(np.uint8([[center_px]]), cv2.COLOR_BGR2HSV)[0][0]
        hue = hsv[0]
        col = "Unknown"
        if hue < 10 or hue > 170: col = "Red"
        elif hue < 25: col = "Orange"
        elif hue < 35: col = "Yellow"
        elif hue < 85: col = "Green"
        elif hue < 130: col = "Blue"
        elif hue < 170: col = "Purple"
        elif hsv[1] < 30: col = "White"
        if hsv[2] < 40: col = "Black"
        speak(f"It is {col}.", priority=True)
        time.sleep(0.5)
        is_navigating = True

    # [Space] Read Text
    if key == ord(' '):
        is_navigating = False
        speak("Reading text...", priority=True)
        roi = frame[y_start:y_end, x_start:x_end]
        clean_roi = enhance_image_for_ocr(roi)
        try:
            res = reader.readtext(clean_roi)
            if res:
                text = " ".join([r[1] for r in res])
                speak(f"Text says: {text}", priority=True)
            else: 
                speak("No text detected.", priority=True)
        except Exception as e:
            speak("Error reading text.", priority=True)
        time.sleep(0.5)
        is_navigating = True

    # [M] Money - Manual Scan
    if key == ord('m'):
        is_navigating = False
        speak("Scanning money...", priority=True)
        
        # Visual scanning animation
        for i in range(y_start, y_end, 20):
            scan = frame.copy()
            cv2.line(scan, (x_start, i), (x_end, i), (0, 0, 255), 2)
            cv2.imshow("Drishti-AI Final", scan)
            cv2.waitKey(1)
        
        # Scan money
        amount = scan_money_in_roi(frame, x_start, y_start, x_end, y_end)
        if amount:
            speak(f"Detected: {amount}.", priority=True)
        else:
            speak("No currency detected. Please place money in the green box.", priority=True)
        
        time.sleep(1)
        is_navigating = True
    
    # [V] Toggle Voice Command Mode
    if key == ord('v'):
        voice_command_active = not voice_command_active
        if voice_command_active:
            speak("Voice commands activated. Say 'Hey Drishti' followed by your command.", priority=True)
        else:
            speak("Voice commands deactivated.", priority=True)
    
    # [H] Manual Trigger for Voice Commands (for testing)
    if key == ord('h'):
        speak("Hello! I'm listening. You can ask me about money, colors, or what I see.", priority=True)
        voice_command_active = True
        time.sleep(2)  # Give time for voice command
    
    # [A] Auto Money Scan (continuous scanning)
    if key == ord('a'):
        speak("Auto money scan activated. Looking for currency...", priority=True)
        auto_scan_start = time.time()
        amount_detected = None
        
        while time.time() - auto_scan_start < 10:
            current_amount = scan_money_in_roi(frame, x_start, y_start, x_end, y_end)
            
            if current_amount and current_amount != amount_detected:
                amount_detected = current_amount
                speak(f"Detected: {amount_detected}", priority=True)
            
            display_frame = frame.copy()
            cv2.putText(display_frame, "AUTO SCAN ACTIVE", (w//2-100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Detected: {amount_detected if amount_detected else 'None'}", 
                       (w//2-120, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press any key to stop", (w//2-100, h-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            cv2.imshow("Drishti-AI Final", display_frame)
            
            if cv2.waitKey(1) & 0xFF != 255:
                break
        
        if amount_detected:
            speak(f"Final detection: {amount_detected}. Auto scan complete.", priority=True)
        else:
            speak("No currency detected during auto scan.", priority=True)

    cv2.imshow("Drishti-AI Final", frame)
    if key == ord('q'):
        is_running = False
        speak("Shutting down. Goodbye!", priority=True)
        break

cap.release()
cv2.destroyAllWindows()
