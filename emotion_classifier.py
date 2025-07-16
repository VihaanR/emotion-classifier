# THe main one
# TODO : emoji addition beside emotion | maybe add audio cues ? 
import cv2 as cv
from fer import FER #pre-trained model for emotion recognition
from datetime import datetime
from collections import deque, Counter #for specialised data structures for smoothening and counting
import mediapipe as mp #for mesh (made by Google :O)
import time
from emotion_overlay import emotion_emoji,overlay_emotion #funcs from emotion_overlay.py
from logger import setup_csv, save_data ,display_graphs#funcs from logger.py

# Camera setup (these work fine on my cam)
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 576)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 432) #resolution setup to 576x432px (optimum as per my cam)

# FER detector initialization for emotion detection
detector = FER() 

# Smoothing buffer for emotion predictions (helps with results changing fast and jumping around too much)
recent_emotion = deque(maxlen=10) 

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawer = mp.solutions.drawing_utils

# Need to log all of my emotions for future use
csv_file, writer = setup_csv()

frame_count = 0

# For the FPS
prev_time = time.time()

#For turning off the cameras , the Top right X in the windows wont work but the x from the keyboard will
print("Starting emotion classifier...O....O....O... Press 'x' button to quit") 

while True:
    ret, frame = cap.read()

    #If frames stop being captured (when camera get disconnected or abrupt shutdown)
    if not ret:
        print("No feed available T_T")
        break

    frame=cv.flip(frame,1)  # mirror feed for the need to make yourself look normal

    frame_count +=1
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #Mediapipe needs RGB whereas OpenCV works with BGR

    # Detecting the emotions 
    result = detector.detect_emotions(frame) 
   
    if result:
        emotion, score = detector.top_emotion(frame)
        recent_emotion.append(emotion)
    else:
        recent_emotion.append(None)

    # Get smoothed emotion prediction (application of FDA concept)
    
    valid_emotions = [e for e in recent_emotion if e is not None]#checking if emotion 'e' is not None based on the 10 in deque 
    if valid_emotions:
       
        common_emotion = Counter(valid_emotions).most_common(1) #selecting the most common emotion from the history
        smoothed_emotion = common_emotion[0][0]#extracting the emotion from the Counter tuple [(emotion, count)]
    else:
        smoothed_emotion = ""

    # Face mesh

    mesh_frame = frame.copy() # a copy created to draw the mesh on it

    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks: #goes through each detected face 
            drawer.draw_landmarks(
                mesh_frame,                                 #frame to mesh upon
                face_landmarks,                             #landmarks points on face
                mp_face_mesh.FACEMESH_TESSELATION,          
                landmark_drawing_spec=None,                 #prevents drawing of landmarks on individual points  
                connection_drawing_spec=drawer.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1) #connecting lines
            )# apply the mesh on the mesh frame
            
            # Save the mesh data and emotion to CSV
            save_data(writer, frame_count, smoothed_emotion, face_landmarks)

    # FPS display
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time))
    prev_time = curr_time
    cv.putText(mesh_frame, f"FPS: {fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) #(frame, text, position, font ,scale,colour,thiccness)

    # Add emoji overlay to main frame (no emoji rn cuz they show up as "???" in the output)
    emoji = emotion_emoji.get(smoothed_emotion, "")
    overlay_emotion(frame, emoji, smoothed_emotion)

    # Display both windows
    cv.imshow("Emotion Live Feed ", frame)
    cv.imshow("Face Mesh Feed", mesh_frame)

    #x key to close
    if cv.waitKey(1) & 0xFF == ord('x'):        
        break

# Termination protocol
print("Shutting down...")
csv_file.close()
cap.release()
cv.destroyAllWindows()
face_mesh.close()
display_graphs()  # Show the graphs after the session ends
print("Execution done")

'''
You will notice emoji being there in the code yet no emoji shows up in the output ,well I was planning to show em beside the text
but they show up as "???" so I removed them for now , however there were many variables containing emojis and I am too lazy to remove them
That is in my TODO list for now 
'''
