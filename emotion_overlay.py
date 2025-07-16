# emotion_overlay.py
#Overlay for emotions 

import cv2 as cv

# Basic emotion to emoji mapping
# TODO:could add more emotions ig 
emotion_emoji = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😡",
    "surprise": "😲",
    "fear": "😨",
    "neutral": "😐"
}

def overlay_emotion(frame, emoji, label):
    #Add  text to frame 
    if emoji:
        text = str(label)

        # Green text works well on most backgrounds hence 0,255,0
        cv.putText(frame, text, (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3) #(frame , text , position , font , scale , colour , thiccness)