# logger.py
# Handles CSV logging for emotion and face mesh data
import csv 
import os #for dealing wiht the directories and files
from datetime import datetime
import pandas as pd #for data analysis
import matplotlib.pyplot as plt #for plotting 


def setup_csv(folder="emotion-classifier", filename="emotion_mesh_log.csv"):
    #Setup CSV logging - creates directory if needed
   
    os.makedirs(folder, exist_ok=True) # Make sure the directory exists(kinda imp)
    
    filepath = os.path.join(folder, filename)
    
    file = open(filepath, mode="w", newline="", encoding="utf-8")  #write mode to provide unique data each session
    writer = csv.writer(file) #write the rows in csv file

    # Add header row 
    writer.writerow(["timestamp", "frame", "emotion", "landmark_index", "x", "y", "z"])

    return file, writer #returns the file and writer to the func

def save_data(writer, frame_count, emotion, landmarks):

    #Log emotion and all face mesh points to CSV 

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Each landmark gets its own row , it creates big files but easier to analyze
    
    for idx, lm in enumerate(landmarks.landmark):
        writer.writerow([timestamp, frame_count, emotion, idx, lm.x, lm.y, lm.z])


def display_graphs(csv_path="emotion-classifier/emotion_mesh_log.csv"):
    
    #Gives stats regarding frequency of detected emotions
    try:
        #get the data from csv
        data = pd.read_csv(csv_path)

        #Plot the Emotion Frequency
        emotions = data['emotion']
        emotion_count = (emotions.value_counts())/468 #convert csv rows to frame count (468 rows per frame)

        plt.figure(figsize=(10, 5))
        emotion_count.plot(kind='bar', color='green')
        plt.title("Emotion Frequency : How Often Each Emotion Was Detected ?")
        plt.xlabel("Emotion")
        plt.ylabel("Detections - No. of frames")
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='-', alpha=0.4) #reference horizontal lines for interpretation
        plt.tight_layout()
        plt.show()
    
    #had to add this cuz I once came across abnormalities in csv file and code shutdown randomly
    except Exception as error:
         
        print(f"CANNOT DISPLAY STATS DUE TO THE ERROR: {error}")
