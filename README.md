# Emotion Classifier

A real-time python project for live facial emotion detection and face mesh visualization using your webcam. This project uses computer vision and deep learning to track and log emotions, overlay results, and analyze facial data for later insights.

---

## 🚀 Features

- Real-time emotion detection with smooth result overlay
- Facial mesh visualization using MediaPipe
- Automatic logging of emotions and mesh points to CSV for analysis
- Simple visualization of emotion statistics after a session

---

## 🛠️ Installation Guide

# Directory Structure:
```
 emotion-classifier/
  ├── emotion_classifier.py          # Main file (camera + overlay + mesh + logging)
  ├── emotion_overlay.py             # Emoji rendering logic
  ├── logger.py                      # Emotion + face mesh logging combined
  ├── requirements.txt               # Dependencies
  └── README.md                      # Project description and usage instructions
```

### 1. Clone the Repository

First, clone this repository to your local machine using Git:

```sh
git clone https://github.com/VihaanR/emotion-classifier.git
cd emotion-classifier
```

### 2. Setup Python

This project requires **Python 3.11** for compatibility with TensorFlow (required by the `fer` library).  
Make sure you have Python 3.11 installed on your system.

### 3. Install Required Packages

Install all dependencies with pip:

```sh
pip install -r requirements.txt
```

---

## 🎬 Running the Project

After installing the dependencies, you can start the project:

```sh
python emotion_classifier.py
```

- BE PATIENT it will likely take some time , also there will be a couple of "warning" messages but ignore those and wait until two cam feeds show up
- The webcam feed will launch, showing real-time emotion predictions and facial mesh overlay.
- Press **'x'** on your keyboard to exit at any time.

---

## 📊 Analyzing Results

When you finish a session (by pressing 'x'), a CSV log of all detected emotions and face mesh points is saved in `emotion-classifier/emotion_mesh_log.csv`

Due to variable file path setup , you might not find the csv upfront in the project folder , it might end up in a subfolder also name `emotion-classifier `

A graph will automatically display showing the frequency of detected emotions.

---

## 📝 Note

- **Python Version:** The project is tested on Python 3.11. Other versions may cause dependency errors (especially with TensorFlow).
- **Webcam Issues:** If your webcam is not detected or the feed freezes after sleep, restart the project.
- **Emoji Overlay:** Emoji display is disabled due to Unicode issues; emotion labels are shown as text instead.
- **Graphs Not Showing:** If the stats graph does not appear, check your matplotlib installation.

---

## 📦 Dependencies

All main dependencies are listed in `requirements.txt`:

- `opencv-python`
- `fer`
- `mediapipe`
- `pandas`
- `matplotlib`

---

## 🤖 Possible Applications

1. **Mental Health Monitoring:** Track emotional states for personal wellness or therapist insights.(the most classic one imo)

2. **Human-Computer Interaction:** Make adaptive interfaces for games, education, or accessibility.Can be a great feature for increasing immersion.

3. **Customer Experience:** Analyze real-time reactions in retail, events, or market research.People often lie but stats don't. Real Time Data Analysis would reveal the reactions of each person.

---

## 🚧 WHAT'S NEXT

Here are some features and experiments I plan to explore and build into the project:

- [ ] **Brightness Regulator:** Inspired by the Google Meet background filter brightness feature. I noticed emotion detection is affected by lighting changes, so I want to add a feature that automatically adjusts the webcam feed's brightness to an optimal level by dynamically altering frame properties.

- [ ] **Audio and Visual Overlay:** After the emoji issue, I'm keen to learn and implement robust overlay of images (like expressive icons) and audio cues to enhance feedback.

- [ ] **Gesture Control:** Integrate gesture recognition to control parts of the application hands-free and bring in another layer of interactivity.

- [ ] **UI:** Design and implement a user interface that wraps all these features into an intuitive and seamless experience.

---

## 🙌 Contributing

Pull requests and suggestions are welcome! Please open an issue for bugs or feature requests.

---

**Questions or feedback?**  
Open an issue, or connect with me on [LinkedIn](https://www.linkedin.com/in/vihaan-raut-423b8b313/).
