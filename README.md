# Player Re-Identification in Single Feed (Option 2)

This project implements real-time **player tracking** using a fine-tuned YOLOv11 model and the DeepSORT tracking algorithm. The goal is to assign **unique, consistent IDs** to each player in a single video feed—even when players go out of frame and re-enter.

## 🎯 Objective

Track players in a 15-second soccer match video (`15sec_input_720p.mp4`) and re-identify them using consistent IDs, even when players temporarily disappear from the frame.

## 📂 Folder Structure
player-tracking
├── single_feed.py # Main script for tracking
├── models/
│ └── best.pt # YOLOv11 pre-trained model
├── data/
│ └── 15sec_input_720p.mp4 # Input video
├── output_option2.mp4 # Output video with IDs
├── requirements.txt # Dependencies
└── README.md

## 🚀 How to Run

1. **Install dependencies**:
```bash
pip install -r requirements.txt
python single_feed.py

View the output in a pop-up window (q to quit).

## Output:
Output video is saved as output_option2.mp4