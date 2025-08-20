#  Traffic Flow Analysis using YOLOv8 + SORT

This project studies traffic flow from a video by detecting, tracking, and counting vehicles in multiple lanes.  
It uses YOLOv8 for object detection and SORT (Simple Online Realtime Tracking) for tracking multiple objects.  
Results are marked on the video and saved in a CSV file.

---

##  Features
- Automatic Video Download: Fetches a traffic video from YouTube (or use your own video).  
- Vehicle Detection: Detects vehicles using YOLOv8.  
- Lane-wise Counting: Divides the road into three lanes and counts vehicles in each lane.  
- Annotated Output Video: Saves an annotated MP4 with bounding boxes, IDs, and lane information.  
- CSV Export: Saves per-vehicle data (Vehicle_ID, Lane, Frame, Timestamp).   
- Summary Report: Shows lane-wise counts in the console.

---

##  Project Structure
```
Traffic_Flow_Analysis/
│
├── data/
│   └── traffic_input.mp4       # Downloaded or manually added input video
│
├── output/
│   ├── vehicle_counts.csv      # Lane-wise vehicle counts
│   └── output_overlay.mp4    # Annotated video
│
├── models/
│   └── sort/
│       └── sort.py             # SORT tracker implementation
│
├── traffic_flow.py             # Main script
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

##  Installation

1. Clone this repository
   ```bash
   git clone https://github.com/your-username/Traffic_Flow_Analysis.git
   cd Traffic_Flow_Analysis
   ```

2. Create a virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

---

##  Dependencies
- Python 3.8+
- OpenCV
- NumPy
- Pandas
- Ultralytics YOLOv8
- SORT (included in repo)
- yt-dlp (for YouTube download)

Install all at once:
```bash
pip install opencv-python numpy pandas ultralytics yt-dlp
```

---

##  Usage

### Option 1: Run with YouTube video
The script will auto-download the default traffic video:
```bash
python traffic_flow_analysis.py
```

### Option 2: Use your own video
1. Place your video in the `data/` folder.  
2. Update the `DOWNLOADED_VIDEO` path in `traffic_flow_analysis.py`.  
3. Run:
   ```bash
   python traffic_flow_analysis.py
   ```

---

##  Output

### 1. Annotated Video
- Bounding boxes with vehicle ID and lane number  
- Lane separator lines drawn  
- Lane-wise counts displayed on screen  

### 2. CSV File
Example `vehicle_counts.csv`:
```csv
Vehicle_ID,Lane,Frame,Timestamp
1,1,15,12:01:30
2,2,20,12:01:31
3,3,35,12:01:32
```

### 3. Console Summary
```
--- Vehicle Count Summary per Lane ---
Lane 1: 34 vehicles
Lane 2: 41 vehicles
Lane 3: 29 vehicles
```

---

##  Notes
- Default: 3 lanes (can be changed in `NUM_LANES`).  
- Model used: `yolov8n.pt` (lightweight). Replace with `yolov8s.pt` / `yolov8m.pt` for higher accuracy.  
- Works best with **fixed camera angle** traffic footage.  

---

##  License
This project is open-source and available under the [MIT License](LICENSE).

---

##  Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection  
- [SORT](https://github.com/abewley/sort) for object tracking  
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video downloads  
