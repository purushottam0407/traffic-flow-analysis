import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from modles.sort.sort import Sort  # SORT tracker
import yt_dlp as ydl  # yt-dlp for reliable downloading

# Constants
VIDEO_URL = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
OUTPUT_VIDEO = "output\\output_overlay.mp4"
DOWNLOADED_VIDEO = "data\\traffic_input.mp4"
NUM_LANES = 3

# --- Download the YouTube video ---
def download_youtube_video(url, output_path=DOWNLOADED_VIDEO):
    ydl_opts = {
        'format': 'best',
        'outtmpl': DOWNLOADED_VIDEO
    }
    with ydl.YoutubeDL(ydl_opts) as y:
        y.download([url])
    return output_path

# --- Define lanes as vertical sections ---
def get_lane(x_center, frame_width):
    lane_width = frame_width // NUM_LANES
    return int(x_center // lane_width + 1)

# --- Check if the object has crossed the counting line in its lane ---
def is_new_count(vehicle_id, counted_ids_per_lane, lane):
    if vehicle_id not in counted_ids_per_lane[lane]:
        counted_ids_per_lane[lane].add(vehicle_id)
        return True
    return False

# --- Draw lane lines ---
def draw_lanes(frame):
    height, width = frame.shape[:2]
    lane_width = width // 3
    for i in range(1, 3):
        cv2.line(frame, (lane_width * i, 0), (lane_width * i, height), (255, 0, 0), 2)
    return frame

def main():
    # Download video
    if not os.path.exists(DOWNLOADED_VIDEO):
        print("Downloading video...")
        download_youtube_video(VIDEO_URL, DOWNLOADED_VIDEO)
        print("Download completed.")
        
    cap = cv2.VideoCapture(DOWNLOADED_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Load YOLOv8 Model (Make sure YOLOv8 is available)
    model = YOLO("yolov8n.pt")

    # Initialize SORT tracker
    tracker = Sort()

    counted_ids_per_lane = {1: set(), 2: set(), 3: set()}
    vehicle_data = []

    frame_count = 0

    # Output video
    out = cv2.VideoWriter(OUTPUT_VIDEO,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = str(datetime.now().strftime('%H:%M:%S'))

        results = model(frame, verbose=False)[0]
        detections = []

        # Only detect vehicles
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls_id = result
            if int(cls_id) in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                detections.append([x1, y1, x2, y2, score])

        # Update tracker
        trackers = tracker.update(np.array(detections))

        draw_lanes(frame)

        # Process each tracked vehicle
        for track in trackers:
            x1, y1, x2, y2, track_id = [int(x) for x in track]
            cx = int((x1 + x2) / 2)
            lane = get_lane(cx, frame_width)

            if is_new_count(track_id, counted_ids_per_lane, lane):
                vehicle_data.append({
                    'Vehicle_ID': track_id,
                    'Lane': lane,
                    'Frame': frame_count,
                    'Timestamp': timestamp
                })

            # Draw box and info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{track_id} L{lane}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display lane counts
        for lane_num in range(1, 4):
            count = len(counted_ids_per_lane[lane_num])
            cv2.putText(frame, f'Lane {lane_num}: {count}', (10, 30 + lane_num * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Traffic Flow", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
     # Save CSV
    df = pd.DataFrame(vehicle_data)
    df.to_csv("output\\vehicle_counts.csv", index=False)

    # Summary
    print("\n--- Vehicle Count Summary per Lane ---")
    for lane_num in range(1, 4):
        print(f"Lane {lane_num}: {len(counted_ids_per_lane[lane_num])} vehicles")
        
# --- Main ---
if __name__ == "__main__":
    main()
    
