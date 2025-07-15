from ultralytics import YOLO
import cv2, os
from collections import defaultdict, deque

#gets absolute path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, 'best.pt')

model = YOLO(weights_path)

live = input("Do you want to use a live camera feed? (y/n): ")
if live == 'y':
    # 2. Open the live camera feed
    cap = cv2.VideoCapture(0)  # 0 for the default camera
# 2. Specify the path to your new video
else:
    print("Using video file...")
    video_path = input("enter the path to your video file: ")
    cap = cv2.VideoCapture(video_path)


# Store the track history in the lambda funciton
track_history = defaultdict(lambda: [])

# Define the vertical line (ROI) for counting
# We'll count cells crossing the horizontal centre of the frame
roi_line_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2 / 3)

# Initialise counters
large_count = 0
small_count = 0

# Track last 20 crossing scenarios
crossing_scenarios = deque(maxlen=20)
triggered = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO tracking
    results = model.track(frame, persist=True)
    
    # Get the boxes and track IDs
    try:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        labels = results[0].boxes.cls.int().cpu().tolist()
    except AttributeError:
        # If no objects are detected, boxes or id might be None
        track_ids = []
        labels = []

    # Visualise the results on the frame
    annotated_frame = results[0].plot()

    # Draw the ROI line on the frame
    cv2.line(annotated_frame, (roi_line_x, 0), (roi_line_x, frame.shape[0]), (0, 255, 0), 2)

    # Plot the tracks and count crossings
    for box, track_id, label in zip(boxes, track_ids, labels):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y), label))
        if len(track) > 3000:  # Keep history for the last 3000 frames
            track.pop(0)

        # Check for ROI crossing (vertical line)
        if len(track) > 1:
            prev_x = track[-2][0]
            curr_x = track[-1][0]
            prev_label = track[-2][2]
            curr_label = track[-1][2]

            # Only count when crossing from left to right
            if prev_x < roi_line_x and curr_x >= roi_line_x:
                # Count large vs small
                if curr_label == 0:  # Assuming 0 = small, 1 = large
                    small_count += 1
                elif curr_label == 1:
                    large_count += 1

                # Scenario: NOT small then large
                scenario = not (prev_label == 0 and curr_label == 1)
                crossing_scenarios.append(scenario)

                # Optional: draw a circle on crossing
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Trigger signal if >5 NOT small then large in last 20 crossings
    if len(crossing_scenarios) == 20 and sum(crossing_scenarios) > 5 and not triggered:
        print("Signal triggered: More than 5 NOT small then large scenarios in last 20 crossings!")
        triggered = True  # Prevent repeated triggers

    # Display the counts on the frame
    cv2.putText(annotated_frame, f"Large: {large_count}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Small: {small_count}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the annotated frame
    cv2.imshow("Directional Cell Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()