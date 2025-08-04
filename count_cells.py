from ultralytics import YOLO
import cv2, os
from collections import defaultdict, deque

#gets absolute path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, 'bestmaug100.pt')

model = YOLO(weights_path)

live = input("Do you want to use a live camera feed? (y/n/test): ").strip().lower()
if live == 'y':
    cap = cv2.VideoCapture(0)
elif live == "test":
    print("using the test video defined in script")
    cap = cv2.VideoCapture(r"C:\Users\anish_0fykajq\OneDrive\Documents\Internships\ABIC\merge_detector\2530fps_junct_30pbfps.mp4")
else:
    print("Using video file...")
    video_path = input("enter the path to your video file: ")
    print(f"Trying to open: {video_path}")
    cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Failed to open video/camera.")
    exit()

# track last frames with lambda
track_history = defaultdict(lambda: [])

#define the vertical line (ROI) for counting
# count cells crossing the horizontal centre of the frame
roi_line_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2 / 3)

# Initialise counters
large_count = 0
small_count = 0

# Track last 20 crossing scenarios
crossing_scenarios = deque(maxlen=20)
triggered = False

# Add these before the while loop
last_cross_label = None
pair_history = deque(maxlen=20)
triggered = False

# Grab the first frame for ROI and line selection
ret, first_frame = cap.read()
if not ret:
    print("Failed to read from video/camera.")
    exit()

# Let user select ROI
roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select ROI")
x, y, w, h = roi

# Let user click to select vertical line position within ROI
def click_event(event, mx, my, flags, param):
    global roi_line_x
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_line_x = mx
        cv2.destroyWindow("Click vertical line position")

roi_frame = first_frame[y:y+h, x:x+w].copy()
cv2.imshow("Click vertical line position", roi_frame)
cv2.setMouseCallback("Click vertical line position", click_event)
cv2.waitKey(0)

# If user didn't click, default to 2/3 of ROI width
try:
    roi_line_x
except NameError:
    roi_line_x = int(w * 2 / 3)

track_history = defaultdict(lambda: [])
large_count = 0
small_count = 0
pair_count = 0
incorrect_pair = 0
#more efficient than a list (doubly linked list)
crossing_scenarios = deque(maxlen=20)
triggered = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Crop to ROI
    roi_frame = frame[y:y+h, x:x+w]

    # Run YOLO tracking
    results = model.track(roi_frame, persist=True)
    
    # Get the boxes and track IDs
    try:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        labels = results[0].boxes.cls.int().cpu().tolist()
    except AttributeError:
        track_ids = []
        labels = []

    # Visualise the results on the frame
    annotated_frame = results[0].plot()

    # Draw the ROI vertical line
    cv2.line(annotated_frame, (roi_line_x, 0), (roi_line_x, annotated_frame.shape[0]), (0, 255, 0), 2)

    #plot the tracks and count crossings
    for box, track_id, label in zip(boxes, track_ids, labels):
        x_box, y_box, w_box, h_box = box
        track = track_history[track_id]
        track.append((float(x_box), float(y_box), label))
        if len(track) > 3000:
            track.pop(0)

        # Check for ROI crossing (vertical line)
        if len(track) > 1:
            prev_x = track[-2][0]
            curr_x = track[-1][0]
            prev_label = track[-2][2]
            curr_label = track[-1][2]
            

            if prev_x < roi_line_x and curr_x >= roi_line_x:
                if curr_label == 0:
                    small_count += 1
                elif curr_label == 1:
                    large_count += 1

                # Pair counting logic
                if last_cross_label is not None:
                    if last_cross_label == 1 and curr_label == 0:  # large then small
                        pair_history.append(False)  # correct pair
                    else:
                        pair_history.append(True)   # incorrect pair
                last_cross_label = curr_label  # update for next crossing

    

                scenario = not (prev_label == 0 and curr_label == 1)
                crossing_scenarios.append(scenario)

    if len(crossing_scenarios) == 20 and sum(crossing_scenarios) > 5 and not triggered:
        print("Signal triggered: More than 5 NOT small then large scenarios in last 20 crossings!")
        triggered = True

    if len(pair_history) == 20 and sum(pair_history) > 3 and not triggered:
        print("Signal triggered: More than 3 incorrect pairs in last 20!")
        triggered = True

    cv2.putText(annotated_frame, f"Large: {large_count}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Small: {small_count}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"Incorrect pairs (last 20): {sum(pair_history)}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Total pairs: {len(pair_history)}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Directional Cell Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()