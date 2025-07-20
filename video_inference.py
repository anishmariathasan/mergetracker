from ultralytics import YOLO
import cv2, os

#gets absolute path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
#for the nano model
#weights_path = os.path.join(script_dir, 'best.pt')
#for the medium model 
weights_path = os.path.join(script_dir, 'bestv11m.pt')
model = YOLO(weights_path)

live = input("Do you want to use a live camera feed? (y/n): ")
if live == 'y':
    # 2. Open the live camera feed
    cap = cv2.VideoCapture(0)  # 0 for the default camera
# 2. Specify the path to your new video
else:
    print("Using video file...(click q to exit)")
    video_path = input("enter the path to your video file: ")
    cap = cv2.VideoCapture(video_path)


# Check if video opened successfully "C:\Users\anish_0fykajq\OneDrive\Documents\Internships\ABIC\merge_detector\2530fps_junct_30pbfps.mp4"
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 3. Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv11n tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, device='0')

    # Visualise the results on the frame
    annotated_frame = results[0].plot()

    # Overlay detection details
    boxes = results[0].boxes
    num_objects = len(boxes)
    cv2.putText(annotated_frame, f"Objects detected: {num_objects}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # display the class name and confidence for each cell
    for i, box in enumerate(boxes):
        cls = int(box.cls[0]) if hasattr(box, 'cls') else -1
        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0
        label = f"ID:{i} Class:{cls} Conf:{conf:.2f}"
        x, y, w, h = box.xywh[0]
        cv2.putText(annotated_frame, label, (int(x), int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the annotated frame
    cv2.imshow("YOLOv11n Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("finished scanning live image feed")