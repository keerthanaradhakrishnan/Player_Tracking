import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO('models/best.pt')
tracker = DeepSort(max_age=30, n_init=2)

cap = cv2.VideoCapture('data/15sec_input_720p.mp4')
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output_option2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
player_cls = None  # we'll discover it below

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    results = model(frame)[0]
    detections = []

    for r in results.boxes.data:
        x1, y1, x2, y2, conf, cls = r.tolist()
        cls = int(cls)
        # Discover player_cls in the first frame
        if frame_count == 1:
            print(f"Detected class {cls} with confidence {conf:.2f}")
        # Once you’ve seen the player class, assign it
        if player_cls is None and cls != 0:
            player_cls = cls
            print(f"Assuming players are class {player_cls}")

        # Filter only players
        if player_cls is not None and cls == player_cls and conf > 0.3:
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, conf, 'player'))

    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        l, t, r, b = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {tid}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Player Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()



