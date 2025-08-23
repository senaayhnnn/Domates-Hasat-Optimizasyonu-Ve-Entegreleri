from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
import math
import time

colors_hsv = {
    'red': [
        ([0, 100, 100], [10, 255, 255]),
        ([160, 100, 100], [179, 255, 255])
    ],
    'orange': [
        ([11, 100, 100], [25, 255, 255])
    ],
    'green': [
        ([35, 60, 60], [85, 255, 255])
    ]
}

color_bgr_map = {
    'red': (0, 0, 255),
    'orange': (0, 165, 255),
    'green': (0, 255, 0),
}

model = YOLO(r"C:\Users\pc\OneDrive\Masaüstü\bombus\runs\segment\train3\weights\best.pt")
cap = cv2.VideoCapture(r"C:\Users\pc\Downloads\Video\input_videos\input_videos\IMG_9388.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
panel_width = 600
panel_color = (0, 255, 0)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
color_counts = {'red': set(), 'orange': set(), 'green': set()}

def classify_hsv_color(hsv_pixel):
    hsv_pixel = np.array(hsv_pixel, dtype=np.uint8)
    for color, ranges in colors_hsv.items():
        for lower, upper in ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            if np.all(hsv_pixel >= lower) and np.all(hsv_pixel <= upper):
                return color
    return None

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))
    hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    results = model(frame_resized)[0]
    output = frame_resized.copy()

    boxes = []
    for box, score in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        boxes.append([x1, y1, x2, y2, score])
    boxes = np.array(boxes)

    tracked_objects = tracker.update(boxes)
    masks = results.masks.data.cpu().numpy() if results.masks is not None else []

    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track.astype(int)

        mask = None
        for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
            bx1, by1, bx2, by2 = box.astype(int)
            iou_x1 = max(x1, bx1)
            iou_y1 = max(y1, by1)
            iou_x2 = min(x2, bx2)
            iou_y2 = min(y2, by2)
            iou_w = max(0, iou_x2 - iou_x1)
            iou_h = max(0, iou_y2 - iou_y1)
            iou_area = iou_w * iou_h
            box_area = (bx2 - bx1) * (by2 - by1)
            if box_area == 0:
                continue
            iou = iou_area / box_area
            if iou > 0.5:
                mask = masks[i]
                break
        if mask is None:
            continue

        mask_box = mask[y1:y2, x1:x2]
        hsv_crop = hsv_frame[y1:y2, x1:x2]

        masked_pixels = hsv_crop[mask_box > 0.5]
        if masked_pixels.size == 0:
            continue

        mean_hsv = np.mean(masked_pixels, axis=0).astype(np.uint8)
        color_class = classify_hsv_color(mean_hsv)
        if color_class is None:
            continue

        box_color = color_bgr_map[color_class]
        color_counts[color_class].add(track_id)

        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(output, f"{color_class.upper()} ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        mask_vis = (mask > 0.5).astype(np.uint8) * 255
        colored_mask = np.zeros_like(output)
        colored_mask[:, :] = box_color
        alpha = 0.3
        mask_indices = mask_vis.astype(bool)
        output[mask_indices] = cv2.addWeighted(output, 1 - alpha, colored_mask, alpha, 0)[mask_indices]

    toplamd = len(color_counts['red']) + len(color_counts['orange']) + len(color_counts['green'])
    if toplamd > 0:
        olgunluk_yuzdesi = (
            len(color_counts['red']) * 1.0 +
            len(color_counts['orange']) * 0.7 +
            len(color_counts['green']) * 0.2
        ) / toplamd * 100
    else:
        olgunluk_yuzdesi = 0

    if olgunluk_yuzdesi < 40:
        feedback = "Hasat tavsiye edilmez, domatesler olgunlasmamis."
    elif olgunluk_yuzdesi < 70:
        feedback = "Hasatin zamani var, domatesler yari olgun."
    elif olgunluk_yuzdesi < 90:
        feedback = "Hasat tavsiye edilir, domatesler buyuk oranda olgun."
    else:
        feedback = "Domatesler tam olgun, hemen hasat edilebilir."

    print(f"Olgunluk Yuzdesi: %{olgunluk_yuzdesi:.2f}")
    print(f"Durum: {feedback}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

  
    info_panel = np.ones((output.shape[0], panel_width, 3), dtype=np.uint8) * 200  # Gri arka plan

    
    cv2.putText(info_panel, 'Hasat Optimizasyonu', (10, 40), font, 1.0, (0, 0, 0), 2)
    cv2.putText(info_panel, f'Yesil Domatesler: {len(color_counts["green"])}', (40, 90), font, 0.9, (0, 128, 0), 2)
    cv2.putText(info_panel, f'Turuncu Domatesler: {len(color_counts["orange"])}', (40, 140), font, 0.9, (0, 140, 255), 2)
    cv2.putText(info_panel, f'Kirmizi Domatesler: {len(color_counts["red"])}', (40, 190), font, 0.9, (0, 0, 255), 2)

    elapsed_time = time.time() - start_time
    radius = 10
    orbit_radius = 15
    center_x = 20
    positions_y = {'green': 90, 'orange': 140, 'red': 190}

    for color in ['green', 'orange', 'red']:
        angle = (elapsed_time * 2 * math.pi) % (2 * math.pi)
        if color == 'orange':
            angle += 2
        elif color == 'red':
            angle += 4

        cx = int(center_x + orbit_radius * math.cos(angle))
        cy = positions_y[color] + int(orbit_radius * math.sin(angle))

        cv2.circle(info_panel, (cx, cy), radius, color_bgr_map[color], -1)

    
    bar_x, bar_y = 10, 320
    bar_width_max = 260
    bar_height = 25
    bar_width = int(bar_width_max * (olgunluk_yuzdesi / 100))

    title_text = "Olgunluk Yuzdesi"
    (title_w, title_h), _ = cv2.getTextSize(title_text, font, 0.7, 2)
    title_x = bar_x + (bar_width_max - title_w) // 2
    title_y = bar_y - 10
    cv2.putText(info_panel, title_text, (title_x, title_y), font, 0.7, (0, 0, 0), 2)

    side_text = f"{olgunluk_yuzdesi:.1f}%"
    (side_w, side_h), _ = cv2.getTextSize(side_text, font, 0.7, 2)
    side_x = bar_x + bar_width_max + 10
    side_y = bar_y + bar_height - 5
    cv2.putText(info_panel, side_text, (side_x, side_y), font, 0.7, (0, 0, 0), 2)

    if olgunluk_yuzdesi < 40:
        bar_color = (0, 0, 255)
    elif olgunluk_yuzdesi < 70:
        bar_color = (0, 165, 255)
    elif olgunluk_yuzdesi < 90:
        bar_color = (0, 255, 255)
    else:
        bar_color = (0, 255, 0)

    cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + bar_width_max, bar_y + bar_height), (180, 180, 180), -1)
    cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), bar_color, -1)
    cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + bar_width_max, bar_y + bar_height), (0, 0, 0), 2)

    feedback_pos = (10, 370)
    cv2.putText(info_panel, feedback, feedback_pos, font, 0.7, (0, 0, 0), 2)

    
    pie_center = (info_panel.shape[1] - 100, 100)  # X sağdan 100 px içeride, Y yukardan 100 px aşağıda
    pie_radius = 80

    counts = [len(color_counts['green']), len(color_counts['orange']), len(color_counts['red'])]
    total_counts = sum(counts)
    if total_counts == 0:
        total_counts = 1  

    start_angle = 0
    for i, count in enumerate(counts):
        color_name = ['green', 'orange', 'red'][i]
        angle = int(360 * count / total_counts)
        end_angle = start_angle + angle
        cv2.ellipse(info_panel, pie_center, (pie_radius, pie_radius), 0, start_angle, end_angle, color_bgr_map[color_name], -1)
        start_angle = end_angle

    cv2.circle(info_panel, pie_center, int(pie_radius * 0.5), (200, 200, 200), -1)

    cv2.putText(info_panel, f'Toplam Domates: {toplamd}', (10, info_panel.shape[0] - 40), font, 0.9, (0, 0, 0), 2)

    combined = np.hstack((output, info_panel))
    cv2.imshow("Optimizasyon", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
