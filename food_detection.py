from PIL import Image
import cv2
import numpy as np
import os
from fastapi import UploadFile


def detect_and_crop_image(image: UploadFile, output_dir):
    # Yolo 로드
    net = cv2.dnn.readNet("yolov2-tiny.weights", "yolov2-tiny.cfg")

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 이미지 가져오기
    content = image.file.read()
    img = cv2.imdecode(np.fromstring(content, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape

    # 이미지 감지
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # 사진 탐색
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 좌표 추출
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            image_to_crop = Image.fromarray(img_bgr)
            food = image_to_crop.crop((x, y, x + w, y + h))
            food.save(os.path.join(output_dir, f'image_{i + 1}.png'))
