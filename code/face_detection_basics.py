import cv2
import mediapipe as mp 
import time

cap = cv2.VideoCapture('videos/mike_harry.mov')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    # imgRGB = cv2.ctvColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(img)
    # print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(img,detection)
            # print(id, detection)
            # print(detection.score)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            img_height, img_width, img_chan = img.shape
            bbox = int(bboxC.xmin * img_width), int(bboxC.ymin * img_height), \
                   int(bboxC.width * img_width), int(bboxC.height * img_height)
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', 
                        (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255,0,255), 2)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,140), cv2.FONT_HERSHEY_PLAIN,
                3, (0,0,255), 2)
    cv2.imshow('image', img)
    cv2.waitKey(1)