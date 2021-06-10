import cv2
import mediapipe as mp 
import time


class FaceDetector():
    def __init__(self, minDetectConf=0.5):

        self.minDetectConf = minDetectConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectConf)


    def find_faces(self, img, draw=True):

        self.results = self.faceDetection.process(img)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                img_height, img_width, img_chan = img.shape
                bbox = int(bboxC.xmin * img_width), int(bboxC.ymin * img_height), \
                    int(bboxC.width * img_width), int(bboxC.height * img_height)
                bboxs.append([bbox, detection.score])

                cv2.rectangle(img, bbox, (255,0,255), 2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', 
                            (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255,0,255), 2)
        return img, bboxs


def main():
    cap = cv2.VideoCapture('test.mov')
    pTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = detector.find_faces(img)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,140), cv2.FONT_HERSHEY_PLAIN,
                    3, (0,0,255), 2)
        cv2.imshow('image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()