import datetime

import cv2
import time
import datetime
import requests
import base64
import json
import argparse
import numpy as np
import os

DirName="C:/DetectCardVideo"
filename = DirName+"/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".avi"
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-d","--device",default=0,type=int)
    args = parser.parse_args()
    selected_device=args.device

    if not os.path.exists(DirName):
        os.mkdir(DirName)

    cap = cv2.VideoCapture(selected_device)

    cap.set(3, 640)
    cap.set(4, 480)

    count = 0

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps=10, frameSize=(640, 480))
    oldtime = time.time()
    while True:
        ret, img = cap.read()
        count = count + 1


        #cv2.VideoWriter.write(img)
        newtime = time.time()

        if (newtime - oldtime > 1):
            print(str(count / (newtime - oldtime)))
            count = 0
            oldtime = newtime

        retval, buffer = cv2.imencode('.jpg', img)
        img_b64 = "data:image/jpg;base64," + base64.b64encode(buffer).decode("utf-8")

        resp = requests.post("http://127.0.0.1:5002/detect_logo",
                             json={"img_path": img_b64}
                             )
        print(resp)
        resp_data = json.loads(resp.text)

        rslts = resp_data['results']

        for rslt in rslts:
            box = rslt['box']
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 4)
            cv2.putText(img, rslt['class_name'] + ":" + str(round(rslt['confidence'], 2)), (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('Webcam', img)
        out.write(img)


        if cv2.waitKey(1) == ord('q'):
            break

    #cv2.VideoWriter.release()
    cap.release()
    cv2.destroyAllWindows()
