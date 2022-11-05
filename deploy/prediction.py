import tensorflow as tf
import numpy as np
import imutils
import time
import dlib
import cv2
import base64
from PIL import Image
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from imutils.video import FPS
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject


def read_label_map():
    item_id = None
    item_name = None
    label_dic = {}
    with open("label_map.pbtxt", "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":")[1].replace("\"", " ")
                item_name = item_name.replace("'", " ").strip()   

            if item_id is not None and item_name is not None:
                label_dic[item_name] = item_id
                item_id = None
                item_name = None
    return label_dic


class Model():
    def __init__(self):
        detect_fn = tf.saved_model.load("model/fnl_model/saved_model")
        self.detect_fn = detect_fn
    
    def predict(self, video, skip, thres, func, vis):
        video_result = open("/tmp/video_in.mp4", "wb")
        video_result.write(base64.b64decode(video))

        skip_fps= skip
        threshold= thres
        function= func 
        option_vis= vis

        vs = cv2.VideoCapture("/tmp/video_in.mp4")
        writer = None
        W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ct = CentroidTracker(maxDisappeared= 40, maxDistance = 50)

        trackers = []
        trackableObjects = {}

        if function=="counter": counters={id:0 for id in list(read_label_map().values())}

        totalFrame = 0
        totalDown = 0
        totalUp = 0

        fps = FPS().start()

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter("/tmp/output.mp4", fourcc, 20.0, (W, H), True)

        while True:

            ret, frame = vs.read()

            if frame is None:
                break
            
            rects = []
            attributes = []

            if totalFrame % skip_fps == 0:
                status = "Detecting"
                trackers = []
                image_np = np.array(frame)

                input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
                detections = self.detect_fn(input_tensor)

                detection_scores = np.array(detections["detection_scores"][0])
                detection_clean = [x for x in detection_scores if x >= threshold]
                
                for x in range(len(detection_clean)):
                    idx = int(detections['detection_classes'][0][x])
                    ymin, xmin, ymax, xmax = np.array(detections['detection_boxes'][0][x])
                    classes = int(np.array(detections['detection_classes'][0][x]))
                    score = detection_clean[x]
                    attribute = [classes, score]
                    box = [xmin, ymin, xmax, ymax] * np.array([W, H, W, H])

                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(frame, rect)

                    trackers.append(tracker)
                    attributes.append(attribute)

            else:
                for tracker in trackers:
                    status = "Watching"
                    tracker.update(frame)
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    rects.append((startX, startY, endX, endY))

                    if option_vis=="label" or option_vis=="id": cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)

                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)
                    
                    if bool(to.attributes)==False and bool(attributes):
                        to.attributes = attributes

                        if not to.counted:
                            if function=="counter":
                                counters[to.attributes[0][0]] += 1
                                to.counted = True

                    if not to.counted:
                        if function=="tracker" or function==None:
                            if direction < 0 and centroid[1] < H//2:
                                totalUp += 1
                                to.counted = True
                            elif direction > 0 and centroid[1] > H//2:
                                totalDown += 1
                                to.counted = True
                            

                trackableObjects[objectID] = to

                if function=="tracker" or function==None: cv2.line(frame, (0, H//2), (W, H//2), (0,0,255), 2)
                                
                if bool(to.attributes):
                    if option_vis=="id" or option_vis==None:
                        text = "ID {}".format(objectID)
                        cv2.putText(frame, text, (centroid[0]-20, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,0,255), -1)
                    if option_vis=="label":
                        text = "{0} at {1:.2f}%".format(list(read_label_map().keys())[to.attributes[0][0]-1],(float(to.attributes[0][1])*100))
                        cv2.putText(frame, text, (centroid[0]-50, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,0,255), -1)
                    if option_vis=="centroid":
                        text = "ID {}".format(objectID)
                        cv2.putText(frame, text, (centroid[0]-20, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,0,255), -1)
                    if option_vis=="only_centroid":
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,0,255), -1)        
        

            if function=="tracker" or function==None:
                info = [("Subiendo", totalUp), ("Bajando", totalDown), ("Estado", status)]
                cv2.rectangle(frame, (5, H - ((len(info)*20) + 20)), ( int(W*0.25) , H - 10), (0, 0, 0), -1)

            if function=="counter":
                info = [("Labels: ", {name:cont for (name,cont) in zip(read_label_map().keys(),list(counters.values()))}), ("Estado", status)]
                cv2.rectangle(frame, (5, H - ((len(info)*20) + 20)), ((len(info[0][1])*125), H - 10), (0, 0, 0), -1)

            for (i, (k,v)) in enumerate(info):
                text = "{}: {}".format(k,v)
                cv2.putText(frame, text, (10, H - ((i*20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            writer.write(frame)
            totalFrame += 1
            fps.update()

        fps.stop()

        writer.release()
        vs.release()

        video_out = open("/tmp/output.mp4", "rb")
        video_read = video_out.read()
        image_64_encode = base64.b64encode(video_read)
        image_64_encode_return = image_64_encode.decode() 
        return image_64_encode_return
