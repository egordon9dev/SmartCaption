import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import TextRenderer
import pickle
from collections import namedtuple
import heapq
import imageio
import matplotlib.pyplot as plt

Caption = namedtuple('Caption', ['character', 'message', 'startTime', 'endTime', 'comments'])
PrioritizedCaption = namedtuple('PrioritizedCaption', ['time', 'counter', 'caption'])

directory = "./data/rock_and_kheart/"

framesDir = directory + "frames/"
framePaths = glob.glob(framesDir + "*.jpg")
print("Reading {} files".format(len(framePaths)), flush=True)

framePaths.sort(key=lambda s: int(s.split("/")[-1][len("frame"):-len(".jpg")]))

captionsPath = directory + "captions.pkl"
objectsPath = directory + "objects.pkl"

with open(captionsPath, 'rb') as captionsFile:
    targetFps = pickle.load(captionsFile)
    allCaptions = pickle.load(captionsFile) # list of (character, message, startTime, endTime[, comments])
with open(objectsPath, 'rb') as objectsFile:
    objects = pickle.load(objectsFile) # Currently: {index: BoundingBox} # list of {"Character": "N x 2 numpy array"} dictionaries

# Priority Queues to store captions
currentCaptions = [] # heap sorted by endTime
futureCaptions = [] # heap sorted by startTime

for i, caption in enumerate(allCaptions):
    futureCaptions.append(PrioritizedCaption(caption.startTime, i, caption))

heapq.heapify(futureCaptions)


frame_list = []
confidences = []
prev_speaker = None
prev_roi = None
color = (255,255,255)
for i, path in enumerate(framePaths):
    if i == 1800: break
    # update priority queues
    while futureCaptions and futureCaptions[0].time <= i:
        _, count, caption = heapq.heappop(futureCaptions)
        heapq.heappush(currentCaptions, PrioritizedCaption(caption.endTime, count, caption))
    while currentCaptions and currentCaptions[0].time < i:
        heapq.heappop(currentCaptions)
    # read frame
    frame = imageio.imread(path)
    # get regions map by ID
    #regionsMap = allObjects[i]
    regionsMap = {}
    # apply captions
    for _, _, caption in currentCaptions:
        if caption.character in regionsMap:
            # apply w/ objection tracking
            region = regionsMap[caption.character]
        else:
            captionWidth, captionHeight = TextRenderer.getCaptionSize(caption.message)
            default_roi = (100, 100, captionWidth, captionHeight)
            if i in objects:
                # ..., roi2, class2, conf2
                roi, class1, confidence, roi2, class2, conf2 = objects[i]
                confidences.append(confidence)
                speaker = class1
                if caption.comments is not None:
                    speaker = caption.comments.strip()
                color = (255, 80, 80) if speaker == "rock" else (80, 80, 255)
                sel_roi = default_roi
                if speaker == class1:
                    sel_roi = roi
                elif speaker == class2:
                    sel_roi = roi2
                if prev_speaker == speaker:
                    damp = .01
                    sel_roi = list(sel_roi)
                    sel_roi[0] = int(damp * sel_roi[0] + (1-damp) * prev_roi[0])
                    sel_roi[1] = int(damp * sel_roi[1] + (1-damp) * prev_roi[1])
                    sel_roi[2] = int(damp * sel_roi[2] + (1-damp) * prev_roi[2])
                    sel_roi[3] = int(damp * sel_roi[3] + (1-damp) * prev_roi[3])
                    sel_roi = tuple(sel_roi)
                prev_speaker = speaker
                prev_roi = sel_roi
                TextRenderer.renderCaption(frame, sel_roi, color, caption.message)
            else:
                confidences.append(0)
                # apply w/o object tracking
                TextRenderer.renderCaption(frame, prev_roi, color, caption.message)
    frame_list.append(frame)
    cv2.imshow("Frame", frame[...,::-1])
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

plt.figure()
plt.clf()
plt.title("Facial Recognition Confidence Values")
plt.xlabel("Time")
plt.ylabel("Confidence")
n_conf = len(confidences)
confidences_avg = np.array(confidences)[:n_conf - n_conf % 30].reshape(-1, 30).mean(1)
plt.plot(confidences_avg)
plt.show()

# ~~~~requires imageio-ffmpeg~~~~
# given a list of frames (numpy arrays), specifically an array of size ((wxhx3)xn) where n is the number of frames,
# convert the sequence of frames into a video
def frames2video(frames):
    imageio.mimwrite('out/videoOutput.mp4', frames, fps=targetFps)

frames2video(frame_list)

