# this auto_portrait_album_creator.py programme is designed to detect human face in an image then put it into a potrait album
# the programme will use os package to move image to the assigning file

import mediapipe as mp
import re
import cv2
import os
import matplotlib.pyplot as plt
import shutil

# define those models
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# define the locations & variable
# mypath="Your photos location/"
mypath=""


### Uncomment and run this section to create directory 
# target = ("human", "non_human")
# index=0
# for t in target:
#     if not os.path.exists(mypath + "/" + t):
#         os.makedirs(mypath + "/" + t)
#     if index==0:
#         trueLocation = mypath + "/" + t
#     elif index==1:
#         falseLocation = mypath + "/" + t
#     index+=1
####


#true means the image belongs to human
trueLocation="D:/GT19060/B_earlyWhatsapp/human"
falseLocation=mypath
print(trueLocation,falseLocation)
IMAGE_FILES=[]

# load jpg/jpeg file to list
file =os.listdir(mypath)[:]
for i in file:
    if re.search("^.*\.jpe*g$",i):
        #j = cv2.cvtColor(cv2.imread(re.search("^.*\.jpg$",i).group()),cv2.COLOR_BGR2RGB)
        #IMAGE_FILES.append(j)
        w=re.search("^.*\.jpe*g$",i).group()
        IMAGE_FILES.append(w)
print(IMAGE_FILES[:5],"\n",len(IMAGE_FILES))


### uncomment to view some image
# plt.subplot(1, len(image),1)
# plt.imshow(image[0])
# plt.subplot(1, len(image),2)
# plt.imshow(image[1])
# plt.subplot(1, len(image),3)
# plt.imshow(image[2])
# plt.show()
###

# classification
# For static images:
# https://developers.google.com/mediapipe/solutions/vision/face_detector (the model can be tuned)
def face(preview):
    with mp_face_detection.FaceDetection(
            model_selection=2, min_detection_confidence=0.6) as face_detection:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(mypath+"/"+file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if preview==True:
                if not results.detections:
                    sourse = mypath + "/" + file
                    newloc = falseLocation + "/" + file
                    #print("From:", sourse, ">>>>>>", "To:", newloc)
                    print("********no change********:")
                else:
                    sourse = mypath + "/" + file
                    newloc = trueLocation + "/" + file
                    print("*From:", sourse, ">>>>>>", "To:", newloc)
            else:
                if not results.detections:
                    sourse = mypath + "/" + file
                    newloc = falseLocation + "/" + file
                    #shutil.move(sourse, newloc)
                    #print("From:", sourse, ">>>>>>", "To:", newloc)
                    print("********no change********:")
                else:
                    sourse=mypath+"/"+file
                    newloc=trueLocation+"/"+file
                    shutil.move(sourse,newloc)
                    print("*From:",sourse,">>>>>>","To:",newloc)
       


def pose(preview):
    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5) as holistic:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(mypath+"/"+file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if preview==True:
                if results.pose_landmarks:
                    sourse = mypath + "/" + file
                    newloc = trueLocation + "/" + file
                    print("*From:", sourse, ">>>>>>", "To:", newloc)

                else:
                    sourse = mypath + "/" + file
                    newloc = falseLocation + "/" + file
                    print("From:", sourse, ">>>>>>", "To:", newloc)
            else:
                if results.pose_landmarks:
                    sourse = mypath + "/" + file
                    newloc = trueLocation + "/" + file
                    shutil.move(sourse, newloc)
                    print("*From:", sourse, ">>>>>>", "To:", newloc)

                else:
                    sourse = mypath + "/" + file
                    newloc = falseLocation + "/" + file
                    shutil.move(sourse, newloc)
                    print("From:", sourse, ">>>>>>", "To:", newloc)


face(preview=False)