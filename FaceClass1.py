import keras
import pickle
import cv2
import numpy as np
import sys
import time
'''This is to supress the tensorflow warnings. If something odd happens, remove them and try to debug form the warnings'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
'''This is to supress the tensorflow warnings. If something odd happens, remove them and try to debug form the warnings'''



class FaceIndentity:

    dir_path=__file__[:-12]
    face_detection_path= "static/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
    proto_path = "static/face_detection_model/deploy.prototxt"
    model_path = 'static/pickle/holly_MobileNet_3(50_class).h5'
    label_path = 'static/pickle/holly_50_classes_lableencoder.pickle'
    #l = []
    def __init__(self):


        self.detector = cv2.dnn.readNetFromCaffe(self.proto_path, self.face_detection_path)

        self.model = keras.models.load_model(self.model_path)

        self.labelencoder = pickle.load(open(self.label_path,'rb'))



    def predict_image(self, image):
        image_np = np.asarray(image)
        try:
            data = self.getFace_CV2DNN(image)
        except Exception as e:
            data = None
            print("Some Error in image")
        return data


    def getFace_CV2DNN(self, image):
        facelist = []
        (h,w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)),1.0, (300,300),(104.0, 177.0, 123.0), swapRB= False, crop = False)
        self.detector.setInput(blob)
        detections = self.detector.forward()
        fH = 0
        fW = 0
        for i in range(0,detections.shape[2]):
            confidence = detections[0,0,i,2]

            if confidence < 0.7:
              continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(startX,startY,endX,endY)
            #cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 2)
            startX=round(startX*(0.9))
            startY=round(startY*(0.9))
            endX=round(endX*(1.1))
            endY=round(endY*(1.1))
            print(startX,startY,endX,endY)
            cv2.rectangle(image, (startX-50, startY-50), (endX+50, endY+50), (0,0,255), 2)
            cv2.imshow("image",image)
            cv2.waitKey(0)

            fH = endX - startX
            fW = endY - startY
            if fH < 20 or fW < 20:
              continue
            facelist.append((startX,startY,endX, endY))

        return self.setLabel(facelist, image)

    def setLabel(self, facelist,image):
        l = []
        c = 0
        ll = len(facelist)
        for (x1,y1,x2,y2) in facelist:
            face = image[y1:y2, x1:x2]

            if(face.shape == (0,0,3)):
                continue
            im = cv2.resize(face, (224, 224)).astype(np.float32) / 255.0
            im = im.reshape(1,224,224,3)
            out = self.model.predict(im)
            label = np.argmax(out)
            name = self.labelencoder.get(label)[5:]
            percentage = np.max(out)

            l.append([name,percentage])

        return l


reg = FaceIndentity()

path='static/image/15.jpg'
path1 = 'static/image/12.jpg'
#image = cv2.imread(sys.argv[1])
image=cv2.imread(path)

start = int(round(time.time() * 1000))
print("Predict list",reg.predict_image(image))
end = int(round(time.time() * 1000))
print(end-start)

image=cv2.imread(path1)

start = int(round(time.time() * 1000))
print("Predict list",reg.predict_image(image))
end = int(round(time.time() * 1000))
print(end-start)
