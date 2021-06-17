import cv2
from imutils.video import WebcamVideoStream

#WebCamVideoStream will take images form the camera

class VideoCamera(object):
    def __init__(self):

        # initializing webcam
        # src will depend on your system
        # start will turn on your camera
        self.stream=WebcamVideoStream(src=0).start()
    
    def __del__(self):


        #to stop the camera
        self.stream.stop()

    def get_frame(self):
        #collecting the image
        image=self.stream.read()

        #creating the detector
        detector=cv2.CascadeClassifier('')




        ret,jpeg=cv2.imencode('.jpg',image)
        data=[]
        #image is stored in bytes
        data.append(jpeg.tobytes())
        return data