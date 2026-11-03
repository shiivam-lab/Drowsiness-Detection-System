import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import numpy as np

facemodel=cv2.CascadeClassifier("face.xml")
eyemodel=cv2.CascadeClassifier("eye.xml")
drowsinessmodel=load_model("drowsiness.h5")

vid=cv2.VideoCapture("vid1.mp4")
i=1

while(vid.isOpened()):
    flag,frame=vid.read()

    if(flag):
        faces=facemodel.detectMultiScale(frame)

        for(x,y,l,w) in faces:
            face=frame[y:y+w,x:x+l]

            eyes=eyemodel.detectMultiScale(face)

            for(ex,ey,el,ew) in eyes:
                eye=face[ey:ey+ew,ex:ex+el]

                cv2.imwrite("temp.jpg",eye)

                eye=load_img("temp.jpg",target_size=(150,150))
                eye=img_to_array(eye)
                eye=np.expand_dims(eye,axis=0)
                eye=eye/255

                pred=drowsinessmodel.predict(eye)[0][0]

                if(pred<0.5):
                    cv2.rectangle(frame,(x+ex,y+ey),(x+ex+el,y+ey+ew),(0,0,255),3)
                    cv2.putText(frame,"Drowsy",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

                    path="data/"+str(i)+".jpg"
                    cv2.imwrite(path,frame[y:y+w,x:x+l])
                    i=i+1

                else:
                    cv2.rectangle(frame,(x+ex,y+ey),(x+ex+el,y+ey+ew),(0,255,0),3)
                    cv2.putText(frame,"Awake",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            cv2.rectangle(frame,(x,y),(x+l,y+w),(255,0,0),2)

        cv2.namedWindow("shivam window",cv2.WINDOW_NORMAL)
        cv2.imshow("shivam window",frame)

        key=cv2.waitKey(20)

        if(key==ord("x")):
            break

    else:
        break

vid.release()
cv2.destroyAllWindows()
