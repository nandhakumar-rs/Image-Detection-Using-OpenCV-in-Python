import cv2
image_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def image_proces(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = image_cascade.detectMultiScale(gray,
                                             scaleFactor=1.3,
                                             minNeighbors=5)
    for x, y, w, h in cascade:
        img = cv2.rectangle(img, (x, y), (w + x, h + y), (0, 255, 0), 5)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    # resize_img = cv2.resize(img ,(int(img.shape[1]/2),(int(img.shape[0]/2))))

    cv2.imshow("show", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = input("Enter the image url to recognize the face : ")
image_proces(image)

# pip install -U numpy