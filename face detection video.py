import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

#You can put the video you want type the adress instead of example.mp4

capture = cv.VideoCapture(r'...\example.mp4')

while True:
    is_True, frame = capture.read()

    #for testing purposes
    #cv.imshow('Video' , frame)

    gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)

    #for testing purposes
    #cv.imshow('Video gray',gray)

    #Faces with 40 pixels or less will not be detected. You can change minSize to change that.
    #You can increase scaleFactor to 1.2 or more to use less gpu,cpu and you can decrease the miNeighbors to 2 or 1 to lower gpu , cpu use.
    #These changes will decrease reliability but will increase the performance.

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,minSize=(40, 40))
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x-10, y-10), (x + w + 10, y + h + 10), (0, 255, 0), thickness=2)
    cv.imshow('face detected video', frame)

    #You can quit with pressing d button

    if cv.waitKey(3) & 0xFF==ord('d'):
        break

cv.destroyAllWindows()
capture.release()
