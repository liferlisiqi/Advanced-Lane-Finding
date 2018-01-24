import cv2

vidcap = cv2.VideoCapture('harder_challenge_video.mp4')
success, image = vidcap.read()
print(success)
count = 1000
cv2.imwrite("harder/%d.jpg" % count, image)
success = True
while success:
    success, image = vidcap.read()
    print(success)
    cv2.imwrite("harder/%d.jpg" % count, image)
    count += 1
