import numpy as np
import cv2
import face_recognition as fr

video_capture = cv2.VideoCapture(0)

image = fr.load_image_file(".jpg")  # Place the image url

image_encoding = fr.face_encodings(image)[0]
given_face_encodings = [image_encoding]
given_face_names = ["Write the person name"]     # Write the person name

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    fc_loc = fr.face_locations(rgb_frame)
    fc_encodings = fr.face_encodings(rgb_frame, fc_loc)
    for (top, right, bottom, left), face_encoding in zip(fc_loc, fc_encodings):
        matches = fr.compare_faces(given_face_encodings, face_encoding)

        name = 'Unknown'
        fc_distances = fr.face_distance(given_face_encodings, face_encoding)
        match_index = np.argmin(fc_distances)

        if matches[match_index]:
            name = given_face_names[match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = 0
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('We found', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
