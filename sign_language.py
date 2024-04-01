import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
finger_tips =[8, 12, 16, 20]
thumb_tip= 4
finger_fold_status=[] # initializing finger_fold_status list

while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    results = hands.process(img)


    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            #accessing the landmarks by their position
            lm_list=[]
            for id ,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

             #Draw blue colored circles around the fingertips
            for tip_id in finger_tips:
                x,y=int(lm_list[tip_id].x * w), int(lm_list[tip_id].y * h)
                cv2.circle(img,(x,y), 15, (255,0,0), cv2.FILLED)
            #Check if fingers are folded or not
            if lm_list[tip_id].x<lm_list[tip_id-3].x:
                cv2.circle(img,(x,y), 15, (0, 255, 0), cv2.FILLED)
                finger_fold_status.append(True)
            else:
                finger_fold_status.append(False)
    
    if all(finger_fold_status):
          #check if thumb is down
            if lm_list[thumb_tip].y>lm_list[thumb_tip-1].y < lm_list[thumb_tip-2].y:
                 print("LIKE")
                 cv2.putText(img, "LIKE", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)


            if lm_list[thumb_tip].y>lm_list[thumb_tip-1].y > lm_list[thumb_tip-2].y:
                 print("DISLIKE")
                 cv2.putText(img, "DISLIKE", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)




            mp_draw.draw_landmarks(img, hand_landmark,
            mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
            mp_draw.DrawingSpec((0,255,0),4,2))
    

    cv2.imshow("hand tracking", img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
    