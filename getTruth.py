import os
import sys
import cv2

x_ps = []
y_ps = []
run_time = 120

# initialization
ONLINE = True
CALIBRATE = False
HD = 1280, 640
BGR_COLOR = {'red': (0, 0, 255),
             'green': (127, 255, 0),
             'blue': (255, 127, 0),
             'yellow': (0, 127, 255),
             'black': (0, 0, 0),
             'white': (255, 255, 255)}
WAIT_DELAY = 1
THRESHOLD_WALL_VS_FLOOR = 80
RELATIVE_TRUTH_PATH = 'truth_rat/'
name = ''


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        x_ps.append(x)
        y_ps.append(y)
        print('choosed:', x, y)
    return x_ps, y_ps


def set_truth(file_name):
    global name, WAIT_DELAY, run_time
    name = os.path.splitext(file_name)[0]
    cap = cv2.VideoCapture(file_name)
    ret, frame = cap.read()

    while frame is not None:
        ret, frame = cap.read()

        if frame is None:  # not logical
            break

        r_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
        cv2.putText(frame, 'Time: ' + str('%.0f sec' % r_time), (350, 250),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])

        if ONLINE:
            cv2.imshow(f'Open Field Trace of {name}', frame)

            if r_time % 2 == 0:
                print("At time:", r_time, "the position is")
                cv2.namedWindow("Key frame")
                cv2.setMouseCallback("Key frame", on_EVENT_LBUTTONDOWN)
                cv2.putText(frame, 'Time: ' + str('%.0f sec' % r_time), (350, 250),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])
                cv2.putText(frame, 'Click the position of rat', (350, 280),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])
                cv2.imshow("Key frame", frame)
                cv2.waitKey(0)
                print(x_ps)
                if x_ps[-1] is not None:
                    file = open(RELATIVE_TRUTH_PATH + 'truth.csv', 'a')
                    file.write(str(r_time) + ',%.1f' % x_ps[-1] + ',%.1f\n' % y_ps[-1])
                    file.close()
                    print("x position:", x_ps[-1], "y position", y_ps[-1], 'saved')
                cv2.destroyWindow("Key frame")

            k = cv2.waitKey(WAIT_DELAY) & 0xff
            if r_time >= run_time:
                break
            if k == 27:
                break
            if k == 32:
                if WAIT_DELAY == 1:
                    WAIT_DELAY = 0  # pause
                else:
                    WAIT_DELAY = 1  # play as fast as possible
    cv2.destroyAllWindows()
    cap.release()


def createCSV():
    if not os.path.exists(RELATIVE_TRUTH_PATH):
        os.makedirs(RELATIVE_TRUTH_PATH)
    file = open(RELATIVE_TRUTH_PATH + 'truth.csv', 'w')
    file.write('key frame(second), x position, y position\n')
    file.close()


createCSV()
set_truth("rat.mp4")
