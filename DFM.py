import time
import cv2
import numpy as np
import datetime
import os
import csv

# initialization
BGR_COLOR = {'red': (0, 0, 255),
             'green': (127, 255, 0),
             'blue': (255, 127, 0),
             'yellow': (0, 127, 255),
             'black': (0, 0, 0),
             'white': (255, 255, 255)}
HD = 1280, 640

# path for the result saving
file_name = 'Scene1.mp4'
RELATIVE_DESTINATION_PATH = file_name + '_result/'
RELATIVE_TRUTH_PATH = 'truth_rat/'


def trace(file_name):
    # initialize
    distance = old_x = old_y = 0
    tem_dis_per_frame = 0
    total_error = 0
    WAIT_DELAY = 1
    start = time.time()

    # capturing the video
    capturedVideo = cv2.VideoCapture(file_name)
    h, w = int(capturedVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capturedVideo.get(cv2.CAP_PROP_FRAME_WIDTH))

    # reading two frames
    count, frame1 = capturedVideo.read()
    count, frame2 = capturedVideo.read()
    imgTrack = np.zeros_like(frame1)

    while capturedVideo.isOpened():
        r_time = capturedVideo.get(cv2.CAP_PROP_POS_MSEC) / 1000.

        # graying two frames
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        blur1 = cv2.GaussianBlur(gray_frame1, (5, 5), 0)
        blur2 = cv2.GaussianBlur(gray_frame2, (5, 5), 0)

        differenceOfFrames = cv2.absdiff(blur1, blur2)
        # find the difference between first fame and second frame
        # differenceOfFrames = cv2.absdiff(gray_frame1, gray_frame2)

        # converting frames from BGR to GRAY , Easy to find contours in the gray scale mode
        # grayFrame = cv2.cvtColor(differenceOfFrames, cv2.COLOR_BGR2GRAY)

        # blurring(frame name, k_size, sigmaX value )
        # blur = cv2.GaussianBlur(differenceOfFrames, (5, 5), 0)
        # cv2.imshow("blur", blur)

        # _ we don't need first variable(src, threshold_value, max , type)
        _, thresh = cv2.threshold(differenceOfFrames, 10, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh", thresh)

        dilated = cv2.dilate(thresh, None, iterations=3)

        # going to find the contours in the dilated image(img, mode, method )
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 1:
            continue

        contour = contours[np.argmax(list(map(cv2.contourArea, contours)))]

        # find the center point of the animal
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        if old_x == 0 and old_y == 0:
            old_x = x
            old_y = y
            tem_contour = np.zeros_like(contour)
        dis_per_frame = np.sqrt(((x - old_x) / float(h)) ** 2 + ((y - old_y) / float(h)) ** 2)


        if cv2.contourArea(contour) > 4000 and dis_per_frame < 0.07:
            # print('good:', cv2.contourArea(contour), dis_per_frame)

            distance += dis_per_frame
            speed = dis_per_frame * 10

            # center_x = x_rec + w_rec / 2
            # center_y = y_rec + h_rec / 2
            # cv2.circle(frame1, (int(center_x), int(center_y)), 5, BGR_COLOR['green'], -1, cv2.LINE_AA)

            # Draw a track line of the animal movement
            imgTrack = cv2.addWeighted(np.zeros_like(imgTrack), 0.85,
                                       cv2.line(imgTrack, (x, y), (old_x, old_y),
                                                (255, 127, int(capturedVideo.get(cv2.CAP_PROP_POS_AVI_RATIO) * 255)),
                                                1, cv2.LINE_AA), 0.98, 0.)

            imgFinal = cv2.add(frame1, imgTrack)

            # get the bound of the contour
            (x_rec, y_rec, w_rec, h_rec) = cv2.boundingRect(contour)
            # show the rectangle and center circle
            cv2.rectangle(imgFinal, (x_rec, y_rec), (x_rec + w_rec, y_rec + h_rec), BGR_COLOR['green'], 2)
            cv2.circle(imgFinal, (x, y), 5, BGR_COLOR['green'], -1, cv2.LINE_AA)

            # visualize the text of the information
            # status, distance, speed and time
            cv2.putText(imgFinal, 'Status: Movement Detected', (350, 250),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])
            cv2.putText(imgFinal, 'Time: ' + str('%.0f sec' % r_time), (350, 280),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])
            cv2.putText(imgFinal, 'Distance: ' + str('%.2f' % distance), (350, 310),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])
            cv2.putText(imgFinal, 'Speed ' + str('%.5f /sec' % speed), (350, 340),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])



            tem_contour = contour
            tem_dis_per_frame = dis_per_frame

            old_x = x
            old_y = y
        else:
            # print('bad:', cv2.contourArea(contour), dis_per_frame)
            if imgTrack is not None:
                imgFinal = cv2.add(frame1, imgTrack)
            dis_per_frame = tem_dis_per_frame
            contour = tem_contour
            x = old_x
            y = old_y

            # get the bound of the contour
            (x_rec, y_rec, w_rec, h_rec) = cv2.boundingRect(contour)
            # show the rectangle and center circle
            cv2.rectangle(imgFinal, (x_rec, y_rec), (x_rec + w_rec, y_rec + h_rec), BGR_COLOR['blue'], 2)
            cv2.circle(imgFinal, (x, y), 5, BGR_COLOR['blue'], -1, cv2.LINE_AA)

            # show the information
            cv2.putText(imgFinal, 'Status: Stopping', (350, 250),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])
            cv2.putText(imgFinal, 'Time: ' + str('%.0f sec' % r_time), (350, 280),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])
            cv2.putText(imgFinal, 'Distance: ' + str('%.2f' % distance), (350, 310),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])
            speed = dis_per_frame * 10
            cv2.putText(imgFinal, 'Speed ' + str('%.5f /sec' % speed), (350, 340),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, BGR_COLOR['white'])

            # print('last:', cv2.contourArea(contour), dis_per_frame)
            # print('warning')


        if r_time % 2 == 0:
            error, total_error = computeError(h, x, y, r_time, total_error)

            file = open(RELATIVE_DESTINATION_PATH + 'results.csv', 'a')
            file.write("rat" + ',%.1f' % r_time + ',%.2f' % distance + ',%.5f' % speed +
                       ',%.1f' % x + ',%.1f' % y + ',%.4f' % error + ',%.4f\n' % total_error)
            file.close()

        # the rule of pause and end
        k = cv2.waitKey(WAIT_DELAY) & 0xff
        if r_time >= 120:
            break
        if k == 27:
            break
        if k == 32:
            if WAIT_DELAY == 1:
                WAIT_DELAY = 0  # pause
            else:
                WAIT_DELAY = 1  # play as fast as possible

        # displaying frame
        cv2.imshow("tracing", imgFinal)

        frame1 = frame2

        count, frame2 = capturedVideo.read()

        if cv2.waitKey(40) == 27:
            break

    cv2.destroyAllWindows()
    capturedVideo.release()
    cv2.imwrite(RELATIVE_DESTINATION_PATH + 'traces/' + 'tracing_[distance]=%.2f' % distance +
                '_[time]=%.1fs' % r_time + '.png', cv2.resize(imgTrack, (max(HD), max(HD))))
    print(file_name + '\tdistance %.2f\t' % distance + 'processing/real time %.1f' % float(
        time.time() - start) + '/%.1f s' % r_time)


def computeError(h, x, y, r_time, total_error):
    error = 0
    print("At time:", r_time, "the position is", [x, y])

    with open(RELATIVE_TRUTH_PATH + "truth.csv", "r") as csvFile:
        reader = csv.reader(csvFile)
        rows = [row for row in reader]

    for time_n in range(len(rows)):
        if rows[time_n][0] == str(r_time):
            print('true x position:', rows[time_n][1], 'true y position:', rows[time_n][2])
            diff_x = (x - float(rows[time_n][1])) / float(h)
            diff_y = (y - float(rows[time_n][2])) / float(h)

            error = np.sqrt(diff_x ** 2 + diff_y ** 2)
            total_error += error
            print('Sqaure error in time', r_time, 'is ', error, 'total error is:', total_error)

    return error, total_error


def creatCSV():
    # create "timing" and "traces" folds
    if not os.path.exists(RELATIVE_DESTINATION_PATH + 'traces'):
        os.makedirs(RELATIVE_DESTINATION_PATH + 'traces')
    # if not os.path.exists(RELATIVE_DESTINATION_PATH + 'timing'):
    #     os.makedirs(RELATIVE_DESTINATION_PATH + 'timing')
    # create "results.csv" and initial with some attributes
    file = open(RELATIVE_DESTINATION_PATH + 'results.csv', 'w')
    file.write('Animal,Time(s),Distance(unit of box side),Speed(unit/s),x,y,Error,Total Error\n')
    file.close()

creatCSV()
trace(file_name)
