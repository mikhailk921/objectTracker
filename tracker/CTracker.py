# -*- coding: utf-8 -*-
## @file tracker.py
# @brief Файл содержит класс трекера
import tracker.CTrack as CTrack
import tracker.HungarianAlg as CHungrianAlg
import math
import time

## @brief Класс объекта
class CObject:
    def __init__(self, x=0, y=0, intens=100, square=10, vx=0, vy=0):
        self.point = (x, y)
        self.vx = vx
        self.vy = vy
        self.timeDetect = time.time()
        self.intens = intens
        self.square = square


## @brief Класс трекера
class CTracker:
    ## @brief - Конструктор класса
    # @param width - ширина
    # @param height - высота
    def __init__(self, width, height, fps=1):
        self.Width = width
        self.Height = height
        self.m_settings = {
            "m_dt": 0.25,
            "m_accelNoiseMag": 0.1,
            "m_distThres": 200,
            "m_maximumAllowedSkippedFrames": 10,
            "m_maxTraceLength": 10,
            "m_useAbandonedDetection": False,
            "m_minStaticFrame": 4,
            "m_maxStaticFrame": 7,
            "ROIset": [(0, 0), (0, 0)],
            "ROI": False,
            "LOW_INTENS": 1,
            "LOW_SQUARE": 1
        }
        self.track = []
        self.m_nextTrackID = 0

    ## @brief - функция обновления трекера
    # @param object - текущие объекты (CObject)
    def Update(self, objects):
        try:
            if self.m_settings["ROI"]:
                self._ROI(objects)
            remove = []
            for i in range(0, len(objects)):
                if objects[i].intens < self.m_settings["LOW_INTENS"] or \
                                objects[i].square < self.m_settings["LOW_SQUARE"]:
                    remove.append(objects[i])

            for i in remove:
                objects.remove(i)

            self._UpdateHungrian(objects)
        except Exception as e:
            print("CTracker:Update", e)

    ## @brief - функция обновления трекера
    # @param object - текущие объекты (CObject)
    def _UpdateHungrian(self, objects):
        try:
            N = len(self.track)
            M = len(objects)
            assignment = []
            for i in range(0, N):
                assignment.append(-1)

            #print("len track:", len(self.track))
            if len(self.track) > 0:
                Cost = []
                for i in range(0, (M * N)):
                    Cost.append(0)

                maxCost = 0
                for i in range(0, self.track.__len__()):
                    for j in range(0, len(objects)):
                        x = self.track[i].m_predictionPoint[0] - objects[j].point[0]
                        y = self.track[i].m_predictionPoint[1] - objects[j].point[1]
                        intens = self.track[i].m_lastObject.intens - objects[j].intens
                        dist = math.sqrt(x * x + y * y + intens * intens)
                        Cost[i + j * N] = dist
                        if dist > maxCost:
                            maxCost = dist

                # решение задачи назначения
                APS = CHungrianAlg.AssignmentProblemSolver()
                APS.Solve(Cost, assignment, N, M)

                # удалние назначений пар с больщой дистанцией
                for i in range(0, len(assignment)):
                    if assignment[i] != -1:
                        if Cost[i + assignment[i] * N] > self.m_settings["m_distThres"]:
                            assignment[i] = -1
                            self.track[i].m_skippedFrames += 1
                    else:
                        self.track[i].m_skippedFrames += 1

                # если трек не обнаружен долгое время, то удаляем
                while True:
                    if len(self.track) == 0:
                        break
                    j = 0
                    for i in range(0, len(self.track)):
                        j = i
                        if self.track[i].m_skippedFrames > self.m_settings["m_maximumAllowedSkippedFrames"]:
                            self.track.pop(i)
                            assignment.pop(i)
                            break
                    if j == len(self.track) - 1:
                        break


            # поиск несоответствующих объектов и начало новых треков
            for i in range(0, len(objects)):
                #print(i, len(objects), assignment.count(i))
                if assignment.count(i) == 0:
                    self.track.append(CTrack.CTrack(objects[i],
                                                        self.m_nextTrackID,
                                                        self.m_settings["m_dt"],
                                                        self.m_settings["m_accelNoiseMag"]))
                    self.m_nextTrackID += 1


            #  обновление фильтра Калмана
            for i in range(0, len(assignment)):
                if assignment[i] != -1:
                    minStaticFrame = 0
                    if self.m_settings["m_useAbandonedDetection"]:
                        minStaticFrame = self.m_settings["m_minStaticFrame"]
                    self.track[i].m_skippedFrames = 0
                    self.track[i].Update(objects[assignment[i]], True,
                                             self.m_settings["m_maxTraceLength"],
                                         minStaticFrame)
                else:
                    self.track[i].Update(CObject(), False, self.m_settings["m_maxTraceLength"], 0)

        except Exception as e:
            print("CTracker:UpdateHungarian", e)

    ## @brief - функция установки ROI
    # @param startX - начало по X
    # @param startY - начало по Y
    # @param endX - конец по X
    # @param endY - конец по Y
    def SetROI(self, startX, startY, endX=0, endY=0):
        if startX >= self.Width / 2 or startY >= self.Height:
            return
        if endX != 0 and endY != 0:
            if startX >= endX or startY >= endY:
                return
            self.m_settings["ROIset"] = [[startX, startY], [endX, endY]]
        else:
            self.m_settings["ROIset"] = [[startX, startY], [self.Width - startX, self.Height - startY]]
        self.m_settings["ROI"] = True

    ## @brief - отключить ROI
    def SetROIoff(self):
        self.m_settings["ROI"] = False

    ## @brief - отключить ROI
    # @param objects - объекты для проверки
    def _ROI(self, objects):
        while True:
            j = 0
            for i in range(0, len(objects)):
                j = i
                if not (objects[i].x > self.m_settings["ROIset"][0][0] and
                        objects[i].x < self.m_settings["ROIset"][0][1] or
                        objects[i].x > self.m_settings["ROIset"][0][0] and
                        objects[i].x < self.m_settings["ROIset"][0][1]):
                    objects.pop(i)
                    break
            if j == len(self.track) - 1:
                return


if __name__ == '__main__':
    import cv2
    import numpy as np
    X = 0
    Y = 0

    def mv_mouseCallback(event, x, y, flags, param):
        global X, Y
        if event == cv2.EVENT_MOUSEMOVE:
            X = x
            Y = y

    try:
        import random

        tracker = CTracker(640, 480, 10)
        frame = np.zeros((480, 640, 3))
        cv2.imshow("Video", frame)
        cv2.waitKey(1)

        rndX = []
        rndY = []
        for i in range(0, 2):
            rndX.append(random.randint(-150, 150))
            rndY.append(random.randint(-100, 100))

        alpha = 0
        rotate = False
        intSqr = 10
        while True:
            alpha += 0.05
            frame = np.zeros((480, 640, 3))

            cv2.setMouseCallback("Video", mv_mouseCallback)
            o = []
            #o.append(CObject(X, Y, 1000, 1000))
            #o.append(CObject(X+100, Y+100, 1000, 1000))
            #o.append(CObject(X+100, Y-100, 1000, 1000))
            #o.append(CObject(X-100, Y+100, 1000, 1000))
            #o.append(CObject(X-100, Y-100, 1000, 1000))

            for i in range(0, 2):
                if rotate:
                    o.append(CObject(X + rndX[i]*math.sin(alpha), Y + rndY[i]*math.cos(-alpha), intSqr, intSqr))
                else:
                    o.append(CObject(X + rndX[i], Y + rndY[i], intSqr, intSqr))

            tracker.Update(o)
            intSqr += 1
            try:
                #cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)
                for i in range(0, len(tracker.track)):
                    if len(tracker.track[i].m_trace) > 4:
                        for j in range(1, (len(tracker.track[i].m_trace) - 2)):
                            cv2.line(frame, (tracker.track[i].m_trace[j+1].x, tracker.track[i].m_trace[j+1].y),
                                     (tracker.track[i].m_trace[j+2].x, tracker.track[i].m_trace[j+2].y),
                                     (255, 0, 0), 3)
                        cv2.circle(frame,
                                   (tracker.track[i].m_trace[len(tracker.track[i].m_trace) - 1].x,
                                    tracker.track[i].m_trace[len(tracker.track[i].m_trace) - 1].y),
                                   5, (255, 0, 0), -1)
                        cv2.circle(frame,
                                   (tracker.track[i].m_trace[len(tracker.track[i].m_trace) - 1].x,
                                    tracker.track[i].m_trace[len(tracker.track[i].m_trace) - 1].y),
                                   2, (0, 0, 255), -1)

                for i in range(0, len(tracker.track)):
                    st = None
                    if tracker.m_settings["m_useAbandonedDetection"]:
                        st = str(tracker.track[i].m_trackID) + " " + \
                             str(tracker.track[i].m_trace[len(tracker.track[i].m_trace)-1].vx) + " " + \
                             str(tracker.track[i].m_trace[len(tracker.track[i].m_trace) - 1].vy) + " "
                        if tracker.track[i].IsStatic():
                            st += str(1)
                        else:
                            st += str(0)
                    else:
                        st = str(tracker.track[i].m_trackID) + " " + \
                             str(tracker.track[i].m_trace[len(tracker.track[i].m_trace)-1].vx) + " " + \
                             str(tracker.track[i].m_trace[len(tracker.track[i].m_trace) - 1].vy) + " " + \
                             str(tracker.track[i].m_trace[len(tracker.track[i].m_trace) - 1].intens) + " " + \
                             str(tracker.track[i].m_trace[len(tracker.track[i].m_trace) - 1].square)

                    p = (tracker.track[i].m_trace[len(tracker.track[i].m_trace) - 1].x + 10,
                        tracker.track[i].m_trace[len(tracker.track[i].m_trace) - 1].y - 10)
                    cv2.putText(frame, st, p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

                for i in range(0, len(o)):
                    cv2.circle(frame, (o[i].x, o[i].y), 3, (0, 0, 255), -1)

            except Exception as e:
                print("draw", e)


            cv2.imshow("Video", frame)
            val = cv2.waitKey(100)
            if val & 0xff == ord('q'):
                exit(0)
            if val & 0xff == ord('w'):
                continue


    except Exception as e:
        print(e)
