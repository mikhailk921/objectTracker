# -*- coding: utf-8 -*-
# @file CTrack.py
# @brief Файл содержит класс описывающий трек
import tracker.KalmanFilter as CKalman
import tracker.KalmanFilterMatr as CKalmanMatr
import tracker.KalmanFilter4d as CKalman4
import copy
import math
import time


## @brief Класс трека
class CTrack:
    ## @brief - Конструктор класса
    # @param object - класс точки (CObject)
    # @param trackID - номер трека
    # @param deltaTime - дельта времени
    # @param accelNoiseMag - шум
    def __init__(self, object, trackID, deltaTime=0.25, accelNoiseMag=0.1):
        object.timeDetect = time.time()
        self.m_trackID = trackID
        self.m_skippedFrames = 0
        self.m_lastObject = copy.deepcopy(object)
        self.m_predictionPoint = copy.deepcopy(object)
        #self.m_outOfTheFrame = False
        #self._m_kalman = CKalman.CKalmanFilter(self.m_predictionPoint, deltaTime, accelNoiseMag)
        self._m_kalman = CKalman4.CKalmanFilterMatr4d(self.m_predictionPoint, deltaTime, accelNoiseMag)
        #self._m_kalman = CKalmanMatr.CKalmanFilterMatr(self.m_predictionPoint, deltaTime, accelNoiseMag)
        self.m_trace = []
        self.m_trace.append(copy.deepcopy(object))
        #self.SpeedPoint = {"x": 0, "y": 0}
        self._m_isStatic = False
        self._m_staticFrames = 0

    ## @brief - функция расчета расстояния между последней и текущей точками
    # @param object - текущая точка (CObject)
    # @return - Вернет расстояние между точками
    def _CalcDist(self, pt):
        try:
            diff = {"x": 0., "y": 0.}
            diff["x"] = self.m_predictionPoint.x - pt.x
            diff["y"] = self.m_predictionPoint.y - pt.y
            return math.sqrt((diff["x"] ** 2) + (diff["y"] ** 2))
        except Exception as e:
            print("CTrack:CalcDist", e)

    ## @brief - функция обновления трека
    # @param object - класс точки (CObject)
    # @param dataCorrect - достоверность точки
    # @param deltaTime - дельта времени
    # @param accelNoiseMag - шум
    def Update(self, object, dataCorrect, max_trace_length, trajLen):
        try:

            pt = copy.deepcopy(object)
            #pt.vx = (pt.x - self.m_lastObject.x)/(pt.timeDetect - self.m_lastObject.timeDetect)
            #pt.vy = (pt.y - self.m_lastObject.y) / (pt.timeDetect - self.m_lastObject.timeDetect)
            self._PointUpdate(pt, dataCorrect)
            if dataCorrect:
                self.m_lastObject = copy.deepcopy(object)
                self.m_trace.append(copy.deepcopy(self.m_predictionPoint))
                self._CheckStatic(trajLen)
            else:
                self.m_trace.append(copy.deepcopy(self.m_predictionPoint))

            if len(self.m_trace) > max_trace_length:
                self.m_trace.pop(len(self.m_trace) - max_trace_length)
        except Exception as e:
            print("CTrack:Update", e)

    ## @brief - функция выполняет обновление трека
    # @param pt - координаты точки
    # @param dataCorrect - признак того, что для этого трека было найдено сопоставелние или нет
    # @param width - ширина экрана
    # @param height - высота экрана
    def _PointUpdate(self, pt, dataCorrect):
        try:
            self._m_kalman.GetPointPrediction()
            if dataCorrect:
                p = self._m_kalman.Update({"x": pt.x, "y": pt.y, "vx": pt.vx, "vy": pt.vy,
                                        "intens": pt.intens, "square": pt.square}, dataCorrect)
            else:
                p = self._m_kalman.Update({"x": self.m_predictionPoint.x, "y": self.m_predictionPoint.y,
                                           "vx": self.m_predictionPoint.vx, "vy": self.m_predictionPoint.vy,
                                           "intens": self.m_predictionPoint.intens,
                                           "square": self.m_predictionPoint.square},
                                          dataCorrect)
            self.m_predictionPoint.x = int(p["x"])
            self.m_predictionPoint.y = int(p["y"])
            self.m_predictionPoint.vx = int(p["vx"])
            self.m_predictionPoint.vy = int(p["vy"])
            self.m_predictionPoint.intens = int(p["intens"])
            self.m_predictionPoint.square = int(p["square"])

        except Exception as e:
            print("CTrack:PointUpdate", e)

    ## @brief - функция запроса состояния точки
    # @return - Вернет true or false
    def IsStatic(self):
        return self._m_isStatic

    ## @brief - функция расчитывает больше или меньше число пропущенных кадров чем параметр
    # @param framesTime - колическтво кадров
    # @return - Вернет true or false
    def IsStaticTimeout(self, framesTime):
        return self._m_staticFrames > framesTime

    ## @brief - функция выполняет проверку на статичность точки (неподвижность)
    # @param trajLen - длина траектории
    # @return - Вернет true or false
    def _CheckStatic(self, trajLen):
        if trajLen == 0 or len(self.m_trace) < trajLen:
            self._m_isStatic = False
            self._m_staticFrames = 0
        else:
            xy = [0, 0, 0, 0]
            self._get_lin_regress(self.m_trace, len(self.m_trace) - trajLen, len(self.m_trace), xy)
            speed = math.sqrt((xy[0] ** 2) * (trajLen ** 2) + (xy[2] ** 2) * (trajLen ** 2))
            speedThresh = 10
            if speed < speedThresh:
                self._m_staticFrames += 1
                self._m_isStatic = True
            else:
                self._m_staticFrames = 0
                self._m_isStatic = False
        return self._m_isStatic


    def _get_lin_regress(self, in_data, start_pos, in_data_size, xy):
        m1 = m2 = m3_x = m4_x = m3_y = m4_y = 0.0
        el_count = in_data_size - start_pos
        for i in range(start_pos, in_data_size):
            m1 += i
            m2 += i * i

            m3_x += in_data[i].x
            m4_x += i * in_data[i].x

            m3_y += in_data[i].y
            m4_y += i * in_data[i].y
        det_1 = 1 / (el_count * m2 - m1 * m1)
        m1 *= -1
        xy[0] = det_1 * (m1 * m3_x + el_count * m4_x)
        xy[1] = det_1 * (m2 * m3_x + m1 * m4_x)

        xy[2] = det_1 * (m1 * m3_y + el_count * m4_y)
        xy[3] = det_1 * (m2 * m3_y + m1 * m4_y)
if __name__ == '__main__':

    class CObject:
        def __init__(self, x=0, y=0, vx=0, vy=0, intens=100, square=10):
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.timeDetect = time.time()
            self.intens = intens
            self.square = square
    try:


        print(time.time())

        object = CObject(10, 15)
        track = [CTrack(object, True)]
        dist = track[0]._CalcDist(CObject(100, 150))

        track.append(CTrack(CObject(120, 105), True, 0.1, 1))
        track[0].Update(CObject(15, 20), True, 10, 0)


    except Exception as e:
        print(e)