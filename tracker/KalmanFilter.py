# -*- coding: utf-8 -*-
## @file KalmanFilter.py
# @brief Файл содержит класс с функциями фильтра Калмана для корректировки и предсказания
import cv2
import numpy as np
import copy

## @brief Класс реализующий фильтр Калмана
class CKalmanFilter:
    ## @brief - Конструктор класса
    # @param pt - класс точки (CObject)
    # @param deltaTime - дельта времени сек
    # @param accelNoiseMag - шум
    def __init__(self, pt, deltaTime = 0.2, accelNoiseMag = 0.5):
        self.m_linearKalman = None
        self.m_initialPoints = []
        self.MIN_INIT_VALS = 4
        self.m_lastPointResult = {"x": pt.x, "y": pt.y, "vx": pt.vx, "vy": pt.vy}
        self.m_initialized = False
        self.m_deltaTime = deltaTime
        self.m_accelNoiseMag = accelNoiseMag
        self.m_initialPoints.append({"x": pt.x, "y": pt.y, "vx": pt.vx, "vy": pt.vy})

    def _get_lin_regress(self, in_data, start_pos, in_data_size, xy):
        try:
            m1 = m2 = m3_x = m4_x = m3_y = m4_y = 0.0
            el_count = in_data_size - start_pos
            for i in range(start_pos, in_data_size):
                m1 += i
                m2 += i * i

                m3_x += in_data[i]["x"]
                m4_x += i * in_data[i]["x"]

                m3_y += in_data[i]["y"]
                m4_y += i * in_data[i]["y"]
            det_1 = 1 / (el_count * m2 - m1 * m1)
            m1 *= -1
            xy["kx"] = det_1 * (m1 * m3_x + el_count * m4_x)
            xy["bx"] = det_1 * (m2 * m3_x + m1 * m4_x)

            xy["ky"] = det_1 * (m1 * m3_y + el_count * m4_y)
            xy["by"] = det_1 * (m2 * m3_y + m1 * m4_y)
        except Exception as e:
            print("Kalman:get_lin_regress", e)

    ## @brief - функция инициализации фильта Калмана
    # @param xy0 - координаты точки
    # @param xyv0 - скорость точки
    def _CreateLinear(self, xy0, xyv0):
        self.m_linearKalman = cv2.KalmanFilter(4, 2, 0)
        self.m_linearKalman.transitionMatrix = np.array(([1, 0, self.m_deltaTime, 0],
                                           [0, 1, 0, self.m_deltaTime],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]), dtype='float32')

        # init...
        self.m_lastPointResult = xy0
        self.m_linearKalman.statePre = np.array((xy0["x"], xy0["y"], xyv0["x"], xyv0["y"]), dtype='float32')
        self.m_linearKalman.statePost = np.array((xy0["x"], xy0["y"], 0, 0), dtype='float32')

        self.m_linearKalman.processNoiseCov = np.array(([pow(self.m_deltaTime, 4.0) / 4, 0., pow(self.m_deltaTime, 3.0) / 2.0, 0.],
                                          [0., pow(self.m_deltaTime, 4.0) / 4, 0., pow(self.m_deltaTime, 3.0) / 2.0],
                                          [pow(self.m_deltaTime, 3.0) / 2.0, 0., pow(self.m_deltaTime, 2.0), 0.],
                                          [0., pow(self.m_deltaTime, 3.0) / 2.0, 0., pow(self.m_deltaTime, 2.0)]),
                                         dtype='float32')

        self.m_linearKalman.processNoiseCov *= self.m_accelNoiseMag

        self.m_linearKalman.measurementMatrix = np.array(([1, 0, 0, 0], [0, 1, 0, 0]), dtype='float32')

        self.m_linearKalman.measurementNoiseCov = np.array(([0.1, 0], [0, 0.1]), dtype='float32')
        self.m_linearKalman.errorCovPost = np.array(([0.1, 0, 0, 0],
                                       [0, 0.1, 0, 0],
                                       [0, 0, 0.1, 0],
                                       [0, 0, 0, 0.1]), dtype='float32')

        self.m_initialized = True

    ## @brief - функция предсказания
    # @return - Вернет dict{"x", "y"}
    def GetPointPrediction(self):
        try:
            if self.m_initialized:
                prediction = self.m_linearKalman.predict()
                self.m_lastPointResult = {"x": prediction[0], "y": prediction[1],
                                          "vx": prediction[2], "vy": prediction[3]}
            else:
                prediction = self.m_lastPointResult
            return prediction

        except Exception as e:
            print("Kalman:GetPointPredict", e)

    ## @brief - функция обновления фильта Калмана
    # @param pt - координаты точки
    # @param dataCorrect - достоверность координат
    # @return - Вернет dict{"x", "y"}
    def Update(self, pt, dataCorrect):
        try:
            if not self.m_initialized:
                if len(self.m_initialPoints) < self.MIN_INIT_VALS:
                    if dataCorrect:
                        self.m_initialPoints.append(copy.deepcopy(pt))
                if len(self.m_initialPoints) == self.MIN_INIT_VALS:

                    xy = {"kx": 0, "bx": 0, "ky": 0, "by": 0}
                    self._get_lin_regress(self.m_initialPoints, 0, self.MIN_INIT_VALS, xy)
                    xy0 = {"x": xy["kx"] * (self.MIN_INIT_VALS - 1) + xy["bx"],
                           "y": xy["ky"] * (self.MIN_INIT_VALS - 1) + xy["by"]}
                    xyv0 = {"x": xy["kx"], "y": xy["ky"]}
                    self._CreateLinear(xy0, xyv0)

            if self.m_initialized:

                measurement = np.array((0, 0), dtype='float32')
                if not dataCorrect:
                    measurement[0] = self.m_lastPointResult["x"]
                    measurement[1] = self.m_lastPointResult["y"]
                else:
                    measurement[0] = pt["x"]
                    measurement[1] = pt["y"]

                estimated = self.m_linearKalman.correct(measurement)

                self.m_lastPointResult["x"] = estimated[0]
                self.m_lastPointResult["y"] = estimated[1]
                self.m_lastPointResult["vx"] = estimated[2]
                self.m_lastPointResult["vy"] = estimated[3]
            else:
                if dataCorrect:
                    self.m_lastPointResult["x"] = int(pt["x"])
                    self.m_lastPointResult["y"] = int(pt["y"])
                    self.m_lastPointResult["vx"] = int(pt["vx"])
                    self.m_lastPointResult["vy"] = int(pt["vy"])
            return self.m_lastPointResult
        except Exception as e:
            print("Kalman:Update", e)

if __name__ == '__main__':
    import time


    class CObject:
        def __init__(self, x=0, y=0, intens=100, square=10, vx=0, vy=0):
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.timeDetect = time.time()
            self.intens = intens
            self.square = square
    try:

        n = np.array((0, 0))
        print(n)


        m_kalman = CKalmanFilter(CObject(1, 1))
        k = 1
        while 1:

            last = m_kalman.GetPointPrediction()
            #print(last)
            last = m_kalman.Update({"x": k, "y": k}, True)
            print(k, last)
            k += 1
    except Exception as e:
        print(e)

