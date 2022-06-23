# -*- coding: utf-8 -*-
## @file KalmanFilterMatr.py
# @brief Файл содержит класс с функциями фильтра Калмана для корректировки и предсказания
import numpy as np
import copy

## @brief Класс реализующий фильтр Калмана
class CKalmanFilterMatr:
    ## @brief - Конструктор класса
    # @param pt - класс точки (CObject)
    # @param deltaTime - дельта времени сек
    # @param accelNoiseMag - шум
    def __init__(self, pt, deltaTime = 0.2, accelNoiseMag = 0.5):
        self.m_initialPoints = []
        self.MIN_INIT_VALS = 4
        self.m_lastPointResult = {"x": pt.x, "y": pt.y, "vx": pt.vx, "vy": pt.vy}
        self.m_initialized = False
        self.m_deltaTime = deltaTime
        self.m_accelNoiseMag = accelNoiseMag
        self.m_initialPoints.append({"x": pt.x, "y": pt.y, "vx": pt.vx, "vy": pt.vy})

        self._statePre = np.zeros(4, dtype='float32')
        self._statePost = np.zeros(4, dtype='float32')
        self._transitionMatrix = np.zeros((4, 4), dtype='float32')
        self._processNoiseCov = np.zeros((4, 4), dtype='float32')
        self._measurementMatrix = np.zeros((2, 4), dtype='float32')
        self._measurementNoiseCov = np.zeros((2, 2), dtype='float32')
        self._errorCovPost = np.zeros((4, 4), dtype='float32')
        self._errorCovPre = np.zeros((4, 4), dtype='float32')

        self._temp1 = np.zeros((4, 4), dtype='float32')
        self._temp2 = np.zeros((2, 4), dtype='float32')
        self._temp3 = np.zeros((2, 2), dtype='float32')
        self._temp4 = np.zeros((2, 4), dtype='float32')
        self._gain = np.zeros((4, 2), dtype='float32')
        self._temp5 = np.zeros(2, dtype='float32')

    def _get_lin_regress(self, in_data, start_pos, in_data_size, xy):
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

    ## @brief - функция инициализации фильта Калмана
    # @param xy0 - координаты точки
    # @param xyv0 - скорость точки
    def _CreateLinear(self, xy0, xyv0):
        self._transitionMatrix = np.array(([1, 0, self.m_deltaTime, 0],
                                           [0, 1, 0, self.m_deltaTime],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]), dtype='float32')

        # init...
        self.m_lastPointResult = xy0
        self._statePre = np.array((xy0["x"], xy0["y"], xyv0["x"], xyv0["y"]), dtype='float32')
        self._statePost = np.array((xy0["x"], xy0["y"], 0, 0), dtype='float32')

        self._processNoiseCov = np.array(([pow(self.m_deltaTime, 4.0) / 4, 0., pow(self.m_deltaTime, 3.0) / 2.0, 0.],
                                                        [0., pow(self.m_deltaTime, 4.0) / 4, 0., pow(self.m_deltaTime, 3.0) / 2.0],
                                                        [pow(self.m_deltaTime, 3.0) / 2.0, 0., pow(self.m_deltaTime, 2.0), 0.],
                                                        [0., pow(self.m_deltaTime, 3.0) / 2.0, 0., pow(self.m_deltaTime, 2.0)]),
                                         dtype='float32')

        self._processNoiseCov *= self.m_accelNoiseMag

        self._measurementMatrix = np.array(([1, 0, 0, 0], [0, 1, 0, 0]), dtype='float32')

        self._measurementNoiseCov = np.array(([0.1, 0], [0, 0.1]), dtype='float32')
        self._errorCovPost = np.array(([0.1, 0, 0, 0],
                                       [0, 0.1, 0, 0],
                                       [0, 0, 0.1, 0],
                                       [0, 0, 0, 0.1]), dtype='float32')

        self.m_initialized = True

    ## @brief - функция предсказания
    # @return - Вернет dict{"x", "y"}
    def GetPointPrediction(self):
        try:
            if self.m_initialized:
                prediction = self._predict()
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

                estimated = self._correct(measurement)

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

    ## @brief - функция реализующая предсказание
    def _predict(self):
        '''# обновляем состояние: x '(k) = A * x (k)
        for i in range(0, 4):
            t = 0
            for k in range(0, 4):
                t += self._transitionMatrix[i][k] * self._statePost[k]
            self._statePre[i] = t

        # transitionMatrix[4][4] * errorCovPost[4][4]
        for i in range(0, 4):
            for j in range(0, 4):
                t = 0
                for k in range(0, 4):
                    t += self._transitionMatrix[i][k] * self._errorCovPost[k][j]
                self._temp1[i][j] = t

        # P '(k) = temp1_T * transitionMatrix (A) + processNoiseCov_T (Q)
        # temp1 to temp1_T
        temp1_T = np.zeros((4, 4), dtype='float32')
        for i in range(0, 4):
            for j in range(0, 4):
                temp1_T[j][i] = self._temp1[i][j]

        # temp1_T * A
        temp12 = np.zeros((4, 4), dtype='float32')
        for i in range(0, 4):
            for j in range(0, 4):
                t = 0
                for k in range(0, 4):
                    t += temp1_T[i][k] * self._transitionMatrix[k][j]
                temp12[i][j] = t

        # processNoiseCov to processNoiseCov_T
        processNoiseCov_T = np.zeros((4, 4), dtype='float32')
        for i in range(0, 4):
            for j in range(0, 4):
                processNoiseCov_T[j][i] = self._processNoiseCov[i][j]

        # temp12 + Q_T
        for i in range(0, 4):
            for j in range(0, 4):
                self._errorCovPre[i][j] = temp12[i][j] + processNoiseCov_T[i][j]

        for i in range(0, 4):
            self._statePost[i] = self._statePre[i]

        for i in range(0, 4):
            for j in range(0, 4):
                self._errorCovPost[i][j] = self._errorCovPre[i][j]

        return self._statePre'''
        # обновляем состояние: x '(k) = A * x (k)
        self._statePre = np.dot(self._transitionMatrix, self._statePost)

        # transitionMatrix[6][6] * errorCovPost[6][6]
        self._temp1 = np.dot(self._transitionMatrix, self._errorCovPost)

        # P '(k) = temp1_T * transitionMatrix (A) + processNoiseCov_T (Q)
        # temp1 to temp1_T
        temp1_T = self._temp1.transpose()

        # temp1_T * A
        temp12 = np.dot(temp1_T, self._transitionMatrix)

        # processNoiseCov to processNoiseCov_T
        processNoiseCov_T = self._processNoiseCov.transpose()

        # temp12 + Q_T
        self._errorCovPre = temp12 + processNoiseCov_T

        self._statePost = self._statePre
        self._errorCovPost = self._errorCovPre

        return self._statePre

    ## @brief - функция реализующая коррекцию
    # @param xy - координаты точки
    def _correct(self, xy):
        z = np.array((xy[0], xy[1]), dtype='float32')

        '''# temp2 =  measurementMatrix[2][4] (H) * errorCovPre[4][4] (P'(k))
        for i in range(0, 2):
            for j in range(0, 4):
                t = 0
                for k in range(0, 4):
                    t += self._measurementMatrix[i][k] * self._errorCovPre[k][j]
                self._temp2[i][j] = t

        # temp3 = temp2 * measurementMatrix_T (Ht) + measurementNoiseCov (R)
        # measurementMatrix to measurementMatrix_T
        measurementMatrix_T = np.zeros((4, 2), dtype='float32')
        for i in range(0, 2):
            for j in range(0, 4):
                measurementMatrix_T[j][i] = self._measurementMatrix[i][j]

        # temp2 * H_T
        temp21 = np.zeros((2, 2), dtype='float32')
        for i in range(0, 2):
            for j in range(0, 2):
                t = 0
                for k in range(0, 4):
                    t += self._temp2[i][k] * measurementMatrix_T[k][j]
                temp21[i][j] = t

        # temp21_T + measurementNoiseCov (R)
        for i in range(0, 2):
            for j in range(0, 2):
                self._temp3[i][j] = temp21[i][j] + self._measurementNoiseCov[i][j]

        # temp4 = inv (temp3) * temp2 = Kt (k)
        # inv(temp3)
        temp3_inv = np.linalg.inv(self._temp3)

        # temp4 = inv (temp3) * temp2
        for i in range(0, 2):
            for j in range(0, 4):
                t = 0
                for k in range(0, 2):
                    t += temp3_inv[i][k] * self._temp2[k][j]
                self._temp4[i][j] = t

        # gain = temp4_T
        for i in range(0, 2):
            for j in range(0, 4):
                self._gain[j][i] = self._temp4[i][j]

        # temp5 =  z(k) - measurementMatrix[2][4] (H) * statePre[4][1] (x'(k))
        # measurementMatrix[2][4] (H) * statePre[4] (x'(k))
        temp52 = np.zeros(2, dtype='float32')
        for i in range(0, 2):
            t = 0
            for k in range(0, 4):
                t += self._measurementMatrix[i][k] * self._statePre[k]
            temp52[i] = t

        # temp5 =  z(k) + temp52 * -1
        for i in range(0, 2):
            self._temp5[i] = z[i] + (temp52[i] * -1)

        # statePost (x(k)) = statePre (x'(k)) +  gain[2][4](K (k)) * temp5[2]
        # gain[2][4](K (k)) * temp5[2]
        temp53 = np.zeros(4, dtype='float32')
        for i in range(0, 4):
            t = 0
            for k in range(0, 2):
                t += self._gain[i][k] * self._temp5[k]
            temp53[i] = t

        # statePre[4] (x'(k)) +  temp53[2]
        for i in range(0, 4):
            self._statePost[i] = self._statePre[i] + temp53[i]

        # errorCovPost P(k) = errorCovPre P'(k) - gain K(k) * temp2
        # gain K(k) * temp2
        temp22 = np.zeros((4, 4), dtype='float32')
        for i in range(0, 4):
            for j in range(0, 4):
                t = 0
                for k in range(0, 2):
                    t += self._gain[i][k] * self._temp2[k][j]
                temp22[i][j] = t

        # errorCovPre P'(k) + temp22 * -1
        for i in range(0, 4):
            for j in range(0, 4):
                self._errorCovPost[i][j] = self._errorCovPre[i][j] + (temp22[i][j] * -1)

        return self._statePost'''

        # temp2 =  measurementMatrix[4][6] (H) * errorCovPre[6][6] (P'(k))
        self._temp2 = np.dot(self._measurementMatrix, self._errorCovPre)

        # temp3 = temp2 * measurementMatrix_T (Ht) + measurementNoiseCov (R)
        # measurementMatrix to measurementMatrix_T
        measurementMatrix_T = self._measurementMatrix.transpose()

        # temp2 * H_T
        temp21 = np.dot(self._temp2, measurementMatrix_T)

        # temp21_T + measurementNoiseCov (R)

        self._temp3 = temp21 + self._measurementNoiseCov

        # temp4 = inv (temp3) * temp2 = Kt (k)
        # inv(temp3)
        temp3_inv = np.linalg.inv(self._temp3)

        # temp4 = inv (temp3) * temp2
        self._temp4 = np.dot(temp3_inv, self._temp2)

        # gain = temp4_T
        self._gain = self._temp4.transpose()

        # temp5 =  z(k) - measurementMatrix[4][6] (H) * statePre[6][1] (x'(k))
        # measurementMatrix[4][6] (H) * statePre[6] (x'(k))
        temp52 = np.dot(self._measurementMatrix, self._statePre)

        # temp5 =  z(k) + temp52 * -1
        self._temp5 = z - temp52

        # statePost (x(k)) = statePre (x'(k)) +  gain[4][6](K (k)) * temp5[4]
        # gain[4][6](K (k)) * temp5[4]
        temp53 = np.dot(self._gain, self._temp5)

        # statePre[4] (x'(k)) +  temp53[2]
        self._statePost = self._statePre + temp53

        # errorCovPost P(k) = errorCovPre P'(k) - gain K(k) * temp2
        # gain K(k) * temp2
        temp22 = np.dot(self._gain, self._temp2)

        # errorCovPre P'(k) + temp22 * -1
        self._errorCovPost = self._errorCovPre - temp22

        return self._statePost



if __name__ == '__main__':
    import time
    class CObject:
        def __init__(self, x=0, y=0, vx=0, vy=0, intens=100, square=10):
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.timeDetect = time.time()
            self.m_intens = intens
            self.m_square = square

    def get_lin_regress(in_data, start_pos, in_data_size, xy):
        m1 = m2 = m3_x = m4_x = 0.0
        el_count = in_data_size - start_pos
        for i in range(start_pos, in_data_size):
            m1 += i
            m2 += i * i

            m3_x += in_data[i]
            m4_x += i * in_data[i]
        det_1 = 1 / (el_count * m2 - m1 ** 2)
        m1 *= -1
        xy["kx"] = det_1 * (m1 * m3_x + el_count * m4_x)
        xy["bx"] = det_1 * (m2 * m3_x + m1 * m4_x)


    data = [0, 10, 20, 39, 82, 165]
    xy = {"kx": 0, "bx": 0}
    get_lin_regress(data, 0, len(data), xy)
    print(xy["kx"], xy["bx"])
    xy0 = xy["kx"] * (len(data) - 1) + xy["bx"]
    xyv0 = xy["kx"]

    n = []
    for i in range(0, len(data)-1):
        n.append(-(data[i] - data[i+1]))
    print(n)
    s = add = 0
    for i in range(0, len(n)):
        add += n[i]

    s = add / len(n)
    print(xy["kx"], xy["bx"])

    try:
        m_kalman = CKalmanFilterMatr(CObject(1, 1))
        k = 10
        while 1:
            last = m_kalman.GetPointPrediction()
            # print(last)
            last = m_kalman.Update({"x": k, "y": k}, True)
            print(k, last)
            k += 10

    except Exception as e:
        print("Kalman:Update", e)