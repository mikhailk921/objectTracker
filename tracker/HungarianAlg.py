# -*- coding: utf-8 -*-
## @file HungarianAlg.py
# @brief Файл содержит класс реализации венгерского алгоритма
import math

## @brief Класс венгерский алгоритм
class AssignmentProblemSolver:
    ## @brief - функция решения задачи назначения
    # @param distMatrixIn - входной массив
    # @param assign - массив назначений
    # @param N - число треков
    # @param M - число объектов
    # @return - Вернет cost
    def Solve(self, distMatrixIn, assign, N, M):
        try:
            cost = [0]
            self._assignmentoptimal(assign, cost, distMatrixIn, N, M)
            return cost
        except Exception as e:
            print("AssignmentProblemSolver:Solve", e)

    ## @brief - функция реализации оптимальных назначений
    # @param assignment - массив назначений
    # @param cost - цена
    # @param distMatrixIn - входной массив
    # @param N - число треков
    # @param M - число объектов
    def _assignmentoptimal(self, assignment, cost, distMatrixIn, N, M):
        nOfRows = N
        nOfColumns = M
        #print(N, M)
        #print(distMatrixIn)
        nOfElements = nOfRows * nOfColumns
        distMatrix = []
        for i in range(0, nOfElements):
            distMatrix.append(0)
        distMatrixEnd = len(distMatrix)

        for row in range(0, nOfElements):
            value = distMatrixIn[row]
            assert (value >= 0)
            distMatrix[row] = value

        coveredColumns = []
        for i in range(0, nOfColumns):
            coveredColumns.append(False)
        coveredRows = []
        for i in range(0, nOfRows):
            coveredRows.append(False)
        starMatrix = []
        for i in range(0, nOfElements):
            starMatrix.append(False)
        primeMatrix = []
        for i in range(0, nOfElements):
            primeMatrix.append(False)
        newStarMatrix = []
        for i in range(0, nOfElements):
            newStarMatrix.append(False)

        if nOfRows <= nOfColumns:
            for row in range(0, nOfRows):
                Temp = row
                minValue = distMatrix[Temp]
                Temp += nOfRows

                while Temp < distMatrixEnd:
                    value = distMatrix[Temp]
                    if value < minValue:
                        minValue = value
                    Temp += nOfRows

                Temp = row
                while Temp < distMatrixEnd:
                    distMatrix[Temp] -= minValue
                    Temp += nOfRows

            # step 1 and 2a
            for row in range(0, nOfRows):
                for col in range(0, nOfColumns):
                    if distMatrix[row + nOfRows * col] == 0:
                        if coveredColumns[col] is False:
                            starMatrix[row + nOfRows * col] = True
                            coveredColumns[col] = True
                            break

        # if nOfRows > nOfColumns
        else:
            for col in range(0, nOfColumns):
                Temp = nOfRows * col
                columnEnd = Temp + nOfRows
                minValue = distMatrix[Temp]
                Temp += 1
                while Temp < nOfRows:
                    value = distMatrix[Temp]
                    Temp += 1
                    if value < minValue:
                        minValue = value

                Temp = nOfRows * col
                while Temp < nOfRows:
                    distMatrix[Temp] -= minValue
                    Temp += 1

            # step 1 and 2a
            for col in range(0, nOfColumns):
                for row in range(0, nOfRows):
                    if distMatrix[row + nOfRows * col] == 0:
                        if coveredRows[row] is False:
                            starMatrix[row + nOfRows * col] = True
                            coveredColumns[col] = True
                            coveredRows[row] = True
                            break

            for row in range(0, nOfRows):
                coveredRows[row] = False

        minDim = 0
        if nOfRows <= nOfColumns:
            minDim = nOfRows
        else:
            minDim = nOfColumns

        # move to step 2b
        self._step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows,
                    nOfRows, nOfColumns, minDim)

        # compute cost and remove invalid assignments
        self._computeassignmentcost(assignment, cost, distMatrixIn, nOfRows)

    ## @brief - функция реализующая один из шагов алгоритма
    def _step2b(self, assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows,
               nOfRows, nOfColumns, minDim):
        # count covered columns
        nOfCoveredColumns = 0
        for col in range(0, nOfColumns):
            if coveredColumns[col] is True:
                nOfCoveredColumns += 1

        if nOfCoveredColumns == minDim:
            # algorithm finished
            self._buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns)
        else:
            # move step 3
            self._step3_5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows,
                         nOfRows, nOfColumns, minDim)

    ## @brief - функция собирает вектор назначений
    def _buildassignmentvector(self, assignment, starMatrix, nOfRows, nOfColumns):
        for row in range(0, nOfRows):
            for col in range(0, nOfColumns):
                if starMatrix[row + nOfRows * col]:
                    assignment[row] = col
                    break

    ## @brief - функция высчитывает стоимость назначений
    def _computeassignmentcost(self, assignment, cost, distMatrixIn, nOfRows):
        for row in range(0, nOfRows):
            col = assignment[row]
            if col >= 0:
                cost[0] += distMatrixIn[row + nOfRows * col]

    ## @brief - функция одного из шагов алгоритма
    def _step3_5(self, assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows,
                nOfRows, nOfColumns, minDim):
        while True:
            # step 3
            zerosFound = True
            while zerosFound:
                zerosFound = False
                for col in range(0, nOfColumns):
                    if coveredColumns[col] is False:
                        for row in range(0, nOfRows):
                            if ((coveredRows[row] is False) and (distMatrix[row + nOfRows * col] == 0)):
                                # prime zero
                                primeMatrix[row + nOfRows * col] = True

                                # find starred zero in current row
                                starCol = 0
                                for i in range(0, nOfColumns):
                                    starCol = i
                                    if starMatrix[row + nOfRows * i] is True:
                                        break
                                    else:
                                        if i == nOfColumns - 1:
                                            starCol += 1

                                if starCol == nOfColumns:  # no starred zero found
                                    # move to step 4
                                    self._step4(assignment, distMatrix, starMatrix, newStarMatrix,
                                               primeMatrix, coveredColumns, coveredRows, nOfRows,
                                               nOfColumns, minDim, row, col)
                                    return
                                else:
                                    coveredRows[row] = True
                                    coveredColumns[starCol] = False
                                    zerosFound = True
                                    break

            # step 5
            h = 9999999999999
            for row in range(0, nOfRows):
                if coveredRows[row] is False:
                    for col in range(0, nOfColumns):
                        if coveredColumns[col] is False:
                            value = distMatrix[row + nOfRows * col]
                            if value < h:
                                h = value

            # add h to each covered row
            for row in range(0, nOfRows):
                if coveredRows[row] is True:
                    for col in range(0, nOfColumns):
                        distMatrix[row + nOfRows * col] += h

            # subtract h from each uncovered column
            for col in range(0, nOfColumns):
                if coveredColumns[col] is False:
                    for row in range(0, nOfRows):
                        distMatrix[row + nOfRows * col] -= h

    ## @brief - функция одного из шагов алгоритма
    def _step4(self, assignment, distMatrix, starMatrix, newStarMatrix,
              primeMatrix, coveredColumns, coveredRows, nOfRows,
              nOfColumns, minDim, row, col):
        nOfElements = nOfRows * nOfColumns

        # generate temporary copy of starMatrix
        for n in range(0, nOfElements):
            newStarMatrix[n] = starMatrix[n]

        # star current zero
        newStarMatrix[row + nOfRows * col] = True

        # find starred zero in current column
        starCol = col
        starRow = 0

        for i in range(0, nOfRows):
            starRow = i
            if starMatrix[i + nOfRows * starCol] is True:
                break
            else:
                if i == nOfRows - 1:
                    starRow += 1


        while starRow < nOfRows:
            # unstar the starred zero
            newStarMatrix[starRow + nOfRows * starCol] = False

            #find primed zero in current row
            primeRow = starRow
            primeCol = 0
            for i in range(0, nOfColumns):
                primeCol = i
                if primeMatrix[primeRow + nOfRows * primeCol] is True:
                    break
                else:
                    if i == nOfColumns - 1:
                        primeCol += 1

            #star the primed zero
            newStarMatrix[primeRow + nOfRows*primeCol] = True

            #find starred zero in current column
            starCol = primeCol
            for i in range(0, nOfRows):
                starRow = i
                if starMatrix[i + nOfRows * starCol] is True:
                    break
                else:
                    if i == nOfRows - 1:
                        starRow += 1

        # use temporary copy as new starMatrix
        # delete all primes, uncover all rows
        for n in range(0, nOfElements):
            primeMatrix[n] = False
            starMatrix[n] = newStarMatrix[n]
        for n in range(0, nOfRows):
            coveredRows[n] = False

        # move to step 2a
        self._step2a(assignment, distMatrix, starMatrix, newStarMatrix,
                    primeMatrix, coveredColumns, coveredRows, nOfRows,
                    nOfColumns, minDim)

    ## @brief - функция одного из шагов алгоритма
    def _step2a(self, assignment, distMatrix, starMatrix, newStarMatrix,
                        primeMatrix, coveredColumns, coveredRows, nOfRows,
                        nOfColumns, minDim):

        for col in range(0, nOfColumns):
            startMatrixTemp = nOfRows * col
            columnEnd = startMatrixTemp + nOfRows

            while startMatrixTemp < columnEnd:
                if starMatrix[startMatrixTemp] is True:

                    coveredColumns[col] = True
                    break
                startMatrixTemp += 1

        self._step2b(assignment, distMatrix, starMatrix, newStarMatrix,
                    primeMatrix, coveredColumns, coveredRows, nOfRows,
                    nOfColumns, minDim)


if __name__ == '__main__':

    try:

        point = {"x": 0, "y": 0}
        tracks = []
        tracks.append({"x": 399.0, "y": 323.0})
        tracks.append({"x": 399.0, "y": 123.0})
        tracks.append({"x": 199.0, "y": 323.0})
        tracks.append({"x": 199.0, "y": 123.0})

        objects = []
        objects.append({"x": 284.0, "y": 443.0})
        objects.append({"x": 284.0, "y": 243.0})
        objects.append({"x": 84.0, "y": 443.0})
        objects.append({"x": 84.0, "y": 243.0})

        N = len(tracks)
        M = len(objects)
        assignment = []
        for i in range(0, N):
            assignment.append(-1)
        Cost = []
        for i in range(0, M * N):
            Cost.append(0)

        maxCost = 0
        for i in range(0, len(tracks)):
            for j in range(0, len(objects)):
                point["x"] = tracks[i]["x"] - objects[j]["x"]
                point["y"] = tracks[i]["y"] - objects[j]["y"]
                dist = math.sqrt(point["x"] ** 2 + point["y"] ** 2)
                Cost[i + j * N] = dist
                if dist > maxCost:
                    maxCost = dist

        APS = AssignmentProblemSolver()
        c = APS.Solve(Cost, assignment, N, M)

        print(assignment)

        for i in range(0, len(assignment)):
            print("Cost[%d] = %d ->  assignment[ %d] = %d" % (assignment[i] * N,
                                                              Cost[i + assignment[i] * N], i, assignment[i]))
            if assignment[i] != -1:
                pass

    except Exception as e:
        print(e)
