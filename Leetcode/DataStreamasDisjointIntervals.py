class SummaryRanges:

    def __init__(self):
        self.intervals = []

    def addNum(self, value: int) -> None:
        if not self.intervals:
            self.intervals.append([value, value])
            return
        for i, interval in enumerate(self.intervals):
            if interval[0] <= value <= interval[1]:
                return
            if interval[0] > value:
                if i == 0:
                    if value + 1 == interval[0]:
                        self.intervals[i][0] = value
                    else:
                        self.intervals.insert(0, [value, value])
                    return
                if self.intervals[i - 1][1] + 1 == value:
                    if value + 1 == interval[0]:
                        self.intervals[i - 1][1] = interval[1]
                        self.intervals.pop(i)
                    else:
                        self.intervals[i - 1][1] = value
                else:
                    if value + 1 == interval[0]:
                        self.intervals[i][0] = value
                    else:
                        self.intervals.insert(i, [value, value])
                return
        if self.intervals[-1][1] + 1 == value:
            self.intervals[-1][1] = value
        else:
            self.intervals.append([value, value])

    def getIntervals(self) -> list[list[int]]:
        return self.intervals
