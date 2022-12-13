import time

class mockHashMap:
    def __init__(self) -> None:
        self.res_dict = collections.defaultdict(list)
        self.start_time = time.time()
        # self.putCallCount = 0
        # self.putCallTrack = [] # Each Element in the list is the call times in ith 5 minutes
        # self.getCallCount = 0
        # self.getCallTrack = []

        #basic idea
        #using two queue, back - front < 300s

        self.last_get_idx = 0
        self.last_get_time = int(time.time())
        self.getBuffer = [0] * 300
        self.last_put_idx = 0
        self.last_put_time = int(time.time())
        self.putBuffer = [0] * 300
    
    def put(self, key, val):
        self.res_dict[key] = val

        current_time = int(time.time())
        diff = min(current_time - self.last_get_time, 300)
        self.last_get_time = current_time

        if diff == 0:
            self.getBuffer[self.last_put_idx] += 1
        else:
            for i in range(diff - 1):
                idx = (self.last_put_idx + 1 + i) % 300
                self.putBuffer[idx] = 0
            idx = (self.last_put_idx + diff) % 300
            self.putBuffer[idx] = 1
            self.last_put_idx = idx

        return self.res_dict[key]
    
    def get(self, key) :
        current_time = int(time.time())
        diff = min(current_time - self.last_get_time, 300)
        self.last_get_time = current_time

        if diff == 0:
            self.getBuffer[self.last_get_idx] += 1
        else:
            for i in range(diff - 1):
                idx = (self.last_get_idx + 1 + i) % 300
                self.getBuffer[idx] = 0
            idx = (self.last_get_idx + diff) % 300
            self.getBuffer[idx] = 1
            self.last_get_idx = idx

        return self.res_dict[key]
    
    def measure_put_load(self):
        return sum(self.putBuffer) / 300
    
    def measure_get_load(self):
        return sum(self.getBuffer) / 300
    

obj = mockHashMap()
def sleeptime(hour, min, sec):
    return hour * 3600 + min * 60 + sec


second = sleeptime(0, 0, 1)
count = 0
while 1 == 1:
    count+=1
    time.sleep(second)
    obj.put(count, count)
    if count % 2 == 0: obj.get(count)
    if count >= 10:
        break
print("get" + str(obj.measure_get_load()))
print("put" + str(obj.measure_put_load()))