    import collections
from typing import List


def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        map = collections.defaultdict(int)
        for i in intervals:
            map[i[0]]+=1
            map[i[1]]-=1
        sort_map = collections.OrderedDict(map)
        res = 0
        rooms = 0
        for item in sort_map.items():
            rooms += item[1]
            res = max(res, rooms)
        return res

dd = {'banana': 3, 'apple':4, 'pear': 1, 'orange': 2}
#按key排序
kd = collections.OrderedDict(sorted(dd.items(), key=lambda t: t[0]))
print (kd)
#按照value排序
vd = collections.OrderedDict(sorted(dd.items(),key=lambda t:t[1]))
print (vd)