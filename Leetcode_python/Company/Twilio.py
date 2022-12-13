from abc import abstractproperty
from ast import Index
import collections
from typing import Collection, List, Optional
from collections import *
import sys
import bisect
import heapq
import math
import re
from typing_extensions import Unpack

class Solution:
    def validateImageSize(self, imageUrls: List[List[str]], maxSize: str):
        sizeMap = {'KB':1e3, "MB":1e6, "GB":1e9}
        maxAmount = 0
        if maxSize != 'None':
            unit = maxSize[-2:].upper()
            maxAmount = int(maxSize[:-2]) * sizeMap[unit]
        else:
            maxAmount = 1 * sizeMap['MB']

        res = []
        for query in imageUrls :
            querySize = int(query[1])
            temp_res = 'TRUE'
            if querySize > maxAmount:
                temp_res = 'FALSE'
            res.append([query[0], temp_res])
        return res
    
    def validatePhoneNumberFormat(address: str):
        counter = collections.Counter(address)
        def check_E164(s:str):
            res = re.search(r"^\+?[1-9]\d{1,14}$", str)
            if res :
                return True
            else :
                return False
        
        def check_wechat(s:str):
            res = re.search(r"[a-zA-Z\d_@\-\+\.]{1,256}$", str)
            if res :
                return True
            else :
                return False
        
        #只有一个冒号
        if counter[" "] > 0:
            return "INVALID_ADDRESS"
    
        if counter[":"] == 1:
            seperate = address.find(":")
            provider = address[:seperate]
            identifier = address[seperate+1:]

            if provider.upper() == "WHATSAPP" or provider.upper() == "MESSENGER":
                if check_E164(identifier):
                    return provider.upper()
                else :
                    return "INVALID_ADDRESS"
    
            elif provider.upper() == "WECHAT":
                if check_wechat(identifier):
                    return "WECHAT"
                else :
                    return "INVALID_ADDRESS"

            else:
                return "INVALID_ADDRESS"

        #没有冒号只有判断号码
        elif counter[":"] == 0:
            if not check_E164(address):
                return "INVALID_ADDRESS"
            else:
                return "SMS"
        else:
            return "INVALID_ADDRESS"
