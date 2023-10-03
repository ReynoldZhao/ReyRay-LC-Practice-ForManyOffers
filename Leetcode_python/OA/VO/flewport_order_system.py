import heapq

class client:
    # def __init__(self) -> None:
    #     self.sell_backlog = []
    #     self.buy_backlog = []
    
    # def buy(self, )
    def getNumberOfBacklogOrders(self, orders: List[List[int]]) -> int:
        buy = []
        sell = []
        for price, amount, type in orders:
            if type == 0:
                while amount > 0 and len(sell) > 0:
                    smallest_sell = sell[0]
                    if smallest_sell[0] > price:
                        break
                    deal = min(amount, smallest_sell[1])
                    amount -= deal
                    smallest_sell[1] -= deal
                    if smallest_sell[1] == 0:
                        heapq.heappop(sell)
                if amount > 0:
                    heapq.heappush(buy, [-price, amount])
            else:
                while amount > 0 and len(buy) > 0:
                    largest_buy = buy[0]
                    if largest_buy[0] < price:
                        break
                    deal = min(amount, largest_buy[1])
                    amount -= deal
                    largest_buy[1] -= deal
                    if largest_buy[1] == 0:
                        heapq.heappop(buy)
                if amount > 0 :
                    heapq.heappush(sell, [price, amount])
        res = sum(t[1] for t in buy) + sum(t[1] for t in sell)
        return res % (10**9 + 7)


