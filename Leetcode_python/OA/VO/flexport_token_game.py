import collections

class Player:
    def __init__(self, token_dic):
        self.token_dic = token_dic
        self.card_list = []
        self.discount_dic = collections.defaultdict(int)
    
class Card:
    def __init__(self, point, cost_token_dic, color):
        self.point = point
        self.cost_token_dic = cost_token_dic
        self.color = color

def can_purchase(player: Player, card: Card):
    gold_token_remain = player.token_dic['A']
    for token, amount in card.cost_token_dic.items():
        if player.token_dic[token] + player.discount_dic[token] + gold_token_remain < amount:
            return False
        if player.token_dic[token] + player.discount_dic[token] < amount:
            gold_token_remain -= (amount - player.token_dic[token] - player.discount_dic[token])
    return True

def make_purchase(player: Player, card: Card):
    gold_token_remain = player.token_dic['A']
    for token, amount in card.cost_token_dic.items():
        amount -= player.discount_dic[token]
        if amount <= player.token_dic[token]:
            player.token_dic[token] -= amount
        else:
            gold_token_remain -= (amount - player.token_dic[token])
            player.token_dic[token] = 0
    player.card_list.append(card)
    player.discount_dic[card.color] += 1
    player.token_dic['A'] = gold_token_remain

token_dic = {'R':3, 'B':3, 'K':3, 'G':3, 'W': 3, 'A': 2}
player = Player(token_dic)
cost_token_dic = {'B': 1, 'G': 2}
card = Card(0, cost_token_dic, 'B')
print(can_purchase(player, card))
make_purchase(player, card)
print(player.token_dic, player.card_list)
print(can_purchase(player, card))
make_purchase(player, card)
print(player.token_dic, player.card_list)
print(can_purchase(player, card))
make_purchase(player, card)
print(player.token_dic, player.card_list)

#                  +-----+
#                 |     |
#        Cost --> |5B   |
#                 |3G   |
#                 +-----+
# Notation: Red - R, Blue - U, Black - B, Green - G, White - W

# The goal for this part is to implement a function that determines if a card can be purchased by a particular player given the current state of the game. How you choose to represent the purchase, game state, and function signature is completely up to you.