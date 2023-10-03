board = {}

def display_board(board):
    print("Current Board:")
    keys = list(board.keys())
    keys.sort()
    for k in keys:
        print("{}: {}".format(k, board[k]))

def place_piece(board, player, row, col):
    board[(row, col)] = player

def check_winner(board, player, row, col):
    directions = [(0, 0), (1, 1), (0, 1), (1, 0), (1, -1), (-1, -1), (0, -1), (-1, 0), (-1, 1)]
    for dir_index in range(1, 5):
        count = 0
        r, c = row, col
        cur_dir_r = directions[dir_index][0]
        cur_dir_c = directions[dir_index][1]
        while (r, c) in board and board[(r, c)] == player:
            count += 1
            if count >= 3:
                return True
            r += cur_dir_r
            c += cur_dir_c
        cur_dir_r = directions[dir_index * 2][0]
        cur_dir_c = directions[dir_index * 2][1]
        r, c = row + cur_dir_r, col + cur_dir_c
        while (r, c) in board and board[(r, c)] == player:
            count += 1
            if count >= 3:
                return True
            r += cur_dir_r
            c += cur_dir_c
    return False

def main():
    player_Red = "R"
    player_Blue = "B"
    current_player = player_Red
    while True:
        display_board(board)
        print("Player {}'s turn. Move to which row?".format(current_player))
        row = int(input())
        print("Player {}'s turn. Move to which column?".format(current_player))
        col = int(input())
        place_piece(board, current_player, row, col)
        if check_winner(board, current_player, row, col):
            display_board(board)
            print("Player {} has won! Congratulations!".format(current_player))
            break
        if current_player == player_Red:
            current_player = player_Blue
        else:
            current_player = player_Red

if __name__ == "__main__":
    main()