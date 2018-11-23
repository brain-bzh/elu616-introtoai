# With this template, 
# we are building an AI that will apply 
# combinatorial game theory tools against a greedy opponent.

# Unless you know what you are doing, 
# you should use this template with a very limited number of pieces of cheese, 
# as it is very demanding in terms of computations.

# The first thing you should do is copy this file (tp3.py) to the AIs folder of pyrat, example:
# cp -r ~/IntroToAI/session4/tp3.py ~/pyrat/AIs/

# A typical use would be:
# python pyrat.py -d 0 -md 0 -p 7 --rat AIs/tp3.py --python AIs/manh.py --nonsymmetric

# If enough computation time is allowed, 
# it is reasonable to grow the number of pieces of cheese up to around 15.
# For example:

# python pyrat.py -d 0 -md 0 -p 13 --rat AIs/tp3.py --python AIs/manh.py --synchronous --tests 100 --nodrawing --nonsymmetric

# In this example, we can obtain scores in the order of: "win_python": 0.07 "win_rat": 0.93

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'


# Useful utility functions to obtain new location after a move
def move(location, move):
    if move == MOVE_UP:
        return (location[0], location[1] + 1)
    if move == MOVE_DOWN:
        return (location[0], location[1] - 1)
    if move == MOVE_LEFT:
        return (location[0] - 1, location[1])
    if move == MOVE_RIGHT:
        return (location[0] + 1, location[1])

# The first things we do is we program the AI of the opponent, so that we know exactly what will be its decision in a given situation
def distance(la, lb):
    ax,ay = la
    bx,by = lb
    return abs(bx - ax) + abs(by - ay)

def turn_of_opponent(opponentLocation, piecesOfCheese):    
    closest_poc = (-1,-1)
    best_distance = -1
    for poc in piecesOfCheese:
        if distance(poc, opponentLocation) < best_distance or best_distance == -1:
            best_distance = distance(poc, opponentLocation)
            closest_poc = poc
    ax, ay = opponentLocation
    bx, by = closest_poc
    if bx > ax:
        return MOVE_RIGHT
    if bx < ax:
        return MOVE_LEFT
    if by > ay:
        return MOVE_UP
    if by < ay:
        return MOVE_DOWN
    pass

# We do not need preprocessing, so we let this function empty
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    pass

# We use a recursive function that goes through the trees of possible plays
# It takes as arguments a given situation, and return a best target piece of cheese for the player, such that aiming to grab this piece of cheese will eventually lead to a maximum score. It also returns the corresponding score
def best_target(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese):

    # First we should check how many pieces of cheese each player has to see if the match is over.
    # It is the case if no pieces of cheese are left, 
    # or if playerScore or opponentScore is more than half the total number
    # playerScore + opponentScore + piecesOfCheese
    totalPieces = len(piecesOfCheese) + playerScore + opponentScore
    if playerScore > totalPieces / 2 or opponentScore > totalPieces / 2 or len(piecesOfCheese) == 0:
        return (-1,-1), playerScore

    # If the match is not over, then the player can aim for any of the remaining pieces of cheese
    # So we will simulate the game to each of the pieces, which will then by recurrence test all
    # the possible trees.

    best_score_so_far = -1
    best_target_so_far = (-1,-1)
    for target in piecesOfCheese:
        end_state = simulate_game_until_target(
            target,playerLocation,opponentLocation,
            playerScore,opponentScore,piecesOfCheese.copy())
        _, score = best_target(*end_state)
        if score > best_score_so_far:
            best_score_so_far = score
            best_target_so_far = target

    return best_target_so_far, best_score_so_far

### FUNCTION TO COMPLETE, 
# Move the agent on the labyrinth using the function move and the different directions
# It suffices to move in the direction of the target. 
# You should only run function move once and you can't move diagonally.
# Without loss of generality, we can suppose it gets there moving vertically first then horizontally
def updatePlayerLocation(target,playerLocation):
    return playerLocation

#FUNCTION TO COMPLETE, 
#CHECK IF EITHER/BOTH PLAYERS ARE ON THE SAME SQUARE OF A CHEESE. 
#If that is the case you have to remove the cheese from the piecesOfCheese list and 
#add points to the score. The players get 1 point if they are alone on the square with a cheese.
#If both players are in the same square and there is a cheese on the square each player gets 0.5 points.
def checkEatCheese(playerLocation,opponentLocation,playerScore,opponentScore,piecesOfCheese):
    return playerScore,opponentScore


#FUNCTION TO COMPLETE
#In this function we simulate what will happen until we reach the target
#You should use the two functions defined before
def simulate_game_until_target(target,playerLocation,opponentLocation,playerScore,opponentScore,piecesOfCheese):

    #While the target cheese has not yet been eaten by either player
    #We simulate how the game will evolve until that happens    
    while target in piecesOfCheese:
        #Update playerLocation (position of your player) using updatePlayerLocation
        #Every time that we move the opponent also moves. update the position of the opponent using turn_of_opponent and move
        #Finally use the function checkEatCheese to see if any of the players is in the same square of a cheese.
    return playerLocation,opponentLocation,playerScore,opponentScore,piecesOfCheese
    

# During our turn we continue going to the next target, unless the piece of cheese it originally contained has been taken
# In such case, we compute the new best target to go to
current_target = (-1,-1)
def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    global current_target
    if current_target not in piecesOfCheese:
        current_target, score = best_target(playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese)
        print("My new target is " + str(current_target) + " and I will finish with " + str(score) + " pieces of cheese")
        
    if current_target[1] > playerLocation[1]:
        return MOVE_UP
    if current_target[1] < playerLocation[1]:
        return MOVE_DOWN
    if current_target[0] > playerLocation[0]:
        return MOVE_RIGHT
    return MOVE_LEFT
