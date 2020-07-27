import numpy as np
import matplotlib.pyplot as plt
import pygame
from matplotlib import style
import time
import Symmetry as sym
import Character as chr

style.use("ggplot")

GRID_SIZE = 10
BLOB_SIZE = 50

EPOCHS = 120_000
movePenalty = 1
enemyPenalty = 300
foodReward = 25

epsilon = 0.9
epsilonDecay = 0.9998
showEvery = 5_000

actionSpace = {0: (-1, -1),
               1: (-1, 1),
               2: (1, -1),
               3: (1, 1),
               (-1, -1): 0,
               (-1, 1): 1,
               (1, -1): 2,
               (1, 1): 3}

lr = 0.1
discount = 0.95

blobColors = {'player': (255, 255, 255), 'food': (0, 255, 0), 'enemy': (0, 0, 255)}

pygame.init()
SCREEN = pygame.display.set_mode((GRID_SIZE * BLOB_SIZE, GRID_SIZE * BLOB_SIZE))


def updateScreen(food, player, enemy):
    SCREEN.fill((0, 0, 0))
    pygame.draw.rect(SCREEN, blobColors['food'], (food.y * BLOB_SIZE, food.x * BLOB_SIZE, BLOB_SIZE, BLOB_SIZE))
    pygame.draw.rect(SCREEN, blobColors['player'], (player.y * BLOB_SIZE, player.x * BLOB_SIZE, BLOB_SIZE, BLOB_SIZE))
    pygame.draw.rect(SCREEN, blobColors['enemy'], (enemy.y * BLOB_SIZE, enemy.x * BLOB_SIZE, BLOB_SIZE, BLOB_SIZE))

    pygame.display.update()


qTable = {}
count = 0
for x1 in range(- GRID_SIZE + 1, GRID_SIZE // 2):
    for y1 in range(- GRID_SIZE + 1, GRID_SIZE // 2):
        for x2 in range(- GRID_SIZE + 1, GRID_SIZE // 2):
            for y2 in range(- GRID_SIZE + 1, GRID_SIZE // 2):
                count += 1
                qTable[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

episodeRewards = []

initTime = time.time()

endGame = False
for epoch in range(EPOCHS):
    player = chr.Character(GRID_SIZE)
    food = chr.Character(GRID_SIZE)
    enemy = chr.Character(GRID_SIZE)
    # Checks that blobs aren't on top of each other, if that's the case take new initial position
    # until they are not in top of each other.
    while player == food or player == enemy or food == enemy:
        player = chr.Character(GRID_SIZE)
        food = chr.Character(GRID_SIZE)
        enemy = chr.Character(GRID_SIZE)
    # After 'showEvery' episodes, the info of the last 'showEvery' episodes is shown
    if epoch % showEvery == 0:
        print(f"on # {epoch}, epsilon: {epsilon}")
        print(f"{showEvery} ep mean {np.mean(episodeRewards[-showEvery:])}")
        # 'show' is set to true meaning that we show the next entire episode with pygame.
        show = True
    else:
        show = False
    # The actual episode reward is set to 0
    episodeReward = 0
    # 200 actions are taken in each episode
    for i in range(200):
        # Observational space is set.
        obs = (player - food, player - enemy)
        # The Rotational Symmetry is broken and the transformed observational space is return.
        tObs = sym.breakOBSRotationalSymmetry(obs, player)

        # The best move in the transformed observational space is taken from the qTable or a random move
        if np.random.random() > epsilon:
            tAction = np.argmax(qTable[tObs])
        else:
            tAction = np.random.randint(0, 4)

        # With the info of the actual quadrant of the player and the transformed action the inverse of the
        # move to take obs to tObs is applied and from that we have the action in the normal observational space
        quadrant = player.quadrant()
        if quadrant > 1:
            moveX, moveY = actionSpace[tAction]
            moveX, moveY = - moveY, moveX
            if quadrant == 2 or quadrant == 4:
                moveX, moveY = - moveY, moveX
                if quadrant == 2:
                    moveX, moveY = - moveY, moveX
            action = actionSpace[(moveX, moveY)]
        else:
            action = tAction

        # The action is applied
        player.action(action)

        # Food and enemy move?
        if player == enemy:
            reward = - enemyPenalty
        elif player == food:
            reward = foodReward
        else:
            reward = - movePenalty

        # The best next move value in the qtable is calcultaed.
        newObs = (player - food, player - enemy)
        newTObs = sym.breakOBSRotationalSymmetry(newObs, player)

        maxFutureQ = np.max(qTable[newTObs])
        currentQ = qTable[tObs][tAction]

        # If we hit an enemy or we find food the games finish
        if reward == foodReward or reward == - enemyPenalty:
            newQ = reward
            endGame = True
        else:
            # QLearning algorithm is applied
            newQ = (1 - lr) * currentQ + lr * (reward + discount * maxFutureQ)
        # Update of the new q value
        qTable[tObs][tAction] = newQ

        if show:
            updateScreen(food, player, enemy)
            pygame.time.wait(80)
        episodeReward += reward
        if endGame:
            endGame = False
            break
    episodeRewards.append(episodeReward)
    epsilon *= epsilonDecay

print(len(episodeRewards))
movingAvg = np.convolve(episodeRewards, np.ones(showEvery) / showEvery, mode="valid")

print("ms ", (time.time() - initTime))
plt.plot(movingAvg)
plt.ylabel(f"reward {showEvery}ma")
plt.xlabel("episode #")
plt.show()
