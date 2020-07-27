def breakOBSRotationalSymmetry(obs, player):
    q = player.quadrant()
    if q == 4:
        return (- obs[0][0], -obs[0][1]), (-obs[1][0], -obs[1][1])
    elif q == 2:
        return (obs[0][1], - obs[0][0]), (obs[1][1], - obs[1][0])
    elif q == 3:
        return (- obs[0][1], obs[0][0]), (-obs[1][1], obs[1][0])
    return obs