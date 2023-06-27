from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import math


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    rival = env.get_robot(1-robot_id)
    #If the robot is not carrying a package, find the closest one and see if reachable:
    if not env.robot_is_occupied(robot_id):
        dist0 = manhattan_distance(robot.position, env.packages[0].position)
        dest0 = manhattan_distance(env.packages[0].position, env.packages[0].destination)
        reward0 = dest0 * 2
        if rival.package:
            dist = dist0
            y = reward0
            x = 1/(dist + 1) + 100*robot.credit
            if dist + dest0 - robot.battery < 0:
                return  x
            z = min(manhattan_distance(robot.position, env.charge_stations[0].position) + manhattan_distance(env.charge_stations[0].position, env.packages[0].position), 
                manhattan_distance(robot.position, env.charge_stations[1].position) + manhattan_distance(env.charge_stations[1].position, env.packages[0].position))
            if robot.credit - y <= 0:
                return 1/(z+1) + 100*robot.credit
            return 100*robot.credit
        else:
            rivalDist0 = manhattan_distance(rival.position, env.packages[0].position)
            rivalDist1 = manhattan_distance(rival.position, env.packages[1].position)
            dist1 = manhattan_distance(robot.position, env.packages[1].position)
            dest1 = manhattan_distance(env.packages[1].position, env.packages[1].destination)
            reward1 = dest1 * 2
            if rivalDist0 > rivalDist1 or dist0 <= rivalDist0:
                #If can pick up and deliver:
                if dist0 + dest0 - robot.battery < 0:
                    return 1/(dist0+1) + 100*robot.credit
                #If needs to charge, see if possible and worth it:
                elif reward0 > robot.credit:
                    chargeDist = min(manhattan_distance(env.packages[0].position, env.charge_stations[0].position), manhattan_distance(env.packages[0].position, env.charge_stations[1].position))
                    #If can pick up package and then charge, do minimum of distance:
                    if dist0 + chargeDist - robot.battery <= 0:
                        dist = min(manhattan_distance(robot.position, env.charge_stations[0].position) + manhattan_distance(env.charge_stations[0].position, env.packages[0].position)
                                    + manhattan_distance(env.packages[0].position, env.packages[0].destination), manhattan_distance(robot.position, env.packages[0].position) 
                                    + manhattan_distance(env.packages[0].position, env.charge_stations[0].position) + manhattan_distance(env.charge_stations[0].position, env.packages[0].destination),
                                    manhattan_distance(robot.position, env.charge_stations[1].position) + manhattan_distance(env.charge_stations[1].position, env.packages[0].position)
                                    + manhattan_distance(env.packages[0].position, env.packages[0].destination), manhattan_distance(robot.position, env.packages[0].position) 
                                    + manhattan_distance(env.packages[0].position, env.charge_stations[1].position) + manhattan_distance(env.charge_stations[1].position, env.packages[0].destination))
                        return 1/(dist+1) + 100*robot.credit
                    #If must charge now:
                    else:
                        dist = min(manhattan_distance(robot.position, env.charge_stations[0].position) + manhattan_distance(env.charge_stations[0].position, env.packages[0].position)
                                    + manhattan_distance(env.packages[0].position, env.packages[0].destination), manhattan_distance(robot.position, env.charge_stations[1].position) + manhattan_distance(env.charge_stations[1].position, env.packages[0].position)
                                    + manhattan_distance(env.packages[0].position, env.packages[0].destination))
                        return 1/(dist+1) + 100*robot.credit
            elif rivalDist1 >= rivalDist0 or dist1 <= rivalDist1:
                #If can pick up and deliver:
                if dist1 + dest1 - robot.battery <= 0:
                    return 1/(dist1+1) + 100*robot.credit
                #If needs to charge, see if possible and worth it:
                elif reward1 > robot.credit:
                    dist = min(manhattan_distance(robot.position, env.charge_stations[0].position) + manhattan_distance(env.charge_stations[0].position, env.packages[1].position),
                               manhattan_distance(robot.position, env.packages[1].position) + manhattan_distance(env.packages[1].position, env.charge_stations[0].position), 
                               manhattan_distance(robot.position, env.charge_stations[1].position) + manhattan_distance(env.charge_stations[1].position, env.packages[1].position),
                               manhattan_distance(robot.position, env.packages[1].position) + manhattan_distance(env.packages[1].position, env.charge_stations[1].position))
                    #If can pick up package and then charge, do minimum of distance:
                    if dist - robot.battery <= 0:
                        return 1/(dist+1) + 100*robot.credit
                    #If must charge now:
                    else:
                        chargeDist = min(manhattan_distance(env.packages[1].position, env.charge_stations[0].position), manhattan_distance(env.packages[1].position, env.charge_stations[1].position))
                        return 1/(chargeDist+1) + 100*robot.credit

    #If the robot is carrying a package, see if the destination is reachable:
    else:
        dist = manhattan_distance(robot.position, robot.package.destination)
        y = manhattan_distance(robot.package.position, robot.package.destination) * 2
        x = y/(dist + 1) + 100*robot.credit
        #The destination is reachable:
        if dist - robot.battery <= 0:
            return  x
        z = min(manhattan_distance(robot.position, env.charge_stations[0].position) + manhattan_distance(env.charge_stations[0].position, robot.package.destination), 
                manhattan_distance(robot.position, env.charge_stations[1].position) + manhattan_distance(env.charge_stations[1].position, robot.package.destination))
        #It is worth it to charge the robot in order to deliver the package:
        if robot.credit - y <= 0:
            return 1/(z+1) + 100*robot.credit
        return 0
    
    #Unable to deliver/pick up a package, and not worth charging:
    return 100*robot.credit


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def rb_minimax(self, env: WarehouseEnv, agent_id, depth, time_limit, current_robot):
        if time.time() > time_limit:
            return None, None
        if depth == 0 or env.done():
            return smart_heuristic(env, current_robot), None
        moves, children = self.successors(env, current_robot)
        moves_return = None
        if agent_id == current_robot:
            currMax = -math.inf
            for i, c in enumerate(children):
                v, _ = self.rb_minimax(c, agent_id, depth-1, time_limit, 1-current_robot)
                if v == None:
                    return None, None
                if v > currMax:
                    currMax = v
                    moves_return = moves[i]
            return currMax, moves_return
        currMin = math.inf
        for i, c in enumerate(children):
            v, _ = self.rb_minimax(c, agent_id, depth-1, time_limit, 1-current_robot)
            if v == None:
                return None, None
            if v < currMin:
                currMin = v
                moves_return = moves[i]
        return currMin, moves_return


    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        limit = time.time() + time_limit - 0.01
        moves, _ = self.successors(env, agent_id)
        moves_return = random.choice(moves)
        depth = 1
        while time.time() < limit and depth <= 4:
            if depth > 2*env.get_robot(agent_id).battery:
                return moves_return
            h, moves = self.rb_minimax(env, agent_id, depth, limit, agent_id)
            depth += 1
            if h != None and h > 0 and moves != None:
                moves_return = moves
            else:
                return moves_return
        return moves_return


class AgentAlphaBeta(Agent):
    def rb_alpha_beta(self, env: WarehouseEnv, agent_id, depth, time_limit, current_robot, alpha, beta):
        if time.time() > time_limit:
            return None, None
        if depth == 0 or env.done():
            return smart_heuristic(env, current_robot), None
        moves, children = self.successors(env, current_robot)
        moves_return = None
        if agent_id == current_robot:
            currMax = -math.inf
            for i, c in enumerate(children):
                v, _ = self.rb_alpha_beta(c, agent_id, depth-1, time_limit, 1-current_robot, alpha, beta)
                if v == None:
                    return None, None
                if v > currMax:
                    currMax = v
                    moves_return = moves[i]
                alpha = max(currMax, alpha)
                if currMax >= beta:
                    return math.inf, None
            return currMax, moves_return
        currMin = math.inf
        for i, c in enumerate(children):
            v, _ = self.rb_alpha_beta(c, agent_id, depth-1, time_limit, 1-current_robot, alpha, beta)
            if v == None:
                return None, None
            if v < currMin:
                currMin = v
                moves_return = moves[i]
            beta = min(currMin, beta)
            if currMin <= alpha:
                return -math.inf, None
        return currMin, moves_return


    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        limit = time.time() + time_limit - 0.01
        moves, _ = self.successors(env, agent_id)
        moves_return = random.choice(moves)
        depth = 1
        while time.time() < limit and depth <= 4:
            if depth > 2*env.get_robot(agent_id).battery:
                return moves_return
            h, moves = self.rb_alpha_beta(env, agent_id, depth, limit, agent_id, -math.inf, math.inf)
            depth += 1
            if h != None and moves != None:
                moves_return = moves
            else:
                return moves_return
        return moves_return

class AgentExpectimax(Agent):
    def expectimax(self, env: WarehouseEnv, agent_id, depth, time_limit, current_robot):
        if time.time() > time_limit:
            return None, None
        if depth == 0 or env.done():
            return smart_heuristic(env, current_robot), None
        moves, children = self.successors(env, current_robot)
        moves_return = None
        if agent_id == current_robot:
            currMax = -math.inf
            for i, c in enumerate(children):
                v, _ = self.expectimax(c, agent_id, depth-1, time_limit, 1-current_robot)
                if v == None:
                    return None, None
                if v > currMax:
                    currMax = v
                    moves_return = moves[i]
            return currMax, moves_return
        #If it is the rival's turn:
        currMin = math.inf
        total = 0
        numMoves = len(children)
        for i, c in enumerate(children):
            v, _ = self.expectimax(c, agent_id, depth-1, time_limit, 1-current_robot)
            if v == None:
                return None, None
            if v < currMin:
                currMin = v
            if moves[i] == 'charge':
                v = v*2
            total += v
            average = total/numMoves
        return average, moves[0]

    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        limit = time.time() + time_limit - 0.01
        moves, _ = self.successors(env, agent_id)
        moves_return = random.choice(moves)
        depth = 1
        while time.time() < limit and depth <= 4:
            if depth > 2*env.get_robot(agent_id).battery:
                return moves_return
            h, moves = self.expectimax(env, agent_id, depth, limit, agent_id)
            depth += 1
            if h != None and h > 0 and moves != None:
                moves_return = moves
            else:
                return moves_return
        return moves_return


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move south", "move west", "move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)