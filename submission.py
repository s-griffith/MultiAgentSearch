from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

#should I be taking into account the other robot's position? If it's free and closer to the package I want to pick up?
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    #If the robot is not carrying a package, find the closest one and see if reachable:
    if not env.robot_is_occupied(robot_id):
        dist = manhattan_distance(robot.position, env.packages[0].position)
        y = (manhattan_distance(env.packages[0].position, env.packages[0].destination) * 2)
        if env.packages[1].on_board:
            dist = min(dist, manhattan_distance(robot.position, env.packages[1].position))
            y = max(y, (manhattan_distance(env.packages[1].position, env.packages[1].destination) * 2))
        x = 1/(dist + 1) + 100*robot.credit
    #If the robot is carrying a package, see if the destination is reachable:
    else:
        dist = manhattan_distance(robot.position, robot.package.destination)
        y = manhattan_distance(robot.package.position, robot.package.destination) * 2
        x = y*(1/(dist + 1)) + 100*robot.credit
    #The package/its destination are reachable:
    if dist - robot.battery <= 0:
        return  x
    #It is worth it to charge the robot in order to pick up/deliver the package:
    if robot.credit - y <= 0:
        z = min(manhattan_distance(robot.position, env.charge_stations[0].position), manhattan_distance(robot.position, env.charge_stations[1].position))
        if z - robot.battery <= 0:
            return 1/(z + 1)
    #Unable to deliver/pick up a package, and not worth charging:
    return 0

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


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