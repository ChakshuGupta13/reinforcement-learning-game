import pygame, ptext
from pygame.constants import KEYDOWN, K_ESCAPE, QUIT
from pygame.locals import *
from sys import exit
from ctypes import windll
from random import choice, seed
from time import time_ns
from copy import deepcopy
from collections import defaultdict
from pprint import PrettyPrinter, pprint

pp = PrettyPrinter(indent=2)

# Seeding the random number generater with the current time in nano-seconds
seed(time_ns())

# Extracting size (width, height) of the current window / screen
WIN_W, WIN_H = windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1)

CELL_D = 1
GRID_D = 8
SCALE_UP = 90
BORDER_W = 1

WHITE = "#FFFCE8"
BLACK = "#3E363F"
ORANGE = "#EDAE49"
RED = "#D1495B"
GREEN_BLUE = "#00798C"
DARK_BLUE = "#003D5B"
BLUE = "#30638E"

pygame.init()
SCREEN = pygame.display.set_mode([0, 0])
FONT = pygame.font.Font("freesansbold.ttf", 11)


def _draw_text(text: str, loc: tuple, angle: int = 0, fontsize: int = 20):
    ptext.draw(text, loc, angle=angle, fontsize=fontsize, color=BLACK)


def _is_cue_for_quit(event) -> bool:
    """
    Determines whether @event represents a cue from the user to quit the code run.
    """
    return (event.type == KEYDOWN and event.key == K_ESCAPE) or (event.type == QUIT)


def _is_cue_for_start(event) -> bool:
    """
    Determines whether @event represents a cue from the user to start the game.
    """
    return event.type == KEYDOWN and event.key == K_RETURN


def _wait_for_cue() -> None:
    """
    Allows the `main` method to wait for the user's cue to begin the game.
    Where the user's s cue is: press the `Enter` key.
    """
    SCREEN.fill(WHITE)
    _draw_text("Press 'Enter' to begin the game.", (0, 0))
    pygame.display.update()

    game_started = False
    while not game_started:
        for event in pygame.event.get():
            if _is_cue_for_quit(event):
                exit()
            if _is_cue_for_start(event):
                game_started = True
                break


def _random_goal() -> tuple:
    """
    Return the goal cell or the terminal state:
    A cell (x, y) where x, y belong in [6, 8] each and (x, y) is not a wall.
    """
    seq = []
    for row in range(6, GRID_D + 1):
        for col in range(6, GRID_D + 1):
            if not _wall(row, col):
                seq.append((row, col))
    return choice(seq)


def _wall(row: int, col: int) -> bool:
    """
    Determines if the cell represented by (@row, @col) is a wall.
    """
    return (
        (row == 2 and col == 3)
        or (row == 3 and col == 3)
        or (row == 3 and col == 5)
        or (row == 3 and col == 6)
        or (row == 3 and col == 7)
        or (row == 3 and col == 8)
        or (row == 5 and col == 2)
        or (row == 5 and col == 3)
        or (row == 5 and col == 4)
        or (row == 5 and col == 5)
        or (row == 5 and col == 7)
        or (row == 6 and col == 7)
        or (row == 6 and col == 5)
        or (row == 7 and col == 5)
        or (row == 7 and col == 2)
        or (row == 8 and col == 2)
    )


class AgentWorldEnv:
    """
    Represents the agent world of the game and its environment.
    """

    def __init__(self, goal: tuple, reward: float = -0.01,) -> None:
        """
        Initialises the representation.
        """
        # Determine all the valid states of the game in which the agent may find itself
        self.states = set()
        for row in range(1, GRID_D + 1):
            for col in range(1, GRID_D + 1):
                if not _wall(row, col):
                    self.states.add((row, col))
        # Set the terminal state via @goal
        self.terminal_state = deepcopy(goal)
        # Set the initial state
        self.initial_state = (1, 1)
        # Set the power state
        self.power_state = (8, 5)
        # Set the `Go to Power State` states
        self.go_to_power_state = [(8, 1), (2, 6), (4, 7)]
        # Set the restart states
        self.restart = [(1, 4), (7, 4)]

        # Determine all the valid actions for the agent which the agent may take from any of the valid states
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        # Set the Reward metrix and Transition Model
        self.rewards = {}
        self.transitions = {}
        for state in self.states:
            self.rewards[state] = 1.0 if state == self.terminal_state else reward
            self.transitions[state] = {}
            for action in self.actions:
                next_state = (state[0] + action[0], state[1] + action[1])
                if next_state in self.go_to_power_state:
                    next_state = deepcopy(self.power_state)
                elif next_state in self.restart:
                    next_state = deepcopy(self.initial_state)
                if next_state in self.states:
                    self.transitions[state][action] = deepcopy(next_state)
                else:
                    self.transitions[state][action] = deepcopy(state)

    def R(self, state: tuple) -> float:
        """
        Return the reward available for the agent in @state.
        """
        return self.rewards[state]

    def T(self, state: tuple, action: tuple) -> tuple:
        """
        Return the next state in which the agent will find itself if it takes @action in @state.
        """
        return self.transitions[state][action]


class QLearningAgent:
    """
    Representation of a Q-Learning agent.
    """

    def __init__(
        self,
        agent_world_env: AgentWorldEnv,
        curiosity_limit: int = 10,
        optimistic_reward: float = 1.0,
        alpha=(lambda n: 60.0 / (59 + n)),
        gamma: float = 0.9,
    ):
        """
        Initialise a Q-Learning agent with the help of its @agent_world_env.
        """
        self.terminal_state = agent_world_env.terminal_state
        self.actions = agent_world_env.actions
        self.transitions = agent_world_env.transitions

        # Set the curiosity limit which defines ...
        # the maxiumum number of times the agent can explore its world before getting greedy.
        self.curiosity_limit = curiosity_limit

        # Set the an optimistic reward for the curiosity of the agent ...
        # equal to the maximum reward possible in any of the valid states.
        self.optimistic_reward = optimistic_reward

        # Initialise the knowledge base @self.Q of the agent about its world.
        self.Q = defaultdict(float)

        # Initialise the table which keeps ...
        # the count about how many times the agent has visited a valid state.
        self.N = defaultdict(float)

        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

        self.alpha = alpha
        self.gamma = gamma

    def f(self, utility, num_exploration) -> float:
        """
        Implementation of a simple exploration function which ... 
        balances the curiosity of the agent against its greed.
        """
        return (
            self.optimistic_reward
            if num_exploration < self.curiosity_limit
            else utility
        )

    def actions_in_state(self, state) -> list:
        """
        Returns a list of actions possible in a valid state.
        """
        return [None] if state == self.terminal_state else self.actions

    def __call__(self, percept: tuple) -> tuple:
        """
        Returns the agent's suitable next action on the basis of ...
        current @percept and its knowledge base @self.Q
        """
        state, reward = percept
        Q, N, prev_state, prev_action, prev_reward = (
            self.Q,
            self.N,
            self.prev_state,
            self.prev_action,
            self.prev_reward,
        )

        if prev_state == self.terminal_state:
            Q[prev_state, None] = reward
        if prev_state is not None:
            N[prev_state, prev_action] += 1
            Q[prev_state, prev_action] += self.alpha(N[prev_state, prev_action]) * (
                prev_reward
                + self.gamma
                * max(Q[state, action] for action in self.actions_in_state(state))
                - Q[prev_state, prev_action]
            )
        self.prev_state, self.prev_reward = state, reward
        max_possible_reward = max(
            self.f(Q[state, action], N[state, action])
            for action in self.actions_in_state(state)
        )
        possible_actions = []
        for action in self.actions_in_state(state):
            if self.f(Q[state, action], N[state, action]) == max_possible_reward:
                possible_actions.append(action)
        self.prev_action = deepcopy(choice(possible_actions))
        return self.prev_action


def _draw_rect(rect, color: str, border_width: int = 0) -> None:
    pygame.draw.rect(SCREEN, color, rect, border_width)


def _draw(
    agentWorldEnv: AgentWorldEnv,
    agent: QLearningAgent,
    path: list,
    path_cost: float,
    final: bool = False,
) -> None:
    SCREEN.fill(WHITE)
    if final:
        _draw_text("Optimal", (0, 0))

    size = CELL_D * SCALE_UP
    for row in range(1, GRID_D + 1):
        for col in range(1, GRID_D + 1):
            left = ((WIN_W - (GRID_D * size)) / 2) + ((col - 1) * size)
            top = ((WIN_H - (GRID_D * size)) / 2) + ((row - 1) * size)
            rect = pygame.Rect(left, top, size, size)
            state = (row, col)

            _draw_rect(rect, BLACK, BORDER_W)
            if _wall(row, col):
                _draw_rect(rect, BLACK)
            if state in agentWorldEnv.go_to_power_state:
                _draw_rect(rect, ORANGE)
            if state in agentWorldEnv.restart:
                _draw_rect(rect, RED)
            if state == agentWorldEnv.power_state:
                _draw_rect(rect, BLUE)
            if state == agentWorldEnv.terminal_state:
                _draw_rect(rect, GREEN_BLUE)
            if agent.prev_state != None and state == agent.prev_state:
                pygame.draw.circle(
                    SCREEN,
                    DARK_BLUE,
                    [(left + (size / 2)), (top + (size / 2))],
                    (size / 4),
                )

            if agent.Q[(state, (0, -1))]:
                _draw_text(
                    "Q_W: {:.2f}".format(agent.Q[(state, (0, -1))]),
                    (left, top + size),
                    90,
                )
            if agent.Q[(state, (0, 1))]:
                _draw_text(
                    "Q_E: {:.2f}".format(agent.Q[(state, (0, 1))]),
                    (left + size, top),
                    -90,
                )
            if agent.Q[(state, (-1, 0))]:
                _draw_text("Q_N: {:.2f}".format(agent.Q[(state, (-1, 0))]), (left, top))
            if agent.Q[(state, (1, 0))]:
                _draw_text(
                    "Q_S: {:.2f}".format(agent.Q[(state, (1, 0))]),
                    (left + size, top + size),
                    180,
                )

            if col == 1:
                row_label = "Row_" + str(row)
                text_out = FONT.render(row_label, True, BLACK)
                text_rect = text_out.get_rect()
                text_rect.update(
                    left - text_rect.width,
                    top + ((size - text_rect.height) / 2),
                    text_rect.width,
                    text_rect.height,
                )
                SCREEN.blit(text_out, text_rect)
            if row == 1:
                col_label = "Col_" + str(col)
                text_out = FONT.render(col_label, True, BLACK)
                text_rect = text_out.get_rect()
                text_rect.update(
                    left + ((size - text_rect.width) / 2),
                    top - text_rect.height,
                    text_rect.width,
                    text_rect.height,
                )
                SCREEN.blit(text_out, text_rect)
    height = ((WIN_H - (GRID_D * size)) / 2) + (GRID_D * size)
    _draw_text("Path: " + "".join(path), (((WIN_W - (GRID_D * size)) / 2), height))
    height += text_rect.height
    _draw_text(
        "Path Cost: " + str(path_cost), (((WIN_W - (GRID_D * size)) / 2), height)
    )


def _direction_char(action: tuple):
    """
    Returns a character representing the direction of @action.
    """
    if action == (0, -1):
        return "W"
    if action == (1, 0):
        return "S"
    if action == (0, 1):
        return "E"
    if action == (-1, 0):
        return "N"


def main() -> None:
    # Waiting for the game to begin...
    _wait_for_cue()

    # Game started!
    # Set a random cell in the predefined range as the goal or the terminal state.
    goal = _random_goal()

    # Create the agent world of the game and its environment with @goal as the terminal state.
    agent_world_env = AgentWorldEnv(goal)

    # Create a Q-Learning agent in @agent_world_env
    agent = QLearningAgent(agent_world_env)

    details_of_paths = []
    for _ in range(30):
        state = agent_world_env.initial_state
        path = []
        path_cost = 0
        while True:
            for event in pygame.event.get():
                if _is_cue_for_quit(event):
                    exit()

            reward = agent_world_env.R(state)
            percept = (state, reward)
            next_action = agent(percept)

            path_cost -= reward
            _draw(agent_world_env, agent, path, path_cost)
            pygame.display.update()

            if next_action is None:
                agent.prev_state = None
                agent.prev_action = None
                agent.prev_reward = None
                break

            state = agent_world_env.T(state, next_action)
            path.append(_direction_char(next_action))

        details_of_paths.append(("".join(path), deepcopy(path_cost), deepcopy(agent.Q)))
        pygame.time.delay(1000)

    details_of_paths.sort(key=lambda x: x[1])
    agent.Q = details_of_paths[0][2]
    _draw(agent_world_env, agent, details_of_paths[0][0], details_of_paths[0][1], True)
    pygame.display.update()
    pygame.time.delay(5000)

    file = open("output.txt", "wt")
    for path_detail in details_of_paths:
        file.write("Path or Policy ($\\pi$) " + path_detail[0] + "\n")
        file.write("Path Cost or Policy Cost ($cost_{\\pi}$): " + str(path_detail[1]) + "\n")
        file.write("Knowledge Base ($Q_{\\pi}$):\n")
        for action in agent_world_env.actions:
            file.write("In the direction of " + str(_direction_char(action)) + ":\n\\hline\n")
            for row in range(1, GRID_D + 1):
                Q_row = []
                for col in range(1, GRID_D + 1):        
                    Q_row.append("{:.5f}".format(path_detail[2][(row, col), action]).rstrip('0').rstrip('.'))
                file.write("&".join(Q_row) + "\\\\\n")
                file.write("\\hline\n")
    file.close


if __name__ == "__main__":
    main()
