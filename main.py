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

INITIAL_POINT = (1, 1)
RESTART_STATES = [(1, 4), (7, 4)]
GO_TO_POWER_STATES = [(8, 1), (2, 6), (4, 7)]
POWER_STATE = (8, 5)

pygame.init()
SCREEN = pygame.display.set_mode([0, 0])
FONT = pygame.font.Font("freesansbold.ttf", 11)


def _random_goal() -> tuple:
    seq = []
    for row in range(6, GRID_D + 1):
        for col in range(6, GRID_D + 1):
            if not _wall(row, col):
                seq.append((row, col))
    return choice(seq)


def _draw_rect(rect, color: str, border_width: int = 0) -> None:
    pygame.draw.rect(SCREEN, color, rect, border_width)


def _wall(row: int, col: int) -> bool:
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


class GameEnv:
    def __init__(
        self,
        states: set,
        terminal_state: tuple,
        actions: list,
        non_terminal_state_reward,
        gamma: float = 0.9,
    ) -> None:
        self.states = deepcopy(states)
        self.terminal_state = deepcopy(terminal_state)
        self.initial_state = deepcopy(INITIAL_POINT)
        self.actions = deepcopy(actions)
        self.gamma = gamma
        self.rewards = {}
        self.transitions = {}

        for state in self.states:
            self.rewards[state] = (
                1.0 if state == self.terminal_state else non_terminal_state_reward
            )
            self.transitions[state] = {}
            for action in self.actions:
                next_state = (state[0] + action[0], state[1] + action[1])
                if next_state in GO_TO_POWER_STATES:
                    next_state = deepcopy(POWER_STATE)
                elif next_state in RESTART_STATES:
                    next_state = deepcopy(INITIAL_POINT)
                if next_state in self.states:
                    self.transitions[state][action] = deepcopy(next_state)

    def R(self, state: tuple) -> float:
        return self.rewards[state]

    def T(self, state: tuple, action: tuple) -> tuple:
        return self.transitions[state][action]


class QLearningAgent:
    def __init__(
        self,
        gameEnv: GameEnv,
        N_e: int = 10,
        R_plus: float = 1.0,
        alpha=(lambda n: 60.0 / (59 + n)),
    ):
        self.gamma = gameEnv.gamma
        self.terminal_state = gameEnv.terminal_state
        self.transitions = gameEnv.transitions

        self.max_explore = N_e  # iteration limit in exploration function
        self.optimistic_reward = R_plus  # large value to assign before iteration limit

        self.Q = defaultdict(float)
        self.N = defaultdict(float)
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

        self.alpha = alpha

    def f(self, utility, num_explore) -> float:
        return self.optimistic_reward if num_explore < self.max_explore else utility

    def actions_in_state(self, state) -> list:
        return (
            [None]
            if state == self.terminal_state
            else list(self.transitions[state].keys())
        )

    def __call__(self, percept: tuple) -> tuple:
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
        if prev_state == self.terminal_state:
            self.prev_state = None
            self.prev_action = None
            self.prev_reward = None
        else:
            self.prev_state, self.prev_reward = state, reward
            max_reward_possible = max(
                self.f(Q[state, action], N[state, action])
                for action in self.actions_in_state(state)
            )
            possible_actions = []
            for action in self.actions_in_state(state):
                if self.f(Q[state, action], N[state, action]) == max_reward_possible:
                    possible_actions.append(action)
            self.prev_action = deepcopy(choice(possible_actions))
        return self.prev_action


def _draw(
    goal: tuple, init: tuple, path: list, path_cost: float, Q: dict, final: bool = False
) -> None:
    SCREEN.fill(WHITE)
    if final:
        ptext.draw("Final", (0, 0), fontsize=40)
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
            if state in GO_TO_POWER_STATES:
                _draw_rect(rect, ORANGE)
            if state in RESTART_STATES:
                _draw_rect(rect, RED)
            if state == POWER_STATE:
                _draw_rect(rect, BLUE)
            if state == goal:
                _draw_rect(rect, GREEN_BLUE)
            if init != None and state == init:
                pygame.draw.circle(
                    SCREEN,
                    DARK_BLUE,
                    [(left + (size / 2)), (top + (size / 2))],
                    (size / 4),
                )

            if Q[(state, (0, -1))]:
                ptext.draw(
                    "Q_W: {:.2f}".format(Q[(state, (0, -1))]),
                    (left, top + size),
                    angle=90,
                    fontsize=20,
                    color=BLACK,
                )
            if Q[(state, (0, 1))]:
                ptext.draw(
                    "Q_E: {:.2f}".format(Q[(state, (0, 1))]),
                    (left + size, top),
                    angle=-90,
                    fontsize=20,
                    color=BLACK,
                )
            if Q[(state, (-1, 0))]:
                ptext.draw(
                    "Q_N: {:.2f}".format(Q[(state, (-1, 0))]),
                    (left, top),
                    fontsize=20,
                    color=BLACK,
                )
            if Q[(state, (1, 0))]:
                ptext.draw(
                    "Q_S: {:.2f}".format(Q[(state, (1, 0))]),
                    (left + 25, top + size - 15),
                    fontsize=20,
                    color=BLACK,
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
    ptext.draw(
        "Path: " + "".join(path),
        (((WIN_W - (GRID_D * size)) / 2), height),
        fontsize=20,
        color=BLACK,
    )
    height += text_rect.height
    ptext.draw(
        "Path Cost: " + str(path_cost),
        (((WIN_W - (GRID_D * size)) / 2), height),
        fontsize=20,
        color=BLACK,
    )


def _dir_char(action: tuple):
    if action == (0, -1):
        return "W"
    if action == (1, 0):
        return "S"
    if action == (0, 1):
        return "E"
    if action == (-1, 0):
        return "N"


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
    ptext.draw("Press 'Enter' to begin the game.", (0, 0), color=BLACK)
    pygame.display.update()
    
    game_started = False
    while not game_started:
        for event in pygame.event.get():
            if _is_cue_for_quit(event):
                exit()
            if _is_cue_for_start(event):
                game_started = True
                break


def main() -> None:
    _wait_for_cue()
    goal = _random_goal()
    states = set()
    actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    non_terminal_state_reward = -0.01
    for row in range(1, GRID_D + 1):
        for col in range(1, GRID_D + 1):
            if not _wall(row, col):
                states.add((row, col))
    gameEnv = GameEnv(states, goal, actions, non_terminal_state_reward)
    agent = QLearningAgent(gameEnv)
    paths = []
    for _ in range(30):
        state = gameEnv.initial_state
        path = []
        path_cost = 0
        while True:
            for event in pygame.event.get():
                if _is_cue_for_quit(event):
                    exit()
            reward = gameEnv.R(state)
            path_cost += reward
            percept = (state, reward)
            next_action = agent(percept)
            _draw(goal, agent.prev_state, path, path_cost, agent.Q)
            pygame.display.flip()
            if next_action is None:
                agent.prev_state = None
                agent.prev_action = None
                agent.prev_reward = None
                break
            path.append(_dir_char(next_action))
            state = gameEnv.T(state, next_action)
        paths.append((path_cost, "".join(path), dict(agent.Q)))
        pygame.time.delay(1000)
    paths.sort(reverse=True,key=lambda x: x[0])
    _draw(goal, agent.prev_state, paths[0][1], paths[0][0], paths[0][2], True)
    file = open("output.txt", "wt")
    file.write(pp.pformat(paths))
    file.close


if __name__ == "__main__":
    main()
