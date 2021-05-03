# Reinforcement Learning Game

## Structure

Our objective is to make the agent capable of learning the given game by playing it many times, but as the game itself is not static because its unique terminal state is chosen randomly before beginning each game (which, hereafter, is referred to as a 'game-play') hence, we make the agent play the same game, i.e. having the same terminal state, a constant `K = 30` number of times and we, hereafter, refer to each such play as a 'iteration' of that game-play.

We have a 8x8 grid as the agent world and its environment abides by the below-mentioned details which describes various types of cells or states in which the agent may or may not find itself:

- Start (Unique) (Non-Terminal): The agent starts every iteration of the game-play from the cell (1, 1).
- Power Position (Unique) (Non-Terminal): There exists only one cell of this type.
- Go to Power Position (Non-Unique) (Non-Terminal): If the agent reaches a cell of this type, then it starts its next move from the Power Position cell.
- Wall: The agent cannot enter these cells, hence, it cannot penetrate through these cells.
- Goal:
  - (Unique) There exists only one cell of this type.
  - (Terminal): If the agent reaches this cell, then the game is finished.
  - Whenever the game starts, we must randomly designate a cell `(i, j) : (6 <= i, j <= 8) and (type(i, j) != Wall)` as the goal for that game play.
- Restart (Non-Unique) (Non-Terminal): If the agent reaches a cell of this type, then it starts its next move from the Start.

## Theory

- Every non-wall cell `(x, y) : (1 <= x, y <= 8) and (type(x, y) != Wall)` is a valid state for the agent.
- The Markov-Decision-Process (MDP) transition model `T` is as follow:
  - If the agent is in a valid state, then only it may move in one of the following four directions, deterministically:
    - Up or North (N) or (-1, 0)
    - Right or East (E) or (0, 1)
    - Down or South (S) or (1, 0)
    - Left or West (W) or (0, -1)
  - But the agent only can move in those directions which result into valid states else it will remain in the same state.

    ```python
    if type(state=(x, y)) = Non-Terminal: 
      if valid(x + a_x, y + a_y) == True:
        T(state=(x, y), a=(a_x, a_y)) = (x + a_x, y + a_y)
      else:
        T(state=(x, y), a=(a_x, a_y)) = (x, y)
    ```

  - If the agent is in the terminal state, then it cannot move further for that iteration, hence:

    ```python
    if type(state) == Non-Terminal:
      actions_in(state) = [N,E,S,W]
    else:
      actions_in(state) = [None]
    ```

  - If the agent reaches either a 'Go to Power Position' cell or a 'Restart' cell, then the agent will find itself in 'Power Position' cell or 'Start' cell respectively before making its next move.
- Every non-terminal state of the agent world have a reward of -0.01 and the terminal state has a reward of 1, hence, the reward function can be represented as follows:

  ```python
  if type(state) == Non-Terminal:
    R(state) = -0.01
  else:
    R(state) = 1.0
  ```

- At the starting of each iteration `i` of a game-play `G`, the agent starts with an empty policy or path (`pi = []`) having cost zero (`cost_pi = 0`) from the 'Start' state (`state = (1, 1)`).
- I have assumed that the environment of the agent world is fully-observable to the agent, that is, specifically, the percept received by the agent from that environment comprises of its current state and the current reward available in that state, deterministically: `percept = (state, reward)`. And, I have also assumed that the agent can also recognise if its current state is the terminal state.
- I chose to design the agent as an active reinforcement learner which uses a Temporal Difference (TD) method namely Q-Learning to learn an action-utility representation `Q`, where `Q(state, action)` represents the value of doing `action` in `state` [Book](http://aima.cs.berkeley.edu/), which forms the knowledge base of the Q-Learning-Agent which when receives a `percept` updates its knowledge base `Q` with the following algorithm and determines its next `action`; adding `action` to its policy `pi` and subtracting `reward` from `cost_pi` because the agent gets the `reward` for being in the `state` but `cost_pi` accounts for the cost paid by the agent until now for being in the `state`. If `action = None`, then it sets the persistent variables `s`, `a` and `r` of the following algorithm to `null`, and adds tuple `(pi, cost_pi, Q)` to `pi_G` where `Q` is the copy of the persistent variable `Q` of the following algorithm and `pi_G` is the collection of details of all the policies followed in all iterations. Note that my implementation is using 0.9 as the default value of the future-discount factor, gamma. And also note that, I have modified the algorithm given in the book to increase the randomness of the curiosity of the agent by using `random_argmax` instead of `argmax` which will randomly choose one of the candidates given by `argmax`.

  ```python
  # Approximation Function
  def f(u: float, n: int) -> Float:
    return (gloabl R_plus) if n < (global N_e) else u
    
  # Learning Factor
  def alpha(n : int) -> float:
    assert n >= 1
    return 60 / (59 + n)
  
  # Refer: ![Book](http://aima.cs.berkeley.edu/)
  def Q-Learning-Agent:
    # Here: s_dash is the current state and r_dash is the current reward
    Input: percept = (s_dash, r_dash)

    # A table of action values indexed by state and action, initially zero
    # A table of frequencies for state–action pairs, initially zero
    # The previous state, action, and reward, initially null
    Persistent: Q, N_sa, s, a, r

    If Terminal(s):
      Q[s, None] = r_dash

    If s is not null:
      Increment N_sa
      Q[s, a] += alpha(N_sa) * (r + (gamma * max(Q[s_dash, a_dash] for a_dash in ACTIONS)) - Q[s, a])

    s = s_dash
    a = random_argmax(f(Q[s_dash, a_dash], N_sa[s_dash, a_dash]) for a_dash in ACTIONS)
    r = r_dash

    # Next action
    return a
  ```

- After all the iterations of the `G`, I sort `pi_G` on the basis of the `pi_G[2]` i.e. `cost_pi` in increasing order, hence, `pi_G[1][1]` will be the most optimal policy with cost `pi_G[1][2]` and the agent's knowledge base will be `pi_G[1][3]`.

## How to play the game?

- First, note that the game implementation depends on Python, pygame and [pygame-text](https://github.com/cosmologicon/pygame-text).
- Now, if all the dependencies are satisfies, run the [`main.py`](https://github.com/ChakshuGupta13/reinforcement-learning-game/blob/main/main.py) with the help of Python: `py main.py` in Windows or `python3 main.py` in Linux.
- You will be asked to press `Enter` to the start the game, so, press `Enter` and observe the learning of the agent.
- Game will automatically exit after its last iteration but if you want to quit in between press `Esc`.
