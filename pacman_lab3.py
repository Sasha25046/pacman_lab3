import os
import sys
import time
import heapq
import random
import csv
import select
import statistics
from collections import deque

MAZE_LAYOUT = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "######.##          ##.######",
    "######.## ###  ### ##.######",
    "#......   #      #   ......#",
    "######.## ######## ##.######",
    "######.##          ##.######",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#...##................##...#",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################"
]

PLAYER_START = (16, 13)
GHOST_STARTS = [(10, 11), (10, 12), (10, 14), (10, 15)]
BENCH_SEED = 42

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

class KeyReader:
    def __enter__(self):
        try:
            import termios
            import tty
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self.interactive = True
        except Exception:
            self.interactive = False
        return self

    def __exit__(self, type, value, traceback):
        if self.interactive:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        if not self.interactive:
            return None
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        if dr:
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # Стрілки
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                if ch3 == 'A': return 'w'
                if ch3 == 'B': return 's'
                if ch3 == 'C': return 'd'
                if ch3 == 'D': return 'a'
            return ch.lower()
        return None

class Maze:
    def __init__(self, layout):
        self.rows = len(layout)
        self.cols = len(layout[0])
        self.grid = [[1 if char == '#' else 0 for char in row] for row in layout]

    def is_wall(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] == 1
        return True

    def get_neighbors(self, pos):
        r, c = pos
        neighbors = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not self.is_wall(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

# Алгоритм 1: BFS (пошук у ширину)
def bfs_search(maze, start, goal):
    if start == goal:
        return [start]
    queue = deque([start])
    came_from = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for nxt in maze.get_neighbors(current):
            if nxt not in came_from:
                came_from[nxt] = current
                queue.append(nxt)

    if goal not in came_from:
        return [start]
    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path

# Алгоритм 2: A* (з підтримкою штрафу за скупчення на Етапі 4)
def a_star_search(maze, start, goal, ghost_positions=None, avoid_weight=0):
    if start == goal:
        return [start]

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for nxt in maze.get_neighbors(current):
            step_cost = 1
            if ghost_positions and avoid_weight > 0:
                for g_pos in ghost_positions:
                    d = manhattan_dist(nxt, g_pos)
                    if d < 3:
                        step_cost += int(avoid_weight / (d + 1))

            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan_dist(nxt, goal)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return [start]

    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path

class Environment:
    def __init__(self, stage=1, use_astar=True):
        self.maze = Maze(MAZE_LAYOUT)
        self.stage = stage
        self.use_astar = use_astar
        self.reset()

    def reset(self):
        self.player_pos = PLAYER_START
        self.player_dir = LEFT
        self.ghost_positions = list(GHOST_STARTS)
        self.ghost_colors = ['\033[91m', '\033[94m', '\033[92m', '\033[95m']
        self.steps = 0
        self.caught = False
        self.trap_events = 0
        self.corridor_blocks = 0
        # Перевірка коректності початкових позицій
        for p in [self.player_pos] + self.ghost_positions:
            assert not self.maze.is_wall(*p), f"Початкова позиція {p} є стіною"

    def apply_human_input(self, key):
        key_map = {'w': UP, 's': DOWN, 'a': LEFT, 'd': RIGHT}
        if key in key_map:
            dr, dc = key_map[key]
            nr, nc = self.player_pos[0] + dr, self.player_pos[1] + dc
            if not self.maze.is_wall(nr, nc):
                self.player_pos = (nr, nc)
                self.player_dir = (dr, dc)

    def simulate_heuristic_player(self):
        valid_moves = self.maze.get_neighbors(self.player_pos)
        if not valid_moves:
            return

        front_pos = (self.player_pos[0] + self.player_dir[0], self.player_pos[1] + self.player_dir[1])
        if front_pos in self.ghost_positions:
            self.corridor_blocks += 1

        safe_moves = [m for m in valid_moves if all(manhattan_dist(m, g) > 1 for g in self.ghost_positions)]
        if len(safe_moves) == 0 and len(valid_moves) > 0:
            self.trap_events += 1

        if random.random() < 0.10:
            best_move = random.choice(valid_moves)
        else:
            def safety_score(move):
                return min(manhattan_dist(move, g) for g in self.ghost_positions)
            best_move = max(valid_moves, key=safety_score)

        dr = best_move[0] - self.player_pos[0]
        dc = best_move[1] - self.player_pos[1]
        self.player_dir = (dr, dc)
        self.player_pos = best_move

    def get_ghost_targets(self):
        targets = []
        pr, pc = self.player_pos
        dr, dc = self.player_dir

        if self.stage == 1:
            targets = [self.player_pos] * 4

        elif self.stage == 2:
            targets.append(self.player_pos)
            t1 = (pr + dr * 3, pc + dc * 3)
            targets.append(t1 if not self.maze.is_wall(*t1) else self.player_pos)
            t2 = (pr - dr * 2, pc - dc * 2)
            targets.append(t2 if not self.maze.is_wall(*t2) else self.player_pos)
            targets.append((1, 1) if self.steps % 20 < 10 else (20, 26))

        elif self.stage == 3:
            base_targets = [
                self.player_pos,
                (pr + dr * 2, pc + dc * 2),
                (pr - dr * 2, pc - dc * 2),
                (pr + dc * 2, pc + dr * 2)
            ]
            assigned = []
            for t in base_targets:
                valid_t = t if not self.maze.is_wall(*t) else self.player_pos
                if valid_t in assigned:
                    for n in self.maze.get_neighbors(valid_t):
                        if n not in assigned:
                            valid_t = n
                            break
                assigned.append(valid_t)
                targets.append(valid_t)

        elif self.stage == 4:
            dists = [(manhattan_dist(g, self.player_pos), idx) for idx, g in enumerate(self.ghost_positions)]
            dists.sort()
            roles = {}
            roles[dists[0][1]] = self.player_pos
            front = (pr + dr * 3, pc + dc * 3)
            roles[dists[1][1]] = front if not self.maze.is_wall(*front) else self.player_pos

            neighbors = self.maze.get_neighbors(self.player_pos)
            roles[dists[2][1]] = neighbors[0] if len(neighbors) > 0 else self.player_pos
            roles[dists[3][1]] = neighbors[-1] if len(neighbors) > 1 else self.player_pos
            targets = [roles[i] for i in range(4)]

        return targets

    def step(self):
        self.steps += 1

        # Гравець міг зайти в клітинку, де вже стоїть переслідувач
        if self.player_pos in self.ghost_positions:
            self.caught = True
            return

        targets = self.get_ghost_targets()
        new_ghost_pos = []

        for i in range(4):
            curr_g = self.ghost_positions[i]
            target = targets[i]

            if self.use_astar:
                if self.stage == 4:
                    other_ghosts = [self.ghost_positions[j] for j in range(4) if j != i]
                    path = a_star_search(self.maze, curr_g, target, other_ghosts, avoid_weight=6)
                else:
                    path = a_star_search(self.maze, curr_g, target)
            else:
                path = bfs_search(self.maze, curr_g, target)

            next_pos = path[0] if path else curr_g
            new_ghost_pos.append(next_pos)

        self.ghost_positions = new_ghost_pos

        if self.player_pos in self.ghost_positions:
            self.caught = True

    def render(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        algo = "A*" if self.use_astar else "BFS"
        print(f"=== Етап {self.stage}/4 | Алгоритм: {algo} | Крок: {self.steps} ===")
        print("Керування: [W/A/S/D] або стрілки. Вихід: [Q]")
        RESET = '\033[0m'
        YELLOW = '\033[93m'

        display = []
        for r in range(self.maze.rows):
            line = []
            for c in range(self.maze.cols):
                pos = (r, c)
                if pos == self.player_pos:
                    line.append(f"{YELLOW}P{RESET}")
                elif pos in self.ghost_positions:
                    g_idx = self.ghost_positions.index(pos)
                    line.append(f"{self.ghost_colors[g_idx]}G{RESET}")
                elif self.maze.is_wall(r, c):
                    line.append("#")
                else:
                    line.append(" ")
            display.append("".join(line))
        print("\n".join(display))

def play_game(stage=4, use_astar=True):
    env = Environment(stage=stage, use_astar=use_astar)
    with KeyReader() as key_reader:
        while not env.caught and env.steps < 500:
            env.render()
            start_time = time.time()
            while time.time() - start_time < 0.35:
                k = key_reader.get_key()
                if k == 'q':
                    print("\nГру перервано.")
                    return
                if k in ['w', 'a', 's', 'd']:
                    env.apply_human_input(k)
                    break
            env.step()

        env.render()
        if env.caught:
            print(f"\n[КІНЕЦЬ ГРИ] Вас спіймали за {env.steps} кроків на Етапі {stage}!")
        else:
            print(f"\n[ПЕРЕМОГА] Ви протрималися 500 кроків!")

def run_benchmark(episodes=50, max_steps=250):
    random.seed(BENCH_SEED) 
    print(f"Запуск наукового бенчмарку (по {episodes} епізодів для кожного етапу: BFS проти A*, seed={BENCH_SEED})...")
    results = []

    for stage in [1, 2, 3, 4]:
        for algo_name, use_astar in [("BFS", False), ("A*", True)]:
            caught_count = 0
            steps_list = []
            traps_total = 0
            corridor_blocks_total = 0

            for _ in range(episodes):
                env = Environment(stage=stage, use_astar=use_astar)
                while not env.caught and env.steps < max_steps:
                    env.simulate_heuristic_player()
                    env.step()
                if env.caught:
                    caught_count += 1
                    steps_list.append(env.steps)
                else:
                    steps_list.append(max_steps)
                traps_total += env.trap_events
                corridor_blocks_total += env.corridor_blocks

            catch_rate = (caught_count / episodes) * 100
            avg_steps = statistics.mean(steps_list)
            std_steps = statistics.pstdev(steps_list)
            total_steps = sum(steps_list)
            avg_traps = traps_total / episodes
            avg_blocks = corridor_blocks_total / episodes
            traps_100 = traps_total / total_steps * 100
            blocks_100 = corridor_blocks_total / total_steps * 100

            results.append({
                "stage": stage,
                "algorithm": algo_name,
                "catch_rate": f"{catch_rate:.1f}%",
                "avg_steps": f"{avg_steps:.2f}",
                "std_steps": f"{std_steps:.2f}",
                "avg_trap_events": f"{avg_traps:.2f}",
                "avg_blocks": f"{avg_blocks:.2f}",
                "traps_per_100_steps": f"{traps_100:.2f}",
                "blocks_per_100_steps": f"{blocks_100:.2f}"
            })
            print(f"Етап {stage} [{algo_name}]: Спіймань = {catch_rate:.1f}%, "
                  f"Сер. кроків = {avg_steps:.2f} (±{std_steps:.2f}), "
                  f"Пастки = {avg_traps:.2f}, Блокування = {avg_blocks:.2f}")

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\n[OK] Результати збережено в results.csv.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        run_benchmark()
    else:
        chosen_stage = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4
        play_game(stage=chosen_stage, use_astar=True)