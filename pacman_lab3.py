"""
Лабораторна робота №3. Евристичний пошук та інтелектуальні агенти.
Власне Pac-Man-подібне лабіринтне середовище (без ресурсів комерційної гри).

Запуск:
  python pacman_lab3.py              гра, етап 4 (останній крок)
  python pacman_lab3.py 3            гра, етап 3 (останній крок)
  python pacman_lab3.py 4.2          гра, етап 4, крок ускладнення 2
  python pacman_lab3.py bench [N]    серія експериментів (N епізодів, типово 50) -> results.csv
  python pacman_lab3.py demo 4.2 [astar|bfs] [seed] [greedy|runner] [--plain] [--frames a-b]
                                     відтворювана демонстрація з автоматичним гравцем
"""
import os
import sys
import time
import heapq
import random
import csv
import select
import statistics
import itertools
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
MAX_STEPS_BENCH = 250
CORNERED_LIMIT = 12          # «загнаний у обмежену область»: власна територія гравця <= 12 клітинок

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Кількість кроків ускладнення всередині кожного етапу
STEPS_IN_STAGE = {1: 3, 2: 3, 3: 2, 4: 3}


def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# ---------------------------------------------------------------- введення з клавіатури
class KeyReader:
    """Неблокуюче читання клавіш: Linux/macOS (termios) та Windows (msvcrt)."""

    def __enter__(self):
        self.interactive = False
        self.windows = os.name == 'nt'
        if self.windows:
            try:
                import msvcrt
                self.msvcrt = msvcrt
                os.system('')            # вмикає ANSI-кольори в консолі Windows
                self.interactive = True
            except ImportError:
                pass
        else:
            try:
                import termios
                import tty
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                self.interactive = True
            except Exception:
                pass
        return self

    def __exit__(self, type, value, traceback):
        if self.interactive and not self.windows:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        if not self.interactive:
            return None
        if self.windows:
            if not self.msvcrt.kbhit():
                return None
            ch = self.msvcrt.getwch()
            if ch in ('\x00', '\xe0'):   # стрілки
                return {'H': 'w', 'P': 's', 'M': 'd', 'K': 'a'}.get(self.msvcrt.getwch())
            return ch.lower()
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if not dr:
            return None
        ch = sys.stdin.read(1)
        if ch == '\x1b':                 # стрілки: ESC [ A/B/C/D
            dr, _, _ = select.select([sys.stdin], [], [], 0.01)
            if not dr:
                return None
            sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return {'A': 'w', 'B': 's', 'C': 'd', 'D': 'a'}.get(ch3)
        return ch.lower()


# ---------------------------------------------------------------- лабіринт
class Maze:
    def __init__(self, layout):
        self.rows = len(layout)
        self.cols = len(layout[0])
        self.grid = [[1 if ch == '#' else 0 for ch in row] for row in layout]
        self.open_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                           if self.grid[r][c] == 0]
        self.nb = {}
        for (r, c) in self.open_cells:
            self.nb[(r, c)] = [(r + dr, c + dc) for dr, dc in DIRECTIONS
                               if not self.is_wall(r + dr, c + dc)]
        # довжини найкоротших шляхів між усіма парами прохідних клітинок
        self.dist = {p: self._bfs_all(p) for p in self.open_cells}

    def _bfs_all(self, s):
        d = {s: 0}
        q = deque([s])
        while q:
            cur = q.popleft()
            for n in self.nb[cur]:
                if n not in d:
                    d[n] = d[cur] + 1
                    q.append(n)
        return d

    def is_wall(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] == 1
        return True

    def get_neighbors(self, pos):
        return self.nb.get(pos, [])

    def degree(self, pos):
        return len(self.nb.get(pos, []))

    def nearest_open(self, cell):
        """Найближча прохідна клітинка до заданої (для цілей, що потрапили у стіну)."""
        r, c = cell
        if not self.is_wall(r, c):
            return cell
        for rad in range(1, self.rows + self.cols):
            ring = []
            for dr in range(-rad, rad + 1):
                dc = rad - abs(dr)
                for ddc in {dc, -dc}:
                    rr, cc = r + dr, c + ddc
                    if not self.is_wall(rr, cc):
                        ring.append((rr, cc))
            if ring:
                return sorted(ring)[0]
        return PLAYER_START


_MAZE = None


def get_maze():
    global _MAZE
    if _MAZE is None:
        _MAZE = Maze(MAZE_LAYOUT)
    return _MAZE


# ---------------------------------------------------------------- пошук шляху
def _reconstruct(came_from, start, goal):
    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.reverse()
    return path


def bfs_search(maze, start, goal, shuffle=False):
    """BFS: найкоротший шлях (без початкової клітинки). Недосяжна ціль -> [start]."""
    if start == goal:
        return [start]
    queue = deque([start])
    came_from = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        nbs = list(maze.get_neighbors(current))
        if shuffle:
            random.shuffle(nbs)
        for nxt in nbs:
            if nxt not in came_from:
                came_from[nxt] = current
                queue.append(nxt)
    if goal not in came_from:
        return [start]
    return _reconstruct(came_from, start, goal)


def a_star_search(maze, start, goal, ghost_positions=None, avoid_weight=0, shuffle=False):
    """A* з манхеттенською евристикою та штрафом за близькість до інших переслідувачів."""
    if start == goal:
        return [start]
    frontier = [(0, 0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    use_penalty = bool(ghost_positions) and avoid_weight > 0
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in maze.get_neighbors(current):
            step_cost = 1
            if use_penalty:
                for g_pos in ghost_positions:
                    d = manhattan_dist(nxt, g_pos)
                    if d < 3:
                        step_cost += int(avoid_weight / (d + 1))
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan_dist(nxt, goal)
                tie = random.random() if shuffle else 0.0
                heapq.heappush(frontier, (priority, tie, nxt) if shuffle else (priority, nxt[0] * 100 + nxt[1] + 0.0, nxt))
                came_from[nxt] = current
    if goal not in came_from:
        return [start]
    return _reconstruct(came_from, start, goal)


# ---------------------------------------------------------------- середовище
class Environment:
    """
    stage: 1..4 — етап ускладнення; sub: крок ускладнення всередині етапу (типово останній).
      Етап 1: незалежні агенти.   1.1 випадкове блукання, 1.2 жадібне зменшення відстані, 1.3 пошук шляху
      Етап 2: ролі без обміну.    2.1 переслідувач+засідка, 2.2 +відсікач перехрестя, 2.3 +фланг через центр
      Етап 3: спільна інформація. 3.1 спільні цілі + «кожен обирає найближчу», 3.2 перекриття виходів з території гравця
      Етап 4: взаємне врахування. 4.1 оптимальний розподіл ролей + штраф за скупчення,
                                  4.2 +кліщі (протилежний фланг відносно переслідувача),
                                  4.3 +адаптація до стилю руху гравця
    """

    def __init__(self, stage=4, sub=None, use_astar=True, crowd_weight=6, shuffle=False):
        self.maze = get_maze()
        self.stage = stage
        self.sub = sub if sub else STEPS_IN_STAGE[stage]
        self.use_astar = use_astar
        self.crowd_weight = crowd_weight
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        self.player_pos = PLAYER_START
        self.player_dir = LEFT
        self.ghost_positions = list(GHOST_STARTS)
        self.ghost_prev = list(GHOST_STARTS)
        self.ghost_colors = ['\033[91m', '\033[94m', '\033[92m', '\033[95m']
        self.steps = 0
        self.caught = False
        self.trap_events = 0          # немає жодного безпечного ходу
        self.corridor_blocks = 0      # клітинку попереду зайняв переслідувач
        self.cornered_steps = 0       # власна територія гравця <= CORNERED_LIMIT клітинок
        self.dir_hist = deque(maxlen=10)
        self.last_plan = None
        for p in [self.player_pos] + self.ghost_positions:
            assert not self.maze.is_wall(*p), f"Початкова позиція {p} є стіною"

    # ---------- допоміжні обчислення (спільна інформація переслідувачів)
    def ahead(self, k):
        """Найдальша прохідна клітинка по прямій попереду гравця, не далі k кроків."""
        r, c = self.player_pos
        dr, dc = self.player_dir
        best = self.player_pos
        for j in range(1, k + 1):
            nr, nc = r + dr * j, c + dc * j
            if self.maze.is_wall(nr, nc):
                break
            best = (nr, nc)
        return best

    def junction_ahead(self):
        """Найближче перехрестя (>=3 виходи), до якого гравець дійде, не повертаючи назад."""
        P = self.player_pos
        back = (P[0] - self.player_dir[0], P[1] - self.player_dir[1])
        nbs = [n for n in self.maze.get_neighbors(P) if n != back] or self.maze.get_neighbors(P)
        seen = {P}
        q = deque()
        for n in nbs:
            seen.add(n)
            q.append(n)
        while q:
            cur = q.popleft()
            if self.maze.degree(cur) >= 3:
                return cur
            for n in self.maze.get_neighbors(cur):
                if n not in seen:
                    seen.add(n)
                    q.append(n)
        return P

    def mirror_target(self):
        r, c = self.player_pos
        return self.maze.nearest_open((self.maze.rows - 1 - r, self.maze.cols - 1 - c))

    def habit(self):
        """Частка кроків гравця, що продовжують попередній напрямок (стиль руху)."""
        h = list(self.dir_hist)
        if len(h) < 4:
            return 0.5
        same = sum(1 for i in range(1, len(h)) if h[i] == h[i - 1])
        return same / (len(h) - 1)

    def territory(self, player_pos=None):
        """Клітинки, до яких гравець дістається раніше за будь-якого переслідувача."""
        m = self.maze
        pdm = m.dist[player_pos or self.player_pos]
        gds = [m.dist[g] for g in self.ghost_positions]
        big = 10 ** 6
        return {c for c, d in pdm.items() if all(d < gd.get(c, big) for gd in gds)}

    def frontier(self):
        """Межа території гравця — «виходи», які треба перекрити; відсортовані за близькістю."""
        terr = self.territory()
        pdm = self.maze.dist[self.player_pos]
        fr = [c for c in terr if any(n not in terr for n in self.maze.get_neighbors(c))]
        fr.sort(key=lambda c: (pdm[c], c))
        return fr

    def pick_spread(self, cands, n, accepted=()):
        picked = []
        chosen = list(accepted)
        for c in cands:
            if len(picked) == n:
                break
            if all(manhattan_dist(c, a) >= 3 for a in chosen):
                picked.append(c)
                chosen.append(c)
        while len(picked) < n:
            picked.append(self.player_pos)
        return picked

    def dedupe(self, items):
        used = set()
        out = []
        for t, role in items:
            if t in used:
                seen = {t}
                q = deque([t])
                found = None
                while q and found is None:
                    cur = q.popleft()
                    for n in self.maze.get_neighbors(cur):
                        if n not in seen:
                            if n not in used:
                                found = n
                                break
                            seen.add(n)
                            q.append(n)
                if found:
                    t = found
            used.add(t)
            out.append((t, role))
        return out

    def _greedy_claim(self, ghost_ids, targets):
        """Кожен агент за чергою (за номером) забирає найближчу вільну ціль."""
        free = list(range(len(targets)))
        res = {}
        for gi in ghost_ids:
            g = self.ghost_positions[gi]
            best = min(free, key=lambda ti: self.maze.dist[g].get(targets[ti][0], 10 ** 6))
            res[gi] = targets[best]
            free.remove(best)
        return res

    def _optimal_assign(self, ghost_ids, targets):
        """Розподіл ролей з мінімальною сумарною відстанню (перебір перестановок)."""
        best, best_cost = None, None
        for perm in itertools.permutations(range(len(targets)), len(ghost_ids)):
            cost = sum(self.maze.dist[self.ghost_positions[gi]].get(targets[ti][0], 10 ** 6)
                       for gi, ti in zip(ghost_ids, perm))
            if best_cost is None or cost < best_cost:
                best, best_cost = perm, cost
        return {gi: targets[ti] for gi, ti in zip(ghost_ids, best)}

    # ---------- вибір цілей для переслідувачів на поточному етапі
    def get_plan(self):
        """Повертає для кожного з 4 переслідувачів: ціль, роль і режим (path/random/greedy)."""
        s, k, P = self.stage, self.sub, self.player_pos
        mk = lambda t, role, mode='path': {'target': t, 'role': role, 'mode': mode}

        if s == 1:
            mode = {1: 'random', 2: 'greedy', 3: 'path'}[k]
            role = {'random': 'блукання', 'greedy': 'жадібний', 'path': 'переслідувач'}[mode]
            return [mk(P, role, mode) for _ in range(4)]

        if s == 2:
            items = [(P, 'переслідувач'), (self.ahead(3), 'засідка'), (P, 'переслідувач'), (P, 'переслідувач')]
            if k >= 2:
                items[2] = (self.junction_ahead(), 'відсікач')
            if k >= 3:
                items[3] = (self.mirror_target(), 'фланг')
            return [mk(t, r) for t, r in items]

        m = self.maze
        chaser = min(range(4), key=lambda i: (m.dist[self.ghost_positions[i]].get(P, 10 ** 6), i))
        others = [i for i in range(4) if i != chaser]

        if s == 3 and k == 1:
            items = self.dedupe([(P, 'переслідувач'), (self.ahead(3), 'засідка'),
                                 (self.junction_ahead(), 'відсікач'), (self.mirror_target(), 'фланг')])
            res = self._greedy_claim(range(4), items)
            return [mk(*res[i]) for i in range(4)]

        fr = self.frontier() or [P]
        if s == 3:                                   # крок 3.2
            seals = self.pick_spread(fr, 3)
            res = self._greedy_claim(others, [(t, 'перекривач') for t in seals])
        else:                                        # етап 4
            if k >= 2:
                cf = m.dist[self.ghost_positions[chaser]]
                near = [c for c in fr if m.dist[P][c] <= 8] or fr
                pincer = max(near, key=lambda c: (cf.get(c, 0), -m.dist[P][c]))
                role0 = 'кліщі'
                if k >= 3 and self.habit() >= 0.7:   # гравець «біжить прямо» -> засідка попереду
                    pincer = self.ahead(2 + int(4 * self.habit()))
                    role0 = 'засідка'
                rest = self.pick_spread(fr, 2, accepted=[pincer])
                targets = [(pincer, role0)] + [(t, 'перекривач') for t in rest]
            else:
                targets = [(t, 'перекривач') for t in self.pick_spread(fr, 3)]
            res = self._optimal_assign(others, targets)
        plan = [None] * 4
        plan[chaser] = mk(P, 'переслідувач')
        for gi in others:
            plan[gi] = mk(*res[gi])
        return plan

    # ---------- керований агент
    def record_dir(self, d):
        self.dir_hist.append(d)

    def apply_human_input(self, key):
        key_map = {'w': UP, 's': DOWN, 'a': LEFT, 'd': RIGHT}
        if key in key_map:
            dr, dc = key_map[key]
            nr, nc = self.player_pos[0] + dr, self.player_pos[1] + dc
            if not self.maze.is_wall(nr, nc):
                self.player_pos = (nr, nc)
                self.player_dir = (dr, dc)
                self.record_dir((dr, dc))

    def simulate_player(self, model='greedy'):
        """Автоматичний гравець. greedy — максимізує мінімальну манхеттенську відстань (10% шуму);
        runner — враховує реальні відстані в лабіринті та розмір власної території (5% шуму)."""
        m = self.maze
        valid = m.get_neighbors(self.player_pos)
        if not valid:
            return
        P = self.player_pos
        front = (P[0] + self.player_dir[0], P[1] + self.player_dir[1])
        if front in self.ghost_positions:
            self.corridor_blocks += 1
        if not any(all(manhattan_dist(mv, g) > 1 for g in self.ghost_positions) for mv in valid):
            self.trap_events += 1
        if len(self.territory()) <= CORNERED_LIMIT:
            self.cornered_steps += 1

        if model == 'greedy':
            if random.random() < 0.10:
                best = random.choice(valid)
            else:
                best = max(valid, key=lambda mv: min(manhattan_dist(mv, g) for g in self.ghost_positions))
        else:
            if random.random() < 0.05:
                best = random.choice(valid)
            else:
                def score(mv):
                    dmin = min(m.dist[g].get(mv, 10 ** 6) for g in self.ghost_positions)
                    return min(dmin, 6) * 5 + len(self.territory(mv)) / 8.0
                best = max(valid, key=score)
        d = (best[0] - P[0], best[1] - P[1])
        self.player_dir = d
        self.player_pos = best
        self.record_dir(d)

    # ---------- функція переходу
    def step(self):
        self.steps += 1
        if self.player_pos in self.ghost_positions:
            self.caught = True
            return
        plan = self.get_plan()
        self.last_plan = plan
        new_pos = []
        for i in range(4):
            g = self.ghost_positions[i]
            item = plan[i]
            nbs = self.maze.get_neighbors(g)
            if item['mode'] == 'random':
                opts = [n for n in nbs if n != self.ghost_prev[i]] or nbs
                nxt = random.choice(opts)
            elif item['mode'] == 'greedy':
                nxt = min(nbs, key=lambda n: manhattan_dist(n, self.player_pos))
            else:
                tgt = item['target']
                if self.use_astar:
                    if self.stage == 4:
                        others = [self.ghost_positions[j] for j in range(4) if j != i]
                        path = a_star_search(self.maze, g, tgt, others, self.crowd_weight, self.shuffle)
                    else:
                        path = a_star_search(self.maze, g, tgt, shuffle=self.shuffle)
                else:
                    path = bfs_search(self.maze, g, tgt, self.shuffle)
                nxt = path[0] if path else g
            new_pos.append(nxt)
        self.ghost_prev = list(self.ghost_positions)
        self.ghost_positions = new_pos
        if self.player_pos in self.ghost_positions:
            self.caught = True

    # ---------- відображення
    def frame(self, plain=False, show_targets=True):
        RESET, YELLOW = '\033[0m', '\033[93m'
        tmark = {}
        if show_targets and self.last_plan:
            for i, it in enumerate(self.last_plan):
                tmark.setdefault(it['target'], chr(ord('a') + i))
        lines = []
        for r in range(self.maze.rows):
            row = []
            for c in range(self.maze.cols):
                pos = (r, c)
                if pos == self.player_pos:
                    row.append('P' if plain else f"{YELLOW}P{RESET}")
                elif pos in self.ghost_positions:
                    gi = self.ghost_positions.index(pos)
                    row.append(str(gi + 1) if plain else f"{self.ghost_colors[gi]}G{RESET}")
                elif self.maze.is_wall(r, c):
                    row.append('#')
                elif pos in tmark:
                    row.append(tmark[pos])
                else:
                    row.append(' ')
            lines.append("".join(row))
        return "\n".join(lines)

    def legend(self):
        if not self.last_plan:
            return ""
        parts = []
        for i, it in enumerate(self.last_plan):
            parts.append(f"{i + 1}:{it['role']}→{chr(ord('a') + i)}{it['target']}")
        return "  ".join(parts)

    def label(self):
        algo = "A*" if self.use_astar else "BFS"
        return f"Етап {self.stage}.{self.sub} | {algo} | крок {self.steps}"

    def render(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"=== {self.label()} ===")
        print("Керування: [W/A/S/D] або стрілки. Вихід: [Q]")
        print(self.frame())
        print(self.legend())


def parse_stage(text):
    if '.' in text:
        a, b = text.split('.', 1)
        return int(a), int(b)
    return int(text), None


# ---------------------------------------------------------------- гра з людиною
def play_game(stage=4, sub=None, use_astar=True):
    env = Environment(stage=stage, sub=sub, use_astar=use_astar)
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
                time.sleep(0.01)
            env.step()
        env.render()
        if env.caught:
            print(f"\n[КІНЕЦЬ ГРИ] Вас спіймали за {env.steps} кроків (етап {env.stage}.{env.sub})!")
        else:
            print("\n[ПЕРЕМОГА] Ви протрималися 500 кроків!")


# ---------------------------------------------------------------- відтворювана демонстрація
def run_demo(args):
    plain = '--plain' in args
    frames = None
    if '--frames' in args:
        i = args.index('--frames')
        a, b = args[i + 1].split('-')
        frames = (int(a), int(b))
        del args[i:i + 2]
    args = [a for a in args if a != '--plain']
    stage, sub = parse_stage(args[0]) if args else (4, None)
    use_astar = not (len(args) > 1 and args[1].lower() == 'bfs')
    seed = int(args[2]) if len(args) > 2 else 1
    model = args[3] if len(args) > 3 else 'runner'
    random.seed(seed)
    env = Environment(stage=stage, sub=sub, use_astar=use_astar)
    while not env.caught and env.steps < MAX_STEPS_BENCH:
        env.simulate_player(model)
        env.step()
        show = frames is None or frames[0] <= env.steps <= frames[1]
        if show:
            if not plain:
                os.system('clear' if os.name == 'posix' else 'cls')
            print(f"=== {env.label()} | seed={seed} | гравець: {model} ===")
            print(env.frame(plain=plain))
            print(env.legend())
            if not plain:
                time.sleep(0.35)
            else:
                print()
    print(f"[{'СПІЙМАНО' if env.caught else 'УТРИМАВСЯ'}] крок {env.steps}")


# ---------------------------------------------------------------- бенчмарк
def run_episode_set(task):
    stage, sub = task['stage'], task['sub']
    random.seed(BENCH_SEED)
    caught_count, steps_list = 0, []
    traps = blocks = cornered = 0
    for _ in range(task['episodes']):
        env = Environment(stage=stage, sub=sub, use_astar=task['astar'],
                          crowd_weight=task['weight'], shuffle=task['shuffle'])
        while not env.caught and env.steps < MAX_STEPS_BENCH:
            env.simulate_player(task['player'])
            env.step()
        if env.caught:
            caught_count += 1
        steps_list.append(env.steps if env.caught else MAX_STEPS_BENCH)
        traps += env.trap_events
        blocks += env.corridor_blocks
        cornered += env.cornered_steps
    n = task['episodes']
    total = sum(steps_list)
    std = statistics.pstdev(steps_list)
    return {
        "player": task['player'], "stage": f"{stage}.{sub}", "algorithm": task['algo_name'],
        "variant": task['variant'], "episodes": n,
        "catch_rate": f"{caught_count / n * 100:.1f}%",
        "avg_steps": f"{statistics.mean(steps_list):.2f}",
        "std_steps": f"{std:.2f}", "sem_steps": f"{std / n ** 0.5:.2f}",
        "avg_trap_events": f"{traps / n:.2f}", "avg_blocks": f"{blocks / n:.2f}",
        "traps_per_100_steps": f"{traps / total * 100:.2f}",
        "blocks_per_100_steps": f"{blocks / total * 100:.2f}",
        "cornered_per_100_steps": f"{cornered / total * 100:.2f}",
    }


def build_tasks(episodes):
    tasks = []

    def add(stage, sub, astar, player, variant='base', weight=6, shuffle=False, name=None):
        tasks.append(dict(stage=stage, sub=sub, astar=astar, player=player, variant=variant,
                          weight=weight, shuffle=shuffle, episodes=episodes,
                          algo_name=name or ("A*" if astar else "BFS")))

    for player in ('greedy', 'runner'):
        for stage in (1, 2, 3, 4):
            for sub in range(1, STEPS_IN_STAGE[stage] + 1):
                if stage == 1 and sub < 3:
                    add(stage, sub, True, player, name='—')       # пошук шляху не використовується
                else:
                    add(stage, sub, False, player)
                    add(stage, sub, True, player)
        # абляція: A* на етапі 4 без штрафу за скупчення
        for sub in (1, 2, 3):
            add(4, sub, True, player, variant='no_penalty', weight=0)
        # перевірка гіпотези про вибір між рівноцінними маршрутами: випадковий порядок сусідів
        for (stage, sub) in ((1, 3), (2, 2), (3, 1), (4, 1), (4, 3)):
            add(stage, sub, False, player, variant='shuffle', shuffle=True)
            add(stage, sub, True, player, variant='shuffle', shuffle=True)
    return tasks


def run_benchmark(episodes=50):
    tasks = build_tasks(episodes)
    print(f"Запуск бенчмарку: {len(tasks)} конфігурацій по {episodes} епізодів (seed={BENCH_SEED}, ліміт {MAX_STEPS_BENCH} кроків)...")
    results = []
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        r = run_episode_set(t)
        results.append(r)
        print(f"[{i}/{len(tasks)}] {r['player']:6s} етап {r['stage']} {r['algorithm']:4s} {r['variant']:10s}: "
              f"спіймань {r['catch_rate']}, кроків {r['avg_steps']} (±{r['std_steps']}), "
              f"пастки/100 {r['traps_per_100_steps']}, блок./100 {r['blocks_per_100_steps']}, "
              f"у кутку/100 {r['cornered_per_100_steps']}  ({time.time() - t0:.0f} c)", flush=True)
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\n[OK] Результати збережено в results.csv.")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "bench":
        run_benchmark(int(argv[1]) if len(argv) > 1 else 50)
    elif argv and argv[0] == "demo":
        run_demo(argv[1:])
    else:
        st, sb = parse_stage(argv[0]) if argv and argv[0].replace('.', '').isdigit() else (4, None)
        play_game(stage=st, sub=sb, use_astar=True)