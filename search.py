import time as timer
import heapq
import random
import numpy as np
import config

action_list = np.array([[0, 0],[-1, 0],[1, 0],[0, -1],[0, 1]], dtype=np.int8)

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1

            if child_loc[0] < 0 or child_loc[0] >= my_map.shape[0] or child_loc[1] < 0 or child_loc[1] >= my_map.shape[1]:
               continue

            if my_map[child_loc[0], child_loc[1]] == 1:
                continue

            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):

    table = {}
    for const in constraints:
        if const['agent'] == agent:

            if const['timestep'] not in table:
                table[const['timestep']] = {True:[], False:[]}

            if 'positive' in const:
                table[const['timestep']][const['positive']].append(const['loc'])
            else:
                table[const['timestep']][False].append(const['loc'])

            # goal location constrain, for Task 2.3
            if 'goal' in const:
                table[const['loc'][0]] = const['timestep']

    return table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):

    if not constraint_table:
        return False

    if next_time in constraint_table:

        # positive check
        if constraint_table[next_time][True]:
            vertex = [const for const in constraint_table[next_time][True] if len(const)==1]
            edge = [const for const in constraint_table[next_time][True] if len(const)==2]

            assert len(vertex) <= 1 and len(edge) <= 1, 'too many positive constraint'

            if len(vertex) > 0 and [next_loc] not in vertex:
                return True

            if len(edge) > 0 and [curr_loc, next_loc] not in edge:
                return True

        # negative check
        if constraint_table[next_time][False]:
            if [next_loc] in constraint_table[next_time][False] or [curr_loc, next_loc] in constraint_table[next_time][False]:
                return True

    # goal constraint, for Task 2.3
    elif next_loc in constraint_table:
        if next_time > constraint_table[next_loc]:
            return True

    return False

def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """


    table = build_constraint_table(constraints, agent)

    # max timestep in constraint table
    max_timestep = 0
    if len(table) > 0:
        max_timestep = max([ key for key in table.keys() if type(key) is int])


    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0

    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'timestep': earliest_goal_timestep, 'parent': None}
    push_node(open_list, root)
    closed_list[(root['loc'], 0)] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)
        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc and curr['timestep'] >= max_timestep:
            path = get_path(curr)
            return path

        # Task 2.4
        if curr['timestep'] >= config.max_steps:
            continue

        for dir in range(5):

            if dir == 4:
                # stay still
                child_loc = curr['loc']
            else:
                # move
                child_loc = move(curr['loc'], dir)

            if child_loc[0] < 0 or child_loc[0] >= my_map.shape[0] or child_loc[1] < 0 or child_loc[1] >= my_map.shape[1]:
                continue

            if my_map[child_loc[0], child_loc[1]] == 1:
                continue

            if is_constrained(curr['loc'], child_loc, curr['timestep']+1, table):
                # print(str(curr['loc'])+ ' ' +str(child_loc) + ' ' +str(curr['timestep']+1))
                continue

            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'timestep': curr['timestep'] + 1,
                    'parent': curr}

            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'], child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'], child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'], child['timestep'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions


def detect_collision(path1, path2):

    length = max(len(path1), len(path2))

    for i in range(length):
        loc1 = get_location(path1, i)
        loc2 = get_location(path2, i)

        # vertex collision
        if loc1 == loc2:
            return {'loc':[loc1], 'timestep': i}

        next_loc1 = get_location(path1, i+1)
        next_loc2 = get_location(path2, i+1)

        # edge collision
        if loc1==next_loc2 and loc2==next_loc1:
            return {'loc':[loc1, loc2], 'timestep': i+1}

    return None


def detect_collisions(paths):

    collisions = []
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):

            collision = detect_collision(paths[i], paths[j])
            if collision:
                collisions.append({'a1':i, 'a2':j, 'loc':collision['loc'], 'timestep':collision['timestep']})

    return collisions




def disjoint_splitting(collision):

    assert len(collision['loc']) == 1 or len(collision['loc']) == 2, 'number of collision location out of range'


    agent = random.choice([collision['a1'], collision['a2']])

    # if it is vertex collision
    if len(collision['loc']) == 1:
        return [{'agent': agent, 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive': True},
                {'agent': agent, 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive': False}]

    # if it is edge collision      
    else:

        if agent == collision['a1']:
            return [{'agent': agent, 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive': True},
                    {'agent': agent, 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive': False}]
        else:
            return [{'agent': agent, 'loc': collision['loc'][::-1], 'timestep': collision['timestep'], 'positive': True},
                    {'agent': agent, 'loc': collision['loc'][::-1], 'timestep': collision['timestep'], 'positive': False}]




class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))

        self.num_of_generated += 1

    def pop_node(self):
        _, _, _, node = heapq.heappop(self.open_list)

        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations
        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}

        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i, root['constraints'])

            if path is None:
                raise BaseException('No solutions')

            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)


        while self.open_list:
            P = self.pop_node()
            # for collision in P['collisions']:
            #     print(disjoint_splitting(collision))

            if not P['collisions']:
                return P['paths']

            collision = random.choice(P['collisions'])
            constraints = disjoint_splitting(collision)


            if timer.time() - self.start_time > 2:
                return None

            for constraint in constraints:
                Q = dict()

                Q['constraints'] = P['constraints'].copy()

                if constraint not in P['constraints']:
                    Q['constraints'].append(constraint)

                if 'positive' in constraint and constraint['positive']:
                    # add negative constraint for all other agent
                    for i in range(self.num_of_agents):
                        if i != constraint['agent']:
                            neg_const = constraint.copy()
                            neg_const['agent'] = i
                            neg_const['positive'] = False
                            # check if it is edge constrain
                            if len(neg_const['loc']) == 2:
                                neg_const['loc'] = neg_const['loc'][::-1]

                            if neg_const not in Q['constraints']:
                                Q['constraints'].append(neg_const)


                Q['paths'] = P['paths'].copy()

                a = constraint['agent']

                path = a_star(self.my_map, self.starts[a], self.goals[a], self.heuristics[a], a, Q['constraints'])

                if path:
                    Q['paths'][a] = path
                    Q['collisions'] = detect_collisions(Q['paths'])
                    Q['cost'] = get_sum_of_cost(Q['paths'])
                    self.push_node(Q)

        return root['paths']


def find_path(env):

    map = np.copy(env.map)

    temp_agents_pos = np.copy(env.agents_pos)
    agents_pos= []
    for pos in temp_agents_pos:
        agents_pos.append(tuple(pos))

    temp_goals_pos = np.copy(env.goals_pos)
    goals_pos = []
    for pos in temp_goals_pos:
        goals_pos.append(tuple(pos))

    solver = CBSSolver(map, agents_pos, goals_pos)
    paths = solver.find_solution()

    if paths is None:
        return None


    max_len = max([len(path) for path in paths])

    for path in paths:
        while len(path) < max_len:
            path.append(path[-1])

    actions = []

    for step in range(1, max_len):
        step_action = []

        for i in range(env.num_agents):
            direction = np.asarray(paths[i][step]) - np.asarray(paths[i][step-1])

            if np.array_equal(direction, action_list[0]):
                step_action.append(0)
            elif np.array_equal(direction, action_list[1]):
                step_action.append(1)
            elif np.array_equal(direction, action_list[2]):
                step_action.append(2)
            elif np.array_equal(direction, action_list[3]):
                step_action.append(3)
            elif np.array_equal(direction, action_list[4]):
                step_action.append(4)
            else:
                raise RuntimeError('no action match')

        if env.num_agents == 1:
            actions.append(step_action[0])
        else:
            actions.append(step_action)

    return actions 