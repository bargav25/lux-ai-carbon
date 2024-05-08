import os
import numpy as np
import torch
from lux.game import Game
#import segmentation_models_pytorch as smp

path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'

model = torch.jit.load(f'{path}/model.pth')

model.eval()

# Input for Neural Network
# Feature map size [14,32,32] and global features size [4,4,4]
def make_input(obs):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    global_features = np.zeros((14,4,4))

    b = np.zeros((14, 32, 32), dtype=np.float32)

    friendly_unit_cnt = 0
    opponent_unit_cnt = 0
    friendly_ctile_cnt = 0
    opponent_ctile_cnt = 0
    total_wood = 0
    total_coal = 0
    total_uranium = 0

    can_mine_coal = 0
    can_mine_uranium = 0

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':

            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])

            # Units
            team = int(strs[2])

            if (team - obs['player']) % 2 == 0:
                friendly_unit_cnt += 1
            else:
                opponent_unit_cnt += 1

            cooldown = float(strs[6])
            idx = (team - obs['player']) % 2 * 3
            b[idx:idx + 3, x, y] = (
                1,
                cooldown / 6,
                (wood + coal + uranium) / 100
            )
        elif input_identifier == 'ct':
            # CityTiles

            team = int(strs[1])

            if (team - obs['player']) % 2 == 0:
                friendly_ctile_cnt += 1
            else:
                opponent_ctile_cnt += 1

            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 6 + (team - obs['player']) % 2 * 2
            b[idx:idx + 2, x, y] = (
                1,
                cities[city_id]
            )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 10, 'coal': 11, 'uranium': 12}[r_type], x, y] = amt / 800
            if r_type == 'wood': total_wood += amt
            elif r_type == 'coal': total_coal += amt
            elif r_type == 'uranium': total_uranium += amt

        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            if team - obs['player'] % 2 == 0:
                if rp >= 50:
                    can_mine_coal = 1
                if rp >= 200:
                    can_mine_uranium = 1

            global_features[(team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10

    # Day/Night Cycle
    global_features[2, :] = obs['step'] % 40 / 40
    # Turns
    global_features[3, :] = obs['step'] / 360
    # Number of friendly unit
    global_features[4, :] = friendly_unit_cnt / 50
    # Number of opponent unit
    global_features[5, :] = opponent_unit_cnt / 50
    # Number of friendly ctiles
    global_features[6, :] = friendly_ctile_cnt / 50
    # Number of opponent unit
    global_features[7, :] = opponent_ctile_cnt / 50
    # Total Wood
    global_features[8, :] = total_wood / 24000
    # Total Coal
    global_features[9, :] = total_coal / 24000
    # Total Uranium
    global_features[10, :] = total_uranium / 12000
    global_features[11, :] = can_mine_coal
    global_features[12, :] = can_mine_uranium
    # Map Size
    global_features[13, :] = width

    # Map Size
    b[13, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b, global_features

game_state = None
def get_game_state(observation):
    global game_state

    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation["player"]
    else:
        game_state._update(observation["updates"])
    return game_state

def get_shift(observation):
    width, height = observation['width'], observation['height']
    shift = (32 - width) // 2
    return shift

def in_city(pos):
    try:
        city = game_state.map.get_cell_by_pos(pos).citytile
        return city is not None and city.team == game_state.id
    except:
        return False


def call_func(obj, method, args=[]):
    return getattr(obj, method)(*args)


unit_actions = [('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'), ('build_city',)]
def get_action(policy, unit, dest, shift):
    action = unit_actions[ np.argmax( policy[:, unit.pos.x + shift, unit.pos.y + shift] )]
    pos = unit.pos.translate(action[-1], 1) or unit.pos
    if pos not in dest or in_city(pos):
        return call_func(unit, *action), pos

    return unit.move('c'), unit.pos


def agent(observation, configuration):
    global game_state
    game_state = get_game_state(observation)
    shift = get_shift(observation)
    player = game_state.players[observation.player]
    actions = []

    # City Actions
    unit_count = len(player.units)
    for city in player.cities.values():
        for city_tile in city.citytiles:
            if city_tile.can_act():
                if unit_count < player.city_tile_count:
                    actions.append(city_tile.build_worker())
                    unit_count += 1
                elif not player.researched_uranium():
                    actions.append(city_tile.research())
                    player.research_points += 1

    # Worker Actions
    state_1, state_2 = make_input(observation)
    dest = []
    with torch.no_grad():
        p = model(torch.from_numpy(state_1).unsqueeze(0).float(), torch.from_numpy(state_2).unsqueeze(0).float())
        policy = p.squeeze(0).numpy()
    for unit in player.units:
        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos)):
            action, pos = get_action(policy, unit, dest, shift)
            actions.append(action)
            dest.append(pos)

    return actions