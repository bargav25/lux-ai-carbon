import numpy as np
import json
from pathlib import Path
import os
import random
from tqdm.notebook import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
# import segmentation_models_pytorch as smp

from model import EnhancedDualInputCNN
from model import UNet

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 42
seed_everything(seed)

arr = np.zeros((20,32,32))
arr = arr.astype("bool")
print(arr.dtype)

def to_label(action, obs):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}[strs[2]]
    elif strs[0] == 'bcity':
        label = 4
    else:
        label = None

    unit_pos = (0,0)

    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    for update in obs["updates"]:
        strs = update.split(" ")
        if strs[0] == "u" and strs[3] == unit_id:
            unit_pos = (int(strs[4]) + x_shift, int(strs[5]) + y_shift)
    return unit_id, label, unit_pos

def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True

def create_dataset_from_json(episode_dir):
    obses = {}
    samples = []
    append = samples.append

    episodes = [path for path in Path(episode_dir).rglob('*.json') if 'info' not in path.name]
    for filepath in tqdm(episodes):
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])

        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i+1][index]['action']
                obs = json_load['steps'][i][0]['observation']

                if depleted_resources(obs):
                    break

                obs['player'] = index
                obs = dict([
                    (k,v) for k,v in obs.items()
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])

                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs

                action_map = np.zeros((5,32,32))
                mask = np.zeros((5,32,32))

                for action in actions:
                    unit_id, label, unit_pos = to_label(action, obs)
                    if label is not None:
                        action_map[label, unit_pos[0], unit_pos[1]] = 1
                        mask[:, unit_pos[0], unit_pos[1]] = 1

                mask = mask.astype('bool')
                action_map = action_map.astype('bool')
                #if len(samples) < 210_000:
                append((obs_id, action_map,mask))

    return obses, samples

episode_dir = 'top_agents'
obses, samples = create_dataset_from_json(episode_dir)
obses = obses
print('obses:', len(obses), 'samples:', len(samples))

obs = obses[samples[0][0]]
width, height = obs['width'], obs['height']
x_shift = (32 - width) // 2
y_shift = (32 - height) // 2
print(samples[0][1][:, 1+x_shift, 6+y_shift])

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


class LuxDataset(Dataset):
    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, action_map, mask = self.samples[idx]
        obs = self.obses[obs_id]
        state_1, state_2 = make_input(obs)

        return state_1, state_2, action_map, mask
    
def train_model(model, dataloaders_dict, optimizer, num_epochs):

    for epoch in range(num_epochs):
        model.cuda()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0

            dataloader = dataloaders_dict[phase]
            for item in tqdm(dataloader, leave=False):
                states_1 = item[0].cuda().float()
                states_2 = item[1].cuda().float()
                actions = item[2].cuda().float()
                mask = item[3].cuda().float()

                optimizer.zero_grad()
                criterion = nn.BCEWithLogitsLoss(weight=mask)

                with torch.set_grad_enabled(phase == 'train'):
                    policy = model(states_1, states_2)
                    loss = criterion(policy, actions)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * len(policy) * mask[mask==0].size()[0]/mask[mask==1].size()[0]

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f}')


model = UNet(14, 5, 14)

# model = EnhancedDualInputCNN()

train, val = train_test_split(samples, test_size = 0.1, random_state = 42)
batch_size = 256
train_loader = DataLoader(
    LuxDataset(obses, train),
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)
val_loader = DataLoader(
    LuxDataset(obses, val),
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)
dataloaders_dict = {"train": train_loader, "val": val_loader}

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
train_model(model, dataloaders_dict, optimizer, num_epochs=10)

traced = torch.jit.trace(model.cpu(), (torch.rand(1, 14, 32, 32), torch.rand(1,14,4,4)))
traced.save('model.pth')
