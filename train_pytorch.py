#!/usr/bin/env python3
"""
五子棋监督学习训练 - PyTorch版 (内存优化)
用法: python3 train_pytorch.py [--epochs 50] [--lr 0.001] [--batch 64]
"""

import os, sys, json, re, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time

BOARD_SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, 2
INPUT_PLANES = 6

# ============ SGF 解析 ============
def parse_sgf_coord(s):
    if not s or len(s) < 2: return None
    c, r = ord(s[0]) - ord('a'), ord(s[1]) - ord('a')
    return (r, c) if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE else None

def parse_sgf_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except: return None
    winner = 0
    re_match = re.search(r'RE\[([^\]]*)\]', content)
    if re_match:
        r = re_match.group(1).lower()
        if '黑' in r or 'b+' in r or '1-0' in r: winner = BLACK
        elif '白' in r or 'w+' in r or '0-1' in r: winner = WHITE
    moves = []
    for m in re.finditer(r';([BW])\[([a-o]{0,2})\]', content):
        coord = parse_sgf_coord(m.group(2))
        if coord: moves.append(coord)
    return (moves, winner) if len(moves) >= 5 else None

def parse_all_sgf(sgf_dir):
    files = glob.glob(os.path.join(sgf_dir, '**', '*.sgf'), recursive=True)
    games = []
    for f in files:
        r = parse_sgf_file(f)
        if r: games.append(r)
    print(f"解析: {len(games)}/{len(files)} 个有效棋谱")
    return games

# ============ 特征提取（纯numpy，速度快） ============
def board_to_features_np(board, player):
    """返回 (6, 15, 15) numpy数组 — 向量化版本"""
    opp = BLACK if player == WHITE else WHITE
    planes = np.zeros((INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    planes[0] = (board == player).astype(np.float32)
    planes[1] = (board == opp).astype(np.float32)
    
    # 最后两步: 用numpy找最后的非零位置
    filled = np.argwhere(board != EMPTY)
    if len(filled) >= 1:
        r, c = filled[-1]
        planes[2, r, c] = 1.0
    if len(filled) >= 2:
        r, c = filled[-2]
        planes[3, r, c] = 1.0
    
    # 合法落子（空位）
    planes[5] = (board == EMPTY).astype(np.float32)
    
    # 威胁评估（简化：用numpy卷积近似）
    player_mask = planes[0].copy()
    padded = np.pad(player_mask, 2, mode='constant')
    threat = np.zeros_like(player_mask)
    for dr in range(5):
        for dc in range(5):
            threat += padded[dr:dr+BOARD_SIZE, dc:dc+BOARD_SIZE]
    threat /= 25.0
    planes[4] = threat * planes[5]
    
    return planes

def game_to_samples(moves, winner):
    """返回 (features_array, moves_array, values_array)"""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    features_list = []
    move_indices = []
    values = []
    current = BLACK
    
    for r, c in moves:
        if board[r, c] != EMPTY: break
        feat = board_to_features_np(board, current)
        features_list.append(feat)
        move_indices.append(r * BOARD_SIZE + c)
        if winner == 0: values.append(0.0)
        elif winner == current: values.append(1.0)
        else: values.append(-1.0)
        board[r, c] = current
        current = BLACK if current == WHITE else WHITE
    
    return features_list, move_indices, values

# ============ 数据集 ============
class GomokuDataset(Dataset):
    def __init__(self, features, moves, values):
        self.features = features  # (N, 6, 15, 15) numpy
        self.moves = moves        # (N,) numpy
        self.values = values      # (N,) numpy
    
    def __len__(self): return len(self.moves)
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]),
                torch.tensor(self.moves[idx], dtype=torch.long),
                torch.tensor(self.values[idx], dtype=torch.float32))

# ============ 神经网络 ============
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))

class GomokuNet(nn.Module):
    def __init__(self, filters=64, res_blocks=4):
        super().__init__()
        self.filters = filters
        self.res_blocks = res_blocks
        self.conv1 = nn.Conv2d(INPUT_PLANES, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.res = nn.Sequential(*[ResBlock(filters) for _ in range(res_blocks)])
        self.policy_conv = nn.Conv2d(filters, filters, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(filters)
        self.policy_fc = nn.Linear(filters * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        self.value_conv = nn.Conv2d(filters, filters, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(filters)
        self.value_fc1 = nn.Linear(filters * BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res(x)
        p = F.relu(self.policy_bn(self.policy_conv(x))).view(x.size(0), -1)
        p = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x))).view(x.size(0), -1)
        v = torch.tanh(self.value_fc2(F.relu(self.value_fc1(v))))
        return p, v.squeeze(-1)

# ============ 权重导出 ============
def export_weights_to_json(model, path):
    s = model.state_dict()
    d = {}
    d['conv1_w'] = s['conv1.weight'].flatten().tolist()
    d['conv1_b'] = s['bn1.bias'].tolist()
    for i in range(model.res_blocks):
        p = f'res.{i}'
        d[f'res{i}_conv1_w'] = s[f'{p}.conv1.weight'].flatten().tolist()
        d[f'res{i}_conv1_b'] = s[f'{p}.bn1.bias'].tolist()
        d[f'res{i}_conv2_w'] = s[f'{p}.conv2.weight'].flatten().tolist()
        d[f'res{i}_conv2_b'] = s[f'{p}.bn2.bias'].tolist()
    d['policy_conv_w'] = s['policy_conv.weight'].flatten().tolist()
    d['policy_conv_b'] = s['policy_bn.bias'].tolist()
    d['policy_fc_w'] = s['policy_fc.weight'].flatten().tolist()
    d['policy_fc_b'] = s['policy_fc.bias'].tolist()
    d['value_conv_w'] = s['value_conv.weight'].flatten().tolist()
    d['value_conv_b'] = s['value_bn.bias'].tolist()
    d['value_fc1_w'] = s['value_fc1.weight'].flatten().tolist()
    d['value_fc1_b'] = s['value_fc1.bias'].tolist()
    d['value_fc2_w'] = s['value_fc2.weight'].flatten().tolist()
    d['value_fc2_b'] = s['value_fc2.bias'].tolist()
    with open(path, 'w') as f: json.dump(d, f)
    print(f"  → JSON权重: {path} ({os.path.getsize(path)/1024/1024:.1f}MB)")

# ============ 训练 ============
def train(args):
    device = torch.device('cpu')
    cache_path = args.cache_path
    
    # 1. 加载/解析数据
    if os.path.exists(cache_path):
        print(f"加载缓存: {cache_path}")
        data = np.load(cache_path)
        features, moves, values = data['features'], data['moves'], data['values']
    else:
        print(f"解析SGF: {args.sgf_dir}")
        games = parse_all_sgf(args.sgf_dir)
        all_f, all_m, all_v = [], [], []
        for i, (mv, w) in enumerate(games):
            f, m, v = game_to_samples(mv, w)
            all_f.extend(f); all_m.extend(m); all_v.extend(v)
            if (i+1) % 1000 == 0:
                print(f"  {i+1}/{len(games)} 局, {len(all_m)} 样本")
        features = np.array(all_f, dtype=np.float32)
        moves = np.array(all_m, dtype=np.int64)
        values = np.array(all_v, dtype=np.float32)
        np.savez_compressed(cache_path, features=features, moves=moves, values=values)
        print(f"缓存: {cache_path} ({os.path.getsize(cache_path)/1024/1024:.0f}MB)")
    
    print(f"样本数: {len(moves)}")
    
    # 限制样本数
    if args.max_samples > 0 and len(moves) > args.max_samples:
        idx = np.random.choice(len(moves), args.max_samples, replace=False)
        features, moves, values = features[idx], moves[idx], values[idx]
        print(f"限制为: {len(moves)} 样本")
    
    # 2. 数据集
    dataset = GomokuDataset(features, moves, values)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    
    # 3. 模型
    model = GomokuNet(64, 4).to(device)
    if args.resume and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        print(f"加载模型: {args.model_path}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    p_crit = nn.CrossEntropyLoss()
    v_crit = nn.MSELoss()
    
    print(f"\n训练: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_p, t_v, t_l, correct, total = 0, 0, 0, 0, 0
        t0 = time.time()
        
        for x, mv, val in loader:
            x = x.to(device)
            optimizer.zero_grad()
            pv, vv = model(x)
            p_loss = p_crit(pv, mv.to(device))
            v_loss = v_crit(vv, val.to(device))
            loss = p_loss + v_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_p += p_loss.item(); t_v += v_loss.item(); t_l += loss.item()
            correct += pv.argmax(1).eq(mv.to(device)).sum().item()
            total += mv.size(0)
        
        scheduler.step()
        nb = len(loader)
        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss:{t_l/nb:.4f} (P:{t_p/nb:.4f} V:{t_v/nb:.4f}) | Acc:{100*correct/total:.1f}% | {dt:.0f}s")
        
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.model_path)
            export_weights_to_json(model, args.json_path)
    
    print("\n✅ 训练完成!")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--sgf_dir', type=str, default='sgf-data/sgf')
    p.add_argument('--model_path', type=str, default='model_pytorch.pt')
    p.add_argument('--json_path', type=str, default='model-weights.json')
    p.add_argument('--cache_path', type=str, default='training_cache.npz')
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max_samples', type=int, default=0, help='最大样本数(0=全部)')
    args = p.parse_args()
    d = os.path.dirname(os.path.abspath(__file__))
    for a in ['sgf_dir','model_path','json_path','cache_path']:
        v = getattr(args, a)
        if not os.path.isabs(v): setattr(args, a, os.path.join(d, v))
    train(args)
