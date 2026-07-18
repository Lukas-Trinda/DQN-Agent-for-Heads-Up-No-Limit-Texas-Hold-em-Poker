import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
from treys import Card, Evaluator
import os
import warnings
from typing import List, Tuple, Dict, Optional
import pickle
import sys

warnings.filterwarnings("ignore", category=FutureWarning)




# ============================================================
# CONFIGURAÇÕES E UTILITÁRIOS
# ============================================================




SUIT_MAP = {0: 's', 1: 'h', 2: 'd', 3: 'c'}
RANK_MAP = {2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7',
            8:'8', 9:'9', 10:'T', 11:'J', 12:'Q', 13:'K', 14:'A'}




class PokerConstants:
    INITIAL_STACK = 1000
    SMALL_BLIND = 5
    BIG_BLIND = 10
    ANTE = 1
    MAX_ROUNDS = 4




class CardUtils:
    @staticmethod
    def create_deck():
        return [(s, r) for s in range(4) for r in range(2, 15)]
   
    @staticmethod
    def deal_cards(deck):
        random.shuffle(deck)
        return deck[:2], deck[2:4], deck[4:9]
   
    @staticmethod
    def convert_cards(cards):
        converted = []
        for s, r in cards:
            try:
                converted.append(Card.new(RANK_MAP[r] + SUIT_MAP[s]))
            except:
                pass
        return converted




class HandEvaluator:
    def __init__(self):
        self.evaluator = Evaluator()
        self.cache = {}
   
    def get_winner(self, player, opponent, board):
        key = (tuple(player), tuple(opponent), tuple(board))
        if key in self.cache:
            return self.cache[key]
       
        if len(board) < 3:
            result = self._simulate_preflop(player, opponent, board)
        else:
            result = self._evaluate_full_board(player, opponent, board)
       
        self.cache[key] = result
        return result
   
    def _evaluate_full_board(self, player, opponent, board):
        try:
            player_cards = CardUtils.convert_cards(player)
            opponent_cards = CardUtils.convert_cards(opponent)
            board_cards = CardUtils.convert_cards(board)
           
            if len(board_cards) >= 3:
                player_score = self.evaluator.evaluate(board_cards, player_cards)
                opponent_score = self.evaluator.evaluate(board_cards, opponent_cards)
                return player_score < opponent_score
            else:
                return self._compare_high_card(player, opponent)
        except:
            return random.choice([True, False])
   
    def _compare_high_card(self, player, opponent):
        player_ranks = sorted([r for s, r in player], reverse=True)
        opponent_ranks = sorted([r for s, r in opponent], reverse=True)
        for p, o in zip(player_ranks, opponent_ranks):
            if p != o:
                return p > o
        return False
   
    def _simulate_preflop(self, player, opponent, board, n_sims=100):
        wins = 0
        deck = CardUtils.create_deck()
        used = set(player + opponent + board)
        deck = [c for c in deck if c not in used]
       
        for _ in range(n_sims):
            random.shuffle(deck)
            needed = 5 - len(board)
            sim_board = list(board) + deck[:needed]
            remaining = deck[needed:]
           
            if len(remaining) >= 2:
                sim_opponent = remaining[:2]
            else:
                sim_opponent = random.sample(deck[:10], 2)
           
            if self._evaluate_full_board(player, sim_opponent, sim_board):
                wins += 1
       
        return wins / n_sims




# ============================================================
# FEATURES EXTRACTOR
# ============================================================




class FeatureExtractor:
    def __init__(self):
        self.evaluator = HandEvaluator()
        self.win_cache = {}
   
    def extract_features(self, player, board, opponent_stats, pot, to_call, is_dealer, stack):
        ranks = [r for s, r in player]
        ranks.sort(reverse=True)
        suits = [s for s, r in player]
       
        # Features básicas
        hand_strength = self._calculate_hand_strength(ranks, suits)
        high_card = ranks[0] / 14.0
        low_card = ranks[1] / 14.0
        is_pair = 1.0 if ranks[0] == ranks[1] else 0.0
        is_suited = 1.0 if suits[0] == suits[1] else 0.0
        gap = abs(ranks[0] - ranks[1]) / 12.0
        board_progress = len(board) / 5.0
        pot_odds = min(1.0, to_call / max(pot, 1))
        stack_ratio = min(1.0, stack / PokerConstants.INITIAL_STACK)
        m_value = min(1.0, stack / max(pot, 20))
        hand_value = (ranks[0] + ranks[1]) / 28.0
        position_feature = 1.0 if is_dealer else 0.0
       
        # Win probability com cache
        win_prob = self._get_win_probability(player, board)
       
        # Opponent tendencies
        fold_rate = min(1.0, opponent_stats.get('fold_rate', 0.0))
        call_rate = min(1.0, opponent_stats.get('call_rate', 0.0))
        raise_rate = min(1.0, opponent_stats.get('raise_rate', 0.0))
        aggression = min(1.0, (raise_rate + 0.1) / (fold_rate + 0.1))
       
        # Improvement potential
        improvement = self._calculate_improvement_potential(ranks, suits, board)
       
        # Draw possibilities
        has_flush_draw = self._has_flush_draw(suits, board)
        has_straight_draw = self._has_straight_draw(ranks, board)
       
        features = np.array([
            hand_strength, high_card, low_card, is_pair,
            is_suited, gap, board_progress, pot_odds,
            stack_ratio, m_value, hand_value, position_feature,
            win_prob, fold_rate, call_rate, raise_rate,
            aggression, improvement, has_flush_draw, has_straight_draw
        ], dtype=np.float32)
       
        return features
   
    def _calculate_hand_strength(self, ranks, suits):
        if ranks[0] == ranks[1]:
            return 0.5 + (ranks[0] - 2) / 24.0
        elif suits[0] == suits[1]:
            return 0.3 + (ranks[0] + ranks[1] - 4) / 48.0
        else:
            return 0.1 + (ranks[0] + ranks[1] - 4) / 48.0
   
    def _get_win_probability(self, player, board):
        key = (tuple(player), tuple(board))
        if key not in self.win_cache:
            self.win_cache[key] = self._calculate_win_prob(player, board)
        return self.win_cache[key]
   
    def _calculate_win_prob(self, player, board, n_sims=30):
        if len(board) < 3:
            return self._estimate_preflop_strength(player)
       
        wins = 0
        deck = CardUtils.create_deck()
        used = set(player + board)
        deck = [c for c in deck if c not in used]
       
        for _ in range(n_sims):
            random.shuffle(deck)
            opp = deck[:2]
            sim_board = list(board)
            while len(sim_board) < 5:
                sim_board.append(deck[len(opp) + len(sim_board) - len(board)])
           
            if self.evaluator.get_winner(player, opp, sim_board):
                wins += 1
       
        return wins / n_sims
   
    def _estimate_preflop_strength(self, player):
        ranks = [r for s, r in player]
        high = max(ranks)
        low = min(ranks)
        suited = player[0][0] == player[1][0]
       
        if high >= 14 and low >= 13:
            return 0.65 if suited else 0.55
        elif high >= 14:
            return 0.55 + (low - 2) * 0.01 if suited else 0.45 + (low - 2) * 0.008
        elif high >= 13:
            return 0.50 + (low - 2) * 0.008 if suited else 0.40 + (low - 2) * 0.006
        elif high == low:
            return 0.40 + (high - 2) * 0.03
        else:
            return 0.20 + (high + low) / 56.0
   
    def _calculate_improvement_potential(self, ranks, suits, board):
        if len(board) < 3:
            gap = abs(ranks[0] - ranks[1])
            suited = suits[0] == suits[1]
            potential = 0.0
            if gap <= 2:
                potential += 0.3
            if suited:
                potential += 0.2
            if ranks[0] >= 12:
                potential += 0.2
            return min(1.0, potential)
        else:
            potential = 0.0
            board_suits = [s for s, r in board]
            if suits[0] in board_suits or suits[1] in board_suits:
                potential += 0.3
            if abs(ranks[0] - ranks[1]) <= 2:
                potential += 0.2
            return min(1.0, potential)
   
    def _has_flush_draw(self, suits, board):
        if len(board) < 3:
            return 0.0
        board_suits = [s for s, r in board]
        suited_count = sum(1 for s in suits if s in board_suits)
        return 0.5 if suited_count >= 2 else 0.0
   
    def _has_straight_draw(self, ranks, board):
        if len(board) < 3:
            return 0.0
        board_ranks = [r for s, r in board]
        all_ranks = sorted(set(ranks + board_ranks))
       
        for i in range(len(all_ranks) - 3):
            if all_ranks[i+3] - all_ranks[i] <= 4:
                return 0.5
        return 0.0




# ============================================================
# OPPONENT MODEL
# ============================================================




class OpponentModel:
    def __init__(self):
        self.history = deque(maxlen=100)
        self.stats = {
            'fold_rate': 0.0,
            'call_rate': 0.0,
            'raise_rate': 0.0,
            'total_hands': 0
        }
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.last_actions = deque(maxlen=5)
   
    def update(self, action: int):
        self.history.append(action)
        self.action_counts[action] += 1
        self.stats['total_hands'] += 1
        self.last_actions.append(action)
       
        total = max(1, self.stats['total_hands'])
        self.stats['fold_rate'] = self.action_counts[0] / total
        self.stats['call_rate'] = self.action_counts[1] / total
        self.stats['raise_rate'] = self.action_counts[2] / total
   
    def get_stats(self) -> Dict:
        return self.stats.copy()




# ============================================================
# REDE NEURAL MELHORADA COM DROPOUT
# ============================================================




class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int = 20, output_dim: int = 3):
        super().__init__()
       
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
       
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
       
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
   
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))




# ============================================================
# POKER GAME
# ============================================================




class PokerGame:
    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.reset()
   
    def reset(self):
        self.deck = CardUtils.create_deck()
        random.shuffle(self.deck)
        self.player_hand = self.deck[:2]
        self.opponent_hand = self.deck[2:4]
        self.board = []
        self.pot = PokerConstants.BIG_BLIND + PokerConstants.SMALL_BLIND
        self.stacks = {
            'player': PokerConstants.INITIAL_STACK,
            'opponent': PokerConstants.INITIAL_STACK
        }
        self.bets = {'player': 0, 'opponent': 0}
        self.current_phase = 0
        self.history = []
        self.done = False
        self.winner = None
        self.is_player_dealer = random.choice([True, False])
       
        self.bets['player'] = PokerConstants.SMALL_BLIND
        self.bets['opponent'] = PokerConstants.BIG_BLIND
        self.stacks['player'] -= PokerConstants.SMALL_BLIND
        self.stacks['opponent'] -= PokerConstants.BIG_BLIND
       
        return self._get_state()
   
    def _get_state(self):
        return {
            'player_hand': self.player_hand,
            'board': self.board.copy(),
            'pot': self.pot,
            'stacks': self.stacks.copy(),
            'bets': self.bets.copy(),
            'phase': self.current_phase,
            'to_call': self._get_to_call(),
            'is_dealer': self.is_player_dealer,
            'history': self.history.copy(),
        }
   
    def _get_to_call(self):
        opponent_bet = self.bets['opponent']
        player_bet = self.bets['player']
        return max(0, opponent_bet - player_bet)
   
    def step(self, action: int, opponent_action: int = None):
        if self.done:
            return self._get_state(), 0, True, {}
       
        if opponent_action is None:
            opponent_action = self._get_opponent_action()
       
        self._execute_action('player', action)
        self._execute_action('opponent', opponent_action)
       
        self.history.append((action, opponent_action))
       
        if self._check_hand_end():
            self.done = True
            winner = self._determine_winner()
            reward = self._calculate_reward(winner)
           
            if winner == 'player':
                self.stacks['player'] += self.pot
            elif winner == 'opponent':
                self.stacks['opponent'] += self.pot
           
            return self._get_state(), reward, True, {'winner': winner}
       
        self._advance_phase()
        reward = -self._get_to_call() * 0.01
       
        return self._get_state(), reward, False, {}
   
    def _execute_action(self, player: str, action: int):
        if action == 0:
            self.done = True
            self.winner = 'opponent' if player == 'player' else 'player'
            return
       
        elif action == 1:
            to_call = self._get_to_call()
            if player == 'player':
                to_call = min(to_call, self.stacks['player'])
                self.stacks['player'] -= to_call
                self.bets['player'] += to_call
                self.pot += to_call
            else:
                to_call = min(to_call, self.stacks['opponent'])
                self.stacks['opponent'] -= to_call
                self.bets['opponent'] += to_call
                self.pot += to_call
       
        elif action == 2:
            raise_amount = self._calculate_raise_amount(player)
            if player == 'player':
                raise_amount = min(raise_amount, self.stacks['player'])
                self.stacks['player'] -= raise_amount
                self.bets['player'] += raise_amount
                self.pot += raise_amount
            else:
                raise_amount = min(raise_amount, self.stacks['opponent'])
                self.stacks['opponent'] -= raise_amount
                self.bets['opponent'] += raise_amount
                self.pot += raise_amount
   
    def _calculate_raise_amount(self, player: str):
        pot = self.pot
        to_call = self._get_to_call()
       
        min_raise = max(PokerConstants.BIG_BLIND, pot // 2)
        max_raise = min(self.stacks[player], pot * 2)
       
        if max_raise < min_raise:
            return max_raise
       
        return random.randint(min_raise, max_raise)
   
    def _advance_phase(self):
        self.current_phase += 1
        if self.current_phase == 1:
            self.board = self.deck[4:7]
        elif self.current_phase == 2:
            self.board.append(self.deck[7])
        elif self.current_phase == 3:
            self.board.append(self.deck[8])
       
        self.bets = {'player': 0, 'opponent': 0}
   
    def _check_hand_end(self):
        if self.done:
            return True
        if self.stacks['player'] <= 0 or self.stacks['opponent'] <= 0:
            return True
        if self.current_phase > 3:
            return True
        return False
   
    def _determine_winner(self):
        if self.winner:
            return self.winner
       
        if self.hand_evaluator.get_winner(self.player_hand, self.opponent_hand, self.board):
            return 'player'
        else:
            return 'opponent'
   
    def _calculate_reward(self, winner: str):
        if winner == 'player':
            return self.pot * 0.1 + 2.0
        elif winner == 'opponent':
            return -self.pot * 0.1 - 2.0
        else:
            return 0.0
   
    def _get_opponent_action(self):
        return random.choice([0, 1, 2])




# ============================================================
# AGENTE DQN COM EARLY STOPPING
# ============================================================




class DQNAgent:
    def __init__(self, state_dim: int = 20, action_dim: int = 3, device: str = 'cpu'):
        self.device = torch.device(device)
        self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
       
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0002)
        self.memory = deque(maxlen=200000)
       
        # Parâmetros com reinício de epsilon
        self.epsilon = 0.5  # Começa mais alto
        self.epsilon_min = 0.02  # Mínimo mais alto para manter exploração
        self.epsilon_decay = 0.9998  # Decaimento mais lento
        self.gamma = 0.99
        self.batch_size = 256
        self.update_target_every = 500  # Mais frequente
        self.step_count = 0
       
        # Opponent modeling
        self.opponent_models = defaultdict(OpponentModel)
        self.current_opponent = None
       
        self.feature_extractor = FeatureExtractor()
       
        # Para early stopping
        self.best_avg_reward = -float('inf')
        self.patience_counter = 0
       
    def remember(self, state, action, reward, next_state, done, opponent_id='default'):
        self.memory.append((state, action, reward, next_state, done))
        if opponent_id:
            self.opponent_models[opponent_id].update(action)
   
    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)
       
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
   
    def act_with_opponent(self, state: np.ndarray, opponent_id: str = None, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)
       
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
           
            if opponent_id and opponent_id in self.opponent_models:
                model = self.opponent_models[opponent_id]
                opp_stats = model.get_stats()
               
                # Ajuste mais sutil para evitar overfitting
                if opp_stats['fold_rate'] > 0.3:
                    q_values[0, 2] += 0.3 * opp_stats['fold_rate']
                if opp_stats['raise_rate'] > 0.3:
                    q_values[0, 1] += 0.2 * opp_stats['raise_rate']
                if opp_stats['call_rate'] > 0.5:
                    q_values[0, 1] -= 0.1 * opp_stats['call_rate']
           
            return torch.argmax(q_values).item()
   
    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
       
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
       
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
       
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
       
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(current_q, target_q)
       
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
       
        # Decaimento mais suave
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
       
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
       
        return loss.item()
   
    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'opponent_models': dict(self.opponent_models)
        }, path)
   
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint.get('epsilon', 0.02)
        self.step_count = checkpoint.get('step_count', 0)
        self.opponent_models = defaultdict(OpponentModel, checkpoint.get('opponent_models', {}))




# ============================================================
# AGENTES OPONENTES
# ============================================================




class BaseAgent:
    def __init__(self, name: str = "Base"):
        self.name = name
        self.opponent_model = OpponentModel()
   
    def act(self, player, board, pot, to_call):
        raise NotImplementedError
   
    def update(self, action: int):
        self.opponent_model.update(action)




class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__("Random")
   
    def act(self, player, board, pot, to_call):
        return random.choice([0, 1, 2])




class HeuristicAgent(BaseAgent):
    def __init__(self):
        super().__init__("Heuristic")
   
    def act(self, player, board, pot, to_call):
        ranks = [r for s, r in player]
        high_rank = max(ranks)
        is_pair = ranks[0] == ranks[1]
        is_suited = player[0][0] == player[1][0]
       
        if high_rank >= 13 or is_pair:
            return 2
        elif high_rank >= 10 or is_suited:
            if to_call <= pot * 0.3:
                return 2
            return 1
        else:
            if to_call <= pot * 0.1:
                return 1
            return 0




class AggressiveAgent(BaseAgent):
    def __init__(self):
        super().__init__("Aggressive")
   
    def act(self, player, board, pot, to_call):
        if to_call > pot * 2:
            return 1
        return 2




class ConservativeAgent(BaseAgent):
    def __init__(self):
        super().__init__("Conservative")
   
    def act(self, player, board, pot, to_call):
        return 1




class FoldAgent(BaseAgent):
    def __init__(self):
        super().__init__("Fold")
   
    def act(self, player, board, pot, to_call):
        if to_call <= 0:
            return 1
        return 0




class ImitatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Imitator")
        self.last_action = 1
   
    def act(self, player, board, pot, to_call):
        self.last_action = 2 if self.last_action == 1 else 1
        return self.last_action




class BlufferAgent(BaseAgent):
    def __init__(self):
        super().__init__("Bluffer")
        self.bluff_probability = 0.3
   
    def act(self, player, board, pot, to_call):
        ranks = [r for s, r in player]
        hand_strength = max(ranks) / 14.0
       
        if hand_strength < 0.3 and random.random() < self.bluff_probability:
            return 2
        elif hand_strength > 0.6:
            return 2
        elif hand_strength > 0.3:
            return 1
        else:
            return 0




# ============================================================
# TREINADOR COM EARLY STOPPING
# ============================================================




class PokerTrainer:
    def __init__(self, agent: DQNAgent, opponents: List[BaseAgent]):
        self.history = {
            'episodes': [],
            'avg_rewards': [],
            'win_rates': [],
            'epsilons': []
        }
        self.agent = agent
        self.opponents = opponents
        self.game = PokerGame()
        self.feature_extractor = FeatureExtractor()
        self.metrics = {
            'wins': 0,
            'losses': 0,
            'total_reward': 0,
            'hand_results': [],
            'best_avg_reward': -float('inf'),
            'best_model': None
        }
        self.patience = 10  # Episódios sem melhora antes de parar
   
    def train_episode(self, opponent_id: str = None) -> Dict:
        if opponent_id is None:
            opponent = random.choice(self.opponents)
            opponent_id = opponent.name
        else:
            opponent = next(o for o in self.opponents if o.name == opponent_id)
       
        state = self.game.reset()
        total_reward = 0
        steps = 0
        done = False
       
        features = self.feature_extractor.extract_features(
            player=state['player_hand'],
            board=state['board'],
            opponent_stats=opponent.opponent_model.get_stats(),
            pot=state['pot'],
            to_call=state['to_call'],
            is_dealer=state['is_dealer'],
            stack=state['stacks']['player']
        )
       
        while not done and steps < 20:
            action = self.agent.act_with_opponent(features, opponent_id, training=True)
           
            opponent_action = opponent.act(
                state['player_hand'],
                state['board'],
                state['pot'],
                state['to_call']
            )
           
            next_state, reward, done, info = self.game.step(action, opponent_action)
           
            next_features = self.feature_extractor.extract_features(
                player=next_state['player_hand'],
                board=next_state['board'],
                opponent_stats=opponent.opponent_model.get_stats(),
                pot=next_state['pot'],
                to_call=next_state['to_call'],
                is_dealer=next_state['is_dealer'],
                stack=next_state['stacks']['player']
            )
           
            self.agent.remember(features, action, reward, next_features, done, opponent_id)
            loss = self.agent.replay()
           
            total_reward += reward
            features = next_features
            state = next_state
            steps += 1
            opponent.update(action)
       
        if done and info.get('winner') == 'player':
            self.metrics['wins'] += 1
        elif done:
            self.metrics['losses'] += 1
       
        self.metrics['total_reward'] += total_reward
        self.metrics['hand_results'].append({
            'opponent': opponent_id,
            'reward': total_reward,
            'steps': steps,
            'winner': info.get('winner', 'none')
        })
       
        return {
            'reward': total_reward,
            'steps': steps,
            'winner': info.get('winner', 'none'),
            'loss': loss if steps > 0 else 0
        }
   
    def train(self, episodes: int = 10000, eval_every: int = 500, patience_limit: int = 10) -> List[Dict]:
        results = []
        no_improvement = 0
        best_avg_reward = -float('inf')
       
        for episode in range(episodes):
            opponent = random.choice(self.opponents)
            result = self.train_episode(opponent.name)
            results.append(result)
           
            if (episode + 1) % eval_every == 0:
                avg_reward = np.mean([r['reward'] for r in results[-eval_every:]])
                win_rate = sum(1 for r in results[-eval_every:] if r['winner'] == 'player') / eval_every
               
               # historia pro grafico 
                self.history['episodes'].append(episode + 1)
                self.history['avg_rewards'].append(avg_reward)
                self.history['win_rates'].append(win_rate)
                self.history['epsilons'].append(self.agent.epsilon)

                print(f"Ep {episode+1}/{episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Win Rate: {win_rate:.1%} | "
                      f"Epsilon: {self.agent.epsilon:.3f}")
               
                # Early stopping
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    no_improvement = 0
                    # Salva melhor modelo
                    self.agent.save('dqn_best_model.pth')
                    print(f"  ✅ Novo melhor modelo! Reward: {avg_reward:.2f}")
                else:
                    no_improvement += 1
                    print(f"  ⏳ Sem melhora por {no_improvement} avaliações")
               
                # Para se não melhorar por muitas avaliações
                if no_improvement >= patience_limit:
                    print(f"\n🛑 Early stopping! Não melhorou por {patience_limit} avaliações.")
                    print(f"Melhor recompensa: {best_avg_reward:.2f}")
                    break
               
                # Salva checkpoint regular
                if (episode + 1) % 2000 == 0:
                    self.agent.save(f'dqn_checkpoint_{episode+1}.pth')
       
        # Carrega melhor modelo
        if os.path.exists('dqn_best_model.pth'):
            print("\n📥 Carregando melhor modelo encontrado...")
            self.agent.load('dqn_best_model.pth')

        with open('training_history.pkl', 'wb') as f: # salva o historico
            pickle.dump(self.history, f)
        print("📁 Histórico de treinamento salvo em 'training_history.pkl'")

        return results
    



# ============================================================
# AVALIADOR
# ============================================================




class PokerEvaluator:
    def __init__(self, agent: DQNAgent, opponents: List[BaseAgent], n_games: int = 1000):
        self.agent = agent
        self.opponents = {opp.name: opp for opp in opponents}
        self.n_games = n_games
        self.feature_extractor = FeatureExtractor()
        self.game = PokerGame()
   
    def evaluate_against_all(self) -> Dict:
        results = {}
       
        for name, opponent in self.opponents.items():
            print(f"\nAvaliando contra {name}...")
            results[name] = self._evaluate_against(opponent)
       
        return results
   
    def _evaluate_against(self, opponent: BaseAgent) -> Dict:
        wins = 0
        total_reward = 0
        results = []
       
        for game_num in range(self.n_games):
            state = self.game.reset()
           
            features = self.feature_extractor.extract_features(
                player=state['player_hand'],
                board=state['board'],
                opponent_stats=opponent.opponent_model.get_stats(),
                pot=state['pot'],
                to_call=state['to_call'],
                is_dealer=state['is_dealer'],
                stack=state['stacks']['player']
            )
           
            done = False
            steps = 0
            reward_sum = 0
           
            while not done and steps < 20:
                action = self.agent.act(features, training=False)
               
                opponent_action = opponent.act(
                    state['player_hand'],
                    state['board'],
                    state['pot'],
                    state['to_call']
                )
               
                next_state, reward, done, info = self.game.step(action, opponent_action)
               
                features = self.feature_extractor.extract_features(
                    player=next_state['player_hand'],
                    board=next_state['board'],
                    opponent_stats=opponent.opponent_model.get_stats(),
                    pot=next_state['pot'],
                    to_call=next_state['to_call'],
                    is_dealer=next_state['is_dealer'],
                    stack=next_state['stacks']['player']
                )
               
                state = next_state
                reward_sum += reward
                steps += 1
           
            if info.get('winner') == 'player':
                wins += 1
           
            total_reward += reward_sum
            results.append({
                'reward': reward_sum,
                'winner': info.get('winner', 'none'),
                'steps': steps
            })
       
        return {
            'win_rate': wins / self.n_games,
            'avg_reward': total_reward / self.n_games,
            'total_reward': total_reward,
            'results': results
        }
   
    def print_results(self, results: Dict):
        print("\n" + "="*70)
        print("📊 RESULTADOS DA AVALIAÇÃO")
        print("="*70)
       
        sorted_results = sorted(
            [(name, data) for name, data in results.items()],
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )
       
        print(f"\n{'Oponente':<15} {'Win Rate':<12} {'Avg Reward':<12} {'Total':<12}")
        print("-"*70)
       
        for name, data in sorted_results:
            print(f"{name:<15} {data['win_rate']:>6.1%}    {data['avg_reward']:>+8.2f}    {data['total_reward']:>+10.1f}")




# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================




def main(seed=None):
    if seed is None:
        seed = random.randint(0, 10000)
    
    # FIXA TODAS AS SEMENTES
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"🎲 Rodando com seed = {seed}")

    print("="*70)
    print("🎰 POKER AI - TREINAMENTO COM EARLY STOPPING")
    print("="*70)
   
    # Cria agentes oponentes
    opponents = [
        RandomAgent(),
        HeuristicAgent(),
        AggressiveAgent(),
        ConservativeAgent(),
        FoldAgent(),
        ImitatorAgent(),
        BlufferAgent()
    ]
   
    print(f"\nOponentes disponíveis: {[opp.name for opp in opponents]}")
   
    # Cria agente DQN
    agent = DQNAgent(state_dim=20)
   
    # Verifica se existe modelo salvo
    model_path = 'dqn_best_model.pth'
    if os.path.exists(model_path):
        print(f"\nCarregando melhor modelo de {model_path}")
        agent.load(model_path)
    elif os.path.exists('dqn_complete_model.pth'):
        print(f"\nCarregando modelo anterior de dqn_complete_model.pth")
        agent.load('dqn_complete_model.pth')
        # Reseta epsilon para explorar mais
        agent.epsilon = 0.3
        print(f"Epsilon resetado para {agent.epsilon}")
    else:
        print("\nIniciando treinamento do zero...")
   
    # Treinador
    trainer = PokerTrainer(agent, opponents)
   
    # Treina com early stopping
    print("\n" + "="*70)
    print("🎯 INICIANDO TREINAMENTO COM EARLY STOPPING")
    print("="*70)
   
    n_episodes = 10000
    results = trainer.train(
        episodes=n_episodes,
        eval_every=500,
        patience_limit=5  # Para se não melhorar em 5 avaliações
    )
   
    # Salva modelo final
    agent.save('dqn_final_model.pth')
    print(f"\n✅ Modelo final salvo em dqn_final_model.pth")
   
    # Avaliação
    print("\n" + "="*70)
    print("🔍 INICIANDO AVALIAÇÃO")
    print("="*70)
   
    evaluator = PokerEvaluator(agent, opponents, n_games=500)
    eval_results = evaluator.evaluate_against_all()
    evaluator.print_results(eval_results)
   
    # Análise estatística
    print("\n" + "="*70)
    print("📈 ANÁLISE ESTATÍSTICA")
    print("="*70)
   
    total_wins = trainer.metrics['wins']
    total_hands = total_wins + trainer.metrics['losses']
    overall_win_rate = total_wins / max(1, total_hands)
   
    print(f"Total de mãos jogadas: {trainer.metrics['wins'] + trainer.metrics['losses']}")
    print(f"Vitórias: {trainer.metrics['wins']}")
    print(f"Derrotas: {trainer.metrics['losses']}")
    print(f"Win Rate Geral: {overall_win_rate:.1%}")
    print(f"Recompensa Total: {trainer.metrics['total_reward']:.1f}")



if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(seed)