import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict

# ============================================================
# IMPORTE AS CLASSES DO SEU CÓDIGO PRINCIPAL
# ============================================================

from CODIGO_DQN import *

def carregar_modelo(caminho='dqn_best_model.pth'):
    """Cria o agente e carrega os pesos salvos."""
    if not os.path.exists(caminho):
        print(f"❌ Arquivo {caminho} não encontrado!")
        print("Verifique se o modelo foi salvo corretamente.")
        return None
    
    agent = DQNAgent(state_dim=20)  # 20 features
    agent.load(caminho)
    agent.epsilon = 0.0  # Desliga exploração para avaliação
    print(f"✅ Modelo carregado de {caminho}")
    return agent

def avaliar_contra_oponente(agent, opponent, n_games=500):
    """
    Avalia o agente contra um oponente específico por 'n_games' mãos.
    Retorna: win_rate, avg_reward, total_reward, lista_de_acoes
    """
    game = PokerGame()
    feature_extractor = FeatureExtractor()
    
    wins = 0
    total_reward = 0.0
    all_actions = []
    
    for _ in range(n_games):
        state = game.reset()
        done = False
        reward_sum = 0.0
        actions_this_hand = []
        
        while not done:
            # Extrai features do estado atual
            features = feature_extractor.extract_features(
                player=state['player_hand'],
                board=state['board'],
                opponent_stats=opponent.opponent_model.get_stats(),
                pot=state['pot'],
                to_call=state['to_call'],
                is_dealer=state['is_dealer'],
                stack=state['stacks']['player']
            )
            
            # Ação do nosso agente (sem exploração)
            action = agent.act(features, training=False)
            actions_this_hand.append(action)
            
            # Ação do oponente
            opponent_action = opponent.act(
                state['player_hand'],
                state['board'],
                state['pot'],
                state['to_call']
            )
            
            # Executa o passo no jogo
            next_state, reward, done, info = game.step(action, opponent_action)
            
            # Atualiza
            reward_sum += reward
            state = next_state
            
            # Atualiza o modelo do oponente (para as estatísticas dele)
            opponent.update(action)
        
        # Fim da mão
        all_actions.extend(actions_this_hand)
        total_reward += reward_sum
        
        if info.get('winner') == 'player':
            wins += 1
    
    win_rate = wins / n_games
    avg_reward = total_reward / n_games
    
    return {
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'total_reward': total_reward,
        'actions': all_actions
    }

def plotar_resultados(resultados, nome_arquivo='resultados_avaliacao.png'):
    """
    Gera o gráfico de barras comparativo (Win Rate e Recompensa Média)
    """
    # Ordena os oponentes pelo Win Rate (do maior para o menor)
    nomes = list(resultados.keys())
    win_rates = [resultados[n]['win_rate'] * 100 for n in nomes]
    avg_rewards = [resultados[n]['avg_reward'] for n in nomes]
    
    # Ordenação
    indices_ordenados = sorted(range(len(nomes)), key=lambda i: win_rates[i], reverse=True)
    nomes_ord = [nomes[i] for i in indices_ordenados]
    win_rates_ord = [win_rates[i] for i in indices_ordenados]
    avg_rewards_ord = [avg_rewards[i] for i in indices_ordenados]
    
    # Cria o gráfico com dois eixos
    x = np.arange(len(nomes_ord))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Barras azuis = Win Rate
    bars1 = ax1.bar(x - width/2, win_rates_ord, width, label='Win Rate (%)', 
                    color='#2E86AB', edgecolor='black')
    ax1.set_xlabel('Oponente', fontsize=13)
    ax1.set_ylabel('Win Rate (%)', color='#2E86AB', fontsize=13)
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nomes_ord, rotation=45, ha='right', fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Barras laranjas = Recompensa Média (eixo direito)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, avg_rewards_ord, width, label='Recompensa Média', 
                    color='#E8AA42', edgecolor='black')
    ax2.set_ylabel('Recompensa Média', color='#E8AA42', fontsize=13)
    ax2.tick_params(axis='y', labelcolor='#E8AA42')
    
    # Mostra os valores em cima das barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Título e legenda
    plt.title('Desempenho do Agente DQN contra 7 Arquétipos de Oponentes', fontsize=15, pad=20)
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico de barras salvo como '{nome_arquivo}'")
    plt.show()

def plotar_distribuicao_acoes(resultados, nome_arquivo='distribuicao_acoes.png'):
    """
    Gera gráfico de pizza com a distribuição de ações (Fold, Call, Raise)
    Usa o oponente 'Random' como exemplo (ou qualquer um que tenha ações registradas)
    """
    # Pega as ações do oponente Random (ou o primeiro da lista)
    opp_alvo = None
    for nome, data in resultados.items():
        if 'Random' in nome:
            opp_alvo = data
            break
    
    if opp_alvo is None:
        # Se não achou Random, pega o primeiro
        opp_alvo = list(resultados.values())[0]
    
    actions = opp_alvo['actions']
    counts = [actions.count(0), actions.count(1), actions.count(2)]  # 0=Fold, 1=Call, 2=Raise
    labels = ['Fold', 'Call', 'Raise']
    colors = ['#E63946', '#2ECC71', '#F1C40F']
    explode = (0.05, 0.05, 0.05)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    patches, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%',
                                        startangle=90, colors=colors, explode=explode,
                                        textprops={'fontsize': 13})
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax.set_title('Distribuição de Ações do Agente (vs Oponente Random)', fontsize=15)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"🍕 Gráfico de pizza salvo como '{nome_arquivo}'")
    plt.show()

def main():
    print("="*60)
    print("🎯 SCRIPT DE AVALIAÇÃO E GERAÇÃO DE GRÁFICOS")
    print("="*60)
    
    # 1. Carrega o modelo
    agent = carregar_modelo('dqn_best_model.pth')
    if agent is None:
        # Tenta carregar o final model
        agent = carregar_modelo('dqn_final_model.pth')
        if agent is None:
            print("❌ Nenhum modelo encontrado. Encerrando.")
            return
    
    # 2. Cria a lista de oponentes
    opponents = [
        RandomAgent(),
        HeuristicAgent(),
        AggressiveAgent(),
        ConservativeAgent(),
        FoldAgent(),
        ImitatorAgent(),
        BlufferAgent()
    ]
    print(f"\n👾 Avaliando contra {len(opponents)} oponentes...")
    print("(500 mãos cada, isso pode levar alguns segundos)\n")
    
    # 3. Avalia contra cada um
    resultados = {}
    for opp in opponents:
        print(f"▶️  Avaliando contra {opp.name}...", end=' ', flush=True)
        resultados[opp.name] = avaliar_contra_oponente(agent, opp, n_games=500)
        win = resultados[opp.name]['win_rate'] * 100
        print(f"Win Rate: {win:.1f}%")
    
    # 4. Mostra a tabela no terminal
    print("\n" + "="*60)
    print("📊 RESUMO DA AVALIAÇÃO")
    print("="*60)
    print(f"{'Oponente':<15} {'Win Rate':<12} {'Recompensa Média':<18}")
    print("-"*60)
    for nome, dados in sorted(resultados.items(), key=lambda x: x[1]['win_rate'], reverse=True):
        print(f"{nome:<15} {dados['win_rate']*100:>6.1f}%     {dados['avg_reward']:>+10.2f}")
    print("="*60)
    
    # 5. Gera os gráficos
    plotar_resultados(resultados)
    plotar_distribuicao_acoes(resultados)
    
    print("\n✅ Todos os gráficos foram gerados com sucesso!")
    print("Arquivos criados: 'resultados_avaliacao.png' e 'distribuicao_acoes.png'")

if __name__ == "__main__":
    main()