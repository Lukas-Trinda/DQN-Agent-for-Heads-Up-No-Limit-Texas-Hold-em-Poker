import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def carregar_historico(caminho='training_history.pkl'):
    """Carrega o histórico salvo durante o treinamento."""
    if not os.path.exists(caminho):
        print(f"❌ Arquivo {caminho} não encontrado!")
        print("Você precisa rodar o treinamento com a modificação do Passo 1 primeiro.")
        return None
    
    with open(caminho, 'rb') as f:
        history = pickle.load(f)
    
    print(f"✅ Histórico carregado: {len(history['episodes'])} pontos de avaliação")
    
    # Mostra as estatísticas básicas
    print(f"\n📊 Estatísticas do histórico:")
    print(f"   - Episódios: {history['episodes'][0]} a {history['episodes'][-1]}")
    print(f"   - Melhor recompensa: {max(history['avg_rewards']):.2f}")
    print(f"   - Melhor win rate: {max(history['win_rates'])*100:.1f}%")
    print(f"   - Epsilon final: {history['epsilons'][-1]:.3f}")
    
    return history

def plotar_curvas(history, nome_arquivo='curva_aprendizado.png'):
    """
    Gera dois gráficos empilhados:
    - Superior: Recompensa média
    - Inferior: Win Rate
    """
    episodios = history['episodes']
    avg_rewards = history['avg_rewards']
    win_rates = history['win_rates']
    
    # Calcula estatísticas
    melhor_recompensa = max(avg_rewards)
    melhor_episodio = episodios[avg_rewards.index(melhor_recompensa)]
    melhor_win_rate = max(win_rates)
    win_rate_final = win_rates[-1] if win_rates else 0
    
    # Cria a figura com 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # ----- Gráfico 1: Recompensa -----
    ax1.plot(episodios, avg_rewards, linewidth=2.5, color='#2E86AB', 
             label='Recompensa Média (janela 500)')
    
    # Linha horizontal da melhor recompensa
    ax1.axhline(y=melhor_recompensa, color='red', linestyle='--', 
                linewidth=1.5, label=f'Melhor: {melhor_recompensa:.2f} (ep. {melhor_episodio})')
    
    # Preenche a área sob a curva (opcional, fica bonito)
    ax1.fill_between(episodios, 0, avg_rewards, alpha=0.1, color='#2E86AB')
    
    ax1.set_ylabel('Recompensa Média', fontsize=13)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Evolução da Recompensa Média durante o Treinamento', fontsize=14, fontweight='bold')
    
    # ----- Gráfico 2: Win Rate -----
    ax2.plot(episodios, np.array(win_rates)*100, linewidth=2.5, color='#2ECC71', 
             label='Win Rate (janela 500)')
    
    # Linha horizontal da média final
    ax2.axhline(y=win_rate_final*100, color='orange', linestyle='--', 
                linewidth=1.5, label=f'Final: {win_rate_final*100:.1f}%')
    
    # Linha horizontal do melhor win rate
    ax2.axhline(y=melhor_win_rate*100, color='purple', linestyle=':', 
                linewidth=1.5, label=f'Melhor: {melhor_win_rate*100:.1f}%')
    
    # Preenche a área sob a curva (opcional)
    ax2.fill_between(episodios, 0, np.array(win_rates)*100, alpha=0.1, color='#2ECC71')
    
    ax2.set_xlabel('Episódios', fontsize=13)
    ax2.set_ylabel('Win Rate (%)', fontsize=13)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Adiciona uma anotação com o early stopping
    ultimo_episodio = episodios[-1]
    ax2.annotate(f'Early Stopping\nEp. {ultimo_episodio}', 
                 xy=(ultimo_episodio, win_rate_final*100),
                 xytext=(ultimo_episodio - 500, win_rate_final*100 + 10),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=10, color='gray',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"\n📈 Curva de aprendizado salva como '{nome_arquivo}'")
    plt.show()

def plotar_epsilon(history, nome_arquivo='decaimento_epsilon.png'):
    """
    (Opcional) Plota o decaimento do epsilon ao longo do treinamento.
    """
    episodios = history['episodes']
    epsilons = history['epsilons']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodios, epsilons, linewidth=2, color='#E63946')
    ax.fill_between(episodios, 0, epsilons, alpha=0.1, color='#E63946')
    ax.set_xlabel('Episódios', fontsize=13)
    ax.set_ylabel('Epsilon (ε)', fontsize=13)
    ax.set_title('Decaimento da Taxa de Exploração (ε-greedy)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(epsilons) * 1.1)
    
    # Anota o epsilon mínimo
    eps_min = min(epsilons)
    ax.axhline(y=eps_min, color='red', linestyle='--', linewidth=1.5, 
               label=f'ε mínimo: {eps_min:.3f}')
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"📉 Gráfico do Epsilon salvo como '{nome_arquivo}'")
    plt.show()

def main():
    print("="*60)
    print("📈 PLOTAGEM DA CURVA DE APRENDIZADO")
    print("="*60)
    
    history = carregar_historico('training_history.pkl')
    if history is None:
        return
    
    # Gera os gráficos principais
    plotar_curvas(history)
    
    # (Opcional) Descomente a linha abaixo se quiser ver o decaimento do epsilon
    # plotar_epsilon(history)
    
    print("\n✅ Gráficos gerados com sucesso!")
    print("Arquivo criado: 'curva_aprendizado.png'")

if __name__ == "__main__":
    main()