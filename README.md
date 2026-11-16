# DQN Agent for Heads-Up No-Limit Texas Hold'em Poker

Agente de Inteligência Artificial baseado em **Deep Q-Learning (DQN)** para jogar Heads-Up No-Limit Texas Hold'em, um dos ambientes mais desafiadores para aprendizado por reforço devido à natureza parcialmente observável, informação imperfeita e ações contínuas.

## 🎯 Objetivo

Desenvolver um agente capaz de aprender estratégias competitivas em poker através de **aprendizado por reforço profundo**, utilizando uma rede neural para aproximar a função Q.

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **PyTorch** (ou TensorFlow — ajuste conforme seu código)
- **NumPy**
- **Gym** (se aplicável)
- **Matplotlib** (para visualização de métricas)

## 🧠 Arquitetura do Agente

- **Estado:** Vetor de features representando a mão do agente, cartas comunitárias, stack sizes, pot e histórico de ações.
- **Ações:** Fold, Call/Check, Raise (com tamanhos discretizados).
- **Rede Neural:** [ex: 3 camadas fully-connected com ativação ReLU, output = Q-values para cada ação]
- **Replay Buffer:** Experiência armazenada para treinamento estável.
- **Target Network:** Atualização periódica para reduzir oscilações no treinamento.
- **Exploração:** Política ε-greedy com decaimento linear.

## 📊 Resultados

<img width="990" height="778" alt="curva_aprendizado (1)" src="https://github.com/user-attachments/assets/e39026d9-86d6-4315-92ec-9c81b5078689" />

Exemplo:
- Após **X episódios** de treinamento, o agente atingiu uma taxa de vitória de **Y%** contra um oponente aleatório.
- Recompensa média (últimos 1000 episódios): **Z bb/100**.

## 🚀 Como Executar

```bash
# Clone o repositório
git clone https://github.com/Lukas-Trinda/DQN-Agent-for-Heads-Up-No-Limit-Texas-Hold-em-Poker.git
cd DQN-Agent-for-Heads-Up-No-Limit-Texas-Hold-em-Poker

# Instale as dependências
pip install -r requirements.txt

# Treine o agente
python train.py

# Avalie o agente
python evaluate.py
