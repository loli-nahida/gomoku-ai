/**
 * 训练模块 - 自我对弈训练神经网络
 */

class TrainingManager {
  constructor(ai) {
    this.ai = ai;
    this.network = ai.network;
    this.trainer = new SelfPlayTrainer(this.network);
    this.isTraining = false;
    this.trainingGames = 0;
    this.trainingHistory = [];
    this.onProgress = null;
    this.onComplete = null;
  }

  /**
   * Run training in background
   */
  async train(numGames = 5, simulationsPerMove = 100) {
    if (this.isTraining) return;
    this.isTraining = true;

    for (let g = 0; g < numGames; g++) {
      if (!this.isTraining) break;

      // Run a self-play game
      const result = this.runSelfPlayGame(simulationsPerMove);
      this.trainingGames++;
      this.trainingHistory.push(result);

      // Update progress
      if (this.onProgress) {
        this.onProgress({
          game: this.trainingGames,
          total: numGames,
          result: result,
          loss: this.calculateLoss()
        });
      }

      // Yield to UI
      await new Promise(r => setTimeout(r, 50));
    }

    // Train on collected data
    if (this.trainer.trainingData.length > 0) {
      const loss = this.network.trainOnGame(this.trainer.trainingData, 0.001);
    }

    this.isTraining = false;
    if (this.onComplete) {
      this.onComplete({
        games: this.trainingGames,
        history: this.trainingHistory
      });
    }
  }

  runSelfPlayGame(simulations) {
    const game = new GomokuGame();
    const history = [];
    let moveCount = 0;

    while (!game.gameOver && moveCount < 225) {
      const player = game.currentPlayer;
      const input = game.toTensorInput(player);

      // Quick heuristic MCTS for self-play (faster)
      const mcts = new MCTS(this.network, simulations, 3.0);
      mcts.useNN = this.network.initialized;
      const result = mcts.run(game);

      history.push({
        input: input,
        policy: result.policy,
        player: player
      });

      // Add noise for exploration
      const validMoves = game.getValidMoves();
      const move = this.selectMove(result.policy, validMoves, game);
      game.makeMove(move[0], move[1]);
      moveCount++;
    }

    // Assign values
    const winner = game.winner;
    for (const entry of history) {
      let value;
      if (winner === 0) value = 0;
      else if (winner === entry.player) value = 1;
      else value = -1;

      this.trainer.trainingData.push({
        input: entry.input,
        targetPolicy: entry.policy,
        targetValue: value
      });
    }

    return { winner, moveCount };
  }

  selectMove(policy, validMoves, game) {
    // Temperature-based selection
    const temp = 1.0;
    const probs = [];
    let sum = 0;

    for (const [r, c] of validMoves) {
      const idx = r * BOARD_SIZE + c;
      const p = Math.pow(policy[idx] + 1e-10, 1 / temp);
      probs.push({ move: [r, c], prob: p });
      sum += p;
    }

    // Normalize and sample
    let rand = Math.random() * sum;
    for (const item of probs) {
      rand -= item.prob;
      if (rand <= 0) return item.move;
    }
    return probs[probs.length - 1].move;
  }

  calculateLoss() {
    if (this.trainer.trainingData.length === 0) return 0;
    let totalLoss = 0;
    const sampleSize = Math.min(10, this.trainer.trainingData.length);
    for (let i = 0; i < sampleSize; i++) {
      const idx = Math.floor(Math.random() * this.trainer.trainingData.length);
      const { input, targetPolicy, targetValue } = this.trainer.trainingData[idx];
      const { policy, value } = this.network.predict(input);
      let policyLoss = 0;
      for (let j = 0; j < targetPolicy.length; j++) {
        if (targetPolicy[j] > 0) {
          policyLoss -= targetPolicy[j] * Math.log(policy[j] + 1e-10);
        }
      }
      totalLoss += policyLoss + (value - targetValue) ** 2;
    }
    return totalLoss / sampleSize;
  }

  stop() {
    this.isTraining = false;
  }

  getStats() {
    return {
      gamesPlayed: this.trainingGames,
      trainingDataSize: this.trainer.trainingData.length,
      isTraining: this.isTraining,
      recentResults: this.trainingHistory.slice(-10)
    };
  }
}

window.TrainingManager = TrainingManager;
