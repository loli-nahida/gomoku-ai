/**
 * AI 引擎 v3 - 纯同步，兼容Termux等低端环境
 */

class GomokuAI {
  constructor() {
    this.network = new NeuralNetwork();
    this.network.init();
    this.useNN = true;
    this.difficulty = 'easy';
    this.searches = {
      easy: 0,         // 纯启发式
      normal: 20,      // 20次模拟
      medium: 50,      // 50次模拟
      hard: 100,       // 100次模拟
      expert: 200,     // 200次模拟
      master: 500,     // 500次模拟
      legend: 1000     // 1000次模拟
    };
  }

  setDifficulty(level) {
    this.difficulty = level;
  }

  setUseNN(use) {
    this.useNN = use;
  }

  /**
   * 获取AI最佳走法（同步）
   */
  getBestMove(game) {
    const startTime = performance.now();
    const simulations = this.searches[this.difficulty] || 0;

    let move;
    let nnEval = 0;

    if (simulations === 0) {
      // 入门：纯启发式，瞬间完成
      move = this.getQuickMove(game);
    } else {
      // 其他难度：跑MCTS
      const mcts = new MCTS(
        this.useNN ? this.network : null,
        simulations,
        3.0
      );
      mcts.useNN = this.useNN;
      const result = mcts.run(game);
      move = result.move;

      if (this.useNN && this.network.initialized) {
        const input = game.toTensorInput(game.currentPlayer);
        const pred = this.network.predict(input);
        nnEval = pred.value;
      }
    }

    const elapsed = (performance.now() - startTime) / 1000;

    return {
      move,
      stats: {
        time: elapsed.toFixed(2),
        simulations,
        nnEval: nnEval.toFixed(3),
        winRate: '-'
      }
    };
  }

  /**
   * 快速启发式走法
   */
  getQuickMove(game) {
    const validMoves = game.getValidMoves();
    if (validMoves.length === 0) return null;

    const player = game.currentPlayer;
    const opp = player === BLACK ? WHITE : BLACK;

    // 1. 检查立即获胜
    for (const [r, c] of validMoves) {
      const g = game.clone();
      g.makeMove(r, c);
      if (g.winner === player) return [r, c];
    }

    // 2. 阻止对手立即获胜
    for (const [r, c] of validMoves) {
      const g = game.clone();
      g.board[r][c] = opp;
      if (g.checkWin(r, c)) return [r, c];
    }

    // 3. 检查活四/冲四
    for (const [r, c] of validMoves) {
      const g = game.clone();
      g.board[r][c] = player;
      const patterns = g.countPatterns(player);
      if (patterns.openFour > 0 || patterns.four > 0) return [r, c];
    }

    // 4. 阻止对手活四/冲四
    for (const [r, c] of validMoves) {
      const g = game.clone();
      g.board[r][c] = opp;
      const patterns = g.countPatterns(opp);
      if (patterns.openFour > 0 || patterns.four > 0) return [r, c];
    }

    // 5. 威胁评估打分
    let bestScore = -1;
    let bestMove = validMoves[0];
    for (const [r, c] of validMoves) {
      let score = game.evaluateThreat(r, c, player) * 2;
      score += game.evaluateThreat(r, c, opp) * 1.8;
      // 中心偏好
      const centerDist = Math.abs(r - 7) + Math.abs(c - 7);
      score += (14 - centerDist) * 0.015;
      // 微小随机扰动
      score += Math.random() * 0.05;
      if (score > bestScore) {
        bestScore = score;
        bestMove = [r, c];
      }
    }
    return bestMove;
  }

  exportWeights() {
    return this.network.exportWeights();
  }

  importWeights(json) {
    this.network.importWeights(json);
  }
}

window.GomokuAI = GomokuAI;
