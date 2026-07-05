/**
 * Monte Carlo Tree Search (MCTS) with PUCT
 * AlphaZero风格的蒙特卡洛树搜索
 */

class MCTSNode {
  constructor(parent, prior, move) {
    this.parent = parent;
    this.move = move; // [r, c]
    this.prior = prior; // P(s,a) from neural network
    this.children = new Map(); // action -> MCTSNode
    this.visitCount = 0; // N(s,a)
    this.totalValue = 0; // W(s,a)
    this.meanValue = 0; // Q(s,a) = W/N
    this.isExpanded = false;
  }

  select(cPuct) {
    let bestScore = -Infinity;
    let bestChild = null;
    const sqrtParent = Math.sqrt(this.visitCount);

    for (const [action, child] of this.children) {
      // PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
      const exploration = cPuct * child.prior * sqrtParent / (1 + child.visitCount);
      const score = child.meanValue + exploration;
      if (score > bestScore) {
        bestScore = score;
        bestChild = child;
      }
    }
    return bestChild;
  }

  expand(moveProbs, validMoves) {
    this.isExpanded = true;
    for (const [r, c] of validMoves) {
      const idx = r * BOARD_SIZE + c;
      const prior = moveProbs[idx];
      const action = `${r},${c}`;
      if (!this.children.has(action)) {
        this.children.set(action, new MCTSNode(this, prior, [r, c]));
      }
    }
  }

  backpropagate(value) {
    this.visitCount++;
    this.totalValue += value;
    this.meanValue = this.totalValue / this.visitCount;
    if (this.parent) {
      this.parent.backpropagate(-value); // Zero-sum game
    }
  }
}

class MCTS {
  constructor(network, simulations = 800, cPuct = 3.0) {
    this.network = network;
    this.simulations = simulations;
    this.cPuct = cPuct;
    this.useNN = true;
  }

  run(game) {
    const root = new MCTSNode(null, 1.0, null);

    // Expand root
    const player = game.currentPlayer;
    const input = game.toTensorInput(player);
    let moveProbs, rootValue;

    if (this.useNN && this.network && this.network.initialized) {
      const result = this.network.predict(input);
      moveProbs = result.policy;
      rootValue = result.value;
    } else {
      // Fallback: heuristic-based prior
      moveProbs = this.heuristicPolicy(game, player);
      rootValue = 0;
    }

    const validMoves = game.getValidMoves();
    root.expand(moveProbs, validMoves);

    // Run simulations
    for (let sim = 0; sim < this.simulations; sim++) {
      let node = root;
      const simGame = game.clone();

      // Selection: traverse tree using PUCT
      while (node.isExpanded && node.children.size > 0) {
        node = node.select(this.cPuct);
        simGame.makeMove(node.move[0], node.move[1]);
      }

      // Evaluation
      let value;
      if (simGame.gameOver) {
        if (simGame.winner === player) value = 1;
        else if (simGame.winner === 0) value = 0;
        else value = -1;
      } else {
        // Expand and evaluate
        const simInput = simGame.toTensorInput(simGame.currentPlayer);
        let simProbs;
        if (this.useNN && this.network && this.network.initialized) {
          const result = this.network.predict(simInput);
          simProbs = result.policy;
          // Value is from current player's perspective, flip for the node's parent player
          value = -result.value;
        } else {
          simProbs = this.heuristicPolicy(simGame, simGame.currentPlayer);
          value = this.rollout(simGame, player);
        }
        const simValidMoves = simGame.getValidMoves();
        node.expand(simProbs, simValidMoves);
      }

      // Backpropagation
      node.backpropagate(value);
    }

    // Extract policy from visit counts
    const policy = new Float32Array(BOARD_SIZE * BOARD_SIZE);
    let totalVisits = 0;
    for (const [action, child] of root.children) {
      const [r, c] = child.move;
      policy[r * BOARD_SIZE + c] = child.visitCount;
      totalVisits += child.visitCount;
    }
    if (totalVisits > 0) {
      for (let i = 0; i < policy.length; i++) policy[i] /= totalVisits;
    }

    // Select best move
    let bestMove = null;
    let bestVisits = -1;
    for (const [action, child] of root.children) {
      if (child.visitCount > bestVisits) {
        bestVisits = child.visitCount;
        bestMove = child.move;
      }
    }

    return { move: bestMove, policy, rootValue };
  }

  /**
   * Heuristic policy when NN is not available
   */
  heuristicPolicy(game, player) {
    const policy = new Float32Array(BOARD_SIZE * BOARD_SIZE);
    const validMoves = game.getValidMoves();
    const opp = player === BLACK ? WHITE : BLACK;

    for (const [r, c] of validMoves) {
      let score = 0;
      
      // Offensive: check patterns created by placing here
      score += game.evaluateThreat(r, c, player) * 2;
      
      // Defensive: check opponent threats blocked
      score += game.evaluateThreat(r, c, opp) * 1.8;
      
      // Center preference
      const centerDist = Math.abs(r - 7) + Math.abs(c - 7);
      score += Math.max(0, (14 - centerDist)) * 0.02;
      
      // Slight random noise
      score += Math.random() * 0.05;
      
      policy[r * BOARD_SIZE + c] = Math.max(score, 0.001);
    }

    // Normalize
    let sum = 0;
    for (let i = 0; i < policy.length; i++) sum += policy[i];
    if (sum > 0) {
      for (let i = 0; i < policy.length; i++) policy[i] /= sum;
    }

    return policy;
  }

  /**
   * Random rollout for when NN is disabled
   */
  rollout(game, perspectivePlayer) {
    const simGame = game.clone();
    let moves = 0;
    while (!simGame.gameOver && moves < 100) {
      const validMoves = simGame.getValidMoves();
      if (validMoves.length === 0) break;
      
      // Smart rollout: pick moves that create/break threats
      let bestMove = null;
      let bestScore = -1;
      for (const [r, c] of validMoves) {
        let s = simGame.evaluateThreat(r, c, simGame.currentPlayer);
        s += simGame.evaluateThreat(r, c, simGame.currentPlayer === BLACK ? WHITE : BLACK) * 0.8;
        s += Math.random() * 0.3;
        if (s > bestScore) {
          bestScore = s;
          bestMove = [r, c];
        }
      }
      simGame.makeMove(bestMove[0], bestMove[1]);
      moves++;
    }

    if (simGame.winner === perspectivePlayer) return 1;
    if (simGame.winner === 0) return 0;
    return -1;
  }
}

window.MCTS = MCTS;
window.MCTSNode = MCTSNode;
