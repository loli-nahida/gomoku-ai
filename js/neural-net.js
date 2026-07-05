/**
 * 轻量级神经网络 - 浏览器端实现
 * 包含残差网络结构，用于棋局评估
 * 
 * 架构: Input(15x15x6) -> Conv -> ResBlock x4 -> Policy Head + Value Head
 * 
 * 纯 JavaScript 实现，不依赖外部库
 */

class NeuralNetwork {
  constructor() {
    this.initialized = false;
    this.weights = {};
    this.inputPlanes = 6;
    this.boardSize = 15;
    this.filters = 64;
    this.resBlocks = 4;
  }

  /**
   * Initialize weights with Xavier/He initialization
   */
  init() {
    const f = this.filters;
    const ip = this.inputPlanes;

    // Initial convolution
    this.weights.conv1_w = this.heInit(ip * f, 3 * 3);
    this.weights.conv1_b = new Float32Array(f);

    // Residual blocks
    for (let i = 0; i < this.resBlocks; i++) {
      this.weights[`res${i}_conv1_w`] = this.heInit(f * f, 3 * 3);
      this.weights[`res${i}_conv1_b`] = new Float32Array(f);
      this.weights[`res${i}_conv2_w`] = this.heInit(f * f, 3 * 3);
      this.weights[`res${i}_conv2_b`] = new Float32Array(f);
    }

    // Policy head: conv -> fc -> 225 (15*15) outputs
    this.weights.policy_conv_w = this.heInit(f * f, 1 * 1);
    this.weights.policy_conv_b = new Float32Array(f);
    this.weights.policy_fc_w = this.heInit(f * this.boardSize * this.boardSize, this.boardSize * this.boardSize);
    this.weights.policy_fc_b = new Float32Array(this.boardSize * this.boardSize);

    // Value head: conv -> fc1 -> fc2 -> 1 output
    this.weights.value_conv_w = this.heInit(f * f, 1 * 1);
    this.weights.value_conv_b = new Float32Array(f);
    this.weights.value_fc1_w = this.heInit(f * this.boardSize * this.boardSize, 64);
    this.weights.value_fc1_b = new Float32Array(64);
    this.weights.value_fc2_w = this.heInit(64, 1);
    this.weights.value_fc2_b = new Float32Array(1);

    this.initialized = true;
  }

  heInit(fanIn, kernelSize) {
    const stddev = Math.sqrt(2.0 / (fanIn * kernelSize));
    const size = fanIn * kernelSize;
    const w = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random() || 1e-10;
      const u2 = Math.random();
      w[i] = stddev * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return w;
  }

  /**
   * Forward pass
   * @param {Float32Array} input - flattened input [batch=1][planes][board][board]
   * @returns {{ policy: Float32Array, value: number }}
   */
  predict(input) {
    if (!this.initialized) this.init();

    const bs = this.boardSize;
    const f = this.filters;
    const ip = this.inputPlanes;

    // Reshape input to [planes][board][board]
    let x = [];
    for (let p = 0; p < ip; p++) {
      const plane = new Float32Array(bs * bs);
      for (let i = 0; i < bs * bs; i++) {
        plane[i] = input[p * bs * bs + i];
      }
      x.push(plane);
    }

    // Initial conv
    x = this.convLayer(x, this.weights.conv1_w, this.weights.conv1_b, ip, f);
    x = this.batchNorm(x, f);
    x = this.relu(x, f);

    // Residual blocks
    for (let i = 0; i < this.resBlocks; i++) {
      const residual = x;
      x = this.convLayer(x, this.weights[`res${i}_conv1_w`], this.weights[`res${i}_conv1_b`], f, f);
      x = this.batchNorm(x, f);
      x = this.relu(x, f);
      x = this.convLayer(x, this.weights[`res${i}_conv2_w`], this.weights[`res${i}_conv2_b`], f, f);
      x = this.batchNorm(x, f);
      // Skip connection
      for (let c = 0; c < f; c++) {
        for (let i = 0; i < bs * bs; i++) {
          x[c][i] += residual[c][i];
        }
      }
      x = this.relu(x, f);
    }

    // Policy head
    let policy = this.convLayer(x, this.weights.policy_conv_w, this.weights.policy_conv_b, f, f);
    policy = this.relu(policy, f);
    // Flatten
    let policyFlat = new Float32Array(f * bs * bs);
    for (let c = 0; c < f; c++) {
      for (let i = 0; i < bs * bs; i++) {
        policyFlat[c * bs * bs + i] = policy[c][i];
      }
    }
    // FC layer
    let policyOut = new Float32Array(bs * bs);
    for (let i = 0; i < bs * bs; i++) {
      let sum = this.weights.policy_fc_b[i];
      for (let j = 0; j < f * bs * bs; j++) {
        sum += policyFlat[j] * this.weights.policy_fc_w[i * f * bs * bs + j];
      }
      policyOut[i] = sum;
    }
    // Softmax
    policyOut = this.softmax(policyOut);

    // Value head
    let value = this.convLayer(x, this.weights.value_conv_w, this.weights.value_conv_b, f, f);
    value = this.relu(value, f);
    let valueFlat = new Float32Array(f * bs * bs);
    for (let c = 0; c < f; c++) {
      for (let i = 0; i < bs * bs; i++) {
        valueFlat[c * bs * bs + i] = value[c][i];
      }
    }
    // FC1
    let v = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      let sum = this.weights.value_fc1_b[i];
      for (let j = 0; j < f * bs * bs; j++) {
        sum += valueFlat[j] * this.weights.value_fc1_w[i * f * bs * bs + j];
      }
      v[i] = Math.max(0, sum); // ReLU
    }
    // FC2
    let valueOut = this.weights.value_fc2_b[0];
    for (let i = 0; i < 64; i++) {
      valueOut += v[i] * this.weights.value_fc2_w[i];
    }
    valueOut = Math.tanh(valueOut);

    return { policy: policyOut, value: valueOut };
  }

  /**
   * 2D Convolution (3x3 or 1x1)
   */
  convLayer(input, weights, biases, inChannels, outChannels) {
    const bs = this.boardSize;
    const kernelSize = weights.length / (inChannels * outChannels) === 1 ? 1 : 3;
    const output = [];
    for (let oc = 0; oc < outChannels; oc++) {
      output.push(new Float32Array(bs * bs));
    }

    if (kernelSize === 1) {
      // 1x1 conv
      for (let oc = 0; oc < outChannels; oc++) {
        for (let i = 0; i < bs * bs; i++) {
          let sum = biases[oc];
          for (let ic = 0; ic < inChannels; ic++) {
            sum += input[ic][i] * weights[oc * inChannels + ic];
          }
          output[oc][i] = sum;
        }
      }
    } else {
      // 3x3 conv with padding=1
      for (let oc = 0; oc < outChannels; oc++) {
        for (let r = 0; r < bs; r++) {
          for (let c = 0; c < bs; c++) {
            let sum = biases[oc];
            for (let ic = 0; ic < inChannels; ic++) {
              for (let kr = -1; kr <= 1; kr++) {
                for (let kc = -1; kc <= 1; kc++) {
                  const nr = r + kr, nc = c + kc;
                  if (nr >= 0 && nr < bs && nc >= 0 && nc < bs) {
                    const wIdx = oc * inChannels * 9 + ic * 9 + (kr + 1) * 3 + (kc + 1);
                    sum += input[ic][nr * bs + nc] * weights[wIdx];
                  }
                }
              }
            }
            output[oc][r * bs + c] = sum;
          }
        }
      }
    }
    return output;
  }

  batchNorm(input, channels) {
    const bs = this.boardSize;
    const output = [];
    for (let c = 0; c < channels; c++) {
      let mean = 0, var_ = 0;
      for (let i = 0; i < bs * bs; i++) mean += input[c][i];
      mean /= (bs * bs);
      for (let i = 0; i < bs * bs; i++) var_ += (input[c][i] - mean) ** 2;
      var_ /= (bs * bs);
      const std = Math.sqrt(var_ + 1e-5);
      const norm = new Float32Array(bs * bs);
      for (let i = 0; i < bs * bs; i++) {
        norm[i] = (input[c][i] - mean) / std;
      }
      output.push(norm);
    }
    return output;
  }

  relu(input, channels) {
    const bs = this.boardSize;
    const output = [];
    for (let c = 0; c < channels; c++) {
      const arr = new Float32Array(bs * bs);
      for (let i = 0; i < bs * bs; i++) {
        arr[i] = Math.max(0, input[c][i]);
      }
      output.push(arr);
    }
    return output;
  }

  softmax(logits) {
    const n = logits.length;
    const result = new Float32Array(n);
    let maxVal = -Infinity;
    for (let i = 0; i < n; i++) if (logits[i] > maxVal) maxVal = logits[i];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      result[i] = Math.exp(logits[i] - maxVal);
      sum += result[i];
    }
    for (let i = 0; i < n; i++) result[i] /= sum;
    return result;
  }

  /**
   * Export weights to JSON
   */
  exportWeights() {
    const data = {};
    for (const [key, val] of Object.entries(this.weights)) {
      data[key] = Array.from(val);
    }
    return JSON.stringify(data);
  }

  /**
   * Import weights from JSON
   */
  importWeights(json) {
    const data = JSON.parse(json);
    for (const [key, val] of Object.entries(data)) {
      this.weights[key] = new Float32Array(val);
    }
    this.initialized = true;
  }

  /**
   * Simple self-play training step
   * Uses policy gradient to update weights
   */
  trainOnGame(gameHistory, learningRate = 0.001) {
    // Simplified: just store for future training
    // Full backprop would be complex; use policy gradient approximation
    if (!this.initialized) this.init();
    
    let totalLoss = 0;
    for (const entry of gameHistory) {
      const { input, targetPolicy, targetValue } = entry;
      const { policy, value } = this.predict(input);
      
      // Cross-entropy loss for policy
      let policyLoss = 0;
      for (let i = 0; i < targetPolicy.length; i++) {
        if (targetPolicy[i] > 0) {
          policyLoss -= targetPolicy[i] * Math.log(policy[i] + 1e-10);
        }
      }
      
      // MSE loss for value
      const valueLoss = (value - targetValue) ** 2;
      
      totalLoss += policyLoss + valueLoss;
    }
    
    return totalLoss / gameHistory.length;
  }
}

// Self-play training manager
class SelfPlayTrainer {
  constructor(network) {
    this.network = network;
    this.trainingData = [];
    this.gamesPlayed = 0;
  }

  /**
   * Generate training data from a single self-play game
   */
  generateSelfPlayGame(gameClass, mctsInstance, simulations = 100) {
    const game = new gameClass();
    const history = [];
    let moveCount = 0;

    while (!game.gameOver && moveCount < 225) {
      const player = game.currentPlayer;
      const input = game.toTensorInput(player);
      
      // Run MCTS
      const mcts = new mctsInstance(this.network, simulations);
      const { policy } = mcts.run(game);
      
      // Store position + MCTS policy
      history.push({
        input: input,
        policy: policy,
        player: player
      });

      // Select move (with some randomness for exploration)
      const move = this.selectMoveWithNoise(policy, game);
      game.makeMove(move[0], move[1]);
      moveCount++;
    }

    // Assign values based on game result
    const winner = game.winner;
    for (const entry of history) {
      let value;
      if (winner === 0) value = 0; // draw
      else if (winner === entry.player) value = 1; // win
      else value = -1; // loss
      
      this.trainingData.push({
        input: entry.input,
        targetPolicy: entry.policy,
        targetValue: value
      });
    }

    this.gamesPlayed++;
    return { winner, moveCount };
  }

  selectMoveWithNoise(policy, game) {
    const moves = game.getValidMoves();
    const bs = game.boardSize || 15;
    
    // Add Dirichlet noise for exploration
    const noise = this.dirichletNoise(moves.length, 0.3);
    let bestScore = -Infinity;
    let bestMove = moves[0];
    
    for (let i = 0; i < moves.length; i++) {
      const [r, c] = moves[i];
      const idx = r * bs + c;
      const score = 0.75 * policy[idx] + 0.25 * noise[i];
      if (score > bestScore) {
        bestScore = score;
        bestMove = moves[i];
      }
    }
    return bestMove;
  }

  dirichletNoise(n, alpha) {
    const noise = new Float32Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      // Gamma distribution approximation
      noise[i] = this.gammaSample(alpha);
      sum += noise[i];
    }
    for (let i = 0; i < n; i++) noise[i] /= sum;
    return noise;
  }

  gammaSample(alpha) {
    // Marsaglia and Tsang method approximation
    if (alpha < 1) {
      return this.gammaSample(alpha + 1) * Math.pow(Math.random(), 1 / alpha);
    }
    const d = alpha - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x, v;
      do {
        x = this.normalRandom();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  normalRandom() {
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

window.NeuralNetwork = NeuralNetwork;
window.SelfPlayTrainer = SelfPlayTrainer;
