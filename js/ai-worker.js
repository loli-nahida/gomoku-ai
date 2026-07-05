/**
 * AI Web Worker - 在独立线程中运行MCTS
 */

// 加载依赖
importScripts('neural-net.js', 'game.js', 'mcts.js');

let network = null;
let initialized = false;

self.onmessage = function(e) {
  const { type, data } = e.data;

  switch(type) {
    case 'init':
      network = new NeuralNetwork();
      if (data.weights) {
        try {
          network.importWeights(data.weights);
          initialized = true;
          self.postMessage({ type: 'ready', success: true });
        } catch(err) {
          self.postMessage({ type: 'ready', success: false, error: err.message });
        }
      } else {
        network.init();
        initialized = true;
        self.postMessage({ type: 'ready', success: true });
      }
      break;

    case 'move':
      if (!initialized) {
        self.postMessage({ type: 'error', error: 'Network not initialized' });
        return;
      }
      
      const { boardState, simulations, useNN, playerColor } = data;
      
      // 重建游戏状态
      const game = new GomokuGame();
      for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
          game.board[r][c] = boardState[r][c];
        }
      }
      // 重建moveHistory和状态
      game.currentPlayer = data.currentPlayer;
      game.moveCount = data.moveCount;
      game.moveHistory = data.moveHistory || [];
      game.gameOver = false;
      game.winner = null;
      game.winLine = null;
      
      const startTime = Date.now();
      
      // 分块运行MCTS，每块后发进度
      const mcts = new MCTS(network, simulations, 3.0);
      mcts.useNN = useNN && initialized;
      
      // 直接运行（Worker中不需要yield）
      const result = mcts.run(game);
      const elapsed = (Date.now() - startTime) / 1000;
      
      // 获取NN评估
      let nnEval = 0;
      if (useNN && initialized) {
        const input = game.toTensorInput(game.currentPlayer);
        const pred = network.predict(input);
        nnEval = pred.value;
      }
      
      self.postMessage({
        type: 'result',
        move: result.move,
        stats: {
          time: elapsed.toFixed(2),
          simulations: simulations,
          nnEval: nnEval.toFixed(3),
          winRate: ((result.rootValue + 1) / 2 * 100).toFixed(1)
        }
      });
      break;
  }
};
