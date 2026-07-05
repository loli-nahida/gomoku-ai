/**
 * 主控制器 v2 - 绑定所有模块，管理游戏流程
 */

(function () {
  'use strict';

  // === State ===
  let game = new GomokuGame();
  let ui = new GomokuUI('board-canvas');
  let ai = new GomokuAI();
  let trainer = new TrainingManager(ai);
  let playerColor = BLACK;
  let aiColor = WHITE;
  let isPlayerTurn = true;
  let isAiThinking = false;
  let showHeatmap = false;
  let lastPolicy = null;
  let stats = { playerWins: 0, aiWins: 0, draws: 0 };

  // === DOM Elements ===
  const $ = (id) => document.getElementById(id);
  const $status = $('game-status');
  const $turnIndicator = $('turn-indicator');
  const $moveCount = $('move-count');
  const $aiTime = $('ai-time');
  const $nnEval = $('nn-eval');
  const $winRate = $('win-rate');
  const $moveHistory = $('move-history');
  const $difficulty = $('difficulty');
  const $playerColor = $('player-color');
  const $useNN = $('use-nn');
  const $btnNewGame = $('btn-new-game');
  const $btnUndo = $('btn-undo');
  const $btnTrain = $('btn-train');
  const $btnHeatmap = $('btn-heatmap');
  const $thinkingOverlay = $('thinking-overlay');
  const $victoryModal = $('victory-modal');
  const $victoryIcon = $('victory-icon');
  const $victoryText = $('victory-text');
  const $victoryDetail = $('victory-detail');
  const $btnPlayAgain = $('btn-play-again');
  const $playerWins = $('player-wins');
  const $aiWins = $('ai-wins');
  const $draws = $('draws');
  const $nnStatus = $('nn-status');
  const $trainStatus = $('train-status');
  const $trainProgress = $('train-progress');
  const $trainGames = $('train-games');
  const $trainBar = $('train-bar');

  // === Init ===
  function init() {
    loadStats();
    loadModel();
    setupCallbacks();
    setupControls();
    resizeBoard();
    newGame();
    updateNnStatus();
    window.addEventListener('resize', debounce(resizeBoard, 200));
  }

  function setupCallbacks() {
    ui.onClickHandler((r, c) => {
      if (!isPlayerTurn || isAiThinking || game.gameOver) return;
      if (!game.isValidMove(r, c)) return;
      playerMove(r, c);
    });
  }

  function setupControls() {
    $btnNewGame.addEventListener('click', newGame);
    $btnUndo.addEventListener('click', undoMove);
    $btnPlayAgain.addEventListener('click', () => {
      $victoryModal.classList.add('hidden');
      newGame();
    });

    $difficulty.addEventListener('change', () => {
      ai.setDifficulty($difficulty.value);
    });

    $playerColor.addEventListener('change', () => {
      playerColor = $playerColor.value === 'black' ? BLACK : WHITE;
      aiColor = playerColor === BLACK ? WHITE : BLACK;
      newGame();
    });

    $useNN.addEventListener('change', () => {
      ai.setUseNN($useNN.checked);
    });

    // Heatmap toggle
    if ($btnHeatmap) {
      $btnHeatmap.addEventListener('click', () => {
        showHeatmap = !showHeatmap;
        $btnHeatmap.classList.toggle('active', showHeatmap);
        $btnHeatmap.textContent = showHeatmap ? '🔥 隐藏热力图' : '🔥 策略热力图';
        ui.setShowHeatmap(showHeatmap);
        ui.setHeatmapData(lastPolicy);
        ui.drawBoard(game);
      });
    }

    // Self-training
    if ($btnTrain) {
      $btnTrain.addEventListener('click', () => {
        if (trainer.isTraining) {
          trainer.stop();
          $btnTrain.textContent = '🏋️ 自我训练';
          updateTrainStatus('训练已停止');
        } else {
          startTraining();
        }
      });
    }

    // Close modal on outside click
    $victoryModal.addEventListener('click', (e) => {
      if (e.target === $victoryModal) $victoryModal.classList.add('hidden');
    });
  }

  // === Board resize ===
  function resizeBoard() {
    const container = document.querySelector('.board-container');
    if (!container) return;
    const maxSize = Math.min(window.innerWidth - 580, window.innerHeight - 200, 640);
    const size = Math.max(360, maxSize);
    ui.resize(size);
    ui.drawBoard(game);
  }

  function newGame() {
    game.reset();
    playerColor = $playerColor.value === 'black' ? BLACK : WHITE;
    aiColor = playerColor === BLACK ? WHITE : BLACK;
    isPlayerTurn = playerColor === BLACK;
    isAiThinking = false;
    lastPolicy = null;
    hideThinking();
    ui.setHeatmapData(null);
    ui.drawBoard(game);
    updateStatus();
    updateMoveHistory();

    if (!isPlayerTurn) {
      setTimeout(() => aiMove(), 300);
    }
  }

  // === Player Move ===
  function playerMove(r, c) {
    game.makeMove(r, c);
    lastPolicy = null;
    ui.setHeatmapData(null);
    ui.drawBoard(game);
    updateStatus();
    updateMoveHistory();
    updateStats();

    if (game.gameOver) {
      handleGameEnd();
      return;
    }

    isPlayerTurn = false;
    setTimeout(() => aiMove(), 100);
  }

  // === AI Move ===
  function aiMove() {
    if (game.gameOver) return;
    isAiThinking = true;
    showThinking();

    // 用setTimeout让UI先更新，再执行AI计算
    setTimeout(() => {
      try {
        const result = ai.getBestMove(game);
        if (!result || !result.move) {
          isAiThinking = false;
          hideThinking();
          return;
        }

        const [r, c] = result.move;
        game.makeMove(r, c);
        ui.drawBoard(game);
        updateStatus();
        updateMoveHistory();
        updateAiStats(result.stats);

        if (game.gameOver) {
          handleGameEnd();
          isAiThinking = false;
          hideThinking();
          return;
        }

        isPlayerTurn = true;
      } catch (err) {
        console.error('AI error:', err);
        const move = ai.getQuickMove(game);
        if (move) {
          game.makeMove(move[0], move[1]);
          ui.drawBoard(game);
          updateStatus();
          updateMoveHistory();
        }
        isPlayerTurn = true;
      }

      isAiThinking = false;
      hideThinking();
    }, 50);
  }

  // === Undo ===
  function undoMove() {
    if (isAiThinking || game.moveHistory.length < 2) return;
    game.undoMove();
    game.undoMove();
    isPlayerTurn = true;
    lastPolicy = null;
    ui.setHeatmapData(null);
    ui.drawBoard(game);
    updateStatus();
    updateMoveHistory();
    $aiTime.textContent = '-';
    $nnEval.textContent = '-';
    $winRate.textContent = '-';
  }

  // === Training ===
  async function startTraining() {
    $btnTrain.textContent = '⏹ 停止训练';
    updateTrainStatus('自我对弈训练中...');

    trainer.onProgress = (info) => {
      const pct = Math.round((info.game / info.total) * 100);
      updateTrainProgress(pct, info.game);
      const winner = info.result.winner === 0 ? '平' : (info.result.winner === BLACK ? '黑' : '白');
      updateTrainStatus(`第 ${info.game} 局完成 (${winner}胜, ${info.result.moveCount}步)`);
    };

    trainer.onComplete = (info) => {
      $btnTrain.textContent = '🏋️ 自我训练';
      updateTrainStatus(`训练完成: ${info.games} 局`);
      saveModel();
      updateNnStatus();
    };

    await trainer.train(10, 80);
    saveModel();
  }

  // === Persistence ===
  function saveModel() {
    try {
      const weights = ai.network.exportWeights();
      localStorage.setItem('gomoku-nn-weights', weights);
      localStorage.setItem('gomoku-training-games', trainer.trainingGames);
    } catch (e) {
      console.warn('Failed to save model:', e);
    }
  }

  function loadModel() {
    try {
      const weights = localStorage.getItem('gomoku-nn-weights');
      if (weights) {
        ai.network.importWeights(weights);
        const games = localStorage.getItem('gomoku-training-games');
        if (games) trainer.trainingGames = parseInt(games);
        console.log('Model loaded from localStorage');
      }
    } catch (e) {
      console.warn('Failed to load model:', e);
    }
  }

  function saveStats() {
    try {
      localStorage.setItem('gomoku-stats', JSON.stringify(stats));
    } catch (e) {}
  }

  function loadStats() {
    try {
      const saved = localStorage.getItem('gomoku-stats');
      if (saved) {
        stats = JSON.parse(saved);
        updateWinRecord();
      }
    } catch (e) {}
  }

  // === Game End ===
  function handleGameEnd() {
    let icon, text, detail;

    if (game.winner === playerColor) {
      stats.playerWins++;
      icon = '🎉';
      text = '恭喜获胜！';
      detail = `你用 ${game.moveCount} 步击败了 AI`;
    } else if (game.winner === aiColor) {
      stats.aiWins++;
      icon = '🤖';
      text = 'AI 获胜';
      detail = `AI 在第 ${game.moveCount} 步获胜`;
    } else {
      stats.draws++;
      icon = '🤝';
      text = '平局';
      detail = '棋盘已满，双方握手言和';
    }

    updateWinRecord();
    saveStats();

    setTimeout(() => {
      $victoryIcon.textContent = icon;
      $victoryText.textContent = text;
      $victoryDetail.textContent = detail;
      $victoryModal.classList.remove('hidden');
    }, 600);
  }

  // === UI Updates ===
  function updateStatus() {
    if (game.gameOver) {
      if (game.winner === playerColor) {
        $status.textContent = '🎉 你赢了！';
        $status.style.color = '#4caf50';
      } else if (game.winner === aiColor) {
        $status.textContent = '🤖 AI 获胜';
        $status.style.color = '#f44336';
      } else {
        $status.textContent = '🤝 平局';
        $status.style.color = '#ff9800';
      }
    } else if (isPlayerTurn) {
      $status.textContent = '轮到你了';
      $status.style.color = '#e0e0e0';
    } else {
      $status.textContent = 'AI 思考中...';
      $status.style.color = '#6c63ff';
    }

    const icon = game.currentPlayer === BLACK ? 'black-stone-icon' : 'white-stone-icon';
    const colorName = game.currentPlayer === BLACK ? '黑棋' : '白棋';
    $turnIndicator.innerHTML = `<span class="stone ${icon}"></span> ${colorName}行棋 · 第 ${game.moveCount + 1} 手`;
  }

  function updateMoveHistory() {
    if (game.moveHistory.length === 0) {
      $moveHistory.innerHTML = '<div class="history-empty">暂无落子记录</div>';
      return;
    }

    let html = '';
    for (let i = 0; i < game.moveHistory.length; i++) {
      const { r, c, player } = game.moveHistory[i];
      const colLetter = String.fromCharCode(65 + c);
      const rowNum = BOARD_SIZE - r;
      const stoneClass = player === BLACK ? 'black-stone-icon' : 'white-stone-icon';
      const playerName = player === playerColor ? '你' : 'AI';
      const isLast = i === game.moveHistory.length - 1;

      html += `
        <div class="history-item${isLast ? ' last-move' : ''}">
          <span class="move-num">${i + 1}.</span>
          <span class="move-stone ${stoneClass}"></span>
          <span class="move-coord">${colLetter}${rowNum}</span>
          <span class="move-player">${playerName}</span>
        </div>`;
    }

    $moveHistory.innerHTML = html;
    $moveHistory.scrollTop = $moveHistory.scrollHeight;
  }

  function updateStats() {
    $moveCount.textContent = game.moveCount;
  }

  function updateAiStats(s) {
    $aiTime.textContent = s.time + 's';
    $nnEval.textContent = s.nnEval;
    $winRate.textContent = s.winRate + '%';
  }

  function updateWinRecord() {
    $playerWins.textContent = stats.playerWins;
    $aiWins.textContent = stats.aiWins;
    $draws.textContent = stats.draws;
  }

  function updateNnStatus() {
    if (!$nnStatus) return;
    const statusSpan = $nnStatus.querySelector('span');
    if (ai.network.initialized) {
      const games = trainer.trainingGames;
      statusSpan.className = 'status-ready';
      statusSpan.textContent = `● 就绪 (${games}局训练)`;
    } else {
      statusSpan.className = 'status-loading';
      statusSpan.textContent = '○ 加载中...';
    }
  }

  function updateTrainStatus(text) {
    if ($trainStatus) $trainStatus.textContent = text;
  }

  function updateTrainProgress(pct, games) {
    if ($trainBar) $trainBar.style.width = pct + '%';
    if ($trainGames) $trainGames.textContent = games;
  }

  function showThinking() {
    $thinkingOverlay.classList.remove('hidden');
  }

  function hideThinking() {
    $thinkingOverlay.classList.add('hidden');
  }

  function debounce(fn, ms) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), ms);
    };
  }

  // === Keyboard shortcuts ===
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.key === 'n' || e.key === 'N') newGame();
    if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      undoMove();
    }
    if (e.key === 'h' || e.key === 'H') {
      if ($btnHeatmap) $btnHeatmap.click();
    }
  });

  // === Start ===
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
