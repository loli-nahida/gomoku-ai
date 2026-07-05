/**
 * Node.js 快速自我对弈训练脚本 v3
 * 仅收集启发式MCTS对弈数据，用于后续训练
 */

const fs = require('fs');

eval(fs.readFileSync('js/game.js', 'utf8').replace(/window\./g, 'global.'));
eval(fs.readFileSync('js/neural-net.js', 'utf8').replace(/window\./g, 'global.'));
eval(fs.readFileSync('js/mcts.js', 'utf8').replace(/window\./g, 'global.'));

const TOTAL_GAMES = parseInt(process.argv[2]) || 50;
const SIMS = parseInt(process.argv[3]) || 30;
const SAVE_PATH = 'model-weights.json';
const DATA_PATH = 'training-data.json';

const nn = new NeuralNetwork();
nn.init();

// Load existing model
if (fs.existsSync(SAVE_PATH)) {
  try {
    nn.importWeights(fs.readFileSync(SAVE_PATH, 'utf8'));
    console.log('✅ 已加载预训练模型\n');
  } catch (e) {
    console.log('⚠️  从零开始\n');
  }
}

let blackWins = 0, whiteWins = 0, draws = 0, totalMoves = 0;
const trainingData = [];
const startTime = Date.now();

console.log(`🧠 训练开始 | ${TOTAL_GAMES}局 | ${SIMS}次模拟/步\n`);

for (let g = 1; g <= TOTAL_GAMES; g++) {
  const game = new GomokuGame();
  const history = [];
  let moves = 0;

  while (!game.gameOver && moves < 225) {
    const player = game.currentPlayer;
    const input = game.toTensorInput(player);

    const mcts = new MCTS(null, SIMS, 3.0);
    mcts.useNN = false;
    const result = mcts.run(game);

    history.push({ input, policy: result.policy, player });

    const validMoves = game.getValidMoves();
    let move = result.move;
    if (Math.random() < 0.2 && validMoves.length > 1) {
      move = validMoves[Math.floor(Math.random() * validMoves.length)];
    }
    game.makeMove(move[0], move[1]);
    moves++;
  }

  totalMoves += moves;
  if (game.winner === BLACK) blackWins++;
  else if (game.winner === WHITE) whiteWins++;
  else draws++;

  for (const entry of history) {
    const value = game.winner === 0 ? 0 : (game.winner === entry.player ? 1 : -1);
    trainingData.push({ input: entry.input, targetPolicy: entry.policy, targetValue: value });
  }

  const elapsed = (Date.now() - startTime) / 1000;
  const bar = '█'.repeat(Math.round(g / TOTAL_GAMES * 30)) + '░'.repeat(30 - Math.round(g / TOTAL_GAMES * 30));
  const w = game.winner === BLACK ? '●' : game.winner === WHITE ? '○' : '—';
  process.stdout.write(`\r[${bar}] ${g}/${TOTAL_GAMES} ${w}${moves}步 ●${blackWins} ○${whiteWins} —${draws} ${(g/elapsed).toFixed(1)}局/s`);
}

console.log('\n\n💾 保存模型和训练数据...');

// Save model
fs.writeFileSync(SAVE_PATH, nn.exportWeights());

// Save training metadata
fs.writeFileSync(DATA_PATH, JSON.stringify({
  games: TOTAL_GAMES,
  samples: trainingData.length,
  blackWins, whiteWins, draws,
  avgMoves: (totalMoves / TOTAL_GAMES).toFixed(1),
  timestamp: new Date().toISOString()
}, null, 2));

const totalTime = (Date.now() - startTime) / 1000;

console.log(`
╔══════════════════════════════════════════╗
║            训练完成                       ║
╠══════════════════════════════════════════╣
║  局数:    ${String(TOTAL_GAMES).padEnd(6)}                          ║
║  黑胜:    ${String(blackWins).padEnd(6)} (${(blackWins/TOTAL_GAMES*100).toFixed(1)}%)                    ║
║  白胜:    ${String(whiteWins).padEnd(6)} (${(whiteWins/TOTAL_GAMES*100).toFixed(1)}%)                    ║
║  平局:    ${String(draws).padEnd(6)} (${(draws/TOTAL_GAMES*100).toFixed(1)}%)                    ║
║  样本数:  ${String(trainingData.length).padEnd(6)}                          ║
║  耗时:    ${totalTime.toFixed(1).padEnd(6)}s                         ║
║  模型:    ${SAVE_PATH.padEnd(20)}           ║
╚══════════════════════════════════════════╝
`);
