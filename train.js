/**
 * Node.js 自我对弈训练脚本 v4 - 内存优化版
 * 每500局自动保存，及时释放内存
 */

const fs = require('fs');

eval(fs.readFileSync('js/game.js', 'utf8').replace(/window\./g, 'global.'));
eval(fs.readFileSync('js/neural-net.js', 'utf8').replace(/window\./g, 'global.'));
eval(fs.readFileSync('js/mcts.js', 'utf8').replace(/window\./g, 'global.'));

const TOTAL_GAMES = parseInt(process.argv[2]) || 100;
const SIMS = parseInt(process.argv[3]) || 30;
const SAVE_PATH = 'model-weights.json';
const SAVE_INTERVAL = 500;

const nn = new NeuralNetwork();
nn.init();

if (fs.existsSync(SAVE_PATH)) {
  try {
    nn.importWeights(fs.readFileSync(SAVE_PATH, 'utf8'));
    console.log('✅ 已加载预训练模型\n');
  } catch (e) {
    console.log('⚠️  从零开始\n');
  }
}

let blackWins = 0, whiteWins = 0, draws = 0, totalMoves = 0;
let totalSamples = 0;
const startTime = Date.now();

console.log(`🧠 训练开始 | ${TOTAL_GAMES}局 | ${SIMS}次模拟/步\n`);

function save() {
  fs.writeFileSync(SAVE_PATH, nn.exportWeights());
  global.gc && global.gc();
}

for (let g = 1; g <= TOTAL_GAMES; g++) {
  const game = new GomokuGame();
  let moves = 0;

  while (!game.gameOver && moves < 225) {
    const mcts = new MCTS(null, SIMS, 3.0);
    mcts.useNN = false;
    const result = mcts.run(game);
    const validMoves = game.getValidMoves();
    let move = result.move;
    if (Math.random() < 0.15 && validMoves.length > 1) {
      move = validMoves[Math.floor(Math.random() * validMoves.length)];
    }
    game.makeMove(move[0], move[1]);
    moves++;
    totalSamples++;
  }

  totalMoves += moves;
  if (game.winner === BLACK) blackWins++;
  else if (game.winner === WHITE) whiteWins++;
  else draws++;

  // 进度条
  const elapsed = (Date.now() - startTime) / 1000;
  const bar = '█'.repeat(Math.round(g / TOTAL_GAMES * 30)) + '░'.repeat(30 - Math.round(g / TOTAL_GAMES * 30));
  const w = game.winner === BLACK ? '●' : game.winner === WHITE ? '○' : '—';
  process.stdout.write(`\r[${bar}] ${g}/${TOTAL_GAMES} ${w}${moves}步 ●${blackWins} ○${whiteWins} —${draws} ${(g/elapsed).toFixed(1)}局/s`);

  // 定期保存
  if (g % SAVE_INTERVAL === 0 || g === TOTAL_GAMES) {
    save();
  }
}

const totalTime = (Date.now() - startTime) / 1000;
save();

console.log(`
╔══════════════════════════════════════════╗
║            训练完成                       ║
╠══════════════════════════════════════════╣
║  局数:    ${String(TOTAL_GAMES).padEnd(6)}                          ║
║  黑胜:    ${String(blackWins).padEnd(6)} (${(blackWins/TOTAL_GAMES*100).toFixed(1)}%)                    ║
║  白胜:    ${String(whiteWins).padEnd(6)} (${(whiteWins/TOTAL_GAMES*100).toFixed(1)}%)                    ║
║  平局:    ${String(draws).padEnd(6)} (${(draws/TOTAL_GAMES*100).toFixed(1)}%)                    ║
║  样本数:  ${String(totalSamples).padEnd(6)}                          ║
║  耗时:    ${totalTime.toFixed(1).padEnd(6)}s                         ║
║  模型:    ${SAVE_PATH.padEnd(20)}           ║
╚══════════════════════════════════════════╝
`);
