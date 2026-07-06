/**
 * 测试训练后的神经网络 vs 启发式MCTS
 * 对弈N局，统计胜率
 */
const fs = require('fs');

eval(fs.readFileSync('js/game.js', 'utf8').replace(/window\./g, 'global.'));
eval(fs.readFileSync('js/neural-net.js', 'utf8').replace(/window\./g, 'global.'));
eval(fs.readFileSync('js/mcts.js', 'utf8').replace(/window\./g, 'global.'));

const GAMES = 20;
const NN_SIMS = 40;    // 神经网络方的MCTS模拟次数
const HEU_SIMS = 40;   // 启发式方的MCTS模拟次数

const nn = new NeuralNetwork();
if (fs.existsSync('model-weights.json')) {
  nn.importWeights(fs.readFileSync('model-weights.json', 'utf8'));
  console.log('✅ 已加载训练模型\n');
} else {
  console.log('❌ 未找到模型权重');
  process.exit(1);
}

let nnBlackWins = 0, nnWhiteWins = 0, heuBlackWins = 0, heuWhiteWins = 0, draws = 0;

console.log(`🧠 测试: 神经网络 vs 启发式MCTS | ${GAMES}局 | ${NN_SIMS}/${HEU_SIMS}次模拟\n`);

for (let g = 1; g <= GAMES; g++) {
  const game = new GomokuGame();
  let moves = 0;
  // 偶数局NN执黑，奇数局NN执白
  const nnPlaysBlack = g % 2 === 1;

  while (!game.gameOver && moves < 225) {
    const isNNTurn = (game.currentPlayer === BLACK && nnPlaysBlack) || 
                     (game.currentPlayer === WHITE && !nnPlaysBlack);
    
    let move;
    if (isNNTurn) {
      const mcts = new MCTS(nn, NN_SIMS, 3.0);
      mcts.useNN = true;
      const result = mcts.run(game);
      move = result.move;
    } else {
      const mcts = new MCTS(null, HEU_SIMS, 3.0);
      mcts.useNN = false;
      const result = mcts.run(game);
      move = result.move;
    }
    
    if (!move) break;
    game.makeMove(move[0], move[1]);
    moves++;
  }

  const nnWins = (game.winner === BLACK && nnPlaysBlack) || (game.winner === WHITE && !nnPlaysBlack);
  const heuWins = (game.winner === BLACK && !nnPlaysBlack) || (game.winner === WHITE && nnPlaysBlack);
  
  if (nnWins) {
    if (nnPlaysBlack) nnBlackWins++; else nnWhiteWins++;
  } else if (heuWins) {
    if (nnPlaysBlack) heuBlackWins++; else heuWhiteWins++;
  } else draws++;

  const nnTotal = nnBlackWins + nnWhiteWins;
  const heuTotal = heuBlackWins + heuWhiteWins;
  const w = game.winner === BLACK ? '●' : game.winner === WHITE ? '○' : '—';
  const nnSide = nnPlaysBlack ? '●' : '○';
  process.stdout.write(`\r[${g}/${GAMES}] ${w}${moves}步 NN${nnSide}:${nnTotal}胜 启发:${heuTotal}胜 平:${draws}   `);
}

console.log(`
╔══════════════════════════════════════════════╗
║           测试结果                            ║
╠══════════════════════════════════════════════╣
║  总局数:  ${String(GAMES).padEnd(6)}                            ║
║  NN胜:    ${String(nnBlackWins + nnWhiteWins).padEnd(6)} (${((nnBlackWins + nnWhiteWins)/GAMES*100).toFixed(1)}%)                       ║
║  启发胜:  ${String(heuBlackWins + heuWhiteWins).padEnd(6)} (${((heuBlackWins + heuWhiteWins)/GAMES*100).toFixed(1)}%)                       ║
║  平局:    ${String(draws).padEnd(6)} (${(draws/GAMES*100).toFixed(1)}%)                       ║
║  NN执黑胜: ${String(nnBlackWins).padEnd(5)}                           ║
║  NN执白胜: ${String(nnWhiteWins).padEnd(5)}                           ║
╚══════════════════════════════════════════════╝
`);
