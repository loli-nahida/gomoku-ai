/**
 * SGF Parser for Gomoku - Compact Binary Output
 * 
 * Output format (.bin):
 *   Header: uint32 numSamples
 *   Per sample (fixed size):
 *     input[1350]: uint8 (0 or 1, 6 planes * 15 * 15)
 *     targetPolicy[225]: uint8 (one-hot, index of move)
 *     targetValue: int8 (-1, 0, or 1)
 * 
 * Total per sample: 1350 + 1 + 1 = 1352 bytes
 * 241k samples ≈ 313 MB (much more manageable)
 */

const fs = require('fs');
const path = require('path');

const SGF_DIR = path.join(__dirname, 'sgf-data', 'sgf');
const OUTPUT_BIN = path.join(__dirname, 'training-data.bin');
const OUTPUT_META = path.join(__dirname, 'training-data-meta.json');
const BS = 15;
const INPUT_SIZE = 6 * BS * BS; // 1350
const POLICY_SIZE = BS * BS;     // 225
const SAMPLE_SIZE = INPUT_SIZE + 1 + 1; // 1352 bytes

// GBK hex values for RE field
const RE_HEX = {
  'b0d7caa4': 2,  // 白胜 -> WHITE wins
  'badacaa4': 1,  // 黑胜 -> BLACK wins
  'bacdc6e5': 0,  // 和棋 -> Draw
};

function parseSGFFromBuffer(buffer) {
  const reIdx = buffer.indexOf(Buffer.from('RE['));
  if (reIdx < 0) return null;
  const reEnd = buffer.indexOf(Buffer.from(']'), reIdx);
  if (reEnd < 0) return null;

  const reBytes = buffer.slice(reIdx + 3, reEnd);
  const hex = reBytes.toString('hex');
  const winner = RE_HEX[hex];
  if (winner === undefined) return null;

  const text = buffer.toString('latin1');
  const moves = [];
  const moveRegex = /;([BW])\[([a-o]{2})\]/g;
  let match;
  while ((match = moveRegex.exec(text)) !== null) {
    const col = match[2].charCodeAt(0) - 97;
    const row = match[2].charCodeAt(1) - 97;
    if (row >= 0 && row < BS && col >= 0 && col < BS) {
      moves.push({ color: match[1] === 'B' ? 1 : 2, row, col });
    }
  }

  if (moves.length < 10) return null;
  if (moves[0].color !== 1) return null;

  return { moves, winner };
}

function generateInputPlanes(board, currentPlayer, moves, moveIndex) {
  const planes = new Uint8Array(INPUT_SIZE);
  const opp = currentPlayer === 1 ? 2 : 1;
  const offset2 = BS * BS;

  for (let r = 0; r < BS; r++) {
    for (let c = 0; c < BS; c++) {
      const idx = r * BS + c;
      if (board[r][c] === currentPlayer) planes[idx] = 1;
      else if (board[r][c] === opp) planes[offset2 + idx] = 1;
    }
  }

  if (moveIndex > 0) {
    const m = moves[moveIndex - 1];
    planes[2 * offset2 + m.row * BS + m.col] = 1;
  }
  if (moveIndex > 1) {
    const m = moves[moveIndex - 2];
    planes[3 * offset2 + m.row * BS + m.col] = 1;
  }

  for (let r = 0; r < BS; r++) {
    for (let c = 0; c < BS; c++) {
      if (board[r][c] === 0) {
        if (evaluateThreat(board, r, c, currentPlayer) > 0.3) {
          planes[4 * offset2 + r * BS + c] = 1;
        }
        planes[5 * offset2 + r * BS + c] = 1;
      }
    }
  }

  return planes;
}

function evaluateThreat(board, r, c, player) {
  const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
  let maxThreat = 0;
  for (const [dr, dc] of dirs) {
    let count = 0, open = 0;
    for (let d = 1; d <= 4; d++) {
      const nr = r + dr * d, nc = c + dc * d;
      if (nr < 0 || nr >= BS || nc < 0 || nc >= BS) break;
      if (board[nr][nc] === player) count++;
      else if (board[nr][nc] === 0) { open++; break; } else break;
    }
    for (let d = 1; d <= 4; d++) {
      const nr = r - dr * d, nc = c - dc * d;
      if (nr < 0 || nr >= BS || nc < 0 || nc >= BS) break;
      if (board[nr][nc] === player) count++;
      else if (board[nr][nc] === 0) { open++; break; } else break;
    }
    let threat = 0;
    if (count >= 4) threat = 1.0;
    else if (count === 3 && open === 2) threat = 0.8;
    else if (count === 3 && open === 1) threat = 0.5;
    else if (count === 2 && open === 2) threat = 0.3;
    else if (count === 2 && open === 1) threat = 0.15;
    maxThreat = Math.max(maxThreat, threat);
  }
  return maxThreat;
}

async function main() {
  console.log('=== SGF Parser for Gomoku Training Data (Binary) ===');

  const files = fs.readdirSync(SGF_DIR).filter(f => f.endsWith('.sgf'));
  console.log(`Found ${files.length} SGF files`);

  const fd = fs.openSync(OUTPUT_BIN, 'w');
  // Write placeholder header (4 bytes)
  const headerBuf = Buffer.alloc(4);
  fs.writeSync(fd, headerBuf, 0, 4);

  let parsed = 0, skipped = 0, totalSamples = 0;
  const sampleBuf = Buffer.alloc(SAMPLE_SIZE);

  for (const file of files) {
    try {
      const buffer = fs.readFileSync(path.join(SGF_DIR, file));
      const result = parseSGFFromBuffer(buffer);
      if (!result) { skipped++; continue; }

      const { moves, winner } = result;
      const board = Array.from({ length: BS }, () => new Uint8Array(BS));

      for (let i = 0; i < moves.length; i++) {
        const move = moves[i];
        const currentPlayer = move.color;
        const input = generateInputPlanes(board, currentPlayer, moves, i);

        // Pack into buffer: input[1350] + policyIdx[1] + value[1]
        input.copy(sampleBuf, 0);
        sampleBuf[INPUT_SIZE] = move.row * BS + move.col; // policy target index
        
        let targetValue;
        if (winner === 0) targetValue = 0;
        else if (winner === currentPlayer) targetValue = 1;
        else targetValue = -1;
        sampleBuf[INPUT_SIZE + 1] = targetValue + 1; // 0=loss, 1=draw, 2=win

        fs.writeSync(fd, sampleBuf);
        totalSamples++;

        board[move.row][move.col] = currentPlayer;
      }

      parsed++;
      if (parsed % 1000 === 0) {
        console.log(`  Processed ${parsed} games, ${totalSamples} samples...`);
      }
    } catch (e) {
      skipped++;
    }
  }

  // Write actual sample count in header
  fs.writeSync(fd, Buffer.alloc(0)); // flush
  fs.closeSync(fd);

  // Rewrite header with actual count
  const fd2 = fs.openSync(OUTPUT_BIN, 'r+');
  const countBuf = Buffer.alloc(4);
  countBuf.writeUInt32LE(totalSamples, 0);
  fs.writeSync(fd2, countBuf, 0, 4, 0);
  fs.closeSync(fd2);

  const fileSize = fs.statSync(OUTPUT_BIN).size;
  console.log(`\nResults:`);
  console.log(`  Parsed: ${parsed} games`);
  console.log(`  Skipped: ${skipped} games`);
  console.log(`  Total training samples: ${totalSamples}`);
  console.log(`  Output: ${OUTPUT_BIN} (${(fileSize / 1024 / 1024).toFixed(1)} MB)`);
  console.log(`  Sample size: ${SAMPLE_SIZE} bytes`);
  console.log(`\nDone!`);

  // Write metadata
  fs.writeFileSync(OUTPUT_META, JSON.stringify({
    numSamples: totalSamples,
    sampleSize: SAMPLE_SIZE,
    inputSize: INPUT_SIZE,
    policySize: POLICY_SIZE,
    boardSize: BS,
    inputPlanes: 6,
    description: 'Binary training data: input[1350](uint8, 0/1) + policyIdx(uint8, 0-224) + value(uint8, 0=loss/1=draw/2=win)'
  }, null, 2));
}

main().catch(e => { console.error(e); process.exit(1); });
