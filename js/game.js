/**
 * 五子棋游戏核心逻辑
 * Board: 15x15, BLACK=1, WHITE=2, EMPTY=0
 */
const EMPTY = 0, BLACK = 1, WHITE = 2;
const BOARD_SIZE = 15;

class GomokuGame {
  constructor() {
    this.reset();
  }

  reset() {
    this.board = Array.from({ length: BOARD_SIZE }, () => new Uint8Array(BOARD_SIZE));
    this.currentPlayer = BLACK;
    this.moveHistory = [];
    this.gameOver = false;
    this.winner = null;
    this.winLine = null;
    this.moveCount = 0;
  }

  clone() {
    const g = new GomokuGame();
    for (let r = 0; r < BOARD_SIZE; r++) g.board[r] = new Uint8Array(this.board[r]);
    g.currentPlayer = this.currentPlayer;
    g.moveHistory = [...this.moveHistory];
    g.gameOver = this.gameOver;
    g.winner = this.winner;
    g.winLine = this.winLine ? [...this.winLine] : null;
    g.moveCount = this.moveCount;
    return g;
  }

  isValidMove(r, c) {
    return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && this.board[r][c] === EMPTY && !this.gameOver;
  }

  makeMove(r, c) {
    if (!this.isValidMove(r, c)) return false;
    this.board[r][c] = this.currentPlayer;
    this.moveHistory.push({ r, c, player: this.currentPlayer });
    this.moveCount++;

    // Check win
    if (this.checkWin(r, c)) {
      this.gameOver = true;
      this.winner = this.currentPlayer;
    } else if (this.moveCount >= BOARD_SIZE * BOARD_SIZE) {
      this.gameOver = true;
      this.winner = 0; // draw
    }

    this.currentPlayer = this.currentPlayer === BLACK ? WHITE : BLACK;
    return true;
  }

  undoMove() {
    if (this.moveHistory.length === 0) return false;
    const last = this.moveHistory.pop();
    this.board[last.r][last.c] = EMPTY;
    this.currentPlayer = last.player;
    this.moveCount--;
    this.gameOver = false;
    this.winner = null;
    this.winLine = null;
    return true;
  }

  checkWin(r, c) {
    const player = this.board[r][c];
    const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (const [dr, dc] of dirs) {
      const line = [{ r, c }];
      for (let d = 1; d < 5; d++) {
        const nr = r + dr * d, nc = c + dc * d;
        if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE || this.board[nr][nc] !== player) break;
        line.push({ r: nr, c: nc });
      }
      for (let d = 1; d < 5; d++) {
        const nr = r - dr * d, nc = c - dc * d;
        if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE || this.board[nr][nc] !== player) break;
        line.push({ r: nr, c: nc });
      }
      if (line.length >= 5) {
        this.winLine = line;
        return true;
      }
    }
    return false;
  }

  getValidMoves() {
    const moves = [];
    if (this.gameOver) return moves;
    // Only check cells near existing stones for efficiency
    const visited = new Set();
    const range = 2;
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (this.board[r][c] !== EMPTY) {
          for (let dr = -range; dr <= range; dr++) {
            for (let dc = -range; dc <= range; dc++) {
              const nr = r + dr, nc = c + dc;
              if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE && this.board[nr][nc] === EMPTY) {
                const key = nr * BOARD_SIZE + nc;
                if (!visited.has(key)) {
                  visited.add(key);
                  moves.push([nr, nc]);
                }
              }
            }
          }
        }
      }
    }
    // If board is empty, return center
    if (moves.length === 0) {
      const mid = Math.floor(BOARD_SIZE / 2);
      return [[mid, mid]];
    }
    return moves;
  }

  /**
   * Get feature planes for neural network input
   * 6 planes: current player stones, opponent stones, 
   *           last move, second-to-last move, threat patterns, legal moves
   */
  getFeaturePlanes(player) {
    const planes = Array.from({ length: 6 }, () => 
      Array.from({ length: BOARD_SIZE }, () => new Float32Array(BOARD_SIZE))
    );
    const opp = player === BLACK ? WHITE : BLACK;

    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (this.board[r][c] === player) planes[0][r][c] = 1;
        else if (this.board[r][c] === opp) planes[1][r][c] = 1;
      }
    }

    // Last move
    if (this.moveHistory.length > 0) {
      const last = this.moveHistory[this.moveHistory.length - 1];
      planes[2][last.r][last.c] = 1;
    }
    if (this.moveHistory.length > 1) {
      const prev = this.moveHistory[this.moveHistory.length - 2];
      planes[3][prev.r][prev.c] = 1;
    }

    // Threat detection for current player
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (this.board[r][c] === EMPTY) {
          planes[4][r][c] = this.evaluateThreat(r, c, player);
          planes[5][r][c] = 1; // legal move
        }
      }
    }

    return planes;
  }

  evaluateThreat(r, c, player) {
    const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];
    let maxThreat = 0;
    for (const [dr, dc] of dirs) {
      let count = 0, open = 0;
      for (let d = 1; d <= 4; d++) {
        const nr = r + dr * d, nc = c + dc * d;
        if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
        if (this.board[nr][nc] === player) count++;
        else if (this.board[nr][nc] === EMPTY) { open++; break; }
        else break;
      }
      for (let d = 1; d <= 4; d++) {
        const nr = r - dr * d, nc = c - dc * d;
        if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
        if (this.board[nr][nc] === player) count++;
        else if (this.board[nr][nc] === EMPTY) { open++; break; }
        else break;
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

  /**
   * Heuristic evaluation for a position
   */
  evaluatePosition(player) {
    const opp = player === BLACK ? WHITE : BLACK;
    let score = 0;
    const patterns = this.countPatterns(player);
    const oppPatterns = this.countPatterns(opp);

    score += patterns.five * 100000;
    score += patterns.openFour * 10000;
    score += patterns.four * 1000;
    score += patterns.openThree * 500;
    score += patterns.three * 100;
    score += patterns.openTwo * 50;
    score += patterns.two * 10;

    score -= oppPatterns.five * 100000;
    score -= oppPatterns.openFour * 10000;
    score -= oppPatterns.four * 1000;
    score -= oppPatterns.openThree * 500;
    score -= oppPatterns.three * 100;
    score -= oppPatterns.openTwo * 50;
    score -= oppPatterns.two * 10;

    return score;
  }

  countPatterns(player) {
    const counts = { five: 0, openFour: 0, four: 0, openThree: 0, three: 0, openTwo: 0, two: 0 };
    const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];

    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (this.board[r][c] !== player) continue;
        for (const [dr, dc] of dirs) {
          // Avoid double counting
          const pr = r - dr, pc = c - dc;
          if (pr >= 0 && pr < BOARD_SIZE && pc >= 0 && pc < BOARD_SIZE && this.board[pr][pc] === player) continue;

          let len = 0;
          let nr = r, nc = c;
          while (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE && this.board[nr][nc] === player) {
            len++;
            nr += dr;
            nc += dc;
          }

          // Check open ends
          const beforeOpen = (pr >= 0 && pr < BOARD_SIZE && pc >= 0 && pc < BOARD_SIZE && this.board[pr][pc] === EMPTY);
          const afterOpen = (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE && this.board[nr][nc] === EMPTY);

          if (len >= 5) counts.five++;
          else if (len === 4) {
            if (beforeOpen && afterOpen) counts.openFour++;
            else if (beforeOpen || afterOpen) counts.four++;
          } else if (len === 3) {
            if (beforeOpen && afterOpen) counts.openThree++;
            else if (beforeOpen || afterOpen) counts.three++;
          } else if (len === 2) {
            if (beforeOpen && afterOpen) counts.openTwo++;
            else if (beforeOpen || afterOpen) counts.two++;
          }
        }
      }
    }
    return counts;
  }

  // Serialization for NN input
  toTensorInput(player) {
    const planes = this.getFeaturePlanes(player);
    const flat = [];
    for (let p = 0; p < 6; p++) {
      for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
          flat.push(planes[p][r][c]);
        }
      }
    }
    return flat;
  }
}

// Export
window.GomokuGame = GomokuGame;
window.BOARD_SIZE = BOARD_SIZE;
window.BLACK = BLACK;
window.WHITE = WHITE;
window.EMPTY = EMPTY;
