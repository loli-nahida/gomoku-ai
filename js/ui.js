/**
 * UI Controller v2 - 棋盘绘制、热力图、动画
 */

class GomokuUI {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.boardSize = BOARD_SIZE;
    this.padding = 30;
    this.stoneRadius = 13;
    this.hoverPos = null;
    this.lastMove = null;
    this.winLine = null;
    this.showHeatmap = false;
    this.heatmapData = null;

    this.setupEvents();
    this.draw();
  }

  resize(size) {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = size * dpr;
    this.canvas.height = size * dpr;
    this.canvas.style.width = size + 'px';
    this.canvas.style.height = size + 'px';
    this.ctx.scale(dpr, dpr);
    this.logicalWidth = size;
    this.logicalHeight = size;
    this.cellSize = (size - 2 * this.padding) / (this.boardSize - 1);
    this.stoneRadius = this.cellSize * 0.43;
  }

  setupEvents() {
    this._lastTouchTime = 0;

    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('mouseleave', () => { this.hoverPos = null; this.draw(); });
    this.canvas.addEventListener('click', (e) => this.onClick(e));

    // Touch support — set flag so click handler ignores synthetic click
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this._lastTouchTime = Date.now();
      const touch = e.touches[0];
      const pos = this.getGridPos(touch);
      if (pos && this.onClickCallback) this.onClickCallback(pos.r, pos.c);
    }, { passive: false });
  }

  getGridPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = (this.logicalWidth || this.canvas.width) / rect.width;
    const scaleY = (this.logicalHeight || this.canvas.height) / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    const c = Math.round((x - this.padding) / this.cellSize);
    const r = Math.round((y - this.padding) / this.cellSize);
    if (r >= 0 && r < this.boardSize && c >= 0 && c < this.boardSize) {
      return { r, c };
    }
    return null;
  }

  onMouseMove(e) {
    const pos = this.getGridPos(e);
    this.hoverPos = pos;
    this.draw();
  }

  onClick(e) {
    // Ignore synthetic click right after touch
    if (Date.now() - this._lastTouchTime < 500) return;
    const pos = this.getGridPos(e);
    if (pos && this.onClickCallback) {
      this.onClickCallback(pos.r, pos.c);
    }
  }

  onHover(cb) { this.onHoverCallback = cb; }
  onClickHandler(cb) { this.onClickCallback = cb; }
  setShowHeatmap(show) { this.showHeatmap = show; }
  setHeatmapData(policy) { this.heatmapData = policy; }

  draw() {
    const ctx = this.ctx;
    const w = this.logicalWidth || this.canvas.width;
    const h = this.logicalHeight || this.canvas.height;

    // Background
    ctx.fillStyle = '#dcb468';
    ctx.fillRect(0, 0, w, h);

    // Wood grain
    ctx.strokeStyle = 'rgba(139, 105, 20, 0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 30; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * (h / 30) + Math.sin(i) * 3);
      ctx.lineTo(w, i * (h / 30) + Math.cos(i) * 3);
      ctx.stroke();
    }

    // Grid lines
    ctx.strokeStyle = '#8b6914';
    ctx.lineWidth = 1;
    for (let i = 0; i < this.boardSize; i++) {
      const pos = this.padding + i * this.cellSize;
      ctx.beginPath();
      ctx.moveTo(this.padding, pos);
      ctx.lineTo(this.padding + (this.boardSize - 1) * this.cellSize, pos);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pos, this.padding);
      ctx.lineTo(pos, this.padding + (this.boardSize - 1) * this.cellSize);
      ctx.stroke();
    }

    // Border
    ctx.strokeStyle = '#6b5010';
    ctx.lineWidth = 2;
    ctx.strokeRect(this.padding, this.padding, (this.boardSize - 1) * this.cellSize, (this.boardSize - 1) * this.cellSize);

    // Star points
    const starPoints = [[3,3],[3,7],[3,11],[7,3],[7,7],[7,11],[11,3],[11,7],[11,11]];
    ctx.fillStyle = '#8b6914';
    for (const [r, c] of starPoints) {
      const x = this.padding + c * this.cellSize;
      const y = this.padding + r * this.cellSize;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(3, this.cellSize * 0.1), 0, Math.PI * 2);
      ctx.fill();
    }

    // Coordinates
    ctx.fillStyle = '#6b5010';
    ctx.font = `${Math.max(10, this.cellSize * 0.3)}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < this.boardSize; i++) {
      const x = this.padding + i * this.cellSize;
      ctx.fillText(String.fromCharCode(65 + i), x, 12);
      ctx.fillText(String.fromCharCode(65 + i), x, h - 12);
      const y = this.padding + i * this.cellSize;
      ctx.fillText(String(this.boardSize - i), 12, y);
      ctx.fillText(String(this.boardSize - i), w - 12, y);
    }
  }

  drawHeatmap() {
    if (!this.showHeatmap || !this.heatmapData) return;
    const ctx = this.ctx;

    // Find max for normalization
    let maxVal = 0;
    for (let i = 0; i < this.heatmapData.length; i++) {
      if (this.heatmapData[i] > maxVal) maxVal = this.heatmapData[i];
    }
    if (maxVal <= 0) return;

    for (let r = 0; r < this.boardSize; r++) {
      for (let c = 0; c < this.boardSize; c++) {
        const val = this.heatmapData[r * this.boardSize + c];
        if (val <= 0) continue;
        const normalized = val / maxVal;
        const x = this.padding + c * this.cellSize;
        const y = this.padding + r * this.cellSize;
        const size = this.cellSize * 0.45;

        // Color: low = blue, mid = yellow, high = red
        const hue = (1 - normalized) * 240; // 240=blue, 0=red
        ctx.fillStyle = `hsla(${hue}, 90%, 50%, ${0.2 + normalized * 0.5})`;
        ctx.fillRect(x - size, y - size, size * 2, size * 2);

        // Value text
        if (normalized > 0.1) {
          ctx.fillStyle = `rgba(255,255,255,${0.5 + normalized * 0.5})`;
          ctx.font = `bold ${Math.max(8, this.cellSize * 0.22)}px sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText((val * 100).toFixed(0) + '%', x, y);
        }
      }
    }
  }

  drawStone(r, c, player, isLast = false) {
    const ctx = this.ctx;
    const x = this.padding + c * this.cellSize;
    const y = this.padding + r * this.cellSize;

    // Shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
    ctx.beginPath();
    ctx.arc(x + 2, y + 2, this.stoneRadius, 0, Math.PI * 2);
    ctx.fill();

    // Stone gradient
    const gradient = ctx.createRadialGradient(
      x - this.stoneRadius * 0.3, y - this.stoneRadius * 0.3, this.stoneRadius * 0.1,
      x, y, this.stoneRadius
    );

    if (player === BLACK) {
      gradient.addColorStop(0, '#555');
      gradient.addColorStop(0.7, '#222');
      gradient.addColorStop(1, '#111');
    } else {
      gradient.addColorStop(0, '#fff');
      gradient.addColorStop(0.7, '#e8e8e8');
      gradient.addColorStop(1, '#ccc');
    }

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, this.stoneRadius, 0, Math.PI * 2);
    ctx.fill();

    // Last move marker
    if (isLast) {
      ctx.fillStyle = '#f44336';
      ctx.beginPath();
      ctx.arc(x, y, this.stoneRadius * 0.25, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  drawBoard(game) {
    this.draw();

    // Heatmap (under stones)
    this.drawHeatmap();

    this.lastMove = game.moveHistory.length > 0 ? game.moveHistory[game.moveHistory.length - 1] : null;
    this.winLine = game.winLine;

    // Draw all stones
    for (let r = 0; r < this.boardSize; r++) {
      for (let c = 0; c < this.boardSize; c++) {
        if (game.board[r][c] !== EMPTY) {
          const isLast = this.lastMove && this.lastMove.r === r && this.lastMove.c === c;
          this.drawStone(r, c, game.board[r][c], isLast);
        }
      }
    }

    // Win line highlight
    if (this.winLine) {
      this.ctx.strokeStyle = '#4caf50';
      this.ctx.lineWidth = 3;
      for (const { r, c } of this.winLine) {
        const x = this.padding + c * this.cellSize;
        const y = this.padding + r * this.cellSize;
        this.ctx.beginPath();
        this.ctx.arc(x, y, this.stoneRadius + 3, 0, Math.PI * 2);
        this.ctx.stroke();
      }
    }

    // Hover
    if (this.hoverPos && !game.gameOver) {
      const x = this.padding + this.hoverPos.c * this.cellSize;
      const y = this.padding + this.hoverPos.r * this.cellSize;
      if (game.board[this.hoverPos.r][this.hoverPos.c] === EMPTY) {
        this.ctx.fillStyle = 'rgba(108, 99, 255, 0.25)';
        this.ctx.beginPath();
        this.ctx.arc(x, y, this.stoneRadius, 0, Math.PI * 2);
        this.ctx.fill();
      }
    }
  }
}

window.GomokuUI = GomokuUI;
