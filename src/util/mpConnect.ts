import { NormalizedLandmark } from '@mediapipe/holistic';

function mpConnect(
  ctx: CanvasRenderingContext2D,
  connectors: Array<[NormalizedLandmark, NormalizedLandmark]>,
): void {
  const canvas = ctx.canvas;
  for (const connector of connectors) {
    const from = connector[0];
    const to = connector[1];
    if (from && to) {
      if (
        from.visibility &&
        to.visibility &&
        (from.visibility < 0.1 || to.visibility < 0.1)
      ) {
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
      ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
      ctx.stroke();
    }
  }
}

export default mpConnect;
