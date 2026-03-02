// Canvas interaction utilities for ColumnarVisualizer
import { GraphNode, Point } from '../types';
import { Transform, screenToWorld } from './canvasRenderer';

// Get mouse position relative to canvas
export const getMousePos = (e: MouseEvent, canvas: HTMLCanvasElement): Point => {
  const rect = canvas.getBoundingClientRect();
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top
  };
};

// Find node at given world position
export const getNodeAtPosition = (worldPos: Point, nodes: GraphNode[]): GraphNode | null => {
  for (const node of nodes) {
    const nodeSize = (node.size || 50) / 2;
    const dx = worldPos.x - node.position.x;
    const dy = worldPos.y - node.position.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance <= nodeSize) {
      return node;
    }
  }
  return null;
};

// Handle zoom with mouse wheel
export const handleZoom = (
  e: WheelEvent,
  canvas: HTMLCanvasElement,
  transform: Transform
): Transform => {
  e.preventDefault();

  const mousePos = getMousePos(e, canvas);
  const worldPosBeforeZoom = screenToWorld(mousePos, transform);

  const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
  let newScale = transform.scale * zoomFactor;
  newScale = Math.max(0.3, Math.min(3, newScale));

  if (Math.abs(newScale - transform.scale) < 0.0001) return transform;

  const newTransform = { ...transform, scale: newScale };
  const worldPosAfterZoom = screenToWorld(mousePos, newTransform);
  
  return {
    x: transform.x + (worldPosAfterZoom.x - worldPosBeforeZoom.x) * newScale,
    y: transform.y + (worldPosAfterZoom.y - worldPosBeforeZoom.y) * newScale,
    scale: newScale
  };
};

// Handle panning
export const handlePan = (
  dx: number,
  dy: number,
  transform: Transform
): Transform => {
  return {
    ...transform,
    x: transform.x + dx,
    y: transform.y + dy
  };
};
