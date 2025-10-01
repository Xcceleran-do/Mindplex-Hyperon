// Graph engine service for layout and rendering
import { GraphNode, GraphEdge, Point, LayoutAlgorithm, LayoutOptions, LayoutState } from '../../types';

export interface GraphEngine {
  setData(nodes: GraphNode[], edges: GraphEdge[]): void;
  applyLayout(algorithm: LayoutAlgorithm, options?: LayoutOptions): void;
  render(ctx: CanvasRenderingContext2D, transform: Transform): void;
  handleNodeDrag(nodeId: string, position: Point): void;
  getNodeAtPosition(position: Point, transform: Transform): GraphNode | null;
  getLayoutState(): LayoutState;
  stopLayout(): void;
}

export interface Transform {
  x: number;
  y: number;
  scale: number;
}

export class GraphEngineImpl implements GraphEngine {
  private nodes: GraphNode[] = [];
  private edges: GraphEdge[] = [];
  private layoutState: LayoutState = {
    isAnimating: false,
    progress: 0,
    algorithm: 'force-directed',
    startTime: 0,
    duration: 0
  };
  private animationFrameId: number | null = null;
  private targetPositions: Map<string, Point> = new Map();

  setData(nodes: GraphNode[], edges: GraphEdge[]): void {
    this.nodes = [...nodes];
    this.edges = [...edges];
  }

  applyLayout(algorithm: LayoutAlgorithm, options?: LayoutOptions): void {
    // Stop any existing animation
    this.stopLayout();

    if (this.nodes.length === 0) return;

    // Set default options
    const layoutOptions: Required<LayoutOptions> = {
      iterations: 300,
      springLength: 200,
      springStrength: 0.1,
      repulsionStrength: 1000,
      damping: 0.9,
      animationDuration: 1500,
      centerForce: 0.01,
      ...options
    };

    // Calculate target positions based on algorithm
    let targetPositions: Map<string, Point>;
    switch (algorithm) {
      case 'force-directed':
        targetPositions = this.calculateForceDirectedLayout(layoutOptions);
        break;
      case 'hierarchical':
        targetPositions = this.calculateHierarchicalLayout(layoutOptions);
        break;
      case 'circular':
        targetPositions = this.calculateCircularLayout(layoutOptions);
        break;
      default:
        targetPositions = this.calculateForceDirectedLayout(layoutOptions);
    }

    // Start smooth animation to target positions
    this.startLayoutAnimation(algorithm, targetPositions, layoutOptions.animationDuration);
  }

  render(ctx: CanvasRenderingContext2D, transform: Transform): void {
    // Clear canvas
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    // Set high-quality rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Draw background grid if zoomed in enough
    if (transform.scale > 0.3) {
      this.drawGrid(ctx, transform);
    }

    // Render edges first (so they appear behind nodes)
    this.edges.forEach(edge => this.renderEdge(ctx, edge, transform));
    
    // Render nodes
    this.nodes.forEach(node => this.renderNode(ctx, node, transform));
  }

  handleNodeDrag(nodeId: string, position: Point): void {
    const node = this.nodes.find(n => n.id === nodeId);
    if (node) {
      node.position = { ...position };
    }
  }

  getNodeAtPosition(position: Point, transform: Transform): GraphNode | null {
    const nodeRadius = 20; // Base node radius
    const worldPos = this.screenToWorld(position, transform);
    
    for (const node of this.nodes) {
      const dx = worldPos.x - node.position.x;
      const dy = worldPos.y - node.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance <= nodeRadius) {
        return node;
      }
    }
    return null;
  }

  // Helper methods for coordinate transformation
  private screenToWorld(screenPoint: Point, transform: Transform): Point {
    return {
      x: (screenPoint.x - transform.x) / transform.scale,
      y: (screenPoint.y - transform.y) / transform.scale
    };
  }

  private worldToScreen(worldPoint: Point, transform: Transform): Point {
    return {
      x: worldPoint.x * transform.scale + transform.x,
      y: worldPoint.y * transform.scale + transform.y
    };
  }

  // Node color mapping based on template.html color scheme
  private getNodeColor(node: GraphNode): string {
    if (node.color) return node.color;
    
    switch (node.type) {
      case 'entity': return 'rgba(59, 130, 246, 0.7)'; // Blue
      case 'predicate': return 'rgba(34, 197, 94, 0.7)'; // Green
      case 'value': return 'rgba(245, 101, 101, 0.7)'; // Red
      case 'hypergraph': return 'rgba(168, 85, 247, 0.7)'; // Purple
      default: return 'rgba(59, 130, 246, 0.7)';
    }
  }

  // Edge color mapping
  private getEdgeColor(edge: GraphEdge): string {
    if (edge.color) return edge.color;
    
    switch (edge.type) {
      case 'relation': return 'rgba(107, 114, 128, 0.6)';
      case 'hypergraph_connection': return 'rgba(168, 85, 247, 0.6)';
      default: return 'rgba(107, 114, 128, 0.6)';
    }
  }

  // Render a single node
  private renderNode(ctx: CanvasRenderingContext2D, node: GraphNode, transform: Transform) {
    const screenPos = this.worldToScreen(node.position, transform);
    const radius = (node.size || 20) * transform.scale;
    
    // Skip rendering if node is outside viewport (with margin)
    const margin = radius + 50;
    if (screenPos.x < -margin || screenPos.x > ctx.canvas.width + margin ||
        screenPos.y < -margin || screenPos.y > ctx.canvas.height + margin) {
      return;
    }

    // Draw node circle
    ctx.beginPath();
    ctx.arc(screenPos.x, screenPos.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = this.getNodeColor(node);
    ctx.fill();
    
    // Draw node border
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Draw node label if scale is large enough
    if (transform.scale > 0.5 && node.label) {
      ctx.fillStyle = '#374151';
      ctx.font = `${Math.max(10, 12 * transform.scale)}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      // Add text background for better readability
      const textMetrics = ctx.measureText(node.label);
      const textWidth = textMetrics.width;
      const textHeight = 16 * transform.scale;
      
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.fillRect(
        screenPos.x - textWidth / 2 - 4,
        screenPos.y - textHeight / 2 - 2,
        textWidth + 8,
        textHeight + 4
      );
      
      ctx.fillStyle = '#374151';
      ctx.fillText(node.label, screenPos.x, screenPos.y);
    }
  }

  // Render a single edge
  private renderEdge(ctx: CanvasRenderingContext2D, edge: GraphEdge, transform: Transform) {
    const sourceNode = this.nodes.find(n => n.id === edge.source);
    const targetNode = this.nodes.find(n => n.id === edge.target);
    
    if (!sourceNode || !targetNode) return;

    const sourceScreen = this.worldToScreen(sourceNode.position, transform);
    const targetScreen = this.worldToScreen(targetNode.position, transform);
    
    // Skip rendering if edge is completely outside viewport
    const margin = 50;
    const minX = Math.min(sourceScreen.x, targetScreen.x);
    const maxX = Math.max(sourceScreen.x, targetScreen.x);
    const minY = Math.min(sourceScreen.y, targetScreen.y);
    const maxY = Math.max(sourceScreen.y, targetScreen.y);
    
    if (maxX < -margin || minX > ctx.canvas.width + margin ||
        maxY < -margin || minY > ctx.canvas.height + margin) {
      return;
    }

    // Draw edge line
    ctx.beginPath();
    ctx.moveTo(sourceScreen.x, sourceScreen.y);
    ctx.lineTo(targetScreen.x, targetScreen.y);
    ctx.strokeStyle = this.getEdgeColor(edge);
    ctx.lineWidth = Math.max(1, 2 * transform.scale);
    ctx.stroke();

    // Draw arrow for directed edges
    if (edge.directed) {
      const angle = Math.atan2(targetScreen.y - sourceScreen.y, targetScreen.x - sourceScreen.x);
      const arrowLength = Math.max(8, 12 * transform.scale);
      const arrowAngle = Math.PI / 6;
      
      // Position arrow at edge of target node
      const nodeRadius = (targetNode.size || 20) * transform.scale;
      const arrowX = targetScreen.x - Math.cos(angle) * nodeRadius;
      const arrowY = targetScreen.y - Math.sin(angle) * nodeRadius;
      
      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(
        arrowX - arrowLength * Math.cos(angle - arrowAngle),
        arrowY - arrowLength * Math.sin(angle - arrowAngle)
      );
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(
        arrowX - arrowLength * Math.cos(angle + arrowAngle),
        arrowY - arrowLength * Math.sin(angle + arrowAngle)
      );
      ctx.strokeStyle = this.getEdgeColor(edge);
      ctx.lineWidth = Math.max(1, 2 * transform.scale);
      ctx.stroke();
    }

    // Draw edge label if scale is large enough
    if (transform.scale > 0.7 && edge.label) {
      const midX = (sourceScreen.x + targetScreen.x) / 2;
      const midY = (sourceScreen.y + targetScreen.y) / 2;
      
      ctx.fillStyle = '#6b7280';
      ctx.font = `${Math.max(8, 10 * transform.scale)}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      // Add text background
      const textMetrics = ctx.measureText(edge.label);
      const textWidth = textMetrics.width;
      const textHeight = 12 * transform.scale;
      
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.fillRect(
        midX - textWidth / 2 - 3,
        midY - textHeight / 2 - 1,
        textWidth + 6,
        textHeight + 2
      );
      
      ctx.fillStyle = '#6b7280';
      ctx.fillText(edge.label, midX, midY);
    }
  }

  // Draw subtle background grid
  private drawGrid(ctx: CanvasRenderingContext2D, transform: Transform) {
    const gridSize = 50 * transform.scale;
    const offsetX = transform.x % gridSize;
    const offsetY = transform.y % gridSize;
    
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.03)';
    ctx.lineWidth = 1;
    
    // Vertical lines
    for (let x = offsetX; x < ctx.canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, ctx.canvas.height);
      ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = offsetY; y < ctx.canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(ctx.canvas.width, y);
      ctx.stroke();
    }
  }

  getLayoutState(): LayoutState {
    return { ...this.layoutState };
  }

  stopLayout(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    this.layoutState.isAnimating = false;
    this.targetPositions.clear();
  }

  // Helper method to center layout results around current graph center
  private centerLayoutPositions(positions: Map<string, Point>): Map<string, Point> {
    if (positions.size === 0) return positions;

    // Calculate current graph center
    let currentCenterX = 0, currentCenterY = 0;
    this.nodes.forEach(node => {
      currentCenterX += node.position.x;
      currentCenterY += node.position.y;
    });
    currentCenterX /= this.nodes.length;
    currentCenterY /= this.nodes.length;

    // Calculate new layout center
    let newCenterX = 0, newCenterY = 0;
    positions.forEach(pos => {
      newCenterX += pos.x;
      newCenterY += pos.y;
    });
    newCenterX /= positions.size;
    newCenterY /= positions.size;

    // Offset all positions to maintain current center
    const offsetX = currentCenterX - newCenterX;
    const offsetY = currentCenterY - newCenterY;

    const centeredPositions = new Map<string, Point>();
    positions.forEach((pos, nodeId) => {
      centeredPositions.set(nodeId, {
        x: pos.x + offsetX,
        y: pos.y + offsetY
      });
    });

    return centeredPositions;
  }

  // Force-directed layout using Fruchterman-Reingold algorithm
  private calculateForceDirectedLayout(options: Required<LayoutOptions>): Map<string, Point> {
    const positions = new Map<string, Point>();
    const velocities = new Map<string, Point>();
    
    // Initialize positions and velocities
    this.nodes.forEach(node => {
      positions.set(node.id, { ...node.position });
      velocities.set(node.id, { x: 0, y: 0 });
    });

    // Calculate optimal area and initial temperature
    const area = Math.max(800, Math.sqrt(this.nodes.length) * 100);
    let temperature = area / 10;
    const cooling = temperature / options.iterations;

    for (let iteration = 0; iteration < options.iterations; iteration++) {
      // Calculate repulsive forces between all pairs of nodes
      for (let i = 0; i < this.nodes.length; i++) {
        const nodeA = this.nodes[i];
        const posA = positions.get(nodeA.id)!;
        const velA = velocities.get(nodeA.id)!;

        for (let j = i + 1; j < this.nodes.length; j++) {
          const nodeB = this.nodes[j];
          const posB = positions.get(nodeB.id)!;
          const velB = velocities.get(nodeB.id)!;

          const dx = posA.x - posB.x;
          const dy = posA.y - posB.y;
          const distance = Math.sqrt(dx * dx + dy * dy) || 0.1;

          const repulsiveForce = options.repulsionStrength / (distance * distance);
          const fx = (dx / distance) * repulsiveForce * 20;
          const fy = (dy / distance) * repulsiveForce * 20;

          velA.x += fx;
          velA.y += fy;
          velB.x -= fx;
          velB.y -= fy;
        }
      }

      // Calculate attractive forces for connected nodes
      this.edges.forEach(edge => {
        const sourcePos = positions.get(edge.source);
        const targetPos = positions.get(edge.target);
        const sourceVel = velocities.get(edge.source);
        const targetVel = velocities.get(edge.target);

        if (!sourcePos || !targetPos || !sourceVel || !targetVel) return;

        const dx = targetPos.x - sourcePos.x;
        const dy = targetPos.y - sourcePos.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 0.1;

        const attractiveForce = (distance * distance) / options.springLength * options.springStrength;
        const fx = (dx / distance) * attractiveForce;
        const fy = (dy / distance) * attractiveForce;

        sourceVel.x += fx;
        sourceVel.y += fy;
        targetVel.x -= fx;
        targetVel.y -= fy;
      });

      // Apply center force to prevent nodes from drifting too far
      const centerX = 0;
      const centerY = 0;
      this.nodes.forEach(node => {
        const pos = positions.get(node.id)!;
        const vel = velocities.get(node.id)!;
        
        vel.x -= pos.x * options.centerForce;
        vel.y -= pos.y * options.centerForce;
      });

      // Update positions and apply damping
      this.nodes.forEach(node => {
        const pos = positions.get(node.id)!;
        const vel = velocities.get(node.id)!;

        // Limit velocity by temperature
        const speed = Math.sqrt(vel.x * vel.x + vel.y * vel.y);
        if (speed > temperature) {
          vel.x = (vel.x / speed) * temperature;
          vel.y = (vel.y / speed) * temperature;
        }

        pos.x += vel.x;
        pos.y += vel.y;

        // Apply damping
        vel.x *= options.damping;
        vel.y *= options.damping;
      });

      // Cool down
      temperature = Math.max(0.1, temperature - cooling);
    }

    return this.centerLayoutPositions(positions);
  }

  // Hierarchical layout based on node connections
  private calculateHierarchicalLayout(options: Required<LayoutOptions>): Map<string, Point> {
    const positions = new Map<string, Point>();
    const levels = new Map<string, number>();
    const visited = new Set<string>();

    // Find root nodes (nodes with no incoming edges or most connections)
    const inDegree = new Map<string, number>();
    const outDegree = new Map<string, number>();
    
    this.nodes.forEach(node => {
      inDegree.set(node.id, 0);
      outDegree.set(node.id, 0);
    });

    this.edges.forEach(edge => {
      inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
      outDegree.set(edge.source, (outDegree.get(edge.source) || 0) + 1);
    });

    // Assign levels using BFS from root nodes
    const queue: string[] = [];
    
    // Start with nodes that have no incoming edges
    this.nodes.forEach(node => {
      if (inDegree.get(node.id) === 0) {
        levels.set(node.id, 0);
        queue.push(node.id);
        visited.add(node.id);
      }
    });

    // If no root nodes found, start with the node with most outgoing connections
    if (queue.length === 0) {
      let maxOutDegree = -1;
      let rootNode = '';
      this.nodes.forEach(node => {
        const degree = outDegree.get(node.id) || 0;
        if (degree > maxOutDegree) {
          maxOutDegree = degree;
          rootNode = node.id;
        }
      });
      if (rootNode) {
        levels.set(rootNode, 0);
        queue.push(rootNode);
        visited.add(rootNode);
      }
    }

    // BFS to assign levels
    while (queue.length > 0) {
      const currentNode = queue.shift()!;
      const currentLevel = levels.get(currentNode)!;

      this.edges.forEach(edge => {
        if (edge.source === currentNode && !visited.has(edge.target)) {
          levels.set(edge.target, currentLevel + 1);
          queue.push(edge.target);
          visited.add(edge.target);
        }
      });
    }

    // Handle unvisited nodes (disconnected components)
    this.nodes.forEach(node => {
      if (!visited.has(node.id)) {
        levels.set(node.id, 0);
      }
    });

    // Group nodes by level
    const levelGroups = new Map<number, string[]>();
    levels.forEach((level, nodeId) => {
      if (!levelGroups.has(level)) {
        levelGroups.set(level, []);
      }
      levelGroups.get(level)!.push(nodeId);
    });

    // Position nodes
    const levelHeight = 170;
    const maxLevel = Math.max(...levels.values());
    
    levelGroups.forEach((nodeIds, level) => {
      const y = (level - maxLevel / 2) * levelHeight;
      const nodeWidth = 120;
      const totalWidth = nodeIds.length * nodeWidth;
      const startX = -totalWidth / 2;

      nodeIds.forEach((nodeId, index) => {
        const x = startX + index * nodeWidth + nodeWidth / 2;
        positions.set(nodeId, { x, y });
      });
    });

    return this.centerLayoutPositions(positions);
  }

  // Circular layout arranging nodes in concentric circles
  private calculateCircularLayout(options: Required<LayoutOptions>): Map<string, Point> {
    const positions = new Map<string, Point>();
    
    if (this.nodes.length === 0) return positions;
    if (this.nodes.length === 1) {
      positions.set(this.nodes[0].id, { x: 0, y: 0 });
      return positions;
    }

    // Group nodes by type or connection count for better organization
    const nodeGroups = new Map<string, string[]>();
    
    this.nodes.forEach(node => {
      const key = node.type;
      if (!nodeGroups.has(key)) {
        nodeGroups.set(key, []);
      }
      nodeGroups.get(key)!.push(node.id);
    });

    const groups = Array.from(nodeGroups.values());
    const baseRadius = 130;
    const radiusIncrement = 100;

    if (groups.length === 1) {
      // Single group - arrange in a circle
      const nodeIds = groups[0];
      const radius = Math.max(baseRadius, nodeIds.length * 15);
      
      nodeIds.forEach((nodeId, index) => {
        const angle = (2 * Math.PI * index) / nodeIds.length;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        positions.set(nodeId, { x, y });
      });
    } else {
      // Multiple groups - arrange in concentric circles
      groups.forEach((nodeIds, groupIndex) => {
        const radius = baseRadius + groupIndex * radiusIncrement;
        
        nodeIds.forEach((nodeId, index) => {
          const angle = (2 * Math.PI * index) / nodeIds.length;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;
          positions.set(nodeId, { x, y });
        });
      });
    }

    return this.centerLayoutPositions(positions);
  }

  // Start smooth animation to target positions
  private startLayoutAnimation(algorithm: LayoutAlgorithm, targetPositions: Map<string, Point>, duration: number): void {
    this.targetPositions = targetPositions;
    const startTime = performance.now();
    this.layoutState = {
      isAnimating: true,
      progress: 0,
      algorithm,
      startTime,
      duration
    };

    // Store initial positions for interpolation
    const initialPositions = new Map<string, Point>();
    this.nodes.forEach(node => {
      initialPositions.set(node.id, { ...node.position });
    });

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.max(0, Math.min(elapsed / duration, 1));
      
      // Easing function for smooth animation (ease-out cubic)
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      
      // Interpolate positions
      this.nodes.forEach(node => {
        const initial = initialPositions.get(node.id);
        const target = targetPositions.get(node.id);
        
        if (initial && target) {
          node.position.x = initial.x + (target.x - initial.x) * easedProgress;
          node.position.y = initial.y + (target.y - initial.y) * easedProgress;
        }
      });

      this.layoutState.progress = progress;

      if (progress < 1 && this.layoutState.isAnimating) {
        this.animationFrameId = requestAnimationFrame(animate);
      } else {
        this.layoutState.isAnimating = false;
        this.layoutState.progress = 1;
        this.targetPositions.clear();
        this.animationFrameId = null;
      }
    };

    this.animationFrameId = requestAnimationFrame(animate);
  }
}