import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force';
import { Component, createEffect, createMemo, createSignal, onCleanup, onMount, Show } from 'solid-js';
import type { FilterState, GraphData, GraphNode, HighlightState, Point } from '../../../types';
import { updateHighlightState } from '../../../utils/highlightManager';
import styles from './SemanticAtlas.module.css';
import { buildPropertyColorMap, colorForAtlasSubject, readAtlasColorTheme } from './colorMapping';

type AtlasNodeKind = 'article' | 'property';

type AtlasNode = SimulationNodeDatum & {
  id: string;
  kind: AtlasNodeKind;
  label: string;
  propertyName?: string;
  articleId?: string;
  x: number;
  y: number;
  radius: number;
  source: GraphNode;
  degree: number;
  prominent: boolean;
};

type AtlasEdge = SimulationLinkDatum<AtlasNode> & {
  id: string;
  source: AtlasNode;
  target: AtlasNode;
  property: string;
};

type AtlasLayout = {
  nodes: AtlasNode[];
  edges: AtlasEdge[];
  articleNodes: AtlasNode[];
  propertyNodes: AtlasNode[];
  propertyCount: number;
  articleCount: number;
  edgeCount: number;
};

type CanvasTheme = {
  bg: string;
  grid: string;
  text: string;
  surface: string;
  article: string;
  hover: string;
  selected: string;
  palette: string[];
};

export interface SemanticAtlasProps {
  graphData: GraphData;
  filterState: FilterState;
  onFilterChange: (filter: FilterState) => void;
  onNodeSelect: (node: GraphNode) => void;
  zoomTrigger?: { action: 'in' | 'out' | 'recenter' | null; timestamp: number };
  showOverview?: boolean;
}

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

const formatLabel = (value: string) =>
  value
    .replace(/^"|"$/g, '')
    .replace(/[_-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const sortArticleIds = (a: GraphNode, b: GraphNode) => {
  const articleA = a.metadata.originalExpression || a.label;
  const articleB = b.metadata.originalExpression || b.label;
  const numA = Number.parseInt(articleA.replace(/\D+/g, ''), 10);
  const numB = Number.parseInt(articleB.replace(/\D+/g, ''), 10);

  if (Number.isFinite(numA) && Number.isFinite(numB) && numA !== numB) {
    return numA - numB;
  }
  return articleA.localeCompare(articleB);
};

const initialPoint = (index: number, kind: AtlasNodeKind): Point => {
  const angle = index * GOLDEN_ANGLE;
  const spacing = kind === 'article' ? 48 : 62;
  const radius = spacing * Math.sqrt(index + 1);
  return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
};

const buildLayout = (graphData: GraphData): AtlasLayout => {
  const articleSources = graphData.nodes
    .filter((node) => node.metadata.columnType === 'article')
    .sort(sortArticleIds);
  const propertySources = graphData.nodes.filter(
    (node) => node.metadata.columnType === 'property' && node.label !== 'None',
  );
  const degreeByNode = new Map<string, number>();

  for (const edge of graphData.edges) {
    degreeByNode.set(edge.source, (degreeByNode.get(edge.source) || 0) + 1);
    degreeByNode.set(edge.target, (degreeByNode.get(edge.target) || 0) + 1);
  }

  const articleNodes = articleSources.map<AtlasNode>((node, index) => {
    const point = initialPoint(index * 2, 'article');
    const degree = degreeByNode.get(node.id) || 0;
    return {
      id: node.id,
      kind: 'article',
      label: node.label,
      articleId: node.metadata.originalExpression || node.label,
      x: point.x,
      y: point.y,
      radius: 9 + Math.min(7, Math.sqrt(Math.max(1, degree)) * 1.35),
      source: node,
      degree,
      prominent: false,
    };
  });

  const rankedProperties = propertySources
    .map((node) => ({ node, degree: degreeByNode.get(node.id) || 0 }))
    .sort((a, b) => b.degree - a.degree || a.node.label.localeCompare(b.node.label));
  const prominentCount = Math.min(10, Math.max(6, Math.ceil(rankedProperties.length * 0.15)));
  const prominentIds = new Set(rankedProperties.slice(0, prominentCount).map(({ node }) => node.id));

  const propertyNodes = rankedProperties.map<AtlasNode>(({ node, degree }, index) => {
    const point = initialPoint(index * 2 + 1, 'property');
    return {
      id: node.id,
      kind: 'property',
      label: node.label,
      propertyName: node.metadata.propertyName || 'property',
      x: point.x,
      y: point.y,
      radius: 8 + Math.min(14, Math.sqrt(Math.max(1, degree)) * 2.15),
      source: node,
      degree,
      prominent: prominentIds.has(node.id),
    };
  });

  const nodesById = new Map([...articleNodes, ...propertyNodes].map((node) => [node.id, node]));
  const edges = graphData.edges
    .map<AtlasEdge | null>((edge) => {
      const source = nodesById.get(edge.source);
      const target = nodesById.get(edge.target);
      if (!source || !target) return null;
      return { id: edge.id, source, target, property: edge.label };
    })
    .filter((edge): edge is AtlasEdge => edge !== null);

  return {
    nodes: [...articleNodes, ...propertyNodes],
    edges,
    articleNodes,
    propertyNodes,
    propertyCount: new Set(propertyNodes.map((node) => node.propertyName)).size,
    articleCount: articleNodes.length,
    edgeCount: edges.length,
  };
};

const createLayoutSimulation = (layout: AtlasLayout): Simulation<AtlasNode, AtlasEdge> =>
  forceSimulation<AtlasNode>(layout.nodes)
    .force(
      'links',
      forceLink<AtlasNode, AtlasEdge>(layout.edges)
        .id((node) => node.id)
        .distance((edge) => 132 + Math.min(48, edge.target.degree * 1.4))
        .strength(0.16),
    )
    .force(
      'charge',
      forceManyBody<AtlasNode>()
        .strength((node) => (node.kind === 'property' ? -920 : -560))
        .distanceMin(28)
        .distanceMax(720),
    )
    .force(
      'collision',
      forceCollide<AtlasNode>()
        .radius((node) => node.radius + (node.kind === 'property' ? 25 : 19))
        .strength(1)
        .iterations(3),
    )
    .force('x', forceX<AtlasNode>(0).strength(0.022))
    .force('y', forceY<AtlasNode>(0).strength(0.022))
    .velocityDecay(0.34)
    .alphaDecay(0.025);

const isLinked = (edge: AtlasEdge, node: AtlasNode | null) =>
  Boolean(node && (edge.source.id === node.id || edge.target.id === node.id));

const focusNeighborhood = (layout: AtlasLayout, focusNode: AtlasNode | null) => {
  if (!focusNode) return null;
  const nodeIds = new Set([focusNode.id]);
  for (const edge of layout.edges) {
    if (edge.source.id === focusNode.id) nodeIds.add(edge.target.id);
    if (edge.target.id === focusNode.id) nodeIds.add(edge.source.id);
  }
  return nodeIds;
};

const colorForNode = (node: AtlasNode, theme: CanvasTheme, propertyColors: Map<string, string>) =>
  colorForAtlasSubject(
    { kind: node.kind, label: node.label, propertyName: node.propertyName },
    theme,
    propertyColors,
  );

const drawBackground = (ctx: CanvasRenderingContext2D, width: number, height: number, theme: CanvasTheme) => {
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.fillStyle = theme.grid;
  ctx.globalAlpha = 0.2;
  for (let x = 24; x < width; x += 48) {
    for (let y = 24; y < height; y += 48) {
      ctx.beginPath();
      ctx.arc(x, y, 0.8, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
};

const drawEdge = (
  ctx: CanvasRenderingContext2D,
  edge: AtlasEdge,
  filterState: FilterState,
  highlightState: HighlightState,
  focusNode: AtlasNode | null,
  theme: CanvasTheme,
  propertyColors: Map<string, string>,
  dense: boolean,
) => {
  if (filterState.active && !highlightState.highlightedEdges.has(edge.id)) return;
  const linked = isLinked(edge, focusNode);
  const highlighted = !filterState.active || highlightState.highlightedEdges.has(edge.id);
  const alpha = linked ? 0.78 : filterState.active ? (highlighted ? 0.34 : 0.018) : dense ? 0.075 : 0.15;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = colorForNode(edge.target, theme, propertyColors);
  ctx.lineWidth = linked ? 2.1 : highlighted ? 0.9 : 0.65;
  ctx.beginPath();
  ctx.moveTo(edge.source.x, edge.source.y);
  ctx.lineTo(edge.target.x, edge.target.y);
  ctx.stroke();
  ctx.restore();
};

const drawLabel = (
  ctx: CanvasRenderingContext2D,
  node: AtlasNode,
  scale: number,
  theme: CanvasTheme,
  focused: boolean,
) => {
  const label = formatLabel(node.kind === 'article' ? node.articleId || node.label : node.label);
  const fontSize = Math.max(10, (focused ? 12.5 : 11) / scale);
  ctx.font = `${fontSize}px "Manrope", "Avenir Next", sans-serif`;
  const width = ctx.measureText(label).width;
  const x = node.x + (node.radius + 10 / scale);
  const y = node.y;

  ctx.save();
  ctx.globalAlpha = focused ? 0.94 : 0.8;
  ctx.fillStyle = theme.bg;
  ctx.fillRect(x - 4 / scale, y - 10 / scale, width + 8 / scale, 20 / scale);
  ctx.fillStyle = theme.text;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x, y);
  ctx.restore();
};

const drawNode = (
  ctx: CanvasRenderingContext2D,
  node: AtlasNode,
  filterState: FilterState,
  highlightState: HighlightState,
  hoveredNode: AtlasNode | null,
  selectedNode: AtlasNode | null,
  focusedNode: AtlasNode | null,
  focusedNodeIds: Set<string> | null,
  scale: number,
  theme: CanvasTheme,
  propertyColors: Map<string, string>,
) => {
  const filterHighlighted = !filterState.active || highlightState.highlightedNodes.has(node.id);
  if (filterState.active && node.kind === 'article' && !filterHighlighted) return;
  const hovered = hoveredNode?.id === node.id;
  const selected = selectedNode?.id === node.id;
  const focused = hovered || selected;
  const neighbor = Boolean(focusedNodeIds?.has(node.id) && !focused);
  const focusHighlighted = !focusedNodeIds || focusedNodeIds.has(node.id);
  const nodeColor = colorForNode(node, theme, propertyColors);

  ctx.save();
  ctx.globalAlpha = filterHighlighted && focusHighlighted ? 1 : 0.11;
  ctx.shadowColor = nodeColor;
  ctx.shadowBlur = focused ? 20 / scale : neighbor ? 13 / scale : node.prominent ? 7 / scale : 0;
  ctx.fillStyle = node.kind === 'article' ? theme.surface : nodeColor;
  ctx.strokeStyle = selected ? theme.selected : hovered ? theme.hover : nodeColor;
  ctx.lineWidth = (focused ? 3 : node.kind === 'article' ? 2.2 : 1.4) / scale;
  ctx.beginPath();
  ctx.arc(node.x, node.y, node.radius + (focused ? 2 / scale : 0), 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.stroke();

  if (node.kind === 'article') {
    ctx.fillStyle = theme.article;
    ctx.beginPath();
    ctx.arc(node.x, node.y, Math.max(3.2, node.radius * 0.34), 0, Math.PI * 2);
    ctx.fill();
  }

  if (neighbor) {
    ctx.shadowBlur = 0;
    ctx.strokeStyle = theme.hover;
    ctx.lineWidth = 2.2 / scale;
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.radius + 4 / scale, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();

  const relatedAttribute = focusedNode?.kind === 'article' && neighbor && node.kind === 'property';
  const showLabel = focused || relatedAttribute || (node.kind === 'property' && node.prominent && scale > 0.58);
  if (showLabel) drawLabel(ctx, node, scale, theme, focused);
};

const getNodeAtPoint = (nodes: AtlasNode[], point: Point, scale: number) => {
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    const node = nodes[index];
    const dx = point.x - node.x;
    const dy = point.y - node.y;
    const hitRadius = Math.max(node.radius + 8 / scale, 15 / scale);
    if (dx * dx + dy * dy <= hitRadius * hitRadius) return node;
  }
  return null;
};

const addOrToggleFilter = (node: AtlasNode, filterState: FilterState, multi: boolean): FilterState => {
  if (node.kind === 'article') {
    const articleId = node.articleId || '';
    const currentArticleIds = new Set(filterState.articleIds || []);
    const articleIds = new Set<string>();
    if (multi) {
      currentArticleIds.forEach((id) => articleIds.add(id));
      if (articleIds.has(articleId)) articleIds.delete(articleId);
      else articleIds.add(articleId);
    } else if (!(currentArticleIds.size === 1 && currentArticleIds.has(articleId))) {
      articleIds.add(articleId);
    }
    return {
      active: articleIds.size > 0 || (filterState.propertyFilters?.length || 0) > 0,
      articleIds: Array.from(articleIds),
      propertyFilters: filterState.propertyFilters || [],
    };
  }

  const propertyFilter = { property: node.propertyName || '', value: node.label };
  const currentPropertyFilters = filterState.propertyFilters || [];
  const propertyFilters = [...currentPropertyFilters];
  const existing = propertyFilters.findIndex(
    (filter) => filter.property === propertyFilter.property && filter.value === propertyFilter.value,
  );
  if (existing >= 0) propertyFilters.splice(existing, 1);
  else {
    if (!multi) {
      for (let index = propertyFilters.length - 1; index >= 0; index -= 1) {
        if (propertyFilters[index].property === propertyFilter.property) {
          propertyFilters.splice(index, 1);
        }
      }
    }
    propertyFilters.push(propertyFilter);
  }
  return {
    active: propertyFilters.length > 0 || (filterState.articleIds?.length || 0) > 0,
    articleIds: filterState.articleIds || [],
    propertyFilters,
  };
};

const SemanticAtlas: Component<SemanticAtlasProps> = (props) => {
  let canvasRef: HTMLCanvasElement | undefined;
  let frame = 0;
  let simulation: Simulation<AtlasNode, AtlasEdge> | null = null;
  let autoFitTimer = 0;
  let transform = { x: 0, y: 0, scale: 1 };
  let interaction: 'idle' | 'pan' | 'node' = 'idle';
  let lastPoint: Point = { x: 0, y: 0 };
  let pointerStart: Point = { x: 0, y: 0 };
  let draggedNode: AtlasNode | null = null;
  let moved = false;
  let previousFix: { x: number | null; y: number | null } | null = null;
  let hoveredNode: AtlasNode | null = null;
  let selectedNode: AtlasNode | null = null;

  const [tooltip, setTooltip] = createSignal({ visible: false, x: 0, y: 0, node: null as AtlasNode | null });
  const [focusedNode, setFocusedNode] = createSignal<AtlasNode | null>(null);
  const layout = createMemo(() => buildLayout(props.graphData));

  const interactiveNodes = () => {
    const currentLayout = layout();
    if (!props.filterState.active) return currentLayout.nodes;
    const highlighted = updateHighlightState(props.graphData, props.filterState).highlightedNodes;
    return currentLayout.nodes.filter(
      (node) => node.kind !== 'article' || highlighted.has(node.id),
    );
  };

  const worldFromEvent = (event: MouseEvent | WheelEvent): Point => {
    const rect = canvasRef!.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left - transform.x) / transform.scale,
      y: (event.clientY - rect.top - transform.y) / transform.scale,
    };
  };

  const recenter = () => {
    if (!canvasRef) return;
    const rect = canvasRef.getBoundingClientRect();
    const nodes = layout().nodes;
    if (nodes.length === 0) {
      transform = { x: rect.width / 2, y: rect.height / 2, scale: 1 };
      return;
    }
    const minX = Math.min(...nodes.map((node) => node.x - node.radius - 40));
    const maxX = Math.max(...nodes.map((node) => node.x + node.radius + 120));
    const minY = Math.min(...nodes.map((node) => node.y - node.radius - 50));
    const maxY = Math.max(...nodes.map((node) => node.y + node.radius + 50));
    const graphWidth = Math.max(1, maxX - minX);
    const graphHeight = Math.max(1, maxY - minY);
    const panelReserve = rect.width > 900 ? 340 : 0;
    const usableWidth = Math.max(320, rect.width - panelReserve - 56);
    const usableHeight = Math.max(280, rect.height - 56);
    const scale = Math.min(1.12, Math.max(0.24, Math.min(usableWidth / graphWidth, usableHeight / graphHeight)));
    transform = {
      x: 28 + usableWidth / 2 - ((minX + maxX) / 2) * scale,
      y: rect.height / 2 - ((minY + maxY) / 2) * scale,
      scale,
    };
  };

  let lastZoomTimestamp = 0;
  createEffect(() => {
    const trigger = props.zoomTrigger;
    if (!trigger || trigger.timestamp <= lastZoomTimestamp) return;
    lastZoomTimestamp = trigger.timestamp;
    if (trigger.action === 'recenter') {
      recenter();
      return;
    }
    const factor = trigger.action === 'in' ? 1.18 : trigger.action === 'out' ? 0.84 : 1;
    transform.scale = Math.min(2.8, Math.max(0.22, transform.scale * factor));
  });

  const render = () => {
    if (!canvasRef) return;
    const ctx = canvasRef.getContext('2d');
    if (!ctx) return;
    const rect = canvasRef.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const rootStyle = getComputedStyle(document.documentElement);
    const atlasTheme = readAtlasColorTheme(rootStyle);
    const theme: CanvasTheme = {
      bg: rootStyle.getPropertyValue('--canvas-bg').trim(),
      grid: rootStyle.getPropertyValue('--canvas-grid').trim(),
      text: rootStyle.getPropertyValue('--text-primary').trim(),
      surface: rootStyle.getPropertyValue('--color-surface').trim(),
      article: atlasTheme.article,
      hover: rootStyle.getPropertyValue('--node-hover').trim(),
      selected: rootStyle.getPropertyValue('--node-selected').trim(),
      palette: atlasTheme.palette,
    };
    drawBackground(ctx, rect.width, rect.height, theme);
    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    const currentLayout = layout();
    const propertyColors = buildPropertyColorMap(
      currentLayout.propertyNodes.map((node) => node.propertyName || node.label),
      atlasTheme,
    );
    const focusNode = hoveredNode || selectedNode;
    const focusedNodeIds = focusNeighborhood(currentLayout, focusNode);
    const highlightState = updateHighlightState(props.graphData, props.filterState);
    const dense = currentLayout.edgeCount > 220;
    for (const edge of currentLayout.edges) {
      drawEdge(ctx, edge, props.filterState, highlightState, focusNode, theme, propertyColors, dense);
    }
    for (const node of currentLayout.nodes) {
      drawNode(
        ctx,
        node,
        props.filterState,
        highlightState,
        hoveredNode,
        selectedNode,
        focusNode,
        focusedNodeIds,
        transform.scale,
        theme,
        propertyColors,
      );
    }
    ctx.restore();
  };

  const resize = () => {
    if (!canvasRef) return;
    const rect = canvasRef.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvasRef.width = Math.round(rect.width * dpr);
    canvasRef.height = Math.round(rect.height * dpr);
    recenter();
  };

  const animate = () => {
    render();
    frame = requestAnimationFrame(animate);
  };

  const updateCursor = (cursor: string) => {
    if (canvasRef) canvasRef.style.cursor = cursor;
  };

  const handlePointerDown = (event: PointerEvent) => {
    if (!canvasRef) return;
    canvasRef.setPointerCapture(event.pointerId);
    pointerStart = { x: event.clientX, y: event.clientY };
    lastPoint = pointerStart;
    moved = false;
    const node = getNodeAtPoint(interactiveNodes(), worldFromEvent(event), transform.scale);
    if (node) {
      interaction = 'node';
      draggedNode = node;
      previousFix = { x: node.fx ?? null, y: node.fy ?? null };
      node.fx = node.x;
      node.fy = node.y;
      simulation?.alphaTarget(0.16).restart();
      updateCursor('grabbing');
      return;
    }
    interaction = 'pan';
    updateCursor('grabbing');
  };

  const handlePointerMove = (event: PointerEvent) => {
    if (!canvasRef) return;
    if (interaction !== 'idle') {
      moved = moved || Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 4;
    }
    if (interaction === 'node' && draggedNode) {
      const point = worldFromEvent(event);
      draggedNode.fx = point.x;
      draggedNode.fy = point.y;
      return;
    }
    if (interaction === 'pan') {
      transform.x += event.clientX - lastPoint.x;
      transform.y += event.clientY - lastPoint.y;
      lastPoint = { x: event.clientX, y: event.clientY };
      return;
    }

    const rect = canvasRef.getBoundingClientRect();
    hoveredNode = getNodeAtPoint(interactiveNodes(), worldFromEvent(event), transform.scale);
    updateCursor(hoveredNode ? 'grab' : 'default');
    setTooltip({
      visible: Boolean(hoveredNode),
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
      node: hoveredNode,
    });
  };

  const handlePointerUp = (event: PointerEvent) => {
    if (!canvasRef) return;
    if (interaction === 'node' && draggedNode) {
      if (!moved) {
        draggedNode.fx = previousFix?.x ?? undefined;
        draggedNode.fy = previousFix?.y ?? undefined;
        selectedNode = draggedNode;
        setFocusedNode(draggedNode);
        props.onNodeSelect(draggedNode.source);
        props.onFilterChange(addOrToggleFilter(draggedNode, props.filterState, event.ctrlKey || event.metaKey));
      } else {
        selectedNode = draggedNode;
        setFocusedNode(draggedNode);
        props.onNodeSelect(draggedNode.source);
      }
    }
    simulation?.alphaTarget(0);
    interaction = 'idle';
    draggedNode = null;
    previousFix = null;
    updateCursor('default');
    if (canvasRef.hasPointerCapture(event.pointerId)) canvasRef.releasePointerCapture(event.pointerId);
  };

  const handleDoubleClick = (event: MouseEvent) => {
    const node = getNodeAtPoint(interactiveNodes(), worldFromEvent(event), transform.scale);
    if (!node) return;
    node.fx = undefined;
    node.fy = undefined;
    simulation?.alpha(0.45).restart();
  };

  const handleWheel = (event: WheelEvent) => {
    if (!canvasRef) return;
    event.preventDefault();
    const rect = canvasRef.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    const before = { x: (mouseX - transform.x) / transform.scale, y: (mouseY - transform.y) / transform.scale };
    transform.scale = Math.min(2.8, Math.max(0.22, transform.scale * (event.deltaY < 0 ? 1.08 : 0.92)));
    transform.x = mouseX - before.x * transform.scale;
    transform.y = mouseY - before.y * transform.scale;
  };

  onMount(() => {
    if (!canvasRef) return;
    resize();
    animate();
    window.addEventListener('resize', resize);
    canvasRef.addEventListener('wheel', handleWheel, { passive: false });
    onCleanup(() => {
      cancelAnimationFrame(frame);
      window.clearTimeout(autoFitTimer);
      simulation?.stop();
      window.removeEventListener('resize', resize);
      canvasRef?.removeEventListener('wheel', handleWheel);
    });
  });

  createEffect(() => {
    const currentLayout = layout();
    simulation?.stop();
    hoveredNode = null;
    selectedNode = null;
    setFocusedNode(null);
    simulation = createLayoutSimulation(currentLayout);
    window.clearTimeout(autoFitTimer);
    autoFitTimer = window.setTimeout(recenter, 700);
    queueMicrotask(recenter);
  });

  createEffect(() => {
    const currentLayout = layout();
    const articleIds = props.filterState.articleIds || [];
    const propertyFilters = props.filterState.propertyFilters || [];
    let nextFocusedNode: AtlasNode | null = null;

    if (props.filterState.active && articleIds.length === 1 && propertyFilters.length === 0) {
      nextFocusedNode = currentLayout.articleNodes.find(
        (node) => node.articleId === articleIds[0],
      ) || null;
    } else if (props.filterState.active && articleIds.length === 0 && propertyFilters.length === 1) {
      const [filter] = propertyFilters;
      nextFocusedNode = currentLayout.propertyNodes.find(
        (node) => node.propertyName === filter.property && node.label === filter.value,
      ) || null;
    }

    selectedNode = nextFocusedNode;
    setFocusedNode(nextFocusedNode);
    if (!props.filterState.active) {
      hoveredNode = null;
      setTooltip({ visible: false, x: 0, y: 0, node: null });
    }
  });

  return (
    <div class={styles.atlas}>
      <canvas
        ref={canvasRef}
        class={styles.canvas}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onDblClick={handleDoubleClick}
        onPointerLeave={() => {
          if (interaction === 'idle') {
            hoveredNode = null;
            setTooltip({ visible: false, x: 0, y: 0, node: null });
          }
        }}
      />

      <Show when={props.showOverview !== false}>
        <div class={styles.panel}>
          <div class={styles.card}>
            <div class={styles.eyebrow}>Knowledge landscape</div>
            <div class={styles.title}>Article intelligence map</div>
            <p class={styles.subtitle}>Shared attributes pull related articles together. Larger attribute nodes represent stronger coverage across the portfolio.</p>
            <div class={styles.legend}>
              <span class={styles.legendItem}><i class={styles.articleMark} />Article</span>
              <span class={styles.legendItem}><i class={styles.attributeMark} />Attribute value</span>
            </div>
            <div class={styles.stats}>
              <div class={styles.stat}><span class={styles.statValue}>{layout().articleCount}</span><span class={styles.statLabel}>Articles</span></div>
              <div class={styles.stat}><span class={styles.statValue}>{layout().propertyCount}</span><span class={styles.statLabel}>Signals</span></div>
              <div class={styles.stat}><span class={styles.statValue}>{layout().edgeCount}</span><span class={styles.statLabel}>Facts</span></div>
            </div>
            <Show when={focusedNode()} fallback={<div class={styles.hint}>Drag to arrange · Double-click to release · Scroll to zoom</div>}>
              {(node) => (
                <div class={styles.focus}>
                  <div class={styles.focusLabel}>{node().kind === 'article' ? 'Selected article' : formatLabel(node().propertyName || 'Attribute')}</div>
                  <div class={styles.focusValue}>{formatLabel(node().kind === 'article' ? node().articleId || node().label : node().label)}</div>
                  <div class={styles.focusMeta}>{node().degree} connected fact{node().degree === 1 ? '' : 's'}</div>
                </div>
              )}
            </Show>
          </div>
        </div>
      </Show>

      <div class={`${styles.tooltip} ${tooltip().visible ? '' : styles.tooltipHidden}`} style={{ left: `${tooltip().x}px`, top: `${tooltip().y}px` }}>
        <div class={styles.tooltipInner}>
          <div class={styles.tooltipTitle}>{formatLabel(tooltip().node?.label || '')}</div>
          <div class={styles.tooltipSub}>
            {tooltip().node?.kind === 'article'
              ? `${tooltip().node?.degree || 0} known attributes`
              : `${formatLabel(tooltip().node?.propertyName || 'attribute')} · used by ${tooltip().node?.degree || 0} article(s)`}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SemanticAtlas;
