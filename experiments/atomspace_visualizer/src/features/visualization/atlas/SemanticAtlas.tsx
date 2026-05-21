import { Component, createEffect, createMemo, createSignal, onCleanup, onMount, Show } from 'solid-js';
import type { FilterState, GraphData, GraphNode, Point } from '../../../types';
import styles from './SemanticAtlas.module.css';
import { buildPropertyColorMap, colorForAtlasSubject, readAtlasColorTheme } from './colorMapping';

type AtlasNodeKind = 'article' | 'property';

type AtlasNode = {
  id: string;
  kind: AtlasNodeKind;
  label: string;
  propertyName?: string;
  value?: string;
  articleId?: string;
  x: number;
  y: number;
  radius: number;
  color: string;
  source: GraphNode;
  degree: number;
};

type AtlasEdge = {
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
  ring: string;
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
}

const TARGET_PROPERTIES = new Set(['engagement', 'audience-expertise']);

const formatLabel = (value: string) =>
  value
    .replace(/^"|"$/g, '')
    .replace(/[_-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const colorForNode = (node: AtlasNode, theme: CanvasTheme, propertyColors: Map<string, string>) => {
  return colorForAtlasSubject(
    {
      kind: node.kind,
      label: node.label,
      propertyName: node.propertyName,
    },
    theme,
    propertyColors,
  );
};

const getActiveArticleIds = (filterState: FilterState) => new Set(filterState.articleIds || []);

const getActivePropertyFilters = (filterState: FilterState) =>
  new Set((filterState.propertyFilters || []).map((filter) => `${filter.property}:${filter.value}`));

const shouldHighlightNode = (node: AtlasNode, filterState: FilterState) => {
  if (!filterState.active) {
    return true;
  }

  const articleIds = getActiveArticleIds(filterState);
  const propertyFilters = getActivePropertyFilters(filterState);
  const hasArticleFilter = articleIds.size > 0;
  const hasPropertyFilter = propertyFilters.size > 0;

  if (node.kind === 'article') {
    return !hasArticleFilter || articleIds.has(node.articleId || '');
  }

  const propertyKey = `${node.propertyName}:${node.label}`;
  return !hasPropertyFilter || propertyFilters.has(propertyKey);
};

const shouldHighlightEdge = (edge: AtlasEdge, filterState: FilterState) => {
  if (!filterState.active) {
    return true;
  }

  return shouldHighlightNode(edge.source, filterState) && shouldHighlightNode(edge.target, filterState);
};

const buildLayout = (graphData: GraphData): AtlasLayout => {
  const articleSources = graphData.nodes.filter((node) => node.metadata.columnType === 'article');
  const propertySources = graphData.nodes.filter((node) => node.metadata.columnType === 'property' && node.label !== 'None');
  const degreeByNode = new Map<string, number>();

  for (const edge of graphData.edges) {
    degreeByNode.set(edge.source, (degreeByNode.get(edge.source) || 0) + 1);
    degreeByNode.set(edge.target, (degreeByNode.get(edge.target) || 0) + 1);
  }

  const articleNodes = articleSources.map<AtlasNode>((node, index) => {
    const lane = index % 2 === 0 ? -1 : 1;
    const row = Math.floor(index / 2);
    return {
      id: node.id,
      kind: 'article',
      label: node.label,
      articleId: node.metadata.originalExpression || node.label,
      x: lane * 88,
      y: row * 50 - (articleSources.length * 25) / 2,
      radius: 7 + Math.min(8, (degreeByNode.get(node.id) || 1) * 0.55),
      color: '',
      source: node,
      degree: degreeByNode.get(node.id) || 0,
    };
  });

  const propertiesByName = new Map<string, GraphNode[]>();
  for (const node of propertySources) {
    const propertyName = node.metadata.propertyName || 'property';
    const nodes = propertiesByName.get(propertyName) || [];
    nodes.push(node);
    propertiesByName.set(propertyName, nodes);
  }

  const sortedProperties = Array.from(propertiesByName.entries()).sort(([a], [b]) => {
    const aTarget = TARGET_PROPERTIES.has(a) ? 0 : 1;
    const bTarget = TARGET_PROPERTIES.has(b) ? 0 : 1;
    return aTarget - bTarget || a.localeCompare(b);
  });

  const propertyNodes: AtlasNode[] = [];
  const bandCount = Math.max(1, sortedProperties.length);
  sortedProperties.forEach(([propertyName, nodes], propertyIndex) => {
    const baseAngle = -Math.PI / 2 + (propertyIndex / bandCount) * Math.PI * 2;
    const sortedNodes = [...nodes].sort((a, b) => a.label.localeCompare(b.label));
    const bandRadius = TARGET_PROPERTIES.has(propertyName) ? 310 : 390 + (propertyIndex % 3) * 54;
    const spread = Math.min(Math.PI / 4, 0.14 * Math.max(1, sortedNodes.length));

    sortedNodes.forEach((node, valueIndex) => {
      const valueOffset = sortedNodes.length === 1 ? 0 : (valueIndex / (sortedNodes.length - 1) - 0.5) * spread;
      const angle = baseAngle + valueOffset;
      const degree = degreeByNode.get(node.id) || 0;
      propertyNodes.push({
        id: node.id,
        kind: 'property',
        label: node.label,
        propertyName,
        value: node.metadata.originalExpression || node.label,
        x: Math.cos(angle) * bandRadius,
        y: Math.sin(angle) * bandRadius,
        radius: 11 + Math.min(18, Math.sqrt(degree) * 2.5),
        color: '',
        source: node,
        degree,
      });
    });
  });

  const nodesById = new Map([...articleNodes, ...propertyNodes].map((node) => [node.id, node]));
  const edges = graphData.edges
    .map<AtlasEdge | null>((edge) => {
      const source = nodesById.get(edge.source);
      const target = nodesById.get(edge.target);
      if (!source || !target) {
        return null;
      }
      return {
        id: edge.id,
        source,
        target,
        property: edge.label,
      };
    })
    .filter((edge): edge is AtlasEdge => edge !== null);

  return {
    nodes: [...articleNodes, ...propertyNodes],
    edges,
    articleNodes,
    propertyNodes,
    propertyCount: sortedProperties.length,
    articleCount: articleNodes.length,
    edgeCount: edges.length,
  };
};

const drawBackground = (ctx: CanvasRenderingContext2D, width: number, height: number, theme: CanvasTheme) => {
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.globalAlpha = 0.16;
  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = 1;
  for (let x = -height; x < width + height; x += 48) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x + height, height);
    ctx.stroke();
  }
  ctx.restore();
};

const drawEdge = (
  ctx: CanvasRenderingContext2D,
  edge: AtlasEdge,
  filterState: FilterState,
  hoveredNode: AtlasNode | null,
  theme: CanvasTheme,
  propertyColors: Map<string, string>,
) => {
  const highlighted = shouldHighlightEdge(edge, filterState);
  const hoverLinked = hoveredNode ? edge.source.id === hoveredNode.id || edge.target.id === hoveredNode.id : false;
  const alpha = hoverLinked ? 0.72 : highlighted ? 0.34 : 0.035;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = colorForNode(edge.target, theme, propertyColors);
  ctx.lineWidth = hoverLinked ? 2.4 : highlighted ? 1.45 : 0.8;
  ctx.beginPath();
  const controlX = (edge.source.x + edge.target.x) / 2;
  const controlY = (edge.source.y + edge.target.y) / 2 + (edge.source.x < edge.target.x ? 34 : -34);
  ctx.moveTo(edge.source.x, edge.source.y);
  ctx.quadraticCurveTo(controlX, controlY, edge.target.x, edge.target.y);
  ctx.stroke();
  ctx.restore();
};

const drawNode = (
  ctx: CanvasRenderingContext2D,
  node: AtlasNode,
  filterState: FilterState,
  hoveredNode: AtlasNode | null,
  selectedNode: AtlasNode | null,
  scale: number,
  theme: CanvasTheme,
  propertyColors: Map<string, string>,
) => {
  const highlighted = shouldHighlightNode(node, filterState);
  const hovered = hoveredNode?.id === node.id;
  const selected = selectedNode?.id === node.id;

  ctx.save();
  ctx.globalAlpha = highlighted ? 1 : 0.18;

  const nodeColor = colorForNode(node, theme, propertyColors);
  ctx.shadowColor = nodeColor;
  ctx.shadowBlur = (hovered || selected ? 18 : 8) / scale;

  ctx.fillStyle = node.kind === 'article' ? theme.surface : nodeColor;
  ctx.strokeStyle = selected ? theme.selected : hovered ? theme.hover : nodeColor;
  ctx.lineWidth = (selected || hovered ? 3 : 1.6) / scale;
  ctx.beginPath();
  ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.stroke();

  if (node.kind === 'article') {
    ctx.fillStyle = theme.article;
    ctx.beginPath();
    ctx.arc(node.x, node.y, Math.max(2.5, node.radius * 0.42), 0, Math.PI * 2);
    ctx.fill();
  }

  if (scale > 0.42 || hovered || selected) {
    ctx.font = `${Math.max(10, 12 / scale)}px Inter, system-ui, sans-serif`;
    ctx.textAlign = node.kind === 'article' ? 'center' : node.x < 0 ? 'right' : 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = theme.text;
    const label = node.kind === 'article' ? formatLabel(node.articleId || node.label) : formatLabel(node.label);
    const offset = node.kind === 'article' ? 20 / scale : (node.x < 0 ? -16 : 16) / scale;
    ctx.fillText(label, node.x + offset, node.y);
  }

  ctx.restore();
};

const getNodeAtPoint = (nodes: AtlasNode[], point: Point, scale: number) => {
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    const node = nodes[index];
    const dx = point.x - node.x;
    const dy = point.y - node.y;
    const hitRadius = Math.max(node.radius + 8 / scale, 14 / scale);
    if (dx * dx + dy * dy <= hitRadius * hitRadius) {
      return node;
    }
  }
  return null;
};

const addOrToggleFilter = (node: AtlasNode, filterState: FilterState, multi: boolean): FilterState => {
  if (node.kind === 'article') {
    const articleId = node.articleId || '';
    const articleIds = new Set(filterState.articleIds || []);
    if (!multi) {
      articleIds.clear();
    }
    if (articleIds.has(articleId)) {
      articleIds.delete(articleId);
    } else {
      articleIds.add(articleId);
    }
    return {
      active: articleIds.size > 0 || (filterState.propertyFilters?.length || 0) > 0,
      articleIds: Array.from(articleIds),
      propertyFilters: filterState.propertyFilters || [],
    };
  }

  const propertyFilter = {
    property: node.propertyName || '',
    value: node.label,
  };
  const propertyFilters = multi ? [...(filterState.propertyFilters || [])] : [];
  const existing = propertyFilters.findIndex(
    (filter) => filter.property === propertyFilter.property && filter.value === propertyFilter.value,
  );

  if (existing >= 0) {
    propertyFilters.splice(existing, 1);
  } else {
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
  let transform = { x: 0, y: 0, scale: 1 };
  let isPanning = false;
  let lastPoint: Point = { x: 0, y: 0 };
  let hoveredNode: AtlasNode | null = null;
  let selectedNode: AtlasNode | null = null;

  const [tooltip, setTooltip] = createSignal({ visible: false, x: 0, y: 0, node: null as AtlasNode | null });
  const [focusedNode, setFocusedNode] = createSignal<AtlasNode | null>(null);

  const layout = createMemo(() => buildLayout(props.graphData));

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
    transform = {
      x: rect.width / 2,
      y: rect.height / 2,
      scale: Math.min(1.1, Math.max(0.55, rect.width / 1200)),
    };
  };

  let lastZoomTimestamp = 0;
  createEffect(() => {
    const trigger = props.zoomTrigger;
    if (!trigger || trigger.timestamp <= lastZoomTimestamp) {
      return;
    }
    lastZoomTimestamp = trigger.timestamp;

    if (trigger.action === 'recenter') {
      recenter();
      return;
    }

    const factor = trigger.action === 'in' ? 1.18 : trigger.action === 'out' ? 0.84 : 1;
    transform.scale = Math.min(2.6, Math.max(0.32, transform.scale * factor));
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
      ring: rootStyle.getPropertyValue('--color-border-strong').trim(),
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

    ctx.save();
    ctx.globalAlpha = 0.34;
    ctx.strokeStyle = theme.ring;
    ctx.lineWidth = 1 / transform.scale;
    for (const radius of [250, 360, 470, 580]) {
      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();

    for (const edge of currentLayout.edges) {
      drawEdge(ctx, edge, props.filterState, hoveredNode, theme, propertyColors);
    }

    for (const node of currentLayout.nodes) {
      drawNode(ctx, node, props.filterState, hoveredNode, selectedNode, transform.scale, theme, propertyColors);
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

  const handlePointerDown = (event: PointerEvent) => {
    if (!canvasRef) return;
    const worldPoint = worldFromEvent(event);
    const node = getNodeAtPoint(layout().nodes, worldPoint, transform.scale);

    if (node) {
      selectedNode = node;
      setFocusedNode(node);
      props.onNodeSelect(node.source);
      props.onFilterChange(addOrToggleFilter(node, props.filterState, event.ctrlKey || event.metaKey));
      return;
    }

    isPanning = true;
    lastPoint = { x: event.clientX, y: event.clientY };
  };

  const handlePointerMove = (event: PointerEvent) => {
    if (!canvasRef) return;

    if (isPanning) {
      transform.x += event.clientX - lastPoint.x;
      transform.y += event.clientY - lastPoint.y;
      lastPoint = { x: event.clientX, y: event.clientY };
      return;
    }

    const worldPoint = worldFromEvent(event);
    hoveredNode = getNodeAtPoint(layout().nodes, worldPoint, transform.scale);
    setTooltip({
      visible: Boolean(hoveredNode),
      x: event.clientX,
      y: event.clientY,
      node: hoveredNode,
    });
  };

  const handlePointerUp = () => {
    isPanning = false;
  };

  const handleWheel = (event: WheelEvent) => {
    if (!canvasRef) return;
    event.preventDefault();
    const rect = canvasRef.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    const before = {
      x: (mouseX - transform.x) / transform.scale,
      y: (mouseY - transform.y) / transform.scale,
    };
    const factor = event.deltaY < 0 ? 1.08 : 0.92;
    transform.scale = Math.min(2.6, Math.max(0.32, transform.scale * factor));
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
      window.removeEventListener('resize', resize);
      canvasRef?.removeEventListener('wheel', handleWheel);
    });
  });

  createEffect(() => {
    layout();
    queueMicrotask(recenter);
  });

  return (
    <div class={styles.atlas}>
      <canvas
        ref={canvasRef}
        class={styles.canvas}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={() => {
          isPanning = false;
          hoveredNode = null;
          setTooltip({ visible: false, x: 0, y: 0, node: null });
        }}
      />

      <div class={styles.panel}>
        <div class={styles.card}>
          <div class={styles.eyebrow}>Semantic Atlas</div>
          <div class={styles.title}>Knowledge topology</div>
          <p class={styles.subtitle}>Articles anchor the center while properties orbit by semantic family and evidence strength.</p>
          <div class={styles.stats}>
            <div class={styles.stat}>
              <span class={styles.statValue}>{layout().articleCount}</span>
              <span class={styles.statLabel}>Articles</span>
            </div>
            <div class={styles.stat}>
              <span class={styles.statValue}>{layout().propertyCount}</span>
              <span class={styles.statLabel}>Properties</span>
            </div>
            <div class={styles.stat}>
              <span class={styles.statValue}>{layout().edgeCount}</span>
              <span class={styles.statLabel}>Facts</span>
            </div>
          </div>
          <Show when={focusedNode()}>
            {(node) => (
              <div class={styles.focus}>
                <div class={styles.focusLabel}>{node().kind === 'article' ? 'Selected article' : node().propertyName}</div>
                <div class={styles.focusValue}>{formatLabel(node().kind === 'article' ? node().articleId || node().label : node().label)}</div>
                <div class={styles.focusMeta}>{node().degree} connected fact{node().degree === 1 ? '' : 's'}</div>
              </div>
            )}
          </Show>
        </div>
      </div>

      <div
        class={`${styles.tooltip} ${tooltip().visible ? '' : styles.tooltipHidden}`}
        style={{ left: `${tooltip().x}px`, top: `${tooltip().y}px` }}
      >
        <div class={styles.tooltipInner}>
          <div class={styles.tooltipTitle}>{formatLabel(tooltip().node?.label || '')}</div>
          <div class={styles.tooltipSub}>
            {tooltip().node?.kind === 'article'
              ? `${tooltip().node?.degree || 0} linked properties`
              : `${formatLabel(tooltip().node?.propertyName || 'property')} value across ${tooltip().node?.degree || 0} article(s)`}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SemanticAtlas;
