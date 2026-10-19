// Highlight state management shared by the columnar and atlas visualizers.
import { FilterState, GraphData, HighlightState } from '../types';

type PropertyFilter = { property: string; value: string };

const activeArticleIds = (filterState: FilterState) => {
  const ids = new Set(filterState.articleIds || []);
  if (filterState.articleId) ids.add(filterState.articleId);
  return ids;
};

const activePropertyFilters = (filterState: FilterState): PropertyFilter[] => {
  const filters = [...(filterState.propertyFilters || [])];
  if (
    filterState.property
    && filterState.value
    && !filters.some(
      (filter) => filter.property === filterState.property && filter.value === filterState.value,
    )
  ) {
    filters.push({ property: filterState.property, value: filterState.value });
  }
  return filters;
};

export const updateHighlightState = (
  graphData: GraphData,
  filterState: FilterState,
): HighlightState => {
  const highlightedNodes = new Set<string>();
  const highlightedEdges = new Set<string>();
  const dimmedNodes = new Set<string>();
  const dimmedEdges = new Set<string>();

  if (!filterState.active) {
    return { highlightedNodes, highlightedEdges, dimmedNodes, dimmedEdges };
  }

  const articleIds = activeArticleIds(filterState);
  const propertyFilters = activePropertyFilters(filterState);
  const propertyNodes = graphData.nodes.filter((node) => node.metadata.columnType === 'property');
  const propertyNodeIdsByFilter = propertyFilters.map(
    (filter) => new Set(
      propertyNodes
        .filter(
          (node) => node.metadata.propertyName === filter.property && node.label === filter.value,
        )
        .map((node) => node.id),
    ),
  );

  const matchingArticleNodeIds = new Set(
    graphData.nodes
      .filter((node) => node.metadata.columnType === 'article')
      .filter((node) => {
        const articleId = node.metadata.originalExpression || node.label;
        if (articleIds.size > 0 && !articleIds.has(articleId)) return false;

        // Every selected attribute is mandatory. This intentionally also means
        // that selecting two different values of one property yields no article.
        return propertyNodeIdsByFilter.every((propertyNodeIds) =>
          graphData.edges.some(
            (edge) => edge.source === node.id && propertyNodeIds.has(edge.target),
          ));
      })
      .map((node) => node.id),
  );

  matchingArticleNodeIds.forEach((id) => highlightedNodes.add(id));

  const selectedPropertyNodeIds = new Set(propertyNodeIdsByFilter.flatMap((ids) => [...ids]));
  selectedPropertyNodeIds.forEach((id) => highlightedNodes.add(id));

  for (const edge of graphData.edges) {
    if (!matchingArticleNodeIds.has(edge.source)) continue;
    if (selectedPropertyNodeIds.size > 0 && !selectedPropertyNodeIds.has(edge.target)) continue;
    highlightedEdges.add(edge.id);
    highlightedNodes.add(edge.target);
  }

  for (const node of graphData.nodes) {
    if (!highlightedNodes.has(node.id)) dimmedNodes.add(node.id);
  }
  for (const edge of graphData.edges) {
    if (!highlightedEdges.has(edge.id)) dimmedEdges.add(edge.id);
  }

  return { highlightedNodes, highlightedEdges, dimmedNodes, dimmedEdges };
};
