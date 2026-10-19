import { Component, For, Show, createEffect, createMemo, createSignal, onCleanup } from 'solid-js';
import { GraphData, FilterState } from '../../types';
import styles from './EnhancedLegend.module.css';
import { buildPropertyColorMap, readAtlasColorTheme } from '../../features/visualization/atlas/colorMapping';

const ChevronIcon = (props: { expanded: boolean }) => (
  <svg
    viewBox="0 0 24 24"
    aria-hidden="true"
    class={styles.chevron}
    classList={{ [styles.chevronExpanded]: props.expanded }}
  >
    <path d="M9 6l6 6-6 6" />
  </svg>
);

export interface EnhancedLegendProps {
  graphData: GraphData;
  onFilterChange: (filter: FilterState) => void;
  filterState: FilterState;
}

const EnhancedLegend: Component<EnhancedLegendProps> = (props) => {
  const [isLegendCollapsed, setIsLegendCollapsed] = createSignal(true);
  const [areFiltersCollapsed, setAreFiltersCollapsed] = createSignal(false);
  const [activeFilterCategory, setActiveFilterCategory] = createSignal<string | null>(null);

  let filterContentRef: HTMLDivElement | undefined;

  const formatPropertyName = (name?: string) =>
    (name ?? '')
      .replace(/_/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .replace(/(^|\s)\w/g, (segment) => segment.toUpperCase());

  const propertyColumns = createMemo(() => {
    const columns = new Map<string, Set<string>>();

    for (const node of props.graphData.nodes) {
      if (node.metadata.columnType !== 'property' || !node.metadata.propertyName) {
        continue;
      }

      if (!columns.has(node.metadata.propertyName)) {
        columns.set(node.metadata.propertyName, new Set());
      }

      if (node.label !== 'None') {
        columns.get(node.metadata.propertyName)!.add(node.label);
      }
    }

    return columns;
  });

  const propertyEntries = () =>
    Array.from(propertyColumns().entries()).sort(([a], [b]) => a.localeCompare(b));

  const filterCategories = () => [
    { key: 'articles', label: 'Articles' },
    ...propertyEntries().map(([property]) => ({
      key: property,
      label: formatPropertyName(property)
    }))
  ];

  const getCategoryLabel = (key: string) =>
    key === 'articles' ? 'Articles' : formatPropertyName(key);

  const activeFilterCount = createMemo(() =>
    (props.filterState.articleIds?.length || 0) + (props.filterState.propertyFilters?.length || 0)
  );

  const toggleFilterCategory = (category: string) => {
    setActiveFilterCategory((previous) => (previous === category ? null : category));
  };

  createEffect(() => {
    if (areFiltersCollapsed()) {
      setActiveFilterCategory(null);
      return;
    }

    const handleDocumentClick = (event: MouseEvent) => {
      if (!filterContentRef) return;
      if (!filterContentRef.contains(event.target as Node)) {
        setActiveFilterCategory(null);
      }
    };

    document.addEventListener('click', handleDocumentClick);
    onCleanup(() => document.removeEventListener('click', handleDocumentClick));
  });

  const getColorRepresentations = () => {
    const rootStyle = getComputedStyle(document.documentElement);
    const theme = readAtlasColorTheme(rootStyle);
    const colorMap = new Map<string, Set<string>>();
    const propertyNames = props.graphData.nodes
      .filter((node) => node.metadata.columnType === 'property' && node.label !== 'None')
      .map((node) => node.metadata.propertyName || node.label);
    const propertyColorMap = buildPropertyColorMap(propertyNames, theme);

    const addLabel = (color: string, label: string) => {
      const existing = colorMap.get(color) ?? new Set<string>();
      existing.add(label);
      colorMap.set(color, existing);
    };

    for (const node of props.graphData.nodes) {
      if (node.metadata.columnType === 'article') {
        addLabel(theme.article, 'article');
        continue;
      }

      if (node.metadata.columnType === 'property') {
        if (node.label === 'None') continue;
        const propertyName = node.metadata.propertyName || node.label;
        const color = propertyColorMap.get(propertyName) || theme.article;
        const label = formatPropertyName(node.metadata.propertyName || node.label || 'value');
        addLabel(color, label);
      }
    }

    return Array.from(colorMap.entries())
      .map(([color, labels]) => ({
        color,
        labels: Array.from(labels).sort((a, b) => a.localeCompare(b)).join(', ')
      }))
      .sort((a, b) => a.labels.localeCompare(b.labels));
  };

  const getArticles = () => {
    const articles: string[] = [];

    for (const node of props.graphData.nodes) {
      if (node.metadata.columnType === 'article') {
        articles.push(node.metadata.originalExpression || node.label);
      }
    }

    return articles.sort((a, b) => parseInt(a) - parseInt(b));
  };

  const handleArticleClick = (articleId: string, event: MouseEvent) => {
    const isMultiSelect = event.ctrlKey || event.metaKey;
    const currentArticleIds = new Set(props.filterState.articleIds || []);

    if (isMultiSelect) {
      if (currentArticleIds.has(articleId)) {
        currentArticleIds.delete(articleId);
      } else {
        currentArticleIds.add(articleId);
      }
    } else {
      if (currentArticleIds.size === 1 && currentArticleIds.has(articleId)) {
        currentArticleIds.clear();
      } else {
        currentArticleIds.clear();
        currentArticleIds.add(articleId);
      }
    }

    props.onFilterChange({
      active: currentArticleIds.size > 0 || (props.filterState.propertyFilters?.length || 0) > 0,
      articleIds: Array.from(currentArticleIds),
      propertyFilters: props.filterState.propertyFilters || []
    });
  };

  const handlePropertyClick = (property: string, value: string, event: MouseEvent) => {
    const isMultiSelect = event.ctrlKey || event.metaKey;
    const propertyFilter = { property, value };
    let currentFilters = [...(props.filterState.propertyFilters || [])];

    const index = currentFilters.findIndex((filter) => filter.property === property && filter.value === value);

    if (isMultiSelect) {
      if (index >= 0) {
        currentFilters.splice(index, 1);
      } else {
        currentFilters.push(propertyFilter);
      }
    } else {
      if (index >= 0) {
        currentFilters.splice(index, 1);
      } else {
        currentFilters = currentFilters.filter((filter) => filter.property !== property);
        currentFilters.push(propertyFilter);
      }
    }

    props.onFilterChange({
      active: (props.filterState.articleIds?.length || 0) > 0 || currentFilters.length > 0,
      articleIds: props.filterState.articleIds || [],
      propertyFilters: currentFilters
    });
  };

  const clearFilter = () => {
    props.onFilterChange({
      active: false,
      articleIds: [],
      propertyFilters: []
    });
  };

  return (
    <div class={styles.legendContainer}>
      <div class={styles.toggleGroup}>
        <button class={styles.sectionToggle} onClick={() => setIsLegendCollapsed(!isLegendCollapsed())}>
          <span class={styles.toggleTitle}>
            <span class={styles.toggleAccent} />
            Legend
          </span>
          <ChevronIcon expanded={!isLegendCollapsed()} />
        </button>

        <Show when={!isLegendCollapsed()}>
          <div class={styles.legendContent}>
            <div class={styles.legendSection}>
              <h4>Color representation</h4>
              <For each={getColorRepresentations()}>
                {(entry) => (
                  <div class={styles.legendItem}>
                    <div class={styles.legendColor} style={{ 'background-color': entry.color }} />
                    <span class={styles.colorLabel}>{entry.labels}</span>
                  </div>
                )}
              </For>
            </div>
          </div>
        </Show>

        <button class={styles.sectionToggle} onClick={() => setAreFiltersCollapsed(!areFiltersCollapsed())}>
          <span class={styles.toggleTitle}>
            <span class={styles.toggleAccent} />
            Filters
          </span>
          <span class={styles.toggleRight}>
            <Show when={activeFilterCount() > 0}>
              <span class={styles.filterCount}>{activeFilterCount()}</span>
            </Show>
            <ChevronIcon expanded={!areFiltersCollapsed()} />
          </span>
        </button>

        <Show when={!areFiltersCollapsed()}>
          <div class={styles.legendContent} ref={(element) => (filterContentRef = element)}>
            <Show when={props.filterState.active}>
              <div class={styles.filterStatus}>
                <div>
                  <strong>Active selection</strong>
                  <Show when={props.filterState.articleIds && props.filterState.articleIds.length > 0}>
                    <div>Articles: {props.filterState.articleIds!.join(', ')}</div>
                  </Show>
                  <Show when={props.filterState.propertyFilters && props.filterState.propertyFilters.length > 0}>
                    <div>
                      Properties:{' '}
                      {props.filterState.propertyFilters!
                        .map((filter) => `${formatPropertyName(filter.property)} = ${filter.value}`)
                        .join(', ')}
                    </div>
                  </Show>
                </div>
                <button class={styles.clearButton} onClick={clearFilter}>
                  Clear
                </button>
              </div>
            </Show>

            <div class={styles.legendSection}>
              <h4>Categories</h4>
              <div class={styles.categoryGrid}>
                <For each={filterCategories()}>
                  {(category) => (
                    <button
                      class={styles.categoryButton}
                      classList={{ [styles.categoryButtonActive]: activeFilterCategory() === category.key }}
                      onClick={() => toggleFilterCategory(category.key)}
                    >
                      {category.label}
                    </button>
                  )}
                </For>
              </div>
            </div>

            <Show when={activeFilterCategory()}>
              {(categoryAccessor) => {
                const categoryKey = createMemo(() => categoryAccessor());
                const propertyValues = createMemo(() =>
                  Array.from(propertyColumns().get(categoryKey()) ?? []).sort((a, b) => a.localeCompare(b))
                );

                return (
                  <div class={styles.filterOptions}>
                    <h4>{getCategoryLabel(categoryKey())}</h4>
                    <Show
                      when={categoryKey() === 'articles'}
                      fallback={
                        <div class={styles.itemGrid}>
                          <For each={propertyValues()}>
                            {(value) => (
                              <button
                                class={styles.filterButton}
                                classList={{
                                  [styles.filterButtonActive]: props.filterState.propertyFilters?.some(
                                    (filter) => filter.property === categoryKey() && filter.value === value
                                  )
                                }}
                                onClick={(event) => handlePropertyClick(categoryKey(), value, event)}
                              >
                                {value}
                              </button>
                            )}
                          </For>
                        </div>
                      }
                    >
                      <div class={styles.itemGrid}>
                        <For each={getArticles()}>
                          {(article) => (
                            <button
                              class={styles.filterButton}
                              classList={{
                                [styles.filterButtonActive]: props.filterState.articleIds?.includes(article)
                              }}
                              onClick={(event) => handleArticleClick(article, event)}
                            >
                              {article}
                            </button>
                          )}
                        </For>
                      </div>
                    </Show>
                  </div>
                );
              }}
            </Show>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default EnhancedLegend;
