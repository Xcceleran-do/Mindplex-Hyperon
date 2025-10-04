// Enhanced Legend component with collapsible and clickable items
import { Component, createSignal, For } from 'solid-js';
import { GraphData, FilterState } from '../../types';
import styles from './EnhancedLegend.module.css';

export interface EnhancedLegendProps {
  graphData: GraphData;
  onFilterChange: (filter: FilterState) => void;
  filterState: FilterState;
}

const EnhancedLegend: Component<EnhancedLegendProps> = (props) => {
  const [isCollapsed, setIsCollapsed] = createSignal(true);

  // Extract unique property columns and values
  const getPropertyColumns = () => {
    const columns = new Map<string, Set<string>>();
    
    for (const node of props.graphData.nodes) {
      if (node.metadata.columnType === 'property' && node.metadata.propertyName) {
        if (!columns.has(node.metadata.propertyName)) {
          columns.set(node.metadata.propertyName, new Set());
        }
        if (node.label !== 'None') {
          columns.get(node.metadata.propertyName)!.add(node.label);
        }
      }
    }
    
    return columns;
  };

  // Extract articles
  const getArticles = () => {
    const articles: string[] = [];
    for (const node of props.graphData.nodes) {
      if (node.metadata.columnType === 'article') {
        articles.push(node.metadata.originalExpression || node.label);
      }
    }
    return articles.sort((a, b) => parseInt(a) - parseInt(b));
  };

  const handleArticleClick = (articleId: string, e: MouseEvent) => {
    const isMultiSelect = e.ctrlKey || e.metaKey;
    const currentArticleIds = new Set(props.filterState.articleIds || []);
    
    if (isMultiSelect) {
      // Toggle article in selection
      if (currentArticleIds.has(articleId)) {
        currentArticleIds.delete(articleId);
      } else {
        currentArticleIds.add(articleId);
      }
    } else {
      // Single select - if clicking same article, clear; otherwise select only this one
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

  const handlePropertyClick = (property: string, value: string, e: MouseEvent) => {
    const isMultiSelect = e.ctrlKey || e.metaKey;
    const propertyFilter = { property, value };
    let currentFilters = [...(props.filterState.propertyFilters || [])];
    
    if (isMultiSelect) {
      // Toggle property filter
      const index = currentFilters.findIndex(
        f => f.property === property && f.value === value
      );
      if (index >= 0) {
        currentFilters.splice(index, 1);
      } else {
        currentFilters.push(propertyFilter);
      }
    } else {
      // Single select - if clicking same property, clear; otherwise select only this one
      if (currentFilters.length === 1 && 
          currentFilters[0].property === property && 
          currentFilters[0].value === value) {
        currentFilters = [];
      } else {
        currentFilters = [propertyFilter];
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
      <div class={styles.legendHeader} onClick={() => setIsCollapsed(!isCollapsed())}>
        <h3>Legend & Filters</h3>
        <button class={styles.collapseButton}>
          {isCollapsed() ? '▼' : '▲'}
        </button>
      </div>
      
      {!isCollapsed() && (
        <div class={styles.legendContent}>
          {props.filterState.active && (
            <div class={styles.filterStatus}>
              <div>
                <strong>Active filters:</strong>
                {props.filterState.articleIds && props.filterState.articleIds.length > 0 && (
                  <div>Articles: {props.filterState.articleIds.join(', ')}</div>
                )}
                {props.filterState.propertyFilters && props.filterState.propertyFilters.length > 0 && (
                  <div>
                    Properties: {props.filterState.propertyFilters.map(f => `${f.property}=${f.value}`).join(', ')}
                  </div>
                )}
                <small style={{ display: 'block', 'margin-top': '4px', opacity: 0.7 }}>
                  Hold Ctrl/Cmd to select multiple
                </small>
              </div>
              <button class={styles.clearButton} onClick={clearFilter}>
                Clear All
              </button>
            </div>
          )}

          <div class={styles.legendSection}>
            <h4>Node Types</h4>
            <div class={styles.legendItem}>
              <div class={styles.legendColor} style={{ 'background-color': '#3b82f6' }}></div>
              <span>Article</span>
            </div>
            <div class={styles.legendItem}>
              <div class={styles.legendColor} style={{ 'background-color': '#8b5cf6' }}></div>
              <span>Property Header</span>
            </div>
            <div class={styles.legendItem}>
              <div class={styles.legendColor} style={{ 'background-color': '#10b981' }}></div>
              <span>Property Value</span>
            </div>
            <div class={styles.legendItem}>
              <div class={styles.legendColor} style={{ 'background-color': '#6b7280' }}></div>
              <span>None (Missing)</span>
            </div>
          </div>

          <div class={styles.legendSection}>
            <h4>Articles (click to filter)</h4>
            <div class={styles.itemGrid}>
              <For each={getArticles()}>
                {(article) => (
                  <button
                    class={`${styles.filterButton} ${
                      props.filterState.articleIds?.includes(article)
                        ? styles.active
                        : ''
                    }`}
                    onClick={(e) => handleArticleClick(article, e)}
                  >
                    {article}
                  </button>
                )}
              </For>
            </div>
          </div>

          <For each={Array.from(getPropertyColumns().entries())}>
            {([property, values]) => (
              <div class={styles.legendSection}>
                <h4>{property.replace(/_/g, ' ')} (click to filter)</h4>
                <div class={styles.itemGrid}>
                  <For each={Array.from(values).sort()}>
                    {(value) => (
                      <button
                        class={`${styles.filterButton} ${
                          props.filterState.propertyFilters?.some(
                            f => f.property === property && f.value === value
                          )
                            ? styles.active
                            : ''
                        }`}
                        onClick={(e) => handlePropertyClick(property, value, e)}
                      >
                        {value}
                      </button>
                    )}
                  </For>
                </div>
              </div>
            )}
          </For>
        </div>
      )}
    </div>
  );
};

export default EnhancedLegend;
