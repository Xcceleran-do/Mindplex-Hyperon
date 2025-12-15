// Columnar Graph Transformer - transforms property-based data into columnar layout
import { 
  Triple, 
  GraphNode, 
  GraphEdge, 
  GraphData, 
  GraphMetadata,
  Point 
} from '../../types';

export interface PropertyColumn {
  name: string;
  values: Set<string>;
  position: number;
}

export interface Article {
  id: string;
  properties: Map<string, string>;
}

export class ColumnarTransformer {
  private readonly COLUMN_SPACING = 300;
  private readonly NODE_SPACING = 120;
  private readonly HEADER_HEIGHT = 100;

  transformToColumnar(triples: Triple[]): GraphData {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];

    // Extract articles and their properties
    const articlesMap = this.extractArticles(triples);
    const columns = this.identifyColumns(articlesMap);

    // Create article nodes (first column)
    this.createArticleNodes(articlesMap, nodes);

    // Create property column nodes
    this.createPropertyNodes(columns, nodes);

    // Create edges from articles to properties
    this.createPropertyEdges(articlesMap, columns, edges);

    return {
      nodes,
      edges,
      hypergraphs: [],
      metadata: this.createMetadata(nodes, edges)
    };
  }

  private extractArticles(triples: Triple[]): Map<string, Article> {
    const articlesMap = new Map<string, Article>();

    for (const triple of triples) {
      const subjects = Array.isArray(triple.subject) ? triple.subject : [triple.subject];
      const objects = Array.isArray(triple.object) ? triple.object : [triple.object];

      for (const subject of subjects) {
        if (!articlesMap.has(subject)) {
          articlesMap.set(subject, {
            id: subject,
            properties: new Map<string, string>()
          });
        }

        const article = articlesMap.get(subject)!;
        for (const object of objects) {
          article.properties.set(triple.predicate, object);
        }
      }
    }

    return articlesMap;
  }

  private identifyColumns(articlesMap: Map<string, Article>): PropertyColumn[] {
    const propertyValues = new Map<string, Set<string>>();

    // Collect all unique values for each property
    for (const article of articlesMap.values()) {
      for (const [property, value] of article.properties) {
        if (!propertyValues.has(property)) {
          propertyValues.set(property, new Set<string>());
        }
        propertyValues.get(property)!.add(value);
      }
    }

    // Create columns with positions
    const columns: PropertyColumn[] = [];
    let columnIndex = 1; // Start at 1 because column 0 is for articles

    // Define priority columns that should appear first
    const priorityColumns = ['audience-expertise', 'engagement'];
    
    // Sort properties: priority ones first, then alphabetical
    const sortedProperties = Array.from(propertyValues.keys()).sort((a, b) => {
      const indexA = priorityColumns.indexOf(a);
      const indexB = priorityColumns.indexOf(b);
      
      if (indexA !== -1 && indexB !== -1) return indexA - indexB;
      if (indexA !== -1) return -1;
      if (indexB !== -1) return 1;
      
      return a.localeCompare(b);
    });

    for (const name of sortedProperties) {
      const values = propertyValues.get(name)!;
      columns.push({
        name,
        values,
        position: columnIndex * this.COLUMN_SPACING
      });
      columnIndex++;
    }

    return columns;
  }

  private createArticleNodes(articlesMap: Map<string, Article>, nodes: GraphNode[]): void {
    const articleIds = Array.from(articlesMap.keys()).sort((a, b) => {
      const numA = parseInt(a);
      const numB = parseInt(b);
      return numA - numB;
    });

    articleIds.forEach((articleId, index) => {
      nodes.push({
        id: `article-${articleId}`,
        label: `Article ${articleId}`,
        type: 'entity',
        position: {
          x: 0,
          y: this.HEADER_HEIGHT + index * this.NODE_SPACING
        },
        color: '#3b82f6',
        size: 50,
        metadata: {
          originalExpression: articleId,
          occurrences: 1,
          isGenerated: false,
          columnType: 'article'
        }
      });
    });
  }

  private createPropertyNodes(columns: PropertyColumn[], nodes: GraphNode[]): void {
    for (const column of columns) {
      const isTarget = ['audience-expertise', 'engagement'].includes(column.name);
      
      const label = column.name.replace(/_/g, ' ').toUpperCase() + (isTarget ? ' 🎯' : '');
      
      // Add column header node with better color
      nodes.push({
        id: `header-${column.name}`,
        label: label,
        type: 'predicate',
        position: {
          x: column.position,
          y: 20
        },
        color: this.getHeaderColor(column.name),
        size: isTarget ? 90 : 70, // Larger size for target columns
        metadata: {
          originalExpression: column.name,
          occurrences: 1,
          isGenerated: false,
          columnType: 'header',
          isTarget: isTarget
        }
      });

      // Add value nodes
      const values = Array.from(column.values).sort();
      values.forEach((value, index) => {
        nodes.push({
          id: `${column.name}-${value}`,
          label: value,
          type: 'value',
          position: {
            x: column.position,
            y: this.HEADER_HEIGHT + index * this.NODE_SPACING
          },
          color: this.getPropertyValueColor(column.name),
          size: isTarget ? 60 : 45, // Larger size for target values
          metadata: {
            originalExpression: value,
            occurrences: 1,
            isGenerated: false,
            columnType: 'property',
            propertyName: column.name
          }
        });
      });
    }
  }

  private createPropertyEdges(
    articlesMap: Map<string, Article>,
    columns: PropertyColumn[],
    edges: GraphEdge[]
  ): void {
    for (const [articleId, article] of articlesMap) {
      for (const column of columns) {
        const value = article.properties.get(column.name);
        
        if (value) {
          const targetId = `${column.name}-${value}`;

          edges.push({
            id: `edge-${articleId}-${column.name}`,
            source: `article-${articleId}`,
            target: targetId,
            label: column.name,
            directed: true,
            type: 'relation',
            color: 'rgba(107, 114, 128, 0.3)',
            weight: 1
          });
        }
      }
    }
  }

  private getHeaderColor(propertyName: string): string {
    const colorMap: Record<string, string> = {
      // Target Attributes (High Contrast / Neon)
      'audience-expertise': '#059669', // Darker Emerald (better contrast with white text)
      'engagement': '#be123c',         // Darker Rose (better contrast with white text)

      // Other Attributes (Muted / Pastel)
      'length': '#94a3b8',           // Slate
      'reading-time': '#94a3b8',     // Slate
      'tone': '#a78bfa',             // Soft Purple
      'complexity': '#818cf8',       // Soft Indigo
      'content-type': '#60a5fa',     // Soft Blue
      'date-period': '#9ca3af',      // Gray
      'primary-goal': '#f472b6',     // Soft Pink
      'popularity': '#34d399',       // Soft Green
      'audience-sentiment': '#fbbf24', // Soft Amber
      'authored-by': '#fb7185',      // Soft Rose
      'title': '#4b5563'             // Dark Gray
    };

    return colorMap[propertyName] || '#4b5563';
  }

  private getPropertyValueColor(propertyName: string): string {
    const colorMap: Record<string, string> = {
      // Target Attributes (Bright / Distinct)
      'audience-expertise': '#10b981', // Emerald
      'engagement': '#f43f5e',         // Rose

      // Other Attributes (Lighter versions)
      'length': '#cbd5e1',           // Light Slate
      'reading-time': '#cbd5e1',     // Light Slate
      'tone': '#c4b5fd',             // Light Purple
      'complexity': '#a5b4fc',       // Light Indigo
      'content-type': '#93c5fd',     // Light Blue
      'date-period': '#d1d5db',      // Light Gray
      'primary-goal': '#fbcfe8',     // Light Pink
      'popularity': '#6ee7b7',       // Light Green
      'audience-sentiment': '#fcd34d', // Light Amber
      'authored-by': '#fda4af',      // Light Rose
      'title': '#9ca3af'             // Light Gray
    };

    return colorMap[propertyName] || '#9ca3af';
  }

  private createMetadata(nodes: GraphNode[], edges: GraphEdge[]): GraphMetadata {
    return {
      nodeCount: nodes.length,
      edgeCount: edges.length,
      hypergraphCount: 0,
      lastUpdated: new Date()
    };
  }
}
