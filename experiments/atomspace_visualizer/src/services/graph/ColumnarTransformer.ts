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
  private readonly COLUMN_SPACING = 500;
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
      const headerColors = this.getHeaderColorStats(column.name);

      nodes.push({
        id: `header-${column.name}`,
        label: label,
        type: 'predicate',
        position: {
          x: column.position,
          y: 20
        },
        color: headerColors.bg,
        textColor: headerColors.text,
        strokeColor: headerColors.border,
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
      const valColors = this.getPropertyValueColorStats(column.name);

      values.forEach((value, index) => {
        nodes.push({
          id: `${column.name}-${value}`,
          label: value,
          type: 'value',
          position: {
            x: column.position,
            y: this.HEADER_HEIGHT + index * this.NODE_SPACING
          },
          color: valColors.bg,
          strokeColor: valColors.border,
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

  private generateColor(str: string, type: 'header' | 'value'): { bg: string; text: string; border: string } {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }

    // Hue: Full range 0-360 based on hash
    const h = Math.abs(hash % 360);

    // Saturation: High for vibrant, appealing colors (70-95%)
    const s = 70 + (Math.abs(hash >> 8) % 25);

    // Lightness:
    let l: number;
    let textColor: string;
    let borderColor: string;

    if (type === 'header') {
      // Header: Richer, darker for better contrast with white text, or lighter for black text
      // Let's force a range that guarantees good contrast with one or the other.
      // Dark Mode preference: deeply saturated dark colors (L: 20-40%) with White text
      l = 25 + (Math.abs(hash >> 4) % 20); // 25-45%
      textColor = '#ffffff';
      borderColor = `hsla(${h}, ${s}%, ${l + 20}%, 0.8)`;
    } else {
      // Value: Brighter, pastel (L: 70-90%) - Text is OUTSIDE, so this color is just the circle fill.
      // But if we ever put text inside, it should be black.
      l = 75 + (Math.abs(hash >> 4) % 15); // 75-90%
      textColor = '#1e293b'; // Slate-800
      borderColor = `hsla(${h}, ${s}%, ${l - 20}%, 0.5)`;
    }

    return {
      bg: `hsl(${h}, ${s}%, ${l}%)`,
      text: textColor,
      border: borderColor
    };
  }

  private getHeaderColorStats(propertyName: string) {
    return this.generateColor(propertyName, 'header');
  }

  private getPropertyValueColorStats(propertyName: string) {
    return this.generateColor(propertyName, 'value');
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
