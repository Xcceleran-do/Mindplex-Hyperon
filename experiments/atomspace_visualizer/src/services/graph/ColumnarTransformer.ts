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
  private readonly NODE_SPACING = 80;
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

    for (const [name, values] of propertyValues) {
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
      // Add column header node
      nodes.push({
        id: `header-${column.name}`,
        label: column.name.replace(/_/g, ' ').toUpperCase(),
        type: 'predicate',
        position: {
          x: column.position,
          y: 20
        },
        color: '#8b5cf6',
        size: 60,
        metadata: {
          originalExpression: column.name,
          occurrences: 1,
          isGenerated: false,
          columnType: 'header'
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
          size: 45,
          metadata: {
            originalExpression: value,
            occurrences: 1,
            isGenerated: false,
            columnType: 'property',
            propertyName: column.name
          }
        });
      });

      // Add "None" node at the bottom
      nodes.push({
        id: `${column.name}-None`,
        label: 'None',
        type: 'value',
        position: {
          x: column.position,
          y: this.HEADER_HEIGHT + values.length * this.NODE_SPACING
        },
        color: '#6b7280',
        size: 40,
        metadata: {
          originalExpression: 'None',
          occurrences: 1,
          isGenerated: true,
          columnType: 'property',
          propertyName: column.name
        }
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
        const targetId = value 
          ? `${column.name}-${value}`
          : `${column.name}-None`;

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

  private getPropertyValueColor(propertyName: string): string {
    const colorMap: Record<string, string> = {
      topic: '#10b981',
      length: '#f59e0b',
      tone: '#ef4444',
      writing_style: '#8b5cf6',
      engagement_level: '#06b6d4'
    };

    return colorMap[propertyName] || '#6b7280';
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
