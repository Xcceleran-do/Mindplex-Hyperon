// Metta language parser service
import { ParseResult, ValidationResult, Triple, ParseError, HypergraphStructure } from '../../types';
import { GraphTransformerImpl } from '../graph/GraphTransformer';

export interface MettaParser {
  parse(mettaText: string): ParseResult;
  validateSyntax(mettaText: string): ValidationResult;
  extractTriples(mettaText: string): Triple[];
  handleHypergraph(expression: string): HypergraphStructure[];
}

export class MettaParserImpl implements MettaParser {
  private hypergraphCounter = 0;
  private graphTransformer = new GraphTransformerImpl();

  parse(mettaText: string): ParseResult {
    const errors: ParseError[] = [];

    // First validate syntax and collect errors
    const validation = this.validateSyntax(mettaText);
    errors.push(...validation.errors, ...validation.warnings);

    // Extract triples from the text
    const triples = this.extractTriples(mettaText);

    // Transform triples to graph data using the GraphTransformer
    const graphData = this.graphTransformer.transformTriplestoGraph(triples);

    return {
      nodes: graphData.nodes,
      edges: graphData.edges,
      errors,
      metadata: graphData.metadata
    };
  }

  validateSyntax(mettaText: string): ValidationResult {
    const errors: ParseError[] = [];
    const warnings: ParseError[] = [];

    // Simple validation - just check for parentheses
    const lines = mettaText.split('\n');

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      // Skip empty lines and comments
      if (!line || line.charAt(0) === ';') {
        continue;
      }

      // Check basic structure first
      const hasOpenParen = line.charAt(0) === '(';
      const hasCloseParen = line.charAt(line.length - 1) === ')';

      // If it doesn't start with ( OR doesn't end with ), treat as basic structure issue
      // Exception: if it starts with ( but doesn't end with ), that's an unmatched opening paren
      if (!hasOpenParen || (!hasCloseParen && !hasOpenParen)) {
        errors.push({
          line: i + 1,
          column: 0,
          message: 'Expression must be enclosed in parentheses',
          severity: 'error'
        });
        continue;
      }

      // If basic structure is correct, check for balanced parentheses
      let parenCount = 0;
      let lastOpenParen = -1;
      let hasUnbalancedParens = false;

      for (let j = 0; j < line.length; j++) {
        if (line[j] === '(') {
          parenCount++;
          lastOpenParen = j;
        } else if (line[j] === ')') {
          parenCount--;
          if (parenCount < 0) {
            errors.push({
              line: i + 1,
              column: j + 1,
              message: 'Unmatched closing parenthesis',
              severity: 'error'
            });
            hasUnbalancedParens = true;
            break;
          }
        }
      }

      if (parenCount > 0) {
        errors.push({
          line: i + 1,
          column: lastOpenParen + 1,
          message: 'Unmatched opening parenthesis',
          severity: 'error'
        });
        hasUnbalancedParens = true;
      }

      if (hasUnbalancedParens) {
        continue;
      }

      // Check for minimum content (predicate + at least one argument)
      // Only check this if we have valid parentheses structure
      if (!hasUnbalancedParens && line.length > 2) {
        const content = line.slice(1, -1).trim();
        const parts = content.split(/\s+/);

        if (parts.length < 2) {
          errors.push({
            line: i + 1,
            column: 1,
            message: 'Expression must have at least a predicate and one argument',
            severity: 'error'
          });
        }

        // Warn about potential typos in common predicates
        if (parts.length >= 2) {
          const predicate = parts[0];
          const commonPredicates = ['gender', 'is-parent', 'is-brother', 'is-sister', 'age', 'name'];
          let similarPredicate: string | undefined;
          for (let j = 0; j < commonPredicates.length; j++) {
            if (this.levenshteinDistance(predicate.toLowerCase(), commonPredicates[j]) === 1) {
              similarPredicate = commonPredicates[j];
              break;
            }
          }

          if (similarPredicate && predicate !== similarPredicate) {
            warnings.push({
              line: i + 1,
              column: 1,
              message: `Did you mean "${similarPredicate}"? Found "${predicate}"`,
              severity: 'warning'
            });
          }
        }
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }

  extractTriples(mettaText: string): Triple[] {
    const triples: Triple[] = [];
    const lines = mettaText.split('\n');

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmedLine = line.trim();

      // Skip empty lines and comments
      if (!trimmedLine || trimmedLine.charAt(0) === ';') {
        continue;
      }

      // Parse the expression
      const parsed = this.parseExpression(trimmedLine);
      if (parsed) {
        const triple = this.expressionToTriple(parsed);
        if (triple) {
          triples.push(triple);
        }
      }
    }

    return triples;
  }

  handleHypergraph(expression: string): HypergraphStructure[] {
    const parsed = this.parseExpression(expression);
    if (!parsed) return [];

    const structures: HypergraphStructure[] = [];
    const predicate = parsed.predicate;

    // Find nested expressions (hypergraph components)
    const nestedExpressions: ParsedExpression[] = [];
    for (let i = 0; i < parsed.args.length; i++) {
      const arg = parsed.args[i];
      if (typeof arg === 'object' && arg !== null && 'predicate' in arg) {
        nestedExpressions.push(arg as ParsedExpression);
      }
    }

    if (nestedExpressions.length > 0) {
      // This is a hypergraph structure
      const subjects: string[] = [];
      const objects: string[] = [];

      // For hypergraphs, all nested expression arguments go to objects
      // Simple arguments (if any) go to subjects
      parsed.args.forEach(arg => {
        if (typeof arg === 'string') {
          // Simple argument
          subjects.push(arg);
        } else if (typeof arg === 'object' && arg !== null && 'args' in arg) {
          // Nested expression - extract all arguments and put them in objects
          const parsedArg = arg as ParsedExpression;
          // Add the predicate as well (it's actually an entity in this context)
          objects.push(parsedArg.predicate);
          parsedArg.args.forEach(nestedArg => {
            if (typeof nestedArg === 'string') {
              objects.push(nestedArg);
            }
          });
        }
      });

      this.hypergraphCounter++;
      structures.push({
        id: `hypergraph-${this.hypergraphCounter}`,
        predicate,
        subjects,
        objects,
        intermediateNodeId: `${predicate}-group-${this.hypergraphCounter}`
      });
    }

    return structures;
  }

  private parseExpression(expression: string): ParsedExpression | null {
    const trimmed = expression.trim();
    if (trimmed.charAt(0) !== '(' || trimmed.charAt(trimmed.length - 1) !== ')') {
      return null;
    }

    const content = trimmed.slice(1, -1).trim();
    const tokens = this.tokenize(content);

    if (tokens.length < 2) {
      return null;
    }

    const predicate = tokens[0];
    const args: (string | ParsedExpression)[] = [];

    for (let i = 1; i < tokens.length; i++) {
      const token = tokens[i];
      if (token.charAt(0) === '(' && token.charAt(token.length - 1) === ')') {
        // Nested expression
        const nested = this.parseExpression(token);
        if (nested) {
          args.push(nested);
        }
      } else {
        args.push(token);
      }
    }

    return { predicate, args };
  }

  private tokenize(content: string): string[] {
    const tokens: string[] = [];
    let current = '';
    let parenDepth = 0;
    let inToken = false;

    for (let i = 0; i < content.length; i++) {
      const char = content[i];

      if (char === '(') {
        parenDepth++;
        current += char;
        inToken = true;
      } else if (char === ')') {
        parenDepth--;
        current += char;
        if (parenDepth === 0 && inToken) {
          tokens.push(current.trim());
          current = '';
          inToken = false;
        }
      } else if (char === ' ' && parenDepth === 0) {
        if (current.trim()) {
          tokens.push(current.trim());
          current = '';
        }
        inToken = false;
      } else {
        current += char;
        inToken = true;
      }
    }

    if (current.trim()) {
      tokens.push(current.trim());
    }

    return tokens;
  }

  private expressionToTriple(parsed: ParsedExpression): Triple | null {
    const { predicate, args } = parsed;

    // Check if this is a hypergraph (contains nested expressions)
    let hasNestedExpressions = false;
    for (let i = 0; i < args.length; i++) {
      const arg = args[i];
      if (typeof arg === 'object' && arg !== null && 'predicate' in arg) {
        hasNestedExpressions = true;
        break;
      }
    }

    if (hasNestedExpressions) {
      // Hypergraph structure
      const subjects: string[] = [];
      const objects: string[] = [];

      args.forEach(arg => {
        if (typeof arg === 'string') {
          subjects.push(arg);
        } else if (typeof arg === 'object' && arg !== null && 'args' in arg) {
          const parsedArg = arg as ParsedExpression;
          // Add the predicate as well (it's actually an entity in this context)
          objects.push(parsedArg.predicate);
          parsedArg.args.forEach(nestedArg => {
            if (typeof nestedArg === 'string') {
              objects.push(nestedArg);
            }
          });
        }
      });

      return {
        predicate,
        subject: subjects.length === 1 ? subjects[0] : subjects,
        object: objects.length === 1 ? objects[0] : objects,
        isHypergraph: true
      };
    } else {
      // Simple triple
      const stringArgs: string[] = [];
      for (let i = 0; i < args.length; i++) {
        if (typeof args[i] === 'string') {
          stringArgs.push(args[i] as string);
        }
      }
      if (stringArgs.length >= 2) {
        return {
          predicate,
          subject: stringArgs[0],
          object: stringArgs.slice(1).length === 1 ? stringArgs[1] : stringArgs.slice(1),
          isHypergraph: false
        };
      }
    }

    return null;
  }



  private getOriginalExpression(mettaText: string, triple: Triple, index: number): string {
    const lines = mettaText.split('\n').filter(line =>
      line.trim() && line.trim().charAt(0) !== ';'
    );
    return lines[index] || '';
  }

  private levenshteinDistance(str1: string, str2: string): number {
    const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null));

    for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
    for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;

    for (let j = 1; j <= str2.length; j++) {
      for (let i = 1; i <= str1.length; i++) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1,
          matrix[j - 1][i] + 1,
          matrix[j - 1][i - 1] + indicator
        );
      }
    }

    return matrix[str2.length][str1.length];
  }
}

interface ParsedExpression {
  predicate: string;
  args: (string | ParsedExpression)[];
}