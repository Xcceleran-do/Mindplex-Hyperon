import fs from 'node:fs';
import path from 'node:path';

interface CombinationResult {
  clauses: Array<readonly string[]>;
  normalized: string;
}

const ENGAGEMENT_KEYWORDS = ['engagement', 'audience-expertise'];

function isVariableToken(token: string): boolean {
  return token.startsWith('$');
}

function extractVariableKeys(expr: readonly string[]): Set<string> {
  const vars = new Set<string>();
  for (const token of expr) {
    if (isVariableToken(token)) {
      vars.add(token);
    }
  }
  return vars;
}

function sharedVars(left: Set<string>, right: Set<string>): string[] {
  return Array.from(left).filter((value) => right.has(value));
}

function normalizeVariables(expr: readonly string[], map: Map<string, string>): string[] {
  return expr.map((token) => {
    if (isVariableToken(token)) {
      const existing = map.get(token);
      if (existing) {
        return existing;
      }
      const nextName = `$v${map.size}`;
      map.set(token, nextName);
      return nextName;
    }
    return token;
  });
}

function normalizeCombo(combo: readonly (readonly string[])[]): CombinationResult {
  const variableMap = new Map<string, string>();
  const normalizedClauses = combo.map((clause) => normalizeVariables(clause, variableMap));
  return {
    clauses: normalizedClauses,
    normalized: JSON.stringify(normalizedClauses)
  };
}

function uniqueByKey<T>(items: T[], keyFn: (item: T) => string): T[] {
  const seen = new Set<string>();
  const result: T[] = [];
  for (const item of items) {
    const key = keyFn(item);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(item);
  }
  return result;
}

function canAddClause(candidate: readonly string[], selected: readonly (readonly string[])[], hub: string): boolean {
  if (selected.length === 0) {
    return true;
  }

  const candidateVars = extractVariableKeys(candidate);
  return selected.every((clause) => {
    const clauseVars = extractVariableKeys(clause);
    const overlap = sharedVars(candidateVars, clauseVars);
    return overlap.length === 0 || (overlap.length === 1 && overlap.includes(hub));
  });
}

function pickCombinations(expressions: readonly (readonly string[])[], size: number, hub: string): Array<readonly string[]>[] {
  const combos: Array<readonly string[]>[] = [];
  const visit = (startIndex: number, selected: Array<readonly string[]>): void => {
    if (selected.length === size) {
      combos.push(selected.slice());
      return;
    }
    for (let index = startIndex; index < expressions.length; index += 1) {
      const expression = expressions[index];
      if (!canAddClause(expression, selected, hub)) {
        continue;
      }
      selected.push(expression);
      visit(index + 1, selected);
      selected.pop();
    }
  };
  visit(0, []);
  return combos;
}

function hasRequiredKeyword(clause: readonly string[], keywords: readonly string[]): boolean {
  return keywords.some((keyword) => clause[0] === keyword || clause[0].includes(keyword));
}

export function promoteEngagementConj(conjunction: readonly (readonly string[])[]): Array<readonly string[]> {
  const matches = conjunction.filter((clause) => hasRequiredKeyword(clause, ENGAGEMENT_KEYWORDS));
  const others = conjunction.filter((clause) => !hasRequiredKeyword(clause, ENGAGEMENT_KEYWORDS));
  return [...others, ...matches];
}

export function uniqueCombinationsStar(expressions: readonly (readonly string[])[], size: number): CombinationResult[] {
  if (!Number.isInteger(size) || size <= 0) {
    return [];
  }

  const candidateExpressions = expressions.filter((expr) => Array.isArray(expr) && expr.length >= 1);
  if (candidateExpressions.length < size) {
    return [];
  }

  const hubs = new Set<string>();
  for (const expr of candidateExpressions) {
    for (const token of expr) {
      if (isVariableToken(token)) {
        hubs.add(token);
      }
    }
  }

  const combos: CombinationResult[] = [];
  for (const hub of hubs) {
    const hubMatches = candidateExpressions.filter((expr) => extractVariableKeys(expr).has(hub));
    if (hubMatches.length < size) {
      continue;
    }
    for (const combination of pickCombinations(hubMatches, size, hub)) {
      combos.push(normalizeCombo(combination));
    }
  }

  const uniqueCombos = uniqueByKey(combos, (combo) => combo.normalized);
  return uniqueCombos.map((combo) => ({
    ...combo,
    clauses: promoteEngagementConj(combo.clauses)
  }));
}

function parseCandidateExpressions(input: string): Array<string[]> {
  const normalized = input
    .replace(/;;.*$/gm, '')
    .replace(/\s+/g, ' ')
    .trim();

  const candidateMatches = normalized.match(/\(([^()]+)\)/g) ?? [];
  return candidateMatches
    .map((match) => match.slice(1, -1).trim())
    .filter(Boolean)
    .map((expr) => expr.split(' ').filter(Boolean));
}

export function runMiningRunner(targetPath: string, size = 2): string {
  const absolutePath = path.resolve(targetPath);
  const contents = fs.readFileSync(absolutePath, 'utf8');
  const expressions = parseCandidateExpressions(contents);
  const result = uniqueCombinationsStar(expressions, size);
  return result
    .map((item) => item.clauses.map((clause) => `(${clause.join(' ')})`).join(', '))
    .map((value) => `(${value})`)
    .join('\n');
}

if (require.main === module) {
  const target = process.argv[2];
  if (!target) {
    console.error('Usage: npx tsx mining-runner.ts <path-to-metta-test>');
    process.exit(1);
  }
  console.log(runMiningRunner(target));
}
