import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  MeTTa,
  E,
  S,
  V,
  ValueAtom,
  atomToJs,
  ExpressionAtom,
  GroundedAtom,
  SymbolAtom,
  VariableAtom,
  type Atom,
} from '@metta-ts/hyperon';

// ---------------------------------------------------------------------------
// Override size-atom to handle comma-wrapped collapse results
// Makes TS runner behave identically to PeTTa and metta-ts CLI
// ---------------------------------------------------------------------------
function registerSizeAtom(metta: MeTTa): void {
  metta.registerOperation('size-atom', (args: Atom[]) => {
    const expr = args[0];
    if (!isExpression(expr)) return [ValueAtom(0)];
    
    const children = expr.children();
    const first = children[0];
    
    // Strip the comma head if present (TS runner artifact)
    if (first && isSymbol(first) && first.name() === ',') {
      return [ValueAtom(children.length - 1)];
    }
    return [ValueAtom(children.length)];
  });
}

// ---------------------------------------------------------------------------
// Override tuple-count to handle comma-wrapped collapse results
// Fixes db_size returning 61 instead of 60
// ---------------------------------------------------------------------------
function registerTupleCount(metta: MeTTa): void {
  metta.registerOperation('tuple-count', (args: Atom[]) => {
    const expr = args[0];
    if (!isExpression(expr)) return [ValueAtom(0)];
    
    const children = expr.children();
    const first = children[0];
    
    // Strip the comma head if present (TS runner artifact)
    if (first && isSymbol(first) && first.name() === ',') {
      return [ValueAtom(children.length - 1)];
    }
    return [ValueAtom(children.length)];
  });
}

// ---------------------------------------------------------------------------
// cut-first-char
// ---------------------------------------------------------------------------
function registerCutFirstChar(metta: MeTTa): void {
  metta.registerOperation('cut-first-char', (args: Atom[]) => {
    const input = args[0];
    if (isVariable(input)) return [input];

    if (isSymbol(input)) {
      const stripped = input.name().slice(1);
      return [stripped.startsWith('$') ? V(stripped.slice(1)) : S(stripped)];
    }

    if (isGrounded(input)) {
      const value = input.object().content;
      if (typeof value === 'string') return [ValueAtom(value.slice(1))];
    }

    return [input];
  });
}

// ---------------------------------------------------------------------------
// promote_engagement_conj
// ---------------------------------------------------------------------------
const REQUIRED_KEYWORDS = ['engagement', 'audience-expertise'];

// Use MeTTa's public metatype instead of instanceof. Compatibility packages can
// load more than one copy of the Atom classes, which makes instanceof brittle.
function isExpression(atom: Atom): atom is ExpressionAtom {
  return atom.metatype() === 'Expression';
}

function isSymbol(atom: Atom): atom is SymbolAtom {
  return atom.metatype() === 'Symbol';
}

function isVariable(atom: Atom): atom is VariableAtom {
  return atom.metatype() === 'Variable';
}

function isGrounded(atom: Atom): atom is GroundedAtom {
  return atom.metatype() === 'Grounded';
}

function clauseFunctor(clause: Atom): string | undefined {
  if (!isExpression(clause)) return undefined;
  const head = clause.children()[0];
  return head && isSymbol(head) ? head.name() : undefined;
}

function hasRequiredKeyword(clause: Atom): boolean {
  const functor = clauseFunctor(clause);
  return functor !== undefined && REQUIRED_KEYWORDS.some((kw) => functor.startsWith(kw));
}

function registerPromoteEngagementConj(metta: MeTTa): void {
  metta.registerOperation('promote_engagement_conj', (args: Atom[]) => {
    const conjunction = args[0];
    if (!isExpression(conjunction)) return [conjunction];

    const children = conjunction.children();
    const head = children[0];
    if (!head || !isSymbol(head) || head.name() !== ',') return [conjunction];

    const clauses = children.slice(1);
    const others = clauses.filter((clause) => !hasRequiredKeyword(clause));
    const matches = clauses.filter(hasRequiredKeyword);

    return [E(S(','), ...others, ...matches)];
  });
}

// ---------------------------------------------------------------------------
// unique_combinations_star
// ---------------------------------------------------------------------------
interface ClauseInfo {
  readonly expr: Atom;
  readonly vars: ReadonlySet<string>;
  readonly functor: string;
}

function extractVarKeys(term: Atom, out: Set<string>): void {
  if (isVariable(term)) {
    out.add(term.name());
    return;
  }

  if (isSymbol(term) && term.name().startsWith('$')) {
    out.add(term.name().slice(1));
    return;
  }

  if (isExpression(term)) {
    for (const child of term.children()) extractVarKeys(child, out);
  }
}

function exprFunctor(expr: Atom): string {
  return clauseFunctor(expr) ?? '';
}

function buildInfos(exprs: Atom[]): ClauseInfo[] {
  return exprs.map((expr) => {
    const vars = new Set<string>();
    extractVarKeys(expr, vars);
    return { expr, vars, functor: exprFunctor(expr) };
  });
}

function sharedVars(a: ReadonlySet<string>, b: ReadonlySet<string>): string[] {
  return [...a].filter((v) => b.has(v));
}

function onlyHubShared(hub: string, a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
  const shared = sharedVars(a, b);
  return shared.length === 1 && shared[0] === hub;
}

function combosForHub(pool: ClauseInfo[], hub: string, k: number): ClauseInfo[][] {
  const results: ClauseInfo[][] = [];

  function choose(
    rest: ClauseInfo[],
    remaining: number,
    selected: ClauseInfo[],
    usedFunctors: Set<string>,
  ): void {
    if (remaining === 0) {
      results.push([...selected]);
      return;
    }
    if (rest.length === 0) return;

    const [head, ...tail] = rest;
    const functorOk = head.functor === '' || !usedFunctors.has(head.functor);
    const compatible = selected.every((s) => onlyHubShared(hub, head.vars, s.vars));

    if (functorOk && compatible) {
      const nextUsed =
        head.functor === '' ? usedFunctors : new Set([...usedFunctors, head.functor]);
      choose(tail, remaining - 1, [...selected, head], nextUsed);
    }
    choose(tail, remaining, selected, usedFunctors);
  }

  choose(pool, k, [], new Set());
  return results;
}

function comboSortKey(combo: ClauseInfo[]): string[] {
  return [...combo]
    .map((clause) => clause.expr.toString())
    .sort((a, b) => a.localeCompare(b));
}

function registerUniqueCombinationsStar(metta: MeTTa): void {
  metta.registerOperation('unique_combinations_star', (args: Atom[]) => {
    const input = args[0];
    if (!isExpression(input)) return [E()];

    const inputChildren = input.children();
    const first = inputChildren[0];
    const exprs = (
      first && isSymbol(first) && first.name() === ','
        ? inputChildren.slice(1)
        : inputChildren
    ).filter(isExpression);

    const size = Number(atomToJs(args[1]));
    const k = Number.isFinite(size) ? Math.floor(size) : 0;

    if (k <= 0 || exprs.length < k) {
      return [E()];
    }

    const infos = buildInfos(exprs);
    const hubs = new Set<string>();
    for (const info of infos) {
      for (const v of info.vars) hubs.add(v);
    }

    const seen = new Set<string>();
    const results: Atom[] = [];

    for (const hub of hubs) {
      const pool = infos.filter((i) => i.vars.has(hub));
      if (pool.length < k) continue;

      for (const combo of combosForHub(pool, hub, k)) {
        const key = JSON.stringify(comboSortKey(combo));
        if (seen.has(key)) continue;
        seen.add(key);

        const clauses = [...combo]
          .sort((left, right) => left.expr.toString().localeCompare(right.expr.toString()))
          .map((info) => info.expr);

        const passesFilter = REQUIRED_KEYWORDS.every((kw) =>
          clauses.some((c) => (clauseFunctor(c) ?? '').startsWith(kw)),
        );
        if (!passesFilter) continue;

        results.push(E(S('conjunct'), E(S(','), ...clauses)));
      }
    }

    return [E(...results)];
  });
}

// ---------------------------------------------------------------------------
// Runner: read the target file plus its .metta imports from disk
// ---------------------------------------------------------------------------
const IMPORT_LINE = /^!?\(import!\s+&self\s+([^\s)]+)\)\s*$/;
const PROLOG_IMPORT_LINE = /^!?\(import_prolog_functions_from_file\b.*\)\s*$/;

function resolveMettaImport(target: string, fromDir: string): string {
  const stripped = target.replace(/^"(.*)"$/, '$1');
  const withExt = stripped.endsWith('.metta') ? stripped : `${stripped}.metta`;
  return resolve(fromDir, withExt);
}

function loadWithImports(entryPath: string, seen = new Set<string>()): string {
  const absolute = resolve(entryPath);
  if (seen.has(absolute)) return '';
  seen.add(absolute);

  const dir = dirname(absolute);
  const rawLines = readFileSync(absolute, 'utf8').split('\n');
  const out: string[] = [];

  for (const line of rawLines) {
    const trimmed = line.trim();

    if (PROLOG_IMPORT_LINE.test(trimmed)) continue;

    const importMatch = trimmed.match(IMPORT_LINE);
    if (importMatch) {
      const importTarget = importMatch[1];
      try {
        const importedPath = resolveMettaImport(importTarget, dir);
        out.push(loadWithImports(importedPath, seen));
        continue;
      } catch {
        out.push(line);
        continue;
      }
    }

    out.push(line);
  }

  return out.join('\n');
}

function createMiningRunner(): MeTTa {
  const metta = new MeTTa();
  registerCutFirstChar(metta);
  registerPromoteEngagementConj(metta);
  registerUniqueCombinationsStar(metta);
  registerSizeAtom(metta);    
  registerTupleCount(metta); 
  return metta;
}

function loadMiningLibraries(metta: MeTTa, projectRoot: string): void {
  const seen = new Set<string>();
  const libraryPaths = [
    resolve(projectRoot, 'PeTTa/lib/lib_import.metta'),
    resolve(projectRoot, 'experiments/utils/common-utils.metta'),
    resolve(projectRoot, 'experiments/frequent-pattern-miner/etv-utils.metta'),
    resolve(projectRoot, 'experiments/frequent-pattern-miner/frequent-pattern-miner.metta'),
    resolve(projectRoot, 'experiments/pattern-miner/pattern-miner.metta'),
  ];
  const source = libraryPaths.map((path) => loadWithImports(path, seen)).join('\n');
  metta.run(source);
}

function loadDatasetIntoSpace(metta: MeTTa, datasetPath: string): number {
  const records = metta.parseAll(readFileSync(resolve(datasetPath), 'utf8'));
  const facts: Atom[] = [];
  let recordCount = 0;

  for (const record of records) {
    if (!isExpression(record)) {
      throw new Error(`Dataset record is not an expression: ${record}`);
    }
    const children = record.children();
    const fact = children[0];
    const stv = children[1];
    if (!fact || !isExpression(fact) || !stv || clauseFunctor(stv) !== 'STV') {
      throw new Error(`Dataset record must have the form ((fact ...) (STV strength confidence)): ${record}`);
    }
    // Keep the domain fact for support/STV calculations and an augmented copy
    // for candidate discovery, which explicitly matches the source STV.
    facts.push(fact, E(...fact.children(), stv));
    recordCount += 1;
  }

  if (facts.length === 0) throw new Error(`Dataset contains no facts: ${datasetPath}`);
  metta.run(facts.map((fact) => `!(add-atom &miningDb ${fact})`).join('\n'));
  return recordCount;
}

function mineDataset(
  datasetPath: string,
  minSupport: number,
  conjunctionCount: number,
  projectRoot = dirname(fileURLToPath(import.meta.url)),
): Atom[] {
  if (!Number.isInteger(minSupport) || minSupport < 1) {
    throw new Error('min-support must be a positive integer');
  }
  if (!Number.isInteger(conjunctionCount) || conjunctionCount < 1) {
    throw new Error('conjunction-count must be a positive integer');
  }

  const metta = createMiningRunner();
  loadMiningLibraries(metta, projectRoot);
  loadDatasetIntoSpace(metta, datasetPath);
  return metta.run(`!(pattern-miner &miningDb ${minSupport} ${conjunctionCount})`).at(-1) ?? [];
}

function normalizeGeneratedVariables(source: string): string {
  const variables = new Map<string, string>();
  return source.replace(/\$[A-Za-z_][\w-]*(?:#\d+)?/g, (variable) => {
    let normalized = variables.get(variable);
    if (!normalized) {
      normalized = `$V${variables.size}`;
      variables.set(variable, normalized);
    }
    return normalized;
  });
}

// ---------------------------------------------------------------------------
// CLI entrypoint (ESM-safe)
// ---------------------------------------------------------------------------
const isMainModule =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMainModule) {
  try {
    if (process.argv[2] === '--mine') {
      const datasetPath = process.argv[3];
      const minSupport = Number(process.argv[4]);
      const conjunctionCount = Number(process.argv[5]);
      if (!datasetPath) {
        throw new Error(
          'Usage: npx tsx mining-runner.ts --mine <dataset.metta> <min-support> <conjunction-count>',
        );
      }
      const results = mineDataset(datasetPath, minSupport, conjunctionCount).map((atom) =>
        normalizeGeneratedVariables(atom.toString()),
      );
      // Mining functions may trace intermediate expressions to stdout. Keep a
      // machine-readable final line so the Python worker can ignore that noise.
      console.log(JSON.stringify({ results }));
    } else {
      const target = process.argv[2];
      if (!target) {
        throw new Error('Usage: npx tsx mining-runner.ts <path-to-metta-test>');
      }
      const metta = createMiningRunner();
      const results = metta.run(loadWithImports(target));
      for (const group of results) console.log(group.map(String).join('\n'));
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}

export {
  createMiningRunner,
  loadDatasetIntoSpace,
  loadMiningLibraries,
  mineDataset,
  normalizeGeneratedVariables,
  registerCutFirstChar,
  registerPromoteEngagementConj,
  registerUniqueCombinationsStar,
};
