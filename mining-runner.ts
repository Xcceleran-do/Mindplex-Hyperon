import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MeTTa, E, S, atomToJs, type Atom } from '@metta-ts/hyperon';

function jsToExpr(value: unknown): Atom {
  if (typeof value === 'string') {
    return S(value);
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return S(String(value));
  }
  if (Array.isArray(value)) {
    // FIX: E is variadic ((...children) => ...), not array-taking.
    // Passing a single array made `children` = [theArray], so
    // children.map(c => c.catom) tried to read .catom off the array
    // itself (undefined), which then crashed inside core.expr() trying
    // to read `.ground` off that undefined value. Spread the array so
    // each atom is passed as its own argument, matching E's real signature.
    return E(...value.map(jsToExpr));
  }
  // fallback for anything unexpected
  return S(String(value));
}

// ---------------------------------------------------------------------------
// cut-first-char
// Faithful port of conj_exp.pl's cut-first-char/2: strips the first
// character of a symbol/string, leaving other atom kinds unchanged.
// ---------------------------------------------------------------------------
function registerCutFirstChar(metta: MeTTa): void {
  metta.registerOperation('cut-first-char', (args: Atom[]) => {
    const input = atomToJs(args[0]);
    if (typeof input !== 'string' || input.length === 0) {
      return [args[0]];
    }
    return [jsToExpr(input.slice(1))];
  });
}

// ---------------------------------------------------------------------------
// promote_engagement_conj
// Faithful port: move clauses whose functor is (or starts with) "engagement"
// or "audience-expertise" to the end of the conjunction.
// ---------------------------------------------------------------------------
const REQUIRED_KEYWORDS = ['engagement', 'audience-expertise'];

function clauseFunctor(clause: unknown): string | undefined {
  return Array.isArray(clause) && typeof clause[0] === 'string' ? clause[0] : undefined;
}

function hasRequiredKeyword(clause: unknown): boolean {
  const functor = clauseFunctor(clause);
  return functor !== undefined && REQUIRED_KEYWORDS.some((kw) => functor.startsWith(kw));
}

function registerPromoteEngagementConj(metta: MeTTa): void {
  metta.registerOperation('promote_engagement_conj', (args: Atom[]) => {
    const conjunction = atomToJs(args[0]);
    if (!Array.isArray(conjunction) || conjunction[0] !== ',') {
      return [args[0]];
    }
    const clauses = conjunction.slice(1);
    const others = clauses.filter((c) => !hasRequiredKeyword(c));
    const matches = clauses.filter((c) => hasRequiredKeyword(c));
    return [jsToExpr([',', ...others, ...matches])];
  });
}

// ---------------------------------------------------------------------------
// unique_combinations_star
// Faithful port of conj_exp.pl's unique_combinations_star/3:
//  - build size-K conjunctions where all clauses share exactly one hub
//    variable, and no other variable is shared across any pair of clauses
//  - each combo uses at most one clause per functor
//  - results are deduplicated
//  - filtered to require at least one clause per REQUIRED_KEYWORDS entry
// ---------------------------------------------------------------------------
interface ClauseInfo {
  readonly expr: unknown;
  readonly vars: ReadonlySet<string>;
  readonly functor: string;
}

function extractVarKeys(term: unknown, out: Set<string>): void {
  if (typeof term === 'string' && term.startsWith('$')) {
    out.add(term);
    return;
  }
  if (Array.isArray(term)) {
    for (const t of term) extractVarKeys(t, out);
  }
}

function exprFunctor(expr: unknown): string {
  return Array.isArray(expr) && typeof expr[0] === 'string' ? expr[0] : '';
}

function buildInfos(exprs: unknown[]): ClauseInfo[] {
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

  function choose(rest: ClauseInfo[], remaining: number, selected: ClauseInfo[], usedFunctors: Set<string>): void {
    if (remaining === 0) {
      results.push([...selected]);
      return;
    }
    if (rest.length === 0) return;

    const [head, ...tail] = rest;
    const functorOk = head.functor === '' || !usedFunctors.has(head.functor);
    const compatible = selected.every((s) => onlyHubShared(hub, head.vars, s.vars));

    if (functorOk && compatible) {
      const nextUsed = head.functor === '' ? usedFunctors : new Set([...usedFunctors, head.functor]);
      choose(tail, remaining - 1, [...selected, head], nextUsed);
    }
    choose(tail, remaining, selected, usedFunctors);
  }

  choose(pool, k, [], new Set());
  return results;
}

function comboSortKey(combo: ClauseInfo[]): unknown[] {
  return [...combo]
    .map((c) => c.expr)
    .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
}

function registerUniqueCombinationsStar(metta: MeTTa): void {
  metta.registerOperation('unique_combinations_star', (args: Atom[]) => {
    const exprs = atomToJs(args[0]) as unknown[];
    const sizeRaw = atomToJs(args[1]) as number;
    const k = Number.isInteger(sizeRaw) ? sizeRaw : Math.floor(sizeRaw);

    const candidateExprs = Array.isArray(exprs)
      ? exprs.filter((e) => Array.isArray(e))
       : [];

    if (k <= 0 || candidateExprs.length < k) {
      return [jsToExpr([])];
    }

    const infos = buildInfos(candidateExprs);
    const hubs = new Set<string>();
    for (const info of infos) for (const v of info.vars) hubs.add(v);

    const seen = new Set<string>();
    const results: unknown[] = [];

    for (const hub of hubs) {
      const pool = infos.filter((i) => i.vars.has(hub));
      if (pool.length < k) continue;

      for (const combo of combosForHub(pool, hub, k)) {
        const key = JSON.stringify(comboSortKey(combo));
        if (seen.has(key)) continue;
        seen.add(key);

        const clauses = combo.map((i) => i.expr);
        const passesFilter = REQUIRED_KEYWORDS.every((kw) =>
          clauses.some((c) => (clauseFunctor(c) ?? '').startsWith(kw)),
        );
        if (!passesFilter) continue;

        results.push(['conjunct', [',', ...clauses]]);
      }
    }

    return [jsToExpr(results)];
  });
}

// ---------------------------------------------------------------------------
// Runner: read the target file plus its real .metta imports from disk,
// concatenate them into one real MeTTa program (no text-scraping of facts),
// and execute it for real through metta.run().
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

    // Skip the Prolog consult line entirely - replaced by native TS ops.
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

// ---------------------------------------------------------------------------
// CLI entrypoint (ESM-safe)
// ---------------------------------------------------------------------------
const isMainModule = process.argv[1] !== undefined
  && resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMainModule) {
  const target = process.argv[2];
  if (!target) {
    console.error('Usage: npx tsx mining-runner.ts <path-to-metta-test>');
    process.exit(1);
  }

  const metta = new MeTTa();
  registerCutFirstChar(metta);
  registerPromoteEngagementConj(metta);
  registerUniqueCombinationsStar(metta);

  const combinedSource = loadWithImports(target);
  const results = metta.run(combinedSource);
  for (const group of results) {
    console.log(group.map(String).join('\n'));
  }
}

export { registerCutFirstChar, registerPromoteEngagementConj, registerUniqueCombinationsStar };