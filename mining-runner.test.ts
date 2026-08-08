import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { MeTTa } from '@metta-ts/hyperon';
import {
  mineDataset,
  normalizeGeneratedVariables,
  registerCutFirstChar,
  registerPromoteEngagementConj,
  registerUniqueCombinationsStar,
} from './mining-runner.ts';

function runnerWithOperations(): MeTTa {
  const metta = new MeTTa();
  registerCutFirstChar(metta);
  registerPromoteEngagementConj(metta);
  registerUniqueCombinationsStar(metta);
  return metta;
}

test('registered operations preserve variables and grounded strings', () => {
  const metta = runnerWithOperations();
  assert.equal(String(metta.run('!(cut-first-char l$x)')[0][0]), '$x');
  assert.equal(
    String(
      metta.run(
        '!(promote_engagement_conj (, (engagement $x "Low") (topic $x "AI") (audience-expertise $x "Beginner")))',
      )[0][0],
    ),
    '(, (topic $x "AI") (engagement $x "Low") (audience-expertise $x "Beginner"))',
  );
});

test('unique_combinations_star requires one shared hub and the target predicates', () => {
  const metta = runnerWithOperations();
  const result = String(
    metta.run(
      '!(unique_combinations_star ((topic $x "AI") (engagement $x "Low") (audience-expertise $x "Beginner")) 3)',
    )[0][0],
  );
  assert.equal(
    result,
    '((conjunct (, (audience-expertise $x "Beginner") (engagement $x "Low") (topic $x "AI"))))',
  );
});

test('production mining reads persisted facts and reports exact support', () => {
  const directory = mkdtempSync(join(tmpdir(), 'mettascript-mining-'));
  const dataset = join(directory, 'data.metta');
  writeFileSync(
    dataset,
    [
      '((engagement A1 "Low") (STV 0.8 0.9))',
      '((audience-expertise A1 "Beginner") (STV 0.8 0.9))',
      '((engagement A2 "Low") (STV 0.8 0.9))',
      '((audience-expertise A2 "Beginner") (STV 0.8 0.9))',
    ].join('\n'),
    'utf8',
  );
  try {
    const output = mineDataset(dataset, 2, 2).map((atom) =>
      normalizeGeneratedVariables(atom.toString()),
    );
    assert.equal(output.length, 1);
    assert.match(output[0], /\(supportOf .* \(STV 1\.0 0\.6666666666666666\)\) 2\)/);
    assert.doesNotMatch(output[0], /#\d+/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
