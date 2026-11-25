import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const sha = content => createHash('sha256').update(content).digest('hex');
const bytes = name => readFileSync(join(root, name));
const read = name => bytes(name).toString('utf8').replace(/^\uFEFF/u, '');
const json = name => JSON.parse(read(name));
const csv = name => {
  const [header, ...lines] = read('results/' + name).trimEnd().split(/\r?\n/u);
  const columns = header.split(',');
  return lines.map(line => {
    const cells = line.split(',');
    assert.equal(cells.length, columns.length, name);
    return Object.fromEntries(columns.map((key, i) => [key, cells[i]]));
  });
};
function files(dir = '') {
  return readdirSync(join(root, dir), { withFileTypes: true }).flatMap(entry => {
    const name = dir + entry.name;
    if (entry.isDirectory()) return files(name + '/');
    assert(entry.isFile(), name);
    return [name];
  });
}
const manifest = new Map();
for (const line of read('MANIFEST.sha256').trimEnd().split(/\r?\n/u)) {
  const entry = /^([0-9a-f]{64})  ([\w./-]+)$/u.exec(line);
  assert(entry, 'Malformed manifest line');
  const [, digest, name] = entry;
  assert(!name.startsWith('/') && !name.split('/').includes('..') && !manifest.has(name), name);
  assert.equal(sha(bytes(name)), digest, name);
  manifest.set(name, digest);
}
assert.deepEqual([...manifest.keys()].sort(), files().filter(name => name !== 'MANIFEST.sha256').sort());

const protocol = json('PROTOCOL.json');
const publication = json('PUBLICATION_SCOPE.json');
const tail = json('results/TAIL_RESULTS.json');
assert.deepEqual(protocol.protocol, { H: 0, D: 4, L: 128 });
assert.equal(publication.release_lock_sha256, '4403fe0c314b0d5f4fe773c2f4a34ef5d17914db030f09efb657d86eafbc1a53');
assert.equal(publication.zero_contact_external_test_executed, false);
for (const [name, hash] of Object.entries(publication.source_receipt_sha256_by_relative_path)) {
  assert.equal(manifest.get('workflow/' + name), hash, name);
}

const summary = csv('TAIL_SUMMARY.csv');
const seedRows = csv('TAIL_SEEDS.csv');
const stdRows = csv('TAIL_SEED_SD.csv');
const metrics = ['RMSE', 'MAE', 'MSE', 'R2_y', 'R2_d', 'eta_pers_pct'];
function close(a, b, message) {
  assert(Number.isFinite(a) && Number.isFinite(b) &&
    Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b)), message + ': ' + a + ' vs ' + b);
}
let summaryCount = 0;
let seedCount = 0;
for (const [entity, geometry] of Object.entries(protocol.geometry)) {
  const result = tail[entity];
  const length = geometry.tail[1] - geometry.tail[0];
  const [priorStart, priorEnd] = geometry.historical_target_scoring.range;
  assert.equal(result.release, publication.release_lock_sha256);
  assert.deepEqual(result.rows, geometry.tail);
  assert.deepEqual(result.prior_scored_range, [priorStart, priorEnd]);
  assert.equal(result.checks.continuous_issue, geometry.continuous_issue_start);
  assert.equal(result.subsets.full_tail.N, length);
  if (entity !== 'SRU_H2S') assert.equal(result.subsets.nonoverlap_origins.N, length - (priorEnd - priorStart));
  assert.equal(result.checks.additive_max, 0);
  assert(result.checks.attribution_max <= 5e-11 && result.checks.native_max <= 5e-11);
  assert.equal(result.checks.native_poison_max, 0);
  assert.equal(result.checks.python_poison_max, 0);
  for (const [subset, panel] of Object.entries(result.subsets)) {
    assert.equal(Object.keys(panel.scores).length, 6);
    for (const [model, score] of Object.entries(panel.scores)) {
      const matches = row => row.entity === entity && row.subset === subset && row.model === model;
      const summaries = summary.filter(matches);
      const stds = stdRows.filter(matches);
      const seeds = seedRows.filter(matches);
      assert.equal(summaries.length, 1);
      assert.equal(stds.length, 1);
      assert.equal(seeds.length, ['ElasticNet', 'Transformer_expanded'].includes(model) ? 5 : 1);
      for (const row of [...summaries, ...stds, ...seeds]) assert.equal(Number(row.N), panel.N);
      for (const metric of metrics) {
        const values = seeds.map((row, i) => {
          assert.equal(Number(row.seed), i);
          const value = Number(row[metric]);
          close(value, score.runs[i][metric], metric);
          return value;
        });
        const mean = values.reduce((sum, n) => sum + n, 0) / values.length;
        close(Number(summaries[0][metric]), mean, metric);
        close(score.mean[metric], mean, metric);
        if (values.length === 1) {
          assert.equal(stds[0][metric], '');
        } else {
          const sd = Math.sqrt(values.reduce((sum, n) => sum + (n - mean) ** 2, 0) / (values.length - 1));
          close(Number(stds[0][metric]), sd, metric);
          close(score.sd[metric], sd, metric);
        }
      }
      summaryCount++;
      seedCount += seeds.length;
    }
  }
}
assert.equal(summary.length, summaryCount);
assert.equal(stdRows.length, summaryCount);
assert.equal(seedRows.length, seedCount);

const pareto = csv('PARETO_TABLE.csv');
const scaling = csv('LONG_WINDOW_TABLE.csv');
assert.equal(pareto.length, 15);
assert.equal(scaling.length, 36);
for (const row of pareto) {
  assert(Number(row.median_ns_or_mean_of_seed_medians) > 0);
  close(Number(row.full_tail_RMSE), tail[row.entity].subsets.full_tail.scores[row.model].mean.RMSE, 'Pareto RMSE');
}
for (const entity of Object.keys(tail)) {
  for (const kernel of ['PRISM', 'ARX']) {
    const lengths = scaling.filter(row => row.entity === entity && row.kernel === kernel)
      .map(row => {
        assert(Number(row.median_ns) > 0);
        return Number(row.L);
      });
    assert.deepEqual(lengths, [64, 128, 256, 512, 1024, 2048]);
  }
}
for (const name of ['BENCH_PARETO.json', 'BENCH_SCALING.json']) {
  const evidence = json('results/' + name);
  assert.equal(evidence.release, publication.release_lock_sha256);
  assert(evidence.flags.includes('-mavx2') && evidence.flags.includes('-mfma'));
}
console.log('Verified ' + manifest.size + ' files, ' + summaryCount + ' tail summaries, ' +
  seedCount + ' individual scores, ' + pareto.length + ' Pareto points and ' + scaling.length + ' window timings.');
