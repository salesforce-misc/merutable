use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_compaction_placeholder(c: &mut Criterion) {
    c.bench_function("compaction_noop", |b| {
        b.iter(|| {
            // Placeholder — real benchmarks added when ParquetReader is wired up.
            black_box(42);
        });
    });
}

criterion_group!(benches, bench_compaction_placeholder);
criterion_main!(benches);
