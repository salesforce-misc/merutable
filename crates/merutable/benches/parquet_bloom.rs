use criterion::{Criterion, black_box, criterion_group, criterion_main};
use merutable::parquet::bloom::FastLocalBloom;

fn bench_bloom_add_and_probe(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom");

    group.bench_function("add_10k_keys", |b| {
        b.iter(|| {
            let mut bloom = FastLocalBloom::new(10_000, 10);
            for i in 0u64..10_000 {
                bloom.add(&i.to_le_bytes());
            }
            black_box(bloom);
        });
    });

    group.bench_function("probe_10k_keys", |b| {
        let mut bloom = FastLocalBloom::new(10_000, 10);
        for i in 0u64..10_000 {
            bloom.add(&i.to_le_bytes());
        }
        b.iter(|| {
            let mut found = 0u32;
            for i in 0u64..10_000 {
                if bloom.may_contain(&i.to_le_bytes()) {
                    found += 1;
                }
            }
            black_box(found);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_bloom_add_and_probe);
criterion_main!(benches);
