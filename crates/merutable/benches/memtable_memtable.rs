use bytes::Bytes;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use merutable::memtable::memtable::Memtable;
use merutable::types::sequence::SeqNum;
use merutable::wal::batch::WriteBatch;

fn bench_memtable_insert(c: &mut Criterion) {
    c.bench_function("memtable_insert_10k", |b| {
        b.iter(|| {
            let mem = Memtable::new(SeqNum(1), 256 * 1024 * 1024);
            for i in 0u64..10_000 {
                let mut batch = WriteBatch::new(SeqNum(i + 1));
                batch.put(Bytes::from(i.to_be_bytes().to_vec()), Bytes::from("value"));
                mem.apply_batch(&batch).unwrap();
            }
            black_box(&mem);
        });
    });
}

criterion_group!(benches, bench_memtable_insert);
criterion_main!(benches);
