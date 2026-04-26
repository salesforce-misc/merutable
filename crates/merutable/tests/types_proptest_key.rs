//! Issue #13: property-based tests for `InternalKey` encoding.
//!
//! These tests validate the four load-bearing invariants of the
//! encoding against thousands of randomly generated inputs. Every
//! historical correctness bug in this area (Bug K2 / K3 seq-wraparound,
//! Issue #7 empty-vs-null-prefix collision, Issue #8 seq monotonicity,
//! the v2 terminator width fix) fell inside one of these four
//! invariants. Property tests catch the next one before it ships.
//!
//! **Invariants:**
//!
//! 1. **Round-trip**: `decode(encode(pk, seq, op)) == (pk, seq, op)`
//!    for every supported field-type combination, every legal seq,
//!    both op types.
//!
//! 2. **Memcmp ordering**: the encoded bytes compare lexicographically
//!    the same way `Ord for InternalKey` does. This is what keeps the
//!    skiplist and Parquet page index correct.
//!
//! 3. **Prefix safety**: if `pk_a`'s raw bytes are a strict prefix of
//!    `pk_b`'s raw bytes (for ByteArray composite keys), then
//!    `encode(pk_a, *, *) < encode(pk_b, *, *)` regardless of the
//!    seq / op_type appended. This is the Issue #7 guarantee.
//!
//! 4. **Seek-latest sentinel**: `seek_latest(pk) <= encode(pk, seq, op)`
//!    for any legal `seq` and `op`. Skiplist seeks MUST land at or
//!    before the newest visible version; otherwise `lower_bound` can
//!    miss live entries.
//!
//! Proptest default of 256 cases per property is the workspace baseline;
//! runs in well under a second. CI catches class bugs, not just specimens.

use bytes::Bytes;
use merutable::types::{
    key::InternalKey,
    schema::{ColumnDef, ColumnType, TableSchema},
    sequence::{OpType, SEQNUM_MAX, SeqNum},
    value::FieldValue,
};
use proptest::prelude::*;

// ── Strategies ───────────────────────────────────────────────────────

fn any_op_type() -> impl Strategy<Value = OpType> {
    prop_oneof![Just(OpType::Put), Just(OpType::Delete)]
}

/// Seq values in [0, SEQNUM_MAX]. Biased toward small seqs (the
/// realistic range) and the boundary values where previous bugs lived.
///
/// `SEQNUM_MAX` is reserved as the seek-sentinel tag construction
/// base — real entries should never be written at that seq. Tests
/// that assert seek_latest is a lower bound therefore use
/// `any_real_seq()` (below), which excludes SEQNUM_MAX.
fn any_seq() -> impl Strategy<Value = SeqNum> {
    prop_oneof![
        Just(SeqNum(0)),
        Just(SeqNum(1)),
        Just(SeqNum(SEQNUM_MAX.0)),
        Just(SeqNum(SEQNUM_MAX.0 - 1)),
        (0u64..1_000_000).prop_map(SeqNum),
        (0u64..=SEQNUM_MAX.0).prop_map(SeqNum),
    ]
}

/// Seqs excluding the SEQNUM_MAX sentinel. Use when the property
/// asserts ordering relative to `seek_latest` (which itself is
/// constructed at SEQNUM_MAX and would be nondeterministic against
/// real entries at the same reserved seq).
fn any_real_seq() -> impl Strategy<Value = SeqNum> {
    prop_oneof![
        Just(SeqNum(0)),
        Just(SeqNum(1)),
        Just(SeqNum(SEQNUM_MAX.0 - 1)),
        (0u64..1_000_000).prop_map(SeqNum),
        (0u64..SEQNUM_MAX.0).prop_map(SeqNum),
    ]
}

/// Byte-array contents biased toward failure-mode-targeting shapes:
/// empty, all-null, leading/trailing nulls, all-0xFF (terminator-
/// adjacent), and the mirror Issue #7 regression set.
fn any_bytes() -> impl Strategy<Value = Bytes> {
    prop_oneof![
        Just(Bytes::new()),
        Just(Bytes::from_static(&[0u8])),
        Just(Bytes::from_static(&[0u8, 0u8])),
        Just(Bytes::from_static(&[0xFFu8])),
        Just(Bytes::from_static(&[0u8, 0xFFu8, 0u8])),
        Just(Bytes::from_static(&[0u8, 0x01u8])),
        Just(Bytes::from_static(&[0x01u8, 0u8])),
        prop::collection::vec(any::<u8>(), 0..32).prop_map(Bytes::from),
    ]
}

/// Generate `(TableSchema, pk_a, pk_b)` triples where both PKs share
/// the schema's column types. Covers Int64, ByteArray, and composite
/// (Int64, ByteArray) schemas — the common production shapes.
fn any_schema_and_two_pks() -> impl Strategy<Value = (TableSchema, Vec<FieldValue>, Vec<FieldValue>)>
{
    prop_oneof![
        // Int64 PK.
        (any::<i64>(), any::<i64>()).prop_map(|(a, b)| {
            let s = TableSchema {
                table_name: "t".into(),
                columns: vec![ColumnDef {
                    name: "id".into(),
                    col_type: ColumnType::Int64,
                    nullable: false,

                    ..Default::default()
                }],
                primary_key: vec![0],

                ..Default::default()
            };
            (s, vec![FieldValue::Int64(a)], vec![FieldValue::Int64(b)])
        }),
        // ByteArray PK.
        (any_bytes(), any_bytes()).prop_map(|(a, b)| {
            let s = TableSchema {
                table_name: "t".into(),
                columns: vec![ColumnDef {
                    name: "k".into(),
                    col_type: ColumnType::ByteArray,
                    nullable: false,

                    ..Default::default()
                }],
                primary_key: vec![0],

                ..Default::default()
            };
            (s, vec![FieldValue::Bytes(a)], vec![FieldValue::Bytes(b)])
        }),
        // Composite (Int64, ByteArray) PK.
        (any::<i64>(), any_bytes(), any::<i64>(), any_bytes()).prop_map(|(a1, a2, b1, b2)| {
            let s = TableSchema {
                table_name: "t".into(),
                columns: vec![
                    ColumnDef {
                        name: "a".into(),
                        col_type: ColumnType::Int64,
                        nullable: false,

                        ..Default::default()
                    },
                    ColumnDef {
                        name: "b".into(),
                        col_type: ColumnType::ByteArray,
                        nullable: false,

                        ..Default::default()
                    },
                ],
                primary_key: vec![0, 1],

                ..Default::default()
            };
            (
                s,
                vec![FieldValue::Int64(a1), FieldValue::Bytes(a2)],
                vec![FieldValue::Int64(b1), FieldValue::Bytes(b2)],
            )
        }),
    ]
}

/// Single-schema generator for round-trip and seek tests.
fn any_schema_and_pk() -> impl Strategy<Value = (TableSchema, Vec<FieldValue>)> {
    any_schema_and_two_pks().prop_map(|(s, a, _)| (s, a))
}

// ── Invariant 1: Round-trip ──────────────────────────────────────────

proptest! {
    #[test]
    fn roundtrip(
        (schema, pk) in any_schema_and_pk(),
        seq in any_seq(),
        op in any_op_type(),
    ) {
        let k = InternalKey::encode(&pk, seq, op, &schema).unwrap();
        let decoded = InternalKey::decode(k.as_bytes(), &schema).unwrap();
        prop_assert_eq!(decoded.seq, seq);
        prop_assert_eq!(decoded.op_type, op);
        prop_assert_eq!(decoded.pk_values(), pk.as_slice());
    }
}

// ── Invariant 2: Memcmp ordering ─────────────────────────────────────

proptest! {
    #[test]
    fn memcmp_matches_ord(
        (schema, pk_a, pk_b) in any_schema_and_two_pks(),
        seq_a in any_seq(),
        seq_b in any_seq(),
        op_a in any_op_type(),
        op_b in any_op_type(),
    ) {
        let ka = InternalKey::encode(&pk_a, seq_a, op_a, &schema).unwrap();
        let kb = InternalKey::encode(&pk_b, seq_b, op_b, &schema).unwrap();
        // Bytewise memcmp on the raw encoded wire bytes must agree
        // with the InternalKey Ord impl; they share the skiplist.
        prop_assert_eq!(ka.as_bytes().cmp(kb.as_bytes()), ka.cmp(&kb));
    }
}

// ── Invariant 3: Prefix safety ───────────────────────────────────────

proptest! {
    /// For ByteArray PKs specifically: if raw bytes of A are a strict
    /// prefix of B, then `encode(A) < encode(B)` — independent of the
    /// seq and op appended. This is the Issue #7 invariant, the one
    /// that broke with the single-byte terminator and now holds with
    /// the two-byte terminator.
    #[test]
    fn bytearray_prefix_always_sorts_first(
        prefix in prop::collection::vec(any::<u8>(), 0..16),
        extra in prop::collection::vec(any::<u8>(), 1..16),
        seq_a in any_seq(),
        seq_b in any_seq(),
        op_a in any_op_type(),
        op_b in any_op_type(),
    ) {
        let schema = TableSchema {
            table_name: "t".into(),
            columns: vec![ColumnDef {
                name: "k".into(),
                col_type: ColumnType::ByteArray,
                nullable: false,

                ..Default::default()
            }],
            primary_key: vec![0],

            ..Default::default()
        };
        let a_bytes = Bytes::from(prefix.clone());
        let mut b_raw = prefix;
        b_raw.extend_from_slice(&extra);
        let b_bytes = Bytes::from(b_raw);

        let ka = InternalKey::encode(
            &[FieldValue::Bytes(a_bytes)],
            seq_a,
            op_a,
            &schema,
        ).unwrap();
        let kb = InternalKey::encode(
            &[FieldValue::Bytes(b_bytes)],
            seq_b,
            op_b,
            &schema,
        ).unwrap();
        prop_assert!(
            ka < kb,
            "prefix A must sort before B regardless of tag: ka={:?} kb={:?}",
            ka.as_bytes(),
            kb.as_bytes(),
        );
    }
}

// ── Invariant 4: Seek-latest sentinel ────────────────────────────────

proptest! {
    #[test]
    fn seek_latest_is_lower_bound(
        (schema, pk) in any_schema_and_pk(),
        seq in any_real_seq(),
        op in any_op_type(),
    ) {
        let seek = InternalKey::seek_latest(&pk, &schema).unwrap();
        let real = InternalKey::encode(&pk, seq, op, &schema).unwrap();
        // seek_latest is constructed with (SEQNUM_MAX, Put). For any
        // real entry (i.e., seq < SEQNUM_MAX), the inverted_seq of
        // seek_latest is 0 and the real entry's inverted_seq is > 0,
        // so the real entry sorts strictly AFTER the seek. The
        // SEQNUM_MAX boundary itself is a reserved sentinel — not a
        // seq callers are supposed to use — and the proptest
        // strategy excludes it via `any_real_seq()`.
        prop_assert!(
            seek <= real,
            "seek_latest must be <= every real entry; seek={:?} real={:?} seq={} op={:?}",
            seek.as_bytes(),
            real.as_bytes(),
            seq.0,
            op,
        );
    }
}
