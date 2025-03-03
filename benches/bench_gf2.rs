use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use gf2mul::*;

extern crate m4ri_rust;
use m4ri_rust::friendly::*;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("group");
    group.warm_up_time(std::time::Duration::from_nanos(1));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));

    let mut rng = rand::thread_rng();

    macro_rules! bench_mul_m4ri {
        ($id:expr, $a:tt, $b:tt, $d:tt) => {
            group.bench_function($id, |b| {
                let m1 = BinMatrix::random($a, $b);
                let m2 = BinMatrix::random($b, $d);
                b.iter(|| &m1 * &m2);
            })
        };
    }

    macro_rules! bench_mul_own {
        ($id:expr, $a:tt, $b:tt, $d:tt) => {
            group.bench_function($id, |b| {
                let m1 = GF2Mat::random($a, $b, &mut rng);
                let m2 = GF2Mat::random($b, $d, &mut rng);
                let mut prod = GF2Mat::zero($a, $d);
                b.iter(|| {unsafe { prod.clear();addmul(&mut prod, black_box(&m1), black_box(&m2));}});
            })
        };
    }
    
    let n = 10000;
    bench_mul_own!(format!("own mul {}", n), n, n, n);
    bench_mul_m4ri!(format!("m4ri mul {}", n), n, n, n);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);