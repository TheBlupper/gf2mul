use clap::Parser;
use std::{
    hint::black_box,
    path::Path,
    sync::LazyLock,
};
use m4ri_rust::friendly::*;
use gf2mul::*;
use std::arch::x86_64::_rdtsc;

#[derive(Parser)]
struct Args {
    #[clap(long)]
    out_fn: String,
    #[clap(long)]
    from: Option<usize>,
    #[clap(long)]
    to: Option<usize>,
    #[clap(long)]
    nsamples: Option<usize>,
    #[clap(long)]
    step: Option<usize>,
}

static ARGS: LazyLock<Args> = LazyLock::new(|| Args::parse());

fn timestamp() -> u64 {
    unsafe { _rdtsc() }
}

fn bench_m4rm_aligned(mat_sz: usize, nsamples: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let m1 = GF2Mat::random(mat_sz, mat_sz, &mut rng);
    let m2 = GF2Mat::random(mat_sz, mat_sz, &mut rng);

    let start = timestamp();
    for _ in 0..nsamples {
        let mut prod = GF2Mat::zero(mat_sz, mat_sz);
        unsafe { addmul_m4rm(black_box(&mut prod), black_box(&m1), black_box(&m2)); }
    }
    let dur = timestamp() - start;
    dur as f64 / nsamples as f64
}

fn bench_own(mat_sz: usize, nsamples: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let m1 = GF2Mat::random(mat_sz, mat_sz, &mut rng);
    let m2 = GF2Mat::random(mat_sz, mat_sz, &mut rng);

    let start = timestamp();
    for _ in 0..nsamples {
        let mut prod = GF2Mat::zero(mat_sz, mat_sz);
        unsafe { addmul(&mut prod, black_box(&m1), black_box(&m2)); }
    }
    let dur = timestamp() - start;
    dur as f64 / nsamples as f64
}

fn bench_m4ri(mat_sz: usize, nsamples: usize) -> f64 {
    let m1 = BinMatrix::random(mat_sz, mat_sz);
    let m2 = BinMatrix::random(mat_sz, mat_sz);
    let start = timestamp();
    for _ in 0..nsamples {
        let _ = black_box(&m1) * black_box(&m2);
    }
    let dur = timestamp() - start;
    dur as f64 / nsamples as f64
}

fn main() {
    let from = ARGS.from.unwrap_or(16);
    let to = ARGS.to.unwrap_or(2048);
    let mat_szs = (from..to).step_by(ARGS.step.unwrap_or(16*4));

    let funcs: Vec<(&str, fn(usize, usize) -> f64)> = vec![
        ("own_m4rm", bench_m4rm_aligned),
        ("own", bench_own),
        ("m4ri", bench_m4ri),
    ];

    let nsamples = ARGS.nsamples.unwrap_or(10);
    let mut results = Vec::new();
    for (name, func) in funcs.iter() {
        println!("Benchmarking {}", name);
        for mat_sz in mat_szs.clone() {
            println!("mat_sz: {}", mat_sz);
            let cycles = func(mat_sz, nsamples) as f64;
            results.push(Result {
                method_name: name.to_string(),
                mat_sz,
                cycles,
            });
        }
    }

    save_results(&results, &ARGS.out_fn);

}

pub fn save_results(results: &Vec<Result>, name: &str) {
    let f = Path::new(name);
    let f = f.with_extension("json");
    let f = std::fs::File::create(f).unwrap();
    serde_json::to_writer(f, &results).unwrap();
}

#[derive(serde::Serialize)]
pub struct Result {
    pub method_name: String,
    pub mat_sz: usize,
    pub cycles: f64
}
