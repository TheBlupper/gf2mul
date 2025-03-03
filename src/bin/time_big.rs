use gf2mul::*;
use m4ri_rust::friendly::*;


fn main() {
    /*let n = std::env::args().nth(1).unwrap().parse::<usize>().unwrap();
    println!("generating...");
    let m1_m4ri = BinMatrix::random(n, n);
    let m2_m4ri = BinMatrix::random(n, n);
    println!("converting...");
    let m1 = GF2Mat::from_m4ri(&m1_m4ri);
    let m2 = GF2Mat::from_m4ri(&m2_m4ri);
    let mut res_own = GF2Mat::zero(n, n);
    println!("running...");
    let start = std::time::Instant::now();
    unsafe { addmul(&mut res_own, &m1, &m2) }
    let duration = start.elapsed();
    println!("Own: {}.{:03} seconds", duration.as_secs(), duration.subsec_millis());
    /*let start = std::time::Instant::now();
    unsafe { prod.clear();prod.add_product_strassen_unchecked::<false>(&m1, &m2); }
    let duration = start.elapsed();
    println!("Strassen: {}.{:03} seconds", duration.as_secs(), duration.subsec_millis());*/

    let start = std::time::Instant::now();
    let res_m4ri = &m1_m4ri * &m2_m4ri;
    let duration = start.elapsed();
    println!("m4ri: {}.{:03} seconds", duration.as_secs(), duration.subsec_millis());
    println!("checking...");
    assert!(res_own == GF2Mat::from_m4ri(&res_m4ri));*/


}