#![feature(unbounded_shifts)]
#![feature(test)]
#![allow(incomplete_features)]
#![warn(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]

mod decomp2x2;
mod decomp3x3;
mod decomp4x4;
mod decomp5x5;

mod gf2mat;
mod m4rm;
mod mul;
mod decomp_macro;

pub use m4rm::*;
pub use gf2mat::*;
pub use mul::addmul;

extern crate test;
#[cfg(test)]
mod tests {
    use m4ri_rust::friendly::BinMatrix;
    use rand::Rng;

    use crate::mul::addmul;

    use super::*;

    const MAT_TEST_SIZES: &'static [usize] = &[
        // covers all congruency classes % 16
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        256, 512, 513, 647
    ];

    #[test]
    fn test_mul_strassen() {
        let mut rng = rand::thread_rng();
        for _ in 0..1 {
            let n = rng.gen_range(2048..4096);
            let k = rng.gen_range(2048..4096);
            let m = rng.gen_range(2048..4096);

            let m1_m4ri = BinMatrix::random(n, k);
            let m2_m4ri = BinMatrix::random(k, m);
    
            let m1 = GF2Mat::from_m4ri(&m1_m4ri);
            let m2 = GF2Mat::from_m4ri(&m2_m4ri);
            let mut diff = GF2Mat::zero(n, m);
            unsafe { addmul(&mut diff, &m1, &m2); }
            let tgt = GF2Mat::from_m4ri(&(m1_m4ri * m2_m4ri));
            unsafe { diff.add_unchecked(&tgt); }
            //println!("diff:\n{:?}", diff);
            //assert!(false);
            assert!(diff == GF2Mat::zero(n, m));

        }
    }


    #[test]
    fn test_mul_consistent() {
        for mat_sz in MAT_TEST_SIZES {
            println!("Testing mat_sz: {}", mat_sz);
            let m1_m4ri = BinMatrix::random(*mat_sz, *mat_sz);
            let m2_m4ri = BinMatrix::random(*mat_sz, *mat_sz);
            let m1 = GF2Mat::from_m4ri(&m1_m4ri);
            let m2 = GF2Mat::from_m4ri(&m2_m4ri);
            let tgt = GF2Mat::from_m4ri(&(m1_m4ri * m2_m4ri));
            let mut diff = GF2Mat::zero(*mat_sz, *mat_sz);
            unsafe { addmul(&mut diff, &m1, &m2); }
            unsafe { diff.add_unchecked(&tgt); }
            assert!(diff == GF2Mat::zero(*mat_sz, *mat_sz));
        }

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let nrows = rng.gen_range(1..512);
            let conn = rng.gen_range(1..512);
            let ncols = rng.gen_range(1..512);
            let m1_m4ri = BinMatrix::random(nrows, conn);
            let m2_m4ri = BinMatrix::random(conn, ncols);
            let m1 = GF2Mat::from_m4ri(&m1_m4ri);
            let m2 = GF2Mat::from_m4ri(&m2_m4ri);   
            let mut prod = GF2Mat::zero(nrows, ncols);
            unsafe { addmul(&mut prod, &m1, &m2); }
            let tgt = m1_m4ri * m2_m4ri;
            assert!(prod == GF2Mat::from_m4ri(&tgt));
        }
    }
}