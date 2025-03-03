use crate::gf2mat::{GF2MatLike, GF2MatLikeMut};

use crate::m4rm::addmul_m4rm;
use crate::decomp2x2::addmul_decomp2x2;
//use crate::decomp3x3::addmul_decomp3x3;
use crate::decomp4x4::addmul_decomp4x4;
//use crate::decomp5x5::addmul_decomp5x5;

const STRASSEN_CUTOFF: usize = 4096;
const WINDOW_ALIGN: usize = 16;
const U8SZ: usize = u8::BITS as usize;
const _ : () = const {
    assert!(WINDOW_ALIGN*U8SZ*2 <= STRASSEN_CUTOFF);
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddMulAlgo {
    M4RM,
    Decomp2x2,
    Decomp3x3,
    Decomp4x4,
    Decomp5x5,
}

impl AddMulAlgo {
    /// Whether the algorithm doesn't need another one to recurse into
    fn is_independant(&self) -> bool {
        match self {
            AddMulAlgo::M4RM => true,
            _ => false
        }
    }
}

pub unsafe fn addmul<const ALIGN: usize, T, S1, S2>(tgt: &mut T, lhs: &S1, rhs: &S2)
where T: GF2MatLikeMut<ALIGN>,
    S1: GF2MatLike<ALIGN>,
    S2: GF2MatLike<ALIGN> {

    debug_assert_eq!(tgt.nrows(), lhs.nrows());
    debug_assert_eq!(tgt.max_ncols(), rhs.max_ncols());
    debug_assert!(lhs.max_ncols() >= rhs.nrows());

    let (m, k, n) = (lhs.nrows(), rhs.nrows(), rhs.max_ncols());
    let min_dim = m.min(k).min(n);
    if min_dim < STRASSEN_CUTOFF {
        unsafe { addmul_m4rm(tgt, lhs, rhs) };
        return;
    }

    let mut algos = Vec::<AddMulAlgo>::with_capacity(8);
    let mut exp = 1;
    let mut recurse_sz = 2 * WINDOW_ALIGN * U8SZ;
    {
        let mut res_block_sz = min_dim/2;
        while res_block_sz > STRASSEN_CUTOFF {
            recurse_sz *= 2;
            res_block_sz /= 2;
            exp += 1;
        }
    }
    // TODO: finding optimal decompositions for non-powers of 2
    // specifically 2, 3, 4, 5
    // TODO: pad instead of peel sometimes
    let decomp4x4_times = exp/2;
    let decomp2x2_times = exp - 2*decomp4x4_times;
    for _ in 0..decomp4x4_times {
        algos.push(AddMulAlgo::Decomp4x4);
    }
    for _ in 0..decomp2x2_times {
        algos.push(AddMulAlgo::Decomp2x2);
    }
    algos.push(AddMulAlgo::M4RM);
    
    let mm = m - (m % recurse_sz);
    let kk = k - (k % recurse_sz);
    let nn = n - (n % recurse_sz);

    assert!([mm, kk, nn].iter().all(|x| x % (1<<exp)*WINDOW_ALIGN*U8SZ == 0));
    let chunk_k = kk / (U8SZ*WINDOW_ALIGN);
    let chunk_n = nn / (U8SZ*WINDOW_ALIGN);

    let m_rem = m - mm;
    let k_rem = k - kk;
    let n_rem = n - nn;
    {
        let mut tgt_recurse = unsafe { tgt.get_window_mut_unchecked::<WINDOW_ALIGN>(0, 0, mm, chunk_n) };
        let lhs_recurse = unsafe { lhs.get_window_unchecked::<WINDOW_ALIGN>(0, 0, mm, chunk_k) };
        let rhs_recurse = unsafe { rhs.get_window_unchecked::<WINDOW_ALIGN>(0, 0, kk, chunk_n) };

        unsafe { addmul_recurse(&mut tgt_recurse, &lhs_recurse, &rhs_recurse, &algos) };
    }

    let a_nwinchunk_cols = lhs.nchunk_cols()*ALIGN/WINDOW_ALIGN;
    let b_nwinchunk_cols = rhs.nchunk_cols()*ALIGN/WINDOW_ALIGN;
    let lhs_window = unsafe { lhs.get_window_unchecked::<WINDOW_ALIGN>(0, 0, m, a_nwinchunk_cols) };
    if n_rem > 0 {
        let b_last_col = unsafe { rhs.get_window_unchecked::<WINDOW_ALIGN>(0, chunk_n, k, b_nwinchunk_cols - chunk_n) };
        let mut c_last_col = unsafe { tgt.get_window_mut_unchecked::<WINDOW_ALIGN>(0, chunk_n, m, b_nwinchunk_cols - chunk_n) };
        unsafe { addmul_m4rm(&mut c_last_col, &lhs_window, &b_last_col) };
    }

    if m_rem > 0 {
        let a_last_row = unsafe { lhs.get_window_unchecked::<WINDOW_ALIGN>(mm, 0, m_rem, a_nwinchunk_cols) };
        let b_first_col = unsafe { rhs.get_window_unchecked::<WINDOW_ALIGN>(0, 0, k, chunk_n) };
        let mut c_last_row = unsafe { tgt.get_window_mut_unchecked::<WINDOW_ALIGN>(mm, 0, m_rem, chunk_n) };
        unsafe { addmul_m4rm(&mut c_last_row, &a_last_row, &b_first_col) };
    }

    if k_rem > 0 {
        let a_last_col = unsafe { lhs.get_window_unchecked::<WINDOW_ALIGN>(0, chunk_k, mm, a_nwinchunk_cols - chunk_k) };
        let b_last_row = unsafe { rhs.get_window_unchecked::<WINDOW_ALIGN>(kk, 0, k_rem, chunk_n) };
        let mut c_bulk = unsafe { tgt.get_window_mut_unchecked::<WINDOW_ALIGN>(0, 0, mm, chunk_n) };
        unsafe { addmul_m4rm(&mut c_bulk, &a_last_col, &b_last_row) };
    }
}

pub unsafe fn addmul_recurse<const ALIGN: usize, T, S1, S2>(tgt: &mut T, lhs: &S1, rhs: &S2, algos: &[AddMulAlgo])
where T: GF2MatLikeMut<ALIGN>,
    S1: GF2MatLike<ALIGN>,
    S2: GF2MatLike<ALIGN> {
    
    let (algo, algos) = algos.split_first().expect("No algo to recurse into");
    if algos.is_empty() {
        debug_assert!(algo.is_independant(), "Last algo must be independant");
    }

    match algo {
        AddMulAlgo::M4RM => {
            unsafe { addmul_m4rm(tgt, lhs, rhs) }
        },
        AddMulAlgo::Decomp2x2 => {
            unsafe { addmul_decomp2x2::<ALIGN, WINDOW_ALIGN, _, _, _>(tgt, lhs, rhs, algos) };
        },
        AddMulAlgo::Decomp3x3 => {
            // not used atm, and increases compilation time
            //unsafe { addmul_decomp3x3::<ALIGN, WINDOW_ALIGN, _, _, _>(tgt, lhs, rhs, algos) };
            todo!();
        },
        AddMulAlgo::Decomp4x4 => {
            unsafe { addmul_decomp4x4::<ALIGN, WINDOW_ALIGN, _, _, _>(tgt, lhs, rhs, algos) }; 
        },
        AddMulAlgo::Decomp5x5 => {
            // not used atm, and increases compilation time
            //unsafe { addmul_decomp5x5::<ALIGN, WINDOW_ALIGN, _, _, _>(tgt, lhs, rhs, algos) };
            todo!();
        }
    }
}