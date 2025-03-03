use crate::gf2mat::*;
use const_for::const_for;

const fn gray_encode(num: usize) -> usize {
    num ^ (num >> 1)
}

/// here for reference
const fn gray_decode(mut num: usize) -> usize {
    let mut p = num;
    while {
        num >>= 1;
        num != 0
    } {
        p ^= num;
    }
    p
}

const MAX_TBL_SZ: usize = 8;
const MAX_CODE_LEN: usize = 1 << MAX_TBL_SZ;
type GrayTableT = u8;

const _: () = const {
    assert!(
        MAX_TBL_SZ <= GrayTableT::BITS as usize,
        "change GrayTableT to accomodate MAX_TBL_SZ bits"
    );
};

// mapping from number to it's index in the gray code
const GRAY_TBL: [GrayTableT; MAX_CODE_LEN] = const {
    let mut tbl: [GrayTableT; MAX_CODE_LEN] = [0; MAX_CODE_LEN];
    const_for!(i in 0..MAX_CODE_LEN => {
        tbl[gray_encode(i)] = i as GrayTableT;
    });
    tbl
};

// INC_TBL[i] says which bit-index to xor to get from
// the i-th gray code to the (i+1)-th gray code
pub const INC_TBL: [GrayTableT; MAX_CODE_LEN] = const {
    let mut tbl: [GrayTableT; MAX_CODE_LEN] = [0; MAX_CODE_LEN];
    const_for!(i in (0..MAX_TBL_SZ).rev() => {
        const_for!(j in 1..(1<<i)+1 => {
            tbl[j*(1<<(MAX_TBL_SZ-i))-1] = (MAX_TBL_SZ - i) as GrayTableT;
        });
    });
    tbl
};

pub unsafe fn tabulate_m4rm<const ALIGN: usize, T, S>(
    tbl: &mut T,
    src_mat: &S,
    src_row: usize,
    tbl_sz: usize,
) where
    T: GF2MatLikeMut<ALIGN>,
    S: GF2MatLike<ALIGN>,
{
    debug_assert!(tbl_sz <= MAX_TBL_SZ);
    debug_assert!(src_row + tbl_sz <= src_mat.nrows());
    debug_assert!(1 << tbl_sz <= tbl.nrows());
    debug_assert!(tbl.nchunk_cols() == src_mat.nchunk_cols());
    debug_assert!(unsafe { tbl.row_slice_unchecked(0) }
        .iter()
        .all(|&x| x == 0));
    for i in 1..(1 << tbl_sz) {
        let inc_idx = INC_TBL[i - 1] as usize;
        unsafe {
            tbl.add_row_row_from_mat(src_mat, src_row + inc_idx, i - 1, i);
        }
    }
}

pub unsafe fn addmul_m4rm<const ALIGN: usize, T, S1, S2>(tgt: &mut T, lhs: &S1, rhs: &S2)
where T: GF2MatLikeMut<ALIGN>,
    S1: GF2MatLike<ALIGN>,
    S2: GF2MatLike<ALIGN> {
    debug_assert!(tgt.nrows() == lhs.nrows());
    debug_assert!(tgt.nchunk_cols() == rhs.nchunk_cols()); // TODO
    //debug_assert!(tgt.nbyte_cols() >= rhs.nbyte_cols());
    debug_assert!(lhs.max_ncols() >= rhs.nrows());
    debug_assert!(tgt.nrows() != 0 && tgt.nchunk_cols() != 0 && lhs.nchunk_cols() != 0); // TODO

    // here we use the fixed table size of 8, changing this may (will) break things
    const TBL_SZ: usize = 8;
    const _: () = const { assert!(TBL_SZ <= MAX_TBL_SZ); };

    let mut tbl: AlignedGF2Mat<ALIGN> = AlignedGF2Mat::zero(1<<TBL_SZ, rhs.max_ncols());
    for slice_start in (0..rhs.nrows()).step_by(TBL_SZ) {
        let sub_tbl_sz = (rhs.nrows() - slice_start).min(TBL_SZ);
        unsafe { tabulate_m4rm(&mut tbl, rhs, slice_start, sub_tbl_sz); }
        for i in 0..lhs.nrows() {
            let num = unsafe { lhs.row_slice_unchecked(i)[slice_start/u8::BITS as usize] };
            let tbl_row = GRAY_TBL[num as usize] as usize;
            unsafe { tgt.add_row_from_mat(&tbl, tbl_row, i); }
        }
    }
}