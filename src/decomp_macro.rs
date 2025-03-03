macro_rules! decomp_fn {
    ($func_name: ident, $dim:expr, |$a_blks:ident, $b_blks:ident, $c_blks:ident, $tmp_mk:ident, $tmp_mn:ident, $tmp_kn:ident, $algos:ident| $body: block) => {
        use crate::gf2mat::{AlignedGF2Mat, GF2MatLike, GF2MatLikeMut};
        use crate::mul::{addmul_recurse, AddMulAlgo};
        const U8SZ: usize = u8::BITS as usize;
        pub unsafe fn $func_name<const ALIGN: usize, const WINDOW_ALIGN: usize, T, S1, S2>(
            tgt: &mut T,
            lhs: &S1,
            rhs: &S2,
            $algos: &[AddMulAlgo],
        ) where
            T: GF2MatLikeMut<ALIGN>,
            S1: GF2MatLike<ALIGN>,
            S2: GF2MatLike<ALIGN>,
        {
            assert_eq!(ALIGN % WINDOW_ALIGN, 0);

            debug_assert_eq!(tgt.nrows(), lhs.nrows());
            debug_assert_eq!(tgt.nbyte_cols(), rhs.nbyte_cols());
            debug_assert!(lhs.max_ncols() >= rhs.nrows());

            let (m, k, n) = (lhs.nrows(), rhs.nrows(), rhs.max_ncols());

            debug_assert_ne!(m, 0);
            debug_assert_ne!(k, 0);
            debug_assert_ne!(n, 0);
            debug_assert_eq!(k % ($dim * WINDOW_ALIGN * U8SZ), 0);
            debug_assert_eq!(n % ($dim * WINDOW_ALIGN * U8SZ), 0);

            let mm = m / $dim;
            let kk = k / $dim;
            let nn = n / $dim;

            // let bm = mm / (U8SZ*WINDOW_ALIGN);
            let bk = kk / (U8SZ * WINDOW_ALIGN);
            let bn = nn / (U8SZ * WINDOW_ALIGN);

            unsafe {
                let mut $tmp_mk = AlignedGF2Mat::<WINDOW_ALIGN>::zero(mm, kk);
                let mut $tmp_mn = AlignedGF2Mat::<WINDOW_ALIGN>::zero(mm, nn);
                let mut $tmp_kn = AlignedGF2Mat::<WINDOW_ALIGN>::zero(kk, nn);
                let $a_blks = lhs.division_unchecked::<WINDOW_ALIGN, $dim, $dim>(mm, bk);
                let $b_blks = rhs.division_unchecked::<WINDOW_ALIGN, $dim, $dim>(kk, bn);
                let mut $c_blks = tgt.division_mut_unchecked::<WINDOW_ALIGN, $dim, $dim>(mm, bn);

                { $body }
            }
        }
    }
}
pub(crate) use decomp_fn;