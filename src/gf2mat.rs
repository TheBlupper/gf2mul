/*!
In this module we define traits which aim to be generic
over matrices which own their data (AlignedGF2Mat) and
windows into such matrices (both mutable and immutable).

We currently have alignment as a type parameter. This doesn't
really make sense and we would much rather have it as an associated
constant. However to use this when providing functions with
const generics we need the "highly unstable" generic_const_exprs
feature, so we will have to do without for now.
*/

use std::fmt::Debug;
use std::fmt::Formatter;
use std::marker::PhantomData;

use aligned_vec::AVec;

use m4ri_rust::friendly::BinMatrix;
use rand::{rngs::ThreadRng, Rng};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MatMulError {
    #[error("Cannot multiply matrices of dimensions {0}x{1} and {2}x{3}")]
    IncompatibleMatrices(usize, usize, usize, usize),
    #[error("Cannot store matrix product of size {0}x{1} in matrix of size {2}x{3}")]
    DimMismatch(usize, usize, usize, usize),
}

#[derive(Error, Debug)]
pub enum MatAddError {
    #[error("Cannot add matrices of dimensions {0}x{1} and {2}x{3}")]
    IncompatibleMatrices(usize, usize, usize, usize),
    #[error("Cannot store matrix sum of size {0}x{1} in a matrix of size {2}x{3}")]
    DimMismatch(usize, usize, usize, usize),
}

#[derive(Error, Debug)]
pub enum MatAccessError {
    #[error("Bit position {0}x{1} is out of bounds for matrix of size {2}x{3}")]
    OutOfBounds(usize, usize, usize, usize),
}

// TODO: find automatically
const CACHELINE_SZ: usize = 0x80;

pub unsafe trait GF2MatLike<const ALIGN: usize> {
    fn nrows(&self) -> usize;
    fn nchunk_cols(&self) -> usize;
    fn data_ptr(&self) -> *const u8;
    fn row_stride(&self) -> usize;

    fn max_ncols(&self) -> usize {
        self.nchunk_cols() * ALIGN * u8::BITS as usize
    }

    fn nbyte_cols(&self) -> usize {
        self.nchunk_cols() * ALIGN
    }

    unsafe fn row_slice_unchecked(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.nrows());
        let start = i * self.row_stride();
        let len = self.nbyte_cols();
        unsafe { std::slice::from_raw_parts(self.data_ptr().add(start), len) }
    }

    unsafe fn get_window_unchecked<const WINDOW_ALIGN: usize>(
        &self,
        row: usize,
        chunk_col: usize,
        nrows: usize,
        nchunk_cols: usize,
    ) -> GF2MatWindow<WINDOW_ALIGN> {
        debug_assert!(ALIGN % WINDOW_ALIGN == 0);
        debug_assert!(row + nrows <= self.nrows());
        debug_assert!(chunk_col + nchunk_cols <= self.nbyte_cols() / WINDOW_ALIGN);

        GF2MatWindow {
            nrows,
            nchunk_cols: nchunk_cols,
            row_stride: self.row_stride(),
            data: unsafe {
                self.data_ptr()
                    .add(row * self.row_stride() + chunk_col * WINDOW_ALIGN)
            },
            phantom: PhantomData,
        }
    }

    unsafe fn division_unchecked<
        const WINDOW_ALIGN: usize,
        const NBLOCK_ROWS: usize,
        const NBLOCK_COLS: usize,
    >(
        &self,
        nrows: usize,
        nchunk_cols: usize,
    ) -> [[GF2MatWindow<WINDOW_ALIGN>; NBLOCK_COLS]; NBLOCK_ROWS] {
        debug_assert_eq!(ALIGN % WINDOW_ALIGN, 0);
        debug_assert!(nrows * NBLOCK_ROWS <= self.nrows());
        debug_assert!(nchunk_cols * NBLOCK_COLS <= self.nbyte_cols() / WINDOW_ALIGN);
        core::array::from_fn(|block_row| {
            core::array::from_fn(|block_col| {
                unsafe {
                    self.get_window_unchecked(
                        nrows * block_row,
                        nchunk_cols * block_col,
                        nrows,
                        nchunk_cols,
                    )
                }
            })
        })
    }

    fn to_mat(&self) -> AlignedGF2Mat<ALIGN>
    where Self: Sized {
        let mut mat = AlignedGF2Mat::zero(self.nrows(), self.max_ncols());
        mat.copy_from(self);
        mat
    }
}

pub unsafe trait GF2MatLikeMut<const ALIGN: usize>: GF2MatLike<ALIGN> {
    unsafe fn data_ptr_mut(&mut self) -> *mut u8;

    unsafe fn row_slice_mut_unchecked(&mut self, i: usize) -> &mut [u8] {
        debug_assert!(i < self.nrows());
        let start = i * self.row_stride();
        let len = self.nbyte_cols();
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr_mut().add(start), len) }
    }

    unsafe fn distinct_row_slices_mut_immut(&mut self, i: usize, j: usize) -> (&mut [u8], &[u8]) {
        debug_assert!(i != j);
        debug_assert!(i < self.nrows());
        debug_assert!(j < self.nrows());
        let start1 = i * self.row_stride();
        let start2 = j * self.row_stride();
        let len = self.nchunk_cols() * ALIGN;
        unsafe {
            (
                std::slice::from_raw_parts_mut(self.data_ptr_mut().add(start1), len),
                std::slice::from_raw_parts(self.data_ptr().add(start2), len),
            )
        }
    }

    unsafe fn add_row_row_from_mat<S>(
        &mut self,
        src_mat: &S,
        src_mat_row: usize,
        src_row: usize,
        dst_row: usize,
    ) where
        S: GF2MatLike<ALIGN>,
    {
        debug_assert!(dst_row < self.nrows());
        debug_assert!(src_row < self.nrows());
        debug_assert!(src_row != dst_row);
        debug_assert!(src_mat_row < src_mat.nrows());
        debug_assert!(self.nchunk_cols() == src_mat.nchunk_cols());
        let (dst, src1) = unsafe { self.distinct_row_slices_mut_immut(dst_row, src_row) };
        let src2 = unsafe { src_mat.row_slice_unchecked(src_mat_row) };
        unsafe {
            xor_aligned_slices_to::<ALIGN>(dst, src1, src2);
        }
    }

    unsafe fn add_row_from_mat<S>(&mut self, src_mat: &S, src_mat_row: usize, dst_row: usize)
    where
        S: GF2MatLike<ALIGN>,
    {
        debug_assert!(dst_row < self.nrows());
        debug_assert!(src_mat_row < src_mat.nrows());
        debug_assert!(self.nchunk_cols() == src_mat.nchunk_cols());
        let src = unsafe { src_mat.row_slice_unchecked(src_mat_row) };
        let dst = unsafe { self.row_slice_mut_unchecked(dst_row) };
        unsafe {
            xor_aligned_slices::<ALIGN>(dst, src);
        }
    }

    unsafe fn set_to_sum_unchecked<S1, S2>(&mut self, lhs: &S1, rhs: &S2)
    where
        S1: GF2MatLike<ALIGN>,
        S2: GF2MatLike<ALIGN>,
    {
        debug_assert!(self.nrows() == lhs.nrows());
        debug_assert!(self.nrows() == rhs.nrows());
        debug_assert!(self.nchunk_cols() == lhs.nchunk_cols());
        debug_assert!(self.nchunk_cols() == rhs.nchunk_cols());
        for i in 0..self.nrows() {
            let dst = unsafe { self.row_slice_mut_unchecked(i) };
            let src1 = unsafe { lhs.row_slice_unchecked(i) };
            let src2 = unsafe { rhs.row_slice_unchecked(i) };
            unsafe {
                xor_aligned_slices_to::<ALIGN>(dst, src1, src2);
            }
        }
    }

    unsafe fn add_unchecked<S>(&mut self, rhs: &S)
    where
        S: GF2MatLike<ALIGN>,
    {
        debug_assert!(self.nrows() == rhs.nrows());
        debug_assert!(self.nchunk_cols() == rhs.nchunk_cols());
        for i in 0..self.nrows() {
            let dst = unsafe { self.row_slice_mut_unchecked(i) };
            let src = unsafe { rhs.row_slice_unchecked(i) };
            unsafe {
                xor_aligned_slices::<ALIGN>(dst, src);
            }
        }
    }

    fn clear(&mut self) {
        for i in 0..self.nrows() {
            unsafe {
                self.row_slice_mut_unchecked(i)
                    .iter_mut()
                    .for_each(|x| *x = 0);
            }
        }
    }

    fn copy_from<S>(&mut self, src: &S)
    where
        S: GF2MatLike<ALIGN>,
    {
        debug_assert!(self.nrows() == src.nrows());
        debug_assert!(self.nchunk_cols() == src.nchunk_cols());
        let nrows = src.nrows();
        for i in 0..nrows {
            let src_row = unsafe { src.row_slice_unchecked(i) };
            let dst_row = unsafe { self.row_slice_mut_unchecked(i) };
            dst_row.copy_from_slice(src_row);
        }
    }

    unsafe fn get_window_mut_unchecked<const WINDOW_ALIGN: usize>(
        &mut self,
        row: usize,
        chunk_col: usize,
        nrows: usize,
        nchunk_cols: usize,
    ) -> GF2MatWindowMut<WINDOW_ALIGN> {
        debug_assert!(ALIGN % WINDOW_ALIGN == 0);
        debug_assert!(row + nrows <= self.nrows());
        debug_assert!(chunk_col + nchunk_cols <= (self.nchunk_cols() * ALIGN) / WINDOW_ALIGN);
        GF2MatWindowMut {
            nrows,
            nchunk_cols,
            row_stride: self.row_stride(),
            data: unsafe {
                self.data_ptr_mut()
                    .add(row * self.row_stride() + chunk_col * WINDOW_ALIGN)
            },
            phantom: PhantomData,
        }
    }

    unsafe fn division_mut_unchecked<
        const WINDOW_ALIGN: usize,
        const NBLOCK_ROWS: usize,
        const NBLOCK_COLS: usize,
    >(
        &mut self,
        nrows: usize,
        nchunk_cols: usize,
    ) -> [[GF2MatWindowMut<WINDOW_ALIGN>; NBLOCK_COLS]; NBLOCK_ROWS] {
        debug_assert_eq!(ALIGN % WINDOW_ALIGN, 0);
        debug_assert!(nrows * NBLOCK_ROWS <= self.nrows());
        debug_assert!(nchunk_cols * NBLOCK_COLS <= self.nbyte_cols() / WINDOW_ALIGN);
        core::array::from_fn(|block_row| {
            core::array::from_fn(|block_col| {
                GF2MatWindowMut {
                    nrows: nrows,
                    nchunk_cols: nchunk_cols,
                    row_stride: self.row_stride(),
                    data: unsafe {
                        (self.data_ptr_mut()).add(
                            nrows * block_row * self.row_stride()
                                + nchunk_cols * block_col * WINDOW_ALIGN,
                        )
                    },
                    phantom: PhantomData,
                }
            })
        })
    }
}

pub struct AlignedGF2Mat<const ALIGN: usize> {
    nrows: usize,
    ncols: usize,
    row_stride: usize,
    data: AVec<u8>,
}

/// This is what will usually be used, the more generic AlignedGF2Mat is
/// for consitency with the types of the matrix windows
pub type GF2Mat = AlignedGF2Mat<CACHELINE_SZ>;

pub struct GF2MatWindow<'a, const ALIGN: usize> {
    nrows: usize,
    nchunk_cols: usize,
    row_stride: usize,
    data: *const u8,
    phantom: PhantomData<&'a [u8]>,
}

pub struct GF2MatWindowMut<'a, const ALIGN: usize> {
    nrows: usize,
    nchunk_cols: usize,
    row_stride: usize,
    data: *mut u8,
    phantom: PhantomData<&'a mut [u8]>,
}

unsafe impl<const ALIGN: usize> GF2MatLike<ALIGN> for AlignedGF2Mat<ALIGN> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    fn nchunk_cols(&self) -> usize {
        self.row_stride / ALIGN
    }

    #[inline(always)]
    fn row_stride(&self) -> usize {
        self.row_stride
    }

    #[inline(always)]
    fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

unsafe impl<const ALIGN: usize> GF2MatLikeMut<ALIGN> for AlignedGF2Mat<ALIGN> {
    unsafe fn data_ptr_mut(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
}

unsafe impl<const ALIGN: usize> GF2MatLike<ALIGN> for GF2MatWindow<'_, ALIGN> {
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn nchunk_cols(&self) -> usize {
        self.nchunk_cols
    }

    fn data_ptr(&self) -> *const u8 {
        self.data
    }

    fn row_stride(&self) -> usize {
        self.row_stride
    }
}

unsafe impl<const ALIGN: usize> GF2MatLike<ALIGN> for GF2MatWindowMut<'_, ALIGN> {
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn nchunk_cols(&self) -> usize {
        self.nchunk_cols
    }

    fn row_stride(&self) -> usize {
        self.row_stride
    }

    fn data_ptr(&self) -> *const u8 {
        self.data
    }
}

unsafe impl<const ALIGN: usize> GF2MatLikeMut<ALIGN> for GF2MatWindowMut<'_, ALIGN> {
    unsafe fn data_ptr_mut(&mut self) -> *mut u8 {
        self.data
    }
}

impl<const ALIGN: usize> AlignedGF2Mat<ALIGN> {
    pub fn zero(nrows: usize, ncols: usize) -> Self {
        let row_stride = ncols.div_ceil(u8::BITS as usize).div_ceil(ALIGN) * ALIGN;
        let mut data = AVec::with_capacity(ALIGN, nrows * row_stride);
        data.resize(nrows * row_stride, 0);
        Self {
            nrows,
            ncols,
            row_stride,
            data,
        }
    }

    pub fn random(nrows: usize, ncols: usize, rng: &mut ThreadRng) -> Self {
        let mut mat = Self::zero(nrows, ncols);
        mat.data.iter_mut().for_each(|x| *x = rng.gen::<u8>());
        // zero out edges

        let mask = match ncols % u8::BITS as usize {
            0 => 0xffu8,
            rem => (1u8 << rem) - 1,
        };

        let last_limb: usize = (ncols - 1) / u8::BITS as usize;
        for i in 0..nrows {
            mat.data[i * mat.row_stride + last_limb] &= mask;
            for j in ncols.div_ceil(u8::BITS as usize)..mat.row_stride {
                mat.data[i * mat.row_stride + j] = 0;
            }
        }

        mat
    }

    pub fn from_m4ri(bm: &BinMatrix) -> Self {
        let nrows = bm.nrows();
        let ncols = bm.ncols();
        let mut res = AlignedGF2Mat::<ALIGN>::zero(nrows, ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                if bm.bit(i as usize, j as usize) {
                    res.try_set(i, j, true).unwrap();
                }
            }
        }
        res
    }

    unsafe fn get_unchecked(&self, i: usize, j: usize) -> bool {
        let byte_idx = j / u8::BITS as usize;
        let bit_idx = j % u8::BITS as usize;
        let byte = unsafe { self.row_slice_unchecked(i).get_unchecked(byte_idx) };
        (byte >> bit_idx) & 1 == 1
    }

    fn try_get(&self, i: usize, j: usize) -> Result<bool, MatAccessError> {
        if i >= self.nrows || j >= self.ncols {
            Err(MatAccessError::OutOfBounds(i, j, self.nrows, self.ncols))
        } else {
            Ok(unsafe { self.get_unchecked(i, j) })
        }
    }

    unsafe fn set_unchecked(&mut self, i: usize, j: usize, val: bool) {
        let byte_idx = j / u8::BITS as usize;
        let bit_idx = j % u8::BITS as usize;
        let byte = unsafe { self.row_slice_mut_unchecked(i).get_unchecked_mut(byte_idx) };
        if val {
            *byte |= 1 << bit_idx;
        } else {
            *byte &= !(1 << bit_idx);
        }
    }

    fn try_set(&mut self, i: usize, j: usize, val: bool) -> Result<(), MatAccessError> {
        if i >= self.nrows || j >= self.ncols {
            return Err(MatAccessError::OutOfBounds(i, j, self.nrows, self.ncols));
        }
        unsafe {
            self.set_unchecked(i, j, val);
        }
        Ok(())
    }
}

impl PartialEq for GF2Mat {
    fn eq(&self, other: &GF2Mat) -> bool {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            return false;
        }
        // TODO: optimize
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if self.try_get(i, j).unwrap() != other.try_get(i, j).unwrap() {
                    return false;
                }
            }
        }
        true
    }
}

/*
For these functions we use hint::assert_unchecked to remove
edge cases which result in branches in the generated code.

Having the arrays byte-sizes be multiples of 16 allows the compiler
to always vectorize the loop for example
*/

/// assumes the byte size of dst and src are multiples of ALIGN, non-zero, and equal.
/// dst and src can also not overlap
pub unsafe fn xor_aligned_slices<const ALIGN: usize>(dst: &mut [u8], src: &[u8]) {
    unsafe { std::hint::assert_unchecked(dst.len() != 0) };
    unsafe { std::hint::assert_unchecked(src.len() != 0) };
    unsafe { std::hint::assert_unchecked(dst.len() % ALIGN == 0) };
    unsafe { std::hint::assert_unchecked(src.len() % ALIGN == 0) };
    unsafe { std::hint::assert_unchecked(dst.len() == src.len()) };
    for (d, s) in std::iter::zip(dst, src) {
        *d ^= *s;
    }
}

/// Assumes the byte size of dst, src1 and src2 are multiples of ALIGN, non-zero, and equal.
/// dst can also not overlap with either src1 or src2
pub unsafe fn xor_aligned_slices_to<const ALIGN: usize>(dst: &mut [u8], src1: &[u8], src2: &[u8]) {
    unsafe { std::hint::assert_unchecked(src1.len() != 0) };
    unsafe { std::hint::assert_unchecked(src2.len() != 0) };
    unsafe { std::hint::assert_unchecked(dst.len() != 0) };
    unsafe { std::hint::assert_unchecked(src1.len() % ALIGN == 0) };
    unsafe { std::hint::assert_unchecked(src2.len() % ALIGN == 0) };
    unsafe { std::hint::assert_unchecked(dst.len() % ALIGN == 0) };
    unsafe { std::hint::assert_unchecked(src1.len() == src2.len()) };
    unsafe { std::hint::assert_unchecked(src2.len() == dst.len()) };
    for (d, (s1, s2)) in std::iter::zip(dst, std::iter::zip(src1, src2)) {
        *d = *s1 ^ *s2;
    }
}

impl<const ALIGN: usize> Debug for AlignedGF2Mat<ALIGN> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                write!(f, "{}", if self.try_get(i, j).unwrap() { 1 } else { 0 })?;
            }
            if i != self.nrows - 1 {
                write!(f, "\n")?
            };
        }
        write!(f, "]")
    }
}
