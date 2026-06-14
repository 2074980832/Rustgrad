//! Backend abstraction for tensor operations.
//!
//! This module defines a `Backend` trait that could be used to swap between
//! CPU and GPU execution. The current implementation is CPU-only and the
//! trait serves as architectural documentation for where a GPU backend
//! (e.g., wgpu, CUDA via a Rust binding) would slot in.
//!
//! # Adding a GPU backend (future work):
//!
//! 1. Implement the `Backend` trait for a `GpuBackend` struct.
//! 2. Parameterize `Tensor<B: Backend>` so operations dispatch to the
//!    selected backend.
//! 3. Add device memory allocation, kernel launches, and data transfer
//!    primitives inside `GpuBackend`.
//!
//! No external dependencies are introduced — this is pure Rust.
//!
//! Current status: **CPU-only stub** for design reference.

use crate::tensor::Tensor;
use crate::Result;

/// Numeric scalar type used by all backends.
pub type Scalar = f64;

/// Trait for tensor operation backends.
///
/// Each method mirrors a core operation on `Tensor`. A `CpuBackend`
/// performs the work in-process. A hypothetical `GpuBackend` would
/// upload data to device memory, launch kernels, and retrieve results.
pub trait Backend: std::fmt::Debug {
    /// Name of this backend (e.g. "cpu", "gpu").
    fn name(&self) -> &str;

    /// Element-wise addition with broadcasting.
    fn add(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;

    /// Element-wise subtraction with broadcasting.
    fn sub(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;

    /// Element-wise multiplication with broadcasting.
    fn mul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;

    /// Element-wise division with broadcasting.
    fn div(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;

    /// Matrix multiplication (left @ right).
    fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;

    /// Matrix transpose.
    fn transpose(&self, tensor: &Tensor) -> Result<Tensor>;

    /// Sum reduction (all elements → scalar-like tensor).
    fn sum(&self, tensor: &Tensor) -> Result<Tensor>;

    /// Mean reduction (all elements → scalar-like tensor).
    fn mean(&self, tensor: &Tensor) -> Result<Tensor>;

    /// Sum along an axis.
    fn sum_axis(&self, tensor: &Tensor, axis: usize) -> Result<Tensor>;

    /// Mean along an axis.
    fn mean_axis(&self, tensor: &Tensor, axis: usize) -> Result<Tensor>;

    /// Add a bias vector to each row of a matrix.
    fn row_add(&self, matrix: &Tensor, bias: &Tensor) -> Result<Tensor>;
}

/// The single CPU backend used throughout RustGrad.
///
/// All operations are performed in-process using `f64` arithmetic on the
/// host. This backend is a zero-cost indirection: each method delegates
/// directly to the corresponding `Tensor` method.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Creates a new CPU backend.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn add(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        left.add(right)
    }

    fn sub(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        left.sub(right)
    }

    fn mul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        left.mul(right)
    }

    fn div(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        left.div(right)
    }

    fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        left.matmul(right)
    }

    fn transpose(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.transpose()
    }

    fn sum(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.sum()
    }

    fn mean(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.mean()
    }

    fn sum_axis(&self, tensor: &Tensor, axis: usize) -> Result<Tensor> {
        tensor.sum_axis(axis)
    }

    fn mean_axis(&self, tensor: &Tensor, axis: usize) -> Result<Tensor> {
        tensor.mean_axis(axis)
    }

    fn row_add(&self, matrix: &Tensor, bias: &Tensor) -> Result<Tensor> {
        matrix.row_add(bias)
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, CpuBackend};
    use crate::tensor::Tensor;

    #[test]
    fn cpu_backend_exposes_correct_name() {
        assert_eq!(CpuBackend::new().name(), "cpu");
    }

    #[test]
    fn cpu_backend_add_matches_direct_tensor_add() {
        let backend = CpuBackend::new();
        let left = Tensor::vector(vec![1.0, 2.0, 3.0]).expect("valid");
        let right = Tensor::vector(vec![4.0, 5.0, 6.0]).expect("valid");

        let via_backend = backend.add(&left, &right).expect("backend add");
        let direct = left.add(&right).expect("direct add");

        assert_eq!(via_backend.data(), direct.data());
    }

    #[test]
    fn cpu_backend_matmul_matches_direct_tensor_matmul() {
        let backend = CpuBackend::new();
        let left = Tensor::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid");
        let right = Tensor::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0]).expect("valid");

        let via_backend = backend.matmul(&left, &right).expect("backend matmul");
        let direct = left.matmul(&right).expect("direct matmul");

        assert_eq!(via_backend.data(), direct.data());
    }

    #[test]
    fn backend_trait_is_object_safe() {
        // Verify that the trait can be used as a trait object.
        let backend: &dyn Backend = &CpuBackend::new();
        let a = Tensor::scalar(2.0).expect("valid");
        let b = Tensor::scalar(3.0).expect("valid");

        let c = backend.add(&a, &b).expect("trait object add");
        assert_eq!(c.data(), &[5.0]);
    }
}
