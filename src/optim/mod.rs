//! Optimizers for updating trainable parameters.

use crate::tensor::Tensor;
use crate::{Result, RustGradError};

/// Immutable gradients aligned with a model's trainable parameters.
///
/// RustGrad keeps gradients outside `Tensor` for the first teaching version so
/// optimizers can be demonstrated independently from the computation graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GradientSet {
    gradients: Vec<Tensor>,
}

impl GradientSet {
    /// Creates an empty gradient collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a gradient collection from tensors.
    #[must_use]
    pub fn from_tensors(gradients: Vec<Tensor>) -> Self {
        Self { gradients }
    }

    /// Returns the number of stored gradients.
    #[must_use]
    pub fn len(&self) -> usize {
        self.gradients.len()
    }

    /// Returns true when no gradients are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gradients.is_empty()
    }

    /// Returns a gradient by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Tensor> {
        self.gradients.get(index)
    }

    /// Iterates over gradients in parameter order.
    pub fn iter(&self) -> impl Iterator<Item = &Tensor> {
        self.gradients.iter()
    }

    /// Removes all stored gradients.
    pub fn clear(&mut self) {
        self.gradients.clear();
    }
}

impl From<Vec<Tensor>> for GradientSet {
    fn from(gradients: Vec<Tensor>) -> Self {
        Self::from_tensors(gradients)
    }
}

/// Common interface for parameter optimizers.
pub trait Optimizer {
    /// Updates parameters in place using gradients with matching order.
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &GradientSet) -> Result<()>;

    /// Returns the optimizer learning rate.
    fn learning_rate(&self) -> f64;

    /// Updates the optimizer learning rate.
    fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()>;

    /// Returns a stable optimizer name for reports and debugging.
    fn name(&self) -> &str;
}

/// Stochastic gradient descent optimizer.
///
/// The update rule is `parameter = parameter - learning_rate * gradient`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    /// Creates an SGD optimizer.
    pub fn new(learning_rate: f64) -> Result<Self> {
        validate_learning_rate(learning_rate)?;
        Ok(Self { learning_rate })
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &GradientSet) -> Result<()> {
        validate_parameter_gradient_count(parameters.len(), gradients.len())?;

        for (parameter, gradient) in parameters.iter_mut().zip(gradients.iter()) {
            validate_gradient_shape(parameter, gradient)?;

            for (value, &grad) in parameter.data_mut().iter_mut().zip(gradient.data()) {
                *value -= self.learning_rate * grad;
            }
        }

        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        validate_learning_rate(learning_rate)?;
        self.learning_rate = learning_rate;
        Ok(())
    }

    fn name(&self) -> &str {
        "sgd"
    }
}

fn validate_learning_rate(learning_rate: f64) -> Result<()> {
    if learning_rate <= 0.0 || !learning_rate.is_finite() {
        return Err(RustGradError::InvalidArgument {
            name: "learning_rate",
            reason: "learning rate must be finite and greater than zero".to_string(),
        });
    }

    Ok(())
}

fn validate_parameter_gradient_count(parameter_count: usize, gradient_count: usize) -> Result<()> {
    if parameter_count == gradient_count {
        Ok(())
    } else {
        Err(RustGradError::InvalidArgument {
            name: "gradients",
            reason: format!("expected {parameter_count} gradients, got {gradient_count}"),
        })
    }
}

fn validate_gradient_shape(parameter: &Tensor, gradient: &Tensor) -> Result<()> {
    if parameter.dims() == gradient.dims() {
        Ok(())
    } else {
        Err(RustGradError::ShapeMismatch {
            op: "optimizer gradient",
            left: parameter.shape().to_vec(),
            right: gradient.shape().to_vec(),
        })
    }
}
