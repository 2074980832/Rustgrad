//! Reusable training loops and training metrics.

use crate::tensor::Tensor;
use crate::{Result, RustGradError};

/// Shared configuration for small training examples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingConfig {
    epochs: usize,
    learning_rate: f64,
    log_every: usize,
}

impl TrainingConfig {
    /// Creates a training configuration.
    pub fn new(epochs: usize, learning_rate: f64) -> Result<Self> {
        validate_positive("epochs", epochs)?;
        validate_positive_f64("learning_rate", learning_rate)?;

        Ok(Self {
            epochs,
            learning_rate,
            log_every: 1,
        })
    }

    /// Sets the logging interval in epochs.
    pub fn with_log_every(mut self, log_every: usize) -> Result<Self> {
        validate_positive("log_every", log_every)?;
        self.log_every = log_every;
        Ok(self)
    }

    /// Returns the number of training epochs.
    #[must_use]
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// Returns the optimizer learning rate used by the example.
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the epoch interval used for progress logs.
    #[must_use]
    pub fn log_every(&self) -> usize {
        self.log_every
    }

    /// Returns true when a given epoch should be logged.
    #[must_use]
    pub fn should_log(&self, epoch: usize) -> bool {
        epoch == 1 || epoch == self.epochs || epoch.is_multiple_of(self.log_every)
    }
}

/// Training metrics captured for one epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingRecord {
    epoch: usize,
    loss: f64,
    accuracy: Option<f64>,
}

impl TrainingRecord {
    /// Creates a training record.
    pub fn new(epoch: usize, loss: f64, accuracy: Option<f64>) -> Result<Self> {
        validate_positive("epoch", epoch)?;
        validate_metric("loss", loss)?;
        if let Some(value) = accuracy {
            validate_accuracy(value)?;
        }

        Ok(Self {
            epoch,
            loss,
            accuracy,
        })
    }

    /// Returns the epoch index.
    #[must_use]
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Returns the recorded loss.
    #[must_use]
    pub fn loss(&self) -> f64 {
        self.loss
    }

    /// Returns the optional recorded accuracy.
    #[must_use]
    pub fn accuracy(&self) -> Option<f64> {
        self.accuracy
    }
}

/// Append-only history of training records.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainingHistory {
    records: Vec<TrainingRecord>,
}

impl TrainingHistory {
    /// Creates an empty training history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a training history from records.
    #[must_use]
    pub fn from_records(records: Vec<TrainingRecord>) -> Self {
        Self { records }
    }

    /// Appends one record.
    pub fn push(&mut self, record: TrainingRecord) {
        self.records.push(record);
    }

    /// Returns all records.
    #[must_use]
    pub fn records(&self) -> &[TrainingRecord] {
        &self.records
    }

    /// Returns the number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true when no records are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns the last record.
    #[must_use]
    pub fn last(&self) -> Option<&TrainingRecord> {
        self.records.last()
    }

    /// Returns the first recorded loss.
    #[must_use]
    pub fn initial_loss(&self) -> Option<f64> {
        self.records.first().map(TrainingRecord::loss)
    }

    /// Returns the final recorded loss.
    #[must_use]
    pub fn final_loss(&self) -> Option<f64> {
        self.records.last().map(TrainingRecord::loss)
    }

    /// Returns the lowest recorded loss.
    #[must_use]
    pub fn best_loss(&self) -> Option<f64> {
        self.records
            .iter()
            .map(TrainingRecord::loss)
            .min_by(f64::total_cmp)
    }

    /// Returns the highest recorded accuracy.
    #[must_use]
    pub fn best_accuracy(&self) -> Option<f64> {
        self.records
            .iter()
            .filter_map(TrainingRecord::accuracy)
            .max_by(f64::total_cmp)
    }

    /// Returns true when the final loss is lower than the initial loss.
    #[must_use]
    pub fn loss_decreased(&self) -> bool {
        match (self.initial_loss(), self.final_loss()) {
            (Some(initial), Some(final_loss)) => final_loss < initial,
            _ => false,
        }
    }

    /// Returns all loss values in epoch order.
    #[must_use]
    pub fn losses(&self) -> Vec<f64> {
        self.records.iter().map(TrainingRecord::loss).collect()
    }
}

/// Computes mean squared error between two tensors.
pub fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    ensure_same_shape("mean squared error", predictions, targets)?;
    let total: f64 = predictions
        .data()
        .iter()
        .zip(targets.data())
        .map(|(&prediction, &target)| {
            let diff = prediction - target;
            diff * diff
        })
        .sum();

    Ok(total / predictions.len() as f64)
}

/// Computes mean absolute error between two tensors.
pub fn mean_absolute_error(predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    ensure_same_shape("mean absolute error", predictions, targets)?;
    let total: f64 = predictions
        .data()
        .iter()
        .zip(targets.data())
        .map(|(&prediction, &target)| (prediction - target).abs())
        .sum();

    Ok(total / predictions.len() as f64)
}

/// Computes binary classification accuracy after thresholding predictions.
pub fn binary_accuracy(predictions: &Tensor, targets: &Tensor, threshold: f64) -> Result<f64> {
    ensure_same_shape("binary accuracy", predictions, targets)?;
    validate_finite("threshold", threshold)?;

    let correct = predictions
        .data()
        .iter()
        .zip(targets.data())
        .filter(|&(&prediction, &target)| {
            let predicted_class = if prediction >= threshold { 1.0 } else { 0.0 };
            (predicted_class - target).abs() < f64::EPSILON
        })
        .count();

    Ok(correct as f64 / predictions.len() as f64)
}

/// Computes categorical accuracy for batched class scores and one-hot targets.
pub fn categorical_accuracy(predictions: &Tensor, targets: &Tensor) -> Result<f64> {
    ensure_same_shape("categorical accuracy", predictions, targets)?;
    validate_rank_two("predictions", predictions)?;
    validate_rank_two("targets", targets)?;

    let rows = predictions.rows().expect("rank 2 tensors always have rows");
    let cols = predictions
        .cols()
        .expect("rank 2 tensors always have columns");
    let mut correct = 0;

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let predicted = argmax(&predictions.data()[start..end]);
        let target = argmax(&targets.data()[start..end]);
        if predicted == target {
            correct += 1;
        }
    }

    Ok(correct as f64 / rows as f64)
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("dataset tensors never have zero columns")
}

fn ensure_same_shape(op: &'static str, predictions: &Tensor, targets: &Tensor) -> Result<()> {
    if predictions.dims() == targets.dims() {
        Ok(())
    } else {
        Err(RustGradError::ShapeMismatch {
            op,
            left: predictions.shape().to_vec(),
            right: targets.shape().to_vec(),
        })
    }
}

fn validate_rank_two(name: &'static str, tensor: &Tensor) -> Result<()> {
    if tensor.rank() == 2 {
        Ok(())
    } else {
        Err(RustGradError::InvalidArgument {
            name,
            reason: format!("expected rank 2 tensor, got rank {}", tensor.rank()),
        })
    }
}

fn validate_positive(name: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        Err(RustGradError::InvalidArgument {
            name,
            reason: "value must be greater than zero".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_positive_f64(name: &'static str, value: f64) -> Result<()> {
    if value <= 0.0 || !value.is_finite() {
        Err(RustGradError::InvalidArgument {
            name,
            reason: "value must be finite and greater than zero".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_metric(name: &'static str, value: f64) -> Result<()> {
    if value < 0.0 || !value.is_finite() {
        Err(RustGradError::InvalidArgument {
            name,
            reason: "metric must be finite and non-negative".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_accuracy(value: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&value) || !value.is_finite() {
        Err(RustGradError::InvalidArgument {
            name: "accuracy",
            reason: "accuracy must be finite and in [0, 1]".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_finite(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(RustGradError::InvalidArgument {
            name,
            reason: "value must be finite".to_string(),
        })
    }
}
