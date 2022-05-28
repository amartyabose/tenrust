pub type Result<T> = std::result::Result<T, TenRustError>;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TenRustError {
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(&'static str),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(&'static str),
    #[error("Shape error.")]
    ShapeError(#[from] ndarray::ShapeError)
}

