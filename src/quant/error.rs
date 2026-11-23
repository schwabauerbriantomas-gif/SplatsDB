//! Quantization error types.

#[derive(Debug)]
pub enum QuantError {
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
    OddDimension {
        got: usize,
    },
    ZeroDimension,
    InvalidBitWidth {
        got: u8,
    },
    ZeroProjectionCount,
    RotationFailed(String),
    MalformedCode(String),
    /// Input vector holds NaN or +/-Inf at `index`; it can never become a valid code.
    NonFiniteValue {
        index: usize,
    },
    /// A finite input whose rotated radius at `pair` is not finite (`a*a + b*b`
    /// overflowed f32). It can never become a valid code either (F04e).
    NonFiniteRadius {
        pair: usize,
    },
    /// The rotated or the projected query of a finite query is not finite at `index`
    /// (the rotation or the projection overflowed f32): it is rejected before any code
    /// is scored.
    NonFiniteQueryContext {
        index: usize,
    },
    /// The score of a valid code against a valid query came out NaN/Inf (f32 overflow):
    /// an `Err`, never ranked or returned.
    NonFiniteScore,
}

impl std::fmt::Display for QuantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::OddDimension { got } => write!(f, "dimension must be even, got {}", got),
            Self::ZeroDimension => write!(f, "dimension must be non-zero"),
            Self::InvalidBitWidth { got } => write!(f, "bits must be 1-16, got {}", got),
            Self::ZeroProjectionCount => write!(f, "projection count must be non-zero"),
            Self::RotationFailed(r) => write!(f, "rotation failed: {}", r),
            Self::MalformedCode(r) => write!(f, "malformed code: {}", r),
            Self::NonFiniteValue { index } => {
                write!(f, "non-finite value (NaN/Inf) at index {index}")
            }
            Self::NonFiniteRadius { pair } => {
                write!(f, "non-finite radius at pair {pair} (f32 overflow)")
            }
            Self::NonFiniteQueryContext { index } => {
                write!(f, "non-finite query context (NaN/Inf) at index {index}")
            }
            Self::NonFiniteScore => write!(f, "non-finite score (NaN/Inf, f32 overflow)"),
        }
    }
}

impl std::error::Error for QuantError {}

pub type Result<T> = std::result::Result<T, QuantError>;
