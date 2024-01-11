#[derive(Debug)]
pub enum Error {
    UreqError(ureq::Error),
    IoError(std::io::Error),
    TryFromSliceError(std::array::TryFromSliceError),
}

impl From<ureq::Error> for Error {
    fn from(error: ureq::Error) -> Self {
        Error::UreqError(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::IoError(error)
    }
}

impl From<std::array::TryFromSliceError> for Error {
    fn from(error: std::array::TryFromSliceError) -> Self {
        Error::TryFromSliceError(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
