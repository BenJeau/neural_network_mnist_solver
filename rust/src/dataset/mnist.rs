use flate2::read::GzDecoder;
use std::io::Read;

use crate::error::Result;

pub const BASE_URL: &str = "https://ossci-datasets.s3.amazonaws.com/mnist/";
// pub const BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";

const TRAIN_DATA_FILENAME: &str = "train-images-idx3-ubyte.gz";
const TEST_DATA_FILENAME: &str = "t10k-images-idx3-ubyte.gz";
const TRAIN_LABEL_FILENAME: &str = "train-labels-idx1-ubyte.gz";
const TEST_LABEL_FILENAME: &str = "t10k-labels-idx1-ubyte.gz";

const IMAGES_MAGIC_NUMBER: u32 = 2051;
const LABELS_MAGIC_NUMBER: u32 = 2049;
const NUM_TRAIN_IMAGES: u32 = 60_000;
const NUM_TEST_IMAGES: u32 = 10_000;
const IMAGE_ROWS: u32 = 28;
const IMAGE_COLUMNS: u32 = 28;

pub struct MnistDataset {
    pub train: MnistSubDataset,
    pub test: MnistSubDataset,
}

pub struct MnistSubDataset {
    pub images: MnistImages,
    pub labels: MnistLabels,
}

impl MnistSubDataset {
    pub fn from_url(images_url: &str, labels_url: &str) -> Result<MnistSubDataset> {
        let images = MnistImages::from_bytes(&get_decoded_bytes(images_url)?)?;
        let labels = MnistLabels::from_bytes(&get_decoded_bytes(labels_url)?)?;

        if images.images.len() != labels.0.len() {
            println!(
                "Expected {} images, got {}",
                images.images.len(),
                labels.0.len()
            );
        }

        Ok(MnistSubDataset { images, labels })
    }

    fn print_image(&self, index: usize) {
        println!(
            "Sample image label: {} \nSample image:",
            self.labels.0[index]
        );

        self.images.print_image(index);
    }
}

fn get_decoded_bytes(url: &str) -> Result<Vec<u8>> {
    let resp = ureq::get(url).call()?;

    let mut data: Vec<u8> = Vec::new();
    resp.into_reader().take(10_000_000).read_to_end(&mut data)?;

    let mut buffer = Vec::new();
    let mut decoder = GzDecoder::new(&*data);
    decoder.read_to_end(&mut buffer)?;

    Ok(buffer)
}

impl MnistDataset {
    pub fn from_url(base_url: &str) -> Result<MnistDataset> {
        println!("Downloading MNIST dataset from {base_url}");

        let train = MnistSubDataset::from_url(
            &format!("{base_url}{TRAIN_DATA_FILENAME}"),
            &format!("{base_url}{TRAIN_LABEL_FILENAME}"),
        )?;

        if train.images.images.len() != NUM_TRAIN_IMAGES as usize {
            println!(
                "Expected {NUM_TRAIN_IMAGES} train images, got {}",
                train.images.images.len()
            );
        }

        let test = MnistSubDataset::from_url(
            &format!("{base_url}{TEST_DATA_FILENAME}"),
            &format!("{base_url}{TEST_LABEL_FILENAME}"),
        )?;

        if test.images.images.len() != NUM_TEST_IMAGES as usize {
            println!(
                "Expected {NUM_TEST_IMAGES} test images, got {}",
                test.images.images.len()
            );
        }

        Ok(MnistDataset { train, test })
    }
}

pub struct MnistImages {
    pub num_cols: u32,
    pub images: Vec<Vec<u8>>,
}

impl MnistImages {
    fn from_bytes(bytes: &[u8]) -> Result<MnistImages> {
        let magic_number = u32::from_be_bytes(bytes[0..4].try_into()?);
        let num_images = u32::from_be_bytes(bytes[4..8].try_into()?);
        let num_rows = u32::from_be_bytes(bytes[8..12].try_into()?);
        let num_cols = u32::from_be_bytes(bytes[12..16].try_into()?);
        let num_pixels = (num_rows * num_cols) as usize;

        let images = bytes[16..]
            .chunks_exact(num_pixels)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        if magic_number != IMAGES_MAGIC_NUMBER {
            println!("Expected magic number {IMAGES_MAGIC_NUMBER}, got {magic_number}");
        }

        if num_images != images.len() as u32 {
            println!("Expected {num_images} images, got {}", images.len());
        }

        if num_cols != IMAGE_COLUMNS || num_rows != IMAGE_ROWS {
            println!(
                "Expected images of size {IMAGE_ROWS}x{IMAGE_COLUMNS}, got {num_rows}x{num_cols}"
            );
        }

        Ok(MnistImages { num_cols, images })
    }

    fn print_image(&self, index: usize) {
        for row in self.images[index].chunks(self.num_cols as usize) {
            for &pixel in row {
                if pixel == 0 {
                    print!("__");
                } else {
                    print!("##");
                }
            }
            println!();
        }
    }
}

pub struct MnistLabels(pub Vec<u8>);

impl MnistLabels {
    fn from_bytes<'a>(bytes: &'a [u8]) -> Result<MnistLabels> {
        let magic_number = u32::from_be_bytes(bytes[0..4].try_into()?);
        let num_labels = u32::from_be_bytes(bytes[4..8].try_into()?);
        let labels = bytes[8..].to_vec();

        if magic_number != LABELS_MAGIC_NUMBER {
            println!("Expected magic number {LABELS_MAGIC_NUMBER}, got {magic_number}");
        }

        if num_labels != labels.len() as u32 {
            println!("Expected {num_labels} labels, got {}", labels.len());
        }

        Ok(MnistLabels(labels))
    }
}
