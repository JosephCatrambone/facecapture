
use crate::decision_tree::DecisionTree;

use image::{GenericImage, GenericImageView, GrayImage};
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::fmt;
use std::fs;
use std::str::FromStr;

// 32x32 with 10k positive, 90k (noisy) negatives, 3 deep and 50% sample: TP 1773  FP 37438  TN 7298  FN 3099
// 64x64 with 1k positive, (9k negatives), max_depth of 3 + 100 classifiers and 50% per-tree: TP 56  FP 553  TN 317  FN 48
// 64x64 with 1k pos, 9k negative, max depth 2 + 100 classifiers and 50% per tree: TP 66  FP 575  TN 361  FN 37
// 32x32 with 1k pos, 9k negative, max depth 2 + 100 classifiers and 50% per tree: TP 53  FP 548  TN 331  FN 46
// 64X64 with 1k pos, 9k negative, max depth 5 + 50 classifiers and 80% per tree: TP 46  FP 485  TN 447  FN 47
// 64x64 with 9k pos, 9k negative, max depth 5 + 50 classifiers and 50% per tree:
// max depth 2 + 200 classifiers + 20% per tree: 0.50586087
// Swapping from early-out trunk forest to random forest...
// 64x64 with 10k pos, 10k negative, max depth 3, 50 classifiers and 50% per tree. Really good spread.  0.8333269 with TP scaling nicely.

pub struct DetectedFace {
	pub x: usize,
	pub y: usize,
	pub width: usize,
	pub height: usize,
	pub confidence: f32
}

const DETECTOR_SIZE:u32 = 64u32;

#[derive(Debug, Serialize, Deserialize)]
pub struct FaceDetector {
	classifiers: Vec<DecisionTree>,
	weights: Vec<f32>, // Used to determine how much to emphasize the opinion of one of the classifiers.
}

impl FaceDetector {
	pub fn new() -> Self {
		FaceDetector {
			classifiers: vec![],
			weights: vec![],
		}
	}

	fn detect_faces(&self, image_width: usize, image_height: usize, image_data: &Vec<u8>) -> (usize, Vec::<DetectedFace>) {
		let mut results = vec![];

		let base_img = GrayImage::from_raw(image_width as u32, image_height as u32, image_data.clone()).expect("Failed to convert byte array to u8 grey image.");
		let mut max_response_confidence = 0.0;
		let mut max_response_index = 0;

		for scale in 3..7 { // 6 - 11 seems big enough
			assert!(image_width/scale > DETECTOR_SIZE as usize);
			assert!(image_height/scale > DETECTOR_SIZE as usize);
			let img = image::imageops::resize(&base_img, (image_width / scale) as u32, (image_height / scale) as u32, image::imageops::FilterType::Nearest);
			for y in (0..img.height() - DETECTOR_SIZE).step_by(DETECTOR_SIZE as usize/3) {
				for x in (0..img.width() - DETECTOR_SIZE).step_by(DETECTOR_SIZE as usize/3) {
					let roi = img.view(x, y, DETECTOR_SIZE, DETECTOR_SIZE).to_image();
					let p = self.probability(&roi);
					if p > 0.2f32 {
						let f = DetectedFace {
							x: scale * (x as usize),
							y: scale * (y as usize),
							width: scale * (DETECTOR_SIZE as usize),
							height: scale * (DETECTOR_SIZE as usize),
							//confidence: 0.0f32.max( p)
							confidence: p
						};
						if p > max_response_confidence {
							max_response_index = results.len();
							max_response_confidence = p;
						}
						results.push(f);
					}
				}
			}
		}
		//panic!("At the disco");

		(max_response_index, results)
	}

	pub fn detect_all_faces(&self, image_width: usize, image_height: usize, image_data: &Vec<u8>) -> Vec::<DetectedFace> {
		let (_, faces) = self.detect_faces(image_width, image_height, image_data);
		faces
	}

	pub fn detect_face(&self, image_width: usize, image_height: usize, image_data: &Vec<u8>) -> DetectedFace {
		let (idx, mut faces) = self.detect_faces(image_width, image_height, image_data);
		faces.remove(idx)
	}
	
	pub fn probability<T : GenericImageView<Pixel=image::Luma<u8>>>(&self, example:&T) -> f32 {
		let x = self.vectorize_example(example);
		let mut total = 0f32;
		let mut possible = 0f32;
		for (c, w) in self.classifiers.iter().zip(&self.weights) {
			if c.predict(&x) {
				total += *w;
			} else {
				total -= *w;
			}
			possible += (*w).abs();
		}
		((total / possible)+1f32) / 2f32
	}

	pub fn predict<T : GenericImageView<Pixel=image::Luma<u8>>>(&self, example:&T) -> bool {
		self.probability(example) > 0.5
	}
	
	fn vectorize_example<T : GenericImageView<Pixel=image::Luma<u8>>>(&self, example:&T) -> Vec<f32> {
		example.pixels().into_iter().map(|(x, y, px)| { (px.0[0] as f32) / 255.0f32 }).collect()
	}
	
	pub fn train<T : GenericImageView<Pixel=image::Luma<u8>>>(&mut self, examples:Vec<&T>, labels:Vec<bool>) {
		let mut rng = thread_rng();
		let mut examples = examples;
		let mut labels = labels; // We do this seemingly arbitrary reassign to allow us to mutate these refs.
		
		// Constants.
		let sample_rate_per_tree = 0.5f64;
		let max_depth = 3;
		let classifiers_to_keep:usize = 50;

		self.classifiers.clear();
		self.weights.clear();
		
		// Convert all examples once at the beginning rather than once per loop.
		let mut converted_examples = vec![];
		for example in examples.iter() {
			converted_examples.push(self.vectorize_example(*example));
		}
		
		// Make and train our new classifiers.
		for idx in 0..classifiers_to_keep {
			println!("Training classifier {}", idx);
			let mut d = DecisionTree::new();
			
			// Randomly sample some data from examples for training.
			let mut x = vec![];
			let mut y = vec![];
			for (example, label) in converted_examples.iter().zip(&labels) {
				if rng.gen_bool(sample_rate_per_tree) {
					x.push(example);
					y.push(*label);
				}
			}
			
			// And fit the tree.
			d.train(&x, &y, max_depth);
			
			// Store D.
			self.classifiers.push(d);
			self.weights.push(1.0f32);
		}
	}
}

#[cfg(test)]
mod tests {
	// Note this useful idiom: importing names from outer (for mod tests) scope.
	use super::*;
	use std::fs::File;
	use std::io::prelude::*;
	use std::io::BufWriter;
	
	#[test]
	#[ignore] // Run with cargo test -- --ignored
	fn test_train_face_classifier() {
		println!("Starting up.");
		let num_positive_examples = 10000;
		let num_negative_examples = 10000; // Only up to 10k are validated. 2k are thoroughly validated.
		let mut rng = thread_rng();
		
		// Allocate buffers.
		println!("Allocating buffers.");
		let mut examples:Vec<GrayImage> = Vec::with_capacity(num_negative_examples+num_negative_examples);
		let mut labels:Vec<bool> = vec![];
		let mut train_x:Vec<&GrayImage> = vec![];
		let mut train_y:Vec<bool> = vec![];
		let mut test_x:Vec<&GrayImage> = vec![];
		let mut test_y:Vec<bool> = vec![];
		
		// Load data.
		let mut positive = File::open("faces_128x128.dat").unwrap();
		let mut negative = File::open("notfaces_128x128.dat").unwrap();
		let mut img_buffer = vec![0u8; 128*128];
		
		println!("Reading data.");
		while let Ok(bytes_read) = positive.read(&mut img_buffer) {
			let img = image::GrayImage::from_raw(128, 128, img_buffer.clone()).expect("Failed to load gray image from u8 data.");
			let resized = image::imageops::resize(&img, DETECTOR_SIZE, DETECTOR_SIZE, image::imageops::Nearest);
			examples.push(resized);
			labels.push(true);
			if examples.len() > num_positive_examples {
				break;
			}
		}
		while let Ok(bytes_read) = negative.read(&mut img_buffer) {
			let img = image::GrayImage::from_raw(128, 128, img_buffer.clone()).expect("Failed to load gray image from u8 data.");
			let resized = image::imageops::resize(&img, DETECTOR_SIZE, DETECTOR_SIZE, image::imageops::Nearest);
			examples.push(resized);
			labels.push(false);
			if examples.len() > num_negative_examples+num_positive_examples {
				break;
			}
		}
		
		// Split training and test.
		println!("Splitting data.");
		for i in 0..examples.len() {
			if rng.gen_bool(0.85) {
				train_x.push(&examples[i]);
				train_y.push(labels[i]);
			} else {
				test_x.push(&examples[i]);
				test_y.push(labels[i]);
			}
		}
		
		println!("Training...");
		let mut face_detector = FaceDetector::new();
		face_detector.train(train_x, train_y);
		
		println!("Saving...");
		let mut fout = File::create("face_detector.json").unwrap();
		fout.write(serde_json::to_string::<FaceDetector>(&face_detector).unwrap().as_bytes());
		
		println!("Evaluating...");
		let mut probabilities = vec![];
		for x in test_x {
			//let prediction = face_detector.predict(x);
			let proba = face_detector.probability(x);
			probabilities.push(proba);
		}
		
		println!("Testing classifiers.");
		let mut auc = 0.0f32;
		let mut result_csv = File::create("result.csv").unwrap();
		write!(result_csv, "threshold,TPR_tp_over_tp_plus_fn,FPR_fp_over_fp_plus_tn\n");
		for threshold in (0..100) {
			let mut true_positive = 0;
			let mut false_positive = 0;
			let mut true_negative = 0;
			let mut false_negative = 0;
			
			// TP rate on Y.  FP rate on X.
			for (p, y) in probabilities.iter().zip(&test_y) {
				let prediction = *p > (threshold as f32/100f32);
				if *y {
					if prediction {
						true_positive += 1;
					} else {
						false_negative += 1;
					}
				} else {
					if prediction {
						false_positive += 1;
					} else {
						true_negative += 1;
					}
				}
			}

			let tpr = true_positive as f32 / (true_positive + false_negative) as f32;
			let fpr = false_positive as f32 / (false_positive + true_negative) as f32;
			auc += tpr;
			write!(result_csv, "{},{},{}\n", threshold, tpr, fpr);
			println!("{}\tTP:{} FP:{}  TN:{} FN:{}", threshold, true_positive, false_positive, true_negative, false_negative);
		}
		println!("{}", (auc/100f32));
	}
}