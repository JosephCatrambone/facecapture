
use crate::decision_tree::DecisionTree;
use crate::haar_like_feature::{HaarFeature, make_3x3_haar, make_2x2_haar};
use crate::integral_image::IntegralImage;
use crate::MQImage;

use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::borrow::Borrow;
use std::cmp::{max, min};
use std::error::Error;
use std::fmt;
use std::fs;
use std::str::FromStr;


pub struct DetectedFace {
	pub x: usize,
	pub y: usize,
	pub width: usize,
	pub height: usize,
	pub confidence: f32
}

const DETECTOR_SIZE:u32 = 32u32;

#[derive(Debug, Serialize, Deserialize)]
pub struct FaceDetector {
	classifiers: Vec<DecisionTree>,
	weights: Vec<f32>, // Used to determine how much to emphasize the opinion of one of the classifiers.
	features: Vec<HaarFeature>,
}

impl FaceDetector {
	pub fn new() -> Self {
		FaceDetector {
			classifiers: vec![],
			weights: vec![],
			features: vec![],
		}
	}
	
	pub fn detect_face(&self, image_width: usize, image_height: usize, image_data: &Vec<u8>) -> Vec::<DetectedFace> {
		let mut results = vec![];
		
		//let img = IntegralImage::new_from_image_data(image_width, image_height, image_data);
		
		for scale in 6..11 { // 6 - 11 seems big enough
			let img = IntegralImage::new_from_image_data_subsampled(image_width, image_height, image_data, image_width/scale, image_height/scale);
			//fs::write(format!("sample_image_scale_{}_{}x{}.dat", scale, img.width, img.height), img.data);
			for y in (0..img.height - DETECTOR_SIZE as usize).step_by(DETECTOR_SIZE as usize/3) {
				for x in (0..img.width - DETECTOR_SIZE as usize).step_by(DETECTOR_SIZE as usize/3) {
					let roi = img.new_from_region(x, y, x + DETECTOR_SIZE as usize, y + DETECTOR_SIZE as usize);
					//fs::write(format!("roi_scale_{}_x{}_y{}_{}x{}.dat", scale, x, y, img.width, img.height), roi.data);
					//if !self.predict(&roi) { continue; }
					//if self.predict(&roi) {
					//if self.predict_early_out(&roi, Some(2), Some(4)) {
					let p = self.probability(&roi);
					if p > 0.6 {
						let f = DetectedFace {
							x: scale * x,
							y: scale * y,
							width: scale * roi.width,
							height: scale * roi.height,
							//confidence: 0.0f32.max( p)
							confidence: p
						};
						results.push(f);
					}
					//}
				}
			}
		}
		//panic!("At the disco");
		
		results
	}
	
	pub fn probability(&self, example:&IntegralImage) -> f32 {
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
	
	pub fn predict_early_out(&self, example:&IntegralImage, min_positive:Option<u32>, min_negative:Option<u32>) -> bool {
		let min_positive = min_positive.unwrap_or(self.classifiers.len() as u32/2);
		let min_negative = min_negative.unwrap_or(self.classifiers.len() as u32/2);
		let mut positive_count = 0;
		let mut negative_count = 0;
		let x = self.vectorize_example(example);
		for (c, w) in self.classifiers.iter().zip(&self.weights) {
			if c.predict(&x) {
				if *w > 0.0 {
					positive_count += 1;
				} else {
					negative_count += 1;
				}
			} else {
				if *w > 0.0 {
					negative_count += 1;
				} else {
					positive_count += 1;
				}
			}
			
			if positive_count > min_positive {
				return true;
			} else if negative_count > min_negative {
				return false;
			}
		}
		return positive_count > negative_count;
	}
	
	pub fn predict(&self, example:&IntegralImage) -> bool {
		self.probability(example) > 0.5
	}
	
	fn vectorize_example(&self, example:&IntegralImage) -> Vec<f32> {
		let mut x = vec![0f32; self.features.len()];
		for (i, feat) in self.features.iter().enumerate() {
			x[i] = feat.probability(example);
		}
		x
	}
	
	pub fn train(&mut self, examples:Vec<&IntegralImage>, labels:Vec<bool>) {
		let mut examples = examples;
		let mut labels = labels; // We do this seemingly arbitrary reassign to allow us to mutate these refs.
		
		// Constants.
		let max_depth = 3;
		let classifiers_to_keep:usize = 50; // Viola + Jones used 38, but they were smaller.
		
		// Starting features -- these will be culled at the end.  There might be a _LOT_.
		self.features = make_3x3_haar(DETECTOR_SIZE as usize, DETECTOR_SIZE as usize);
		self.features.extend(make_2x2_haar(DETECTOR_SIZE as usize, DETECTOR_SIZE as usize));
		self.classifiers.clear();
		self.weights.clear();
		
		// Convert all examples once at the beginning rather than once per loop.
		let mut converted_examples = vec![];
		for (example, label) in examples.iter().zip(&labels) {
			converted_examples.push(self.vectorize_example(example));
		}
		
		let mut rng = thread_rng();
		
		for _ in 0..classifiers_to_keep {
			let mut d = DecisionTree::new();
			
			// Randomly sample some data from examples for training.
			let mut x = vec![];
			let mut y = vec![];
			for (example, label) in converted_examples.iter().zip(&labels) {
				if rng.gen_bool(0.8) {
					x.push(example);
					y.push(*label);
				}
			}
			
			// And fit the tree.
			d.train(&x, &y, max_depth);
			
			// Now use the classifier to classify all examples.
			let mut errors = 0;
			let mut weight_update_directions = vec![]; // Track the items we got wrong so we can reweight them in the next step.
			for (x, y) in converted_examples.iter().zip(&labels) {
				if d.predict(x) != *y {
					errors += 1;
					weight_update_directions.push(1);
				} else {
					weight_update_directions.push(-1);
				}
			}
			
			// Store D.
			self.classifiers.push(d);
			
			// Amount of say = 0.5 * log ((1-error)/error)
			// If error is HUUUGE this number will be a large NEGATIVE number.
			let new_classifier_weight = 0.5f32 * ((1.0f32 - errors as f32)/errors as f32);
			self.weights.push(new_classifier_weight);
			
			// Recalculate the sample select probabilities.
			// If incorrectly classified:
			// New sample weight = old_sample_weight * e^new_classifier_weight.
			// If correctly classifier, use -new_classifier_weight.
			let starting_sample_weight = 1.0f32 / examples.len() as f32;
			let mut sample_weights = vec![];
			let mut total_sample_weights = 0f32;
			for wud in &weight_update_directions {
				let new_sample_weight = starting_sample_weight * (*wud as f32).exp();
				sample_weights.push(new_sample_weight);
				total_sample_weights += new_sample_weight;  // Accumulate for norm.
			}
			// Normalize:
			sample_weights.iter_mut().map(|w|{ *w /= total_sample_weights });
			
			// Using the new sample weights, construct a new training set.
			let mut new_examples = vec![];
			let mut new_labels = vec![];
			for _ in 0..examples.len() {
				// TODO: To avoid numeric stability issues, should we be rescaling by weight?
				let mut tunneling_energy = rng.gen::<f32>();
				for (i, (x, y)) in examples.iter().zip(&labels).enumerate() {
					if tunneling_energy < sample_weights[i] {
						new_examples.push(*x);
						new_labels.push(*y);
						break;
					} else {
						tunneling_energy -= sample_weights[i];
					}
				}
			}
			
			examples = new_examples;
			labels = new_labels;
		}
		
		// TODO: Compact features.  Pick the elements that are not used by any of the trees and delete them to save space.
	}
}

#[cfg(test)]
mod tests {
	// Note this useful idiom: importing names from outer (for mod tests) scope.
	use super::*;
	use std::fs::File;
	use std::io::prelude::*;
	
	#[test]
	#[ignore] // Run with cargo test -- --ignored
	fn test_train_face_classifier() {
		let num_positive_examples = 5000;
		let num_negative_examples = 30000;
		let mut rng = thread_rng();
		
		// Allocate buffers.
		let mut examples:Vec<IntegralImage> = vec![];
		let mut labels:Vec<bool> = vec![];
		let mut train_x:Vec<&IntegralImage> = vec![];
		let mut train_y:Vec<bool> = vec![];
		let mut test_x:Vec<&IntegralImage> = vec![];
		let mut test_y:Vec<bool> = vec![];
		
		// Load data.
		let mut positive = File::open("faces_128x128.dat").unwrap();
		let mut negative = File::open("notfaces_128x128.dat").unwrap();
		let mut img_buffer = vec![0u8; 128*128];
		
		println!("Reading data.");
		while let Ok(bytes_read) = positive.read(&mut img_buffer) {
			let resized = IntegralImage::new_from_image_data_subsampled(128, 128, &img_buffer, DETECTOR_SIZE as usize, DETECTOR_SIZE as usize);
			examples.push(resized);
			labels.push(true);
			if examples.len() > num_positive_examples {
				break;
			}
		}
		while let Ok(bytes_read) = negative.read(&mut img_buffer) {
			let resized = IntegralImage::new_from_image_data_subsampled(128, 128, &img_buffer, DETECTOR_SIZE as usize, DETECTOR_SIZE as usize);
			examples.push(resized);
			labels.push(false);
			if examples.len() > num_negative_examples+num_positive_examples {
				break;
			}
		}
		
		// Split training and test.
		println!("Splitting data.");
		for i in 0..examples.len() {
			if rng.gen_bool(0.5) {
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
		
		println!("Testing classifiers.");
		let mut true_positive = 0;
		let mut false_positive = 0;
		let mut true_negative = 0;
		let mut false_negative = 0;
		
		for (x, y) in test_x.iter().zip(&test_y) {
			let prediction = face_detector.predict(x);
			let proba = face_detector.probability(x);
			println!("Truth: {}  Prediction: {}  Probability: {}", *y, prediction, proba);
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
		
		println!("TP {}  FP {}  TN {}  FN {}", true_positive, false_positive, true_negative, false_negative);
		let mut fout = File::create("face_detector.json").unwrap();
		fout.write(serde_json::to_string::<FaceDetector>(&face_detector).unwrap().as_bytes());
	}
}