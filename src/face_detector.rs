
use crate::decision_tree::DecisionTree;
use crate::haar_like_feature::{HaarFeature, make_3x3_haar, make_2x2_haar};
use crate::integral_image::IntegralImage;
use crate::MQImage;

use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::borrow::Borrow;
use std::cmp::{max, min, Ordering};
use std::error::Error;
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
		
		for scale in 5..10 { // 6 - 11 seems big enough
			assert!(image_width/scale > DETECTOR_SIZE as usize);
			assert!(image_height/scale > DETECTOR_SIZE as usize);
			let img = IntegralImage::new_from_image_data_subsampled(image_width, image_height, image_data, image_width/scale, image_height/scale);
			//fs::write(format!("sample_image_scale_{}_{}x{}.dat", scale, img.width, img.height), img.data);
			for y in (0..img.height - DETECTOR_SIZE as usize).step_by(DETECTOR_SIZE as usize/3) {
				for x in (0..img.width - DETECTOR_SIZE as usize).step_by(DETECTOR_SIZE as usize/3) {
					let roi = img.new_from_region(x, y, x + DETECTOR_SIZE as usize, y + DETECTOR_SIZE as usize);
					//fs::write(format!("roi_scale_{}_x{}_y{}_{}x{}.dat", scale, x, y, img.width, img.height), roi.data);
					//if !self.predict(&roi) { continue; }
					//if self.predict(&roi) {
					if self.predict_early_out(&roi, Some(10), Some(40)) {
						let p = self.probability(&roi);
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
		let mut rng = thread_rng();
		let mut examples = examples;
		let mut labels = labels; // We do this seemingly arbitrary reassign to allow us to mutate these refs.
		
		// Constants.
		let sample_rate_per_tree = 0.5f64;
		let max_depth = 3;
		let classifiers_to_keep:usize = 200; // Viola + Jones used 38, but they were smaller.
		
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
		
		// Track our error rates for the final shuffle.
		let mut false_negative_rates:Vec<u32> = vec![];
		
		// Make and train our new classifiers.
		for _ in 0..classifiers_to_keep {
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
			
			// Now use the classifier to classify all examples.
			let mut false_negative_rate:u32 = 0;
			let mut errors = 0;
			let mut weight_update_directions = vec![]; // Track the items we got wrong so we can reweight them in the next step.
			for (x, y) in converted_examples.iter().zip(&labels) {
				let prediction = d.predict(x);
				if prediction != *y {
					errors += 1;
					weight_update_directions.push(1);
				} else {
					weight_update_directions.push(-1);
				}
				
				if prediction == false && *y == true {
					false_negative_rate += 1;
				}
			}
			
			// Store D.
			self.classifiers.push(d);
			
			// Store the error rate.
			false_negative_rates.push(false_negative_rate);
			
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
		
		// Sort by false negative rate, lowest first.
		let mut indices:Vec<usize> = (0..(&self.classifiers).len()).collect();
		indices.sort_by(|a, b| {
			// Sort by the performance of the classifier.
			false_negative_rates[*a].cmp(&false_negative_rates[*b])
		});
		let (new_classifiers, new_weights) = indices.iter().map(|i|{ (self.classifiers[*i].clone(), self.weights[*i]) }).unzip();
		self.classifiers = new_classifiers;
		self.weights = new_weights;
		
		// Compact features.  Pick the elements that are not used by any of the trees and delete them to save space.
		/*
		let mut feature_use_count = vec![0u32; self.features.len()]; // Deliberate choice -- we will probably be popping the zeros and don't want to re-update the indices.
		for c in &self.classifiers {
			feature_use_count[c.get_decision_feature()] += 1;
		}
		let mut feature_indices_to_drop = vec![];
		for (feature_index, count) in feature_use_count.iter().enumerate() {
			if *count == 0 {
				feature_indices_to_drop.push(feature_index);
			}
		}
		for index in feature_indices_to_drop {
			self.features.remove(index);
			for c in &self.classifiers {
				if c.get_decision_feature() > index {
					// Reduce c's index by one.
				}
			}
		}
		*/
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
		let num_positive_examples = 10000;
		let num_negative_examples = 100000; // Only up to 10k are validated. 2k are thoroughly validated.
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