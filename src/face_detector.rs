
use super::integral_image::IntegralImage;

use rand::{thread_rng, Rng};
//use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::fmt;
use std::str::FromStr;
use std::error::Error;

pub struct DetectedFace {
	pub x: usize,
	pub y: usize,
	pub width: usize,
	pub height: usize,
	pub confidence: f32
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FaceDetector {
	classifier_ensemble: Vec<WeakClassifier>
}

impl FaceDetector {
	fn new() -> Self {
		FaceDetector {
			classifier_ensemble: vec![WeakClassifier::new_random(128u32)]
		}
	}
	
	fn detect_face(image_width: usize, image_height: usize, image_data: Vec<u8>) -> Vec::<DetectedFace> {
		let mut results = vec![];
		
		results
	}
	
	pub fn predict_with_probability(&self, example:&IntegralImage) -> f32 {
		let mut accum = 0f32;
		for c in &self.classifier_ensemble {
			let prediction = c.predict(example);
			accum += if prediction {
				c.weight
			} else {
				0.0 // NOT -c.weight
			}
		}
		
		accum / self.classifier_ensemble.len() as f32
	}
	
	// Returns true as soon as the weight exceeds the threshold.  Higher threshold means more steps.
	//pub fn predict_with_confidence(&self, example:&IntegralImage, min_confidence:f32) -> bool {
	
	// Faster than predict with probability.  Returns FALSE as soon as the first negative is found.
	// Since we sort classifier_ensemble by the lowest false negative rate first...
	pub fn predict(&self, example:&IntegralImage) -> bool {
		for c in &self.classifier_ensemble {
			if c.predict(example) == false {
				return false;
			}
		}
		return true;
	}
	
	pub fn train(&mut self, examples:Vec<&IntegralImage>, labels:Vec<bool>) {
		let mut classifiers_to_keep:usize = 100;
		let mut candidate_classifiers: Vec<WeakClassifier> = (0..5000).map(|_|{WeakClassifier::new_random(128u32)}).collect();
		let mut classifiers = Vec::<WeakClassifier>::with_capacity(classifiers_to_keep);
		let mut performance = Vec::<(u32, u32, u32, u32, f32)>::with_capacity(classifiers_to_keep);
		
		// Pick the weak learner that minimizes the number of misclassifications for the data.
		while classifiers.len() < classifiers_to_keep && !candidate_classifiers.is_empty() {
			let mut lowest_gini = 10.0f32;
			let mut best_classifier_index = 0usize;
			let mut best_classifier_error_count = 100000;
			let mut best_error_rates = (10000, 100000, 1000000, 1000000, 1f32);
			
			for (idx, classifier) in candidate_classifiers.iter().enumerate() {
				let (true_positives, true_negatives, false_positives, false_negatives, gini) = evaluate_classifier(&examples, &labels, &classifier);
				//if gini < lowest_gini { // Normally we would just compare gini, but we want to emphasize very low false negative rates.
				if gini < lowest_gini && false_negatives > 0 && false_positives > 0 && true_negatives > 0 && true_negatives > 0 { // Hack!
					lowest_gini = gini;
					best_classifier_index = idx;
					best_classifier_error_count = false_negatives + false_positives;
					best_error_rates = (true_positives, true_negatives, false_positives, false_negatives, gini);
					println!("New best Gini:{}.  TP:{}  TN:{}  FP:{}  FN:{}", lowest_gini, best_error_rates.0, best_error_rates.1, best_error_rates.2, best_error_rates.3);
				}
			}
			println!("Selected {}", best_classifier_index);
			
			// Calculate the amount of say for the classifier.
			let total_error = 1e-3f64 + best_classifier_error_count as f64 / examples.len() as f64;
			let amount_of_say = 0.5 * ((1.0 - total_error) / total_error).ln();
			let mut classifier = candidate_classifiers.remove(best_classifier_index);
			classifier.weight = amount_of_say as f32;
			
			// Reweight the examples.
			let base_sample_weight: f32 = 1.0f32 / (1f32 + examples.len() as f32);
			let mut new_weights_temp: Vec<f32> = (&examples).iter().zip(&labels).map(|(x, y)| {
				// For correctly classified examples, new weight is sample_weight * e^-amount_of_say.
				// For incorrectly classified examples, new weight is sample_weight * e^amount_of_say.
				let correct = classifier.predict(x) == *y;
				if correct {
					base_sample_weight * (-classifier.weight).exp()
				} else {
					base_sample_weight * (classifier.weight).exp()
				}
			}).collect();
			let sum_weights: f32 = new_weights_temp.iter().sum();
			let new_weights: Vec<f32> = new_weights_temp.iter().map(|w| { (*w) / sum_weights }).collect();
			
			// Sample from the examples.
			// _Instead of using weighted gini index_, make a new collection via randomly sampling _based on the weight_.
			let mut new_examples: Vec<&IntegralImage> = vec![];
			let mut new_labels: Vec<bool> = vec![];
			
			let mut rng = thread_rng();
			while new_examples.len() < examples.len() {
				// Generate a random number.
				let mut n = rng.gen::<f32>();
				for (idx, w) in new_weights.iter().enumerate() {
					if n < *w {
						new_examples.push(examples[idx]);
						new_labels.push(labels[idx]);
						break;
					} else {
						n -= *w;
					}
				}
			}
			
			classifiers.push(classifier);
			performance.push(best_error_rates);
		}
		
		// Now that we have all our classifiers, we should sort them by their False negative rate with GINI as the tie breaker.
		// Here's what we _could_ do:
		// Combine perf + classifier into a single array in an O(n) operation, sort in O(n log n), and split it out into the classifier ensemble in O(n).
		// OR we could bubble sort them together in O(n^2).
		// Instead, we make a list of the indices in O(n) and define our sort function in terms of references to perf.
		// THEN we use the sorted index list to pull out classifiers.
		let mut classifier_indices:Vec<usize> = (0..classifiers_to_keep).collect();
		classifier_indices.sort_by(|a, b|{
			let (true_positives_a, true_negatives_a, false_positives_a, false_negatives_a, gini_a) = performance[*a];
			let (true_positives_b, true_negatives_b, false_positives_b, false_negatives_b, gini_b) = performance[*b];
			let metric_a = (1+false_negatives_a) as f32 * gini_a;
			let metric_b = (1+false_negatives_b) as f32 * gini_b;
			metric_a.partial_cmp(&metric_b).unwrap()
		});
		
		self.classifier_ensemble = classifier_indices.iter().map(|i|{ classifiers[*i] }).collect();
	}
}

//
// Weak Classifier
//

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakClassifier {
	rectangles: Vec<(u32, u32, u32, u32)>, // x0, y0, x1, y1 * 3
	start_black: bool, // Dark -> light -> dark if false.  Light -> dark -> light if true.
	weight: f32,
}

impl WeakClassifier {
	fn new_random(max_detector_size:u32) -> Self {
		// Patches can take one of three configurations: stacked horizontal, stacked vertical, blocks.
		let mut rng = thread_rng();
		let num_rectangles: u32 = rng.gen::<u32>() % 5;
		let mut rectangles = vec![];
		for _ in 0..num_rectangles {
			// Pick two random points for the x divs.
			let x1 = rng.gen_range(0, max_detector_size-2);
			let x2 = rng.gen_range(x1+1, max_detector_size);
			let y1 = rng.gen_range(0, max_detector_size-2);
			let y2 = rng.gen_range(y1+1, max_detector_size);
			rectangles.push((x1, y1, x2, y2));
		};
		
		let invert_rects = rng.gen_bool(0.5);
		
		WeakClassifier {
			rectangles,
			start_black: invert_rects,
			weight: 0.0
		}
	}
	
	fn predict(&self, integral_image:&IntegralImage) -> bool {
		let mut positive_regions = 0f32;
		let mut negative_regions = 0f32;
		for (i, (ax, ay, bx, by)) in self.rectangles.iter().enumerate() {
			let dy = *by-*ay;
			let dx = *bx-*ax;
			
			if dx == 0 || dy == 0 {
				continue;
			}
			
			let positive_region = (i%2 == 1) == self.start_black; // Offset by one if start_black = true.
			
			let region_size = (dx as i32 *dy as i32).abs() as f32;
			let region_sum = integral_image.get_region_sum(
				*ax as usize, *ay as usize,
				*bx as usize, *by as usize
			);
			
			if positive_region {
				positive_regions += region_sum as f32/region_size;
			} else {
				negative_regions += region_sum as f32/region_size;
			}
		}
		
		return (positive_regions - negative_regions) > 0f32;
	}
}

fn gini_index(true_positive:u32, true_negative:u32, false_positive:u32, false_negative:u32) -> f32 {
	let total_count = true_positive + true_negative + false_positive + false_negative;
	let num_positive = true_positive + false_negative;
	let num_negative = true_negative + false_positive;
	
	let p_positive = num_positive as f32 / total_count as f32;
	let p_negative = num_negative as f32 / total_count as f32;
	
	let p_true_positive = true_positive as f32 / num_positive as f32;
	let p_false_negative = false_negative as f32 / num_positive as f32;
	let p_true_negative = true_negative as f32 / num_negative as f32;
	let p_false_positive = false_positive as f32 / num_negative as f32;
	
	let gini_positive = 1.0f32 - (p_true_positive*p_true_positive + p_false_negative*p_false_negative);
	let gini_negative = 1.0f32 - (p_true_negative*p_true_negative + p_false_positive*p_false_positive);
	return gini_positive*p_positive + gini_negative*p_negative;
}

fn evaluate_classifier(examples:&Vec<&IntegralImage>, labels:&Vec<bool>, classifier:&WeakClassifier) -> (u32, u32, u32, u32, f32) {
	let mut true_positives = 0;
	let mut true_negatives = 0;
	let mut false_positives = 0;
	let mut false_negatives = 0;
	for (x, y) in (*examples).iter().zip(labels) {
		let prediction = classifier.predict(x);
		if *y && prediction {
			true_positives += 1;
		} else if *y && !prediction {
			false_negatives += 1;
		} else if !*y && prediction {
			false_positives += 1;
		} else if !*y && !prediction {
			true_negatives += 1;
		}
	}
	let gini = gini_index(true_positives, true_negatives, false_positives, false_negatives);
	return (true_positives, true_negatives, false_positives, false_negatives, gini);
}

#[cfg(test)]
mod tests {
	// Note this useful idiom: importing names from outer (for mod tests) scope.
	use super::*;
	use std::fs::File;
	use std::io::prelude::*;
	
	macro_rules! assert_approx_eq {
		($a:expr, $b:expr) => {
			assert!(($a as f32 - $b as f32).abs() < 1e-3f32, "{} !~= {}", $a, $b);
		};
	}
	
	#[test]
	fn weak_classifier_sanity() {
		let img_data = vec![
		//	0  1  2  3  4  5  6  7  8  9 10 11
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 1
			0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, // 2
			0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, // 3
			0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, // 4
			0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, // 5
			0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, // 6
			0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, // 7
			0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, // 8
			0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, // 9
		];
		let img = IntegralImage::new_from_image_data(12, 10, &img_data);
		let column_detector = WeakClassifier {
			rectangles: vec![(0, 2, 1, 5),  (2, 2, 3, 6),  (4, 2, 5, 6)],
			start_black: true, // Dark, light, dark.
			weight: 0.0
		};
		let angle_detector = WeakClassifier {
			rectangles: vec![(2, 2, 3, 3), (4, 4, 5, 5)],
			start_black: false, // Light, dark.
			weight: 0.0
		};
		let bar_detector = WeakClassifier {
			rectangles: vec![(0, 0, 11, 1), (0, 2, 11, 3)],
			start_black: false, // Light, dark
			weight: 0.0
		};
		
		assert!(column_detector.predict(&img));
		assert!(angle_detector.predict(&img));
		assert!(!bar_detector.predict(&img));
	}
	
	#[test]
	fn test_gini() {
		/*
		P(truth = yes) = 6/10
		P(truth = no) = 4/10
		P(truth = yes and foo = yes) = 4/6
		P(truth = yes and foo = no) = 2/6
		Gini index = 1 - ((4/6)^2 + (2/6)^2) = 0.45
		P(truth = no and foo = yes) = 0/4
		P(truth = no and foo = no) = 4/4
		Gini index = 1 - (0^2 + 1^2) = 0
		Gini index = (6/10)*0.45 + (4/10)*0 = 0.27
		*/
		assert_approx_eq!(gini_index(4, 4, 0, 2), 0.2666);
		
		let perfect_classifier_score = gini_index(6, 6, 0, 0);
		let okay_classifier_score = gini_index(3, 4, 1, 1);
		let shitty_classifier_score = gini_index(3, 3, 3, 3);
		assert!(perfect_classifier_score < okay_classifier_score);
		assert!(okay_classifier_score < shitty_classifier_score);
		assert!(perfect_classifier_score < shitty_classifier_score);
	}
	
	#[test]
	#[ignore] // Run with cargo test -- --ignored
	fn test_train_face_classifier() {
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
			examples.push(IntegralImage::new_from_image_data(128, 128, &img_buffer));
			labels.push(true);
			if examples.len() > 5000 {
				break;
			}
		}
		while let Ok(bytes_read) = negative.read(&mut img_buffer) {
			examples.push(IntegralImage::new_from_image_data(128, 128, &img_buffer));
			labels.push(false);
			if examples.len() > 25000 {
				break;
			}
		}
		
		// Split training and test.
		println!("Splitting data.");
		for i in 0..examples.len() {
			if rng.gen_bool(0.9) {
				train_x.push(&examples[i]);
				train_y.push(labels[i]);
			} else {
				test_x.push(&examples[i]);
				test_y.push(labels[i]);
			}
		}
		
		println!("Allocating face detector");
		let mut face_detector = FaceDetector::new();
		face_detector.train(train_x, train_y);
		
		println!("Testing classifiers.");
		let mut true_positive = 0;
		let mut false_positive = 0;
		let mut true_negative = 0;
		let mut false_negative = 0;
		
		for (x, y) in test_x.iter().zip(&test_y) {
			let proba = face_detector.predict_with_probability(x);
			let prediction = face_detector.predict(x);
			println("Truth: {}  Prediction: {}  Probability: {}", *y, prediction, proba);
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
	}
}