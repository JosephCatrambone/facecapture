
use super::integral_image::IntegralImage;

use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::cmp::{max, min};
use std::fmt;
use std::str::FromStr;
use std::error::Error;
use std::borrow::Borrow;

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
	classifier_ensemble: Vec<WeakClassifier>
}

impl FaceDetector {
	fn new() -> Self {
		FaceDetector {
			classifier_ensemble: vec![]
		}
	}
	
	pub fn detect_face(&self, image_width: usize, image_height: usize, image_data: &Vec<u8>) -> Vec::<DetectedFace> {
		let mut results = vec![];
		
		//let img = IntegralImage::new_from_image_data(image_width, image_height, image_data);
		
		for scale in 3..16 {
			let img = IntegralImage::new_from_image_data_subsampled(image_width, image_height, image_data, scale, scale);
			for y in (0..img.height - DETECTOR_SIZE as usize).step_by(DETECTOR_SIZE as usize) {
				for x in (0..img.width - DETECTOR_SIZE as usize).step_by(DETECTOR_SIZE as usize) {
					let roi = img.new_from_region(x, y, x + DETECTOR_SIZE as usize, y + DETECTOR_SIZE as usize);
					//if !self.predict(&roi) { continue; }
					let p = self.predict_with_probability(&roi);
					//let p = self.predict(&roi);
					if self.predict(&roi) {
						let f = DetectedFace {
							x: scale*x,
							y: scale*y,
							width: scale*DETECTOR_SIZE as usize,
							height: scale*DETECTOR_SIZE as usize,
							//confidence: 0.0f32.max( p)
							confidence: p
						};
						results.push(f);
					}
				}
			}
		}
		
		results
	}
	
	pub fn predict_with_probability(&self, example:&IntegralImage) -> f32 {
		let mut possible_weight = 0.0f32;
		let mut total_weight = 0.0f32;
		for c in &self.classifier_ensemble {
			possible_weight += c.weight.abs();
			if c.predict(example) {
				total_weight += c.weight;
			} else {
				total_weight -= c.weight;
			}
		}
		((total_weight / possible_weight) + 1f32) / 2f32
	}
	
	// Faster than predict with probability.  Returns FALSE as soon as the first negative is found.
	// Since we sort classifier_ensemble by the lowest false negative rate first...
	pub fn predict(&self, example:&IntegralImage) -> bool {
		self.predict_with_consensus(example, Some(5), Some(16)) // Since we bias for low false negatives, we emphasize early out.
	}
	
	pub fn predict_with_consensus(&self, example:&IntegralImage, minimum_true:Option<u16>, minimum_false:Option<u16>) -> bool {
		let mut yeas = 0;
		let mut nays = 0;
		let min_yeas = minimum_true.unwrap_or((self.classifier_ensemble.len()/2) as u16);
		let min_nays = minimum_false.unwrap_or((self.classifier_ensemble.len()/2) as u16);
		assert!(min_yeas + min_nays < self.classifier_ensemble.len() as u16);
		for c in &self.classifier_ensemble[..(min_yeas+min_nays) as usize] {
			let p = c.predict(example);
			if (p && c.weight > 0.0) || (!p && c.weight < 0.0) {
				yeas += 1;
			} else {
				nays += 1;
			}
			
			if yeas >= min_yeas {
				return true;
			} else if nays >= min_nays {
				return false;
			}
		}
		return yeas > nays;
	}
	
	pub fn train(&mut self, examples:Vec<&IntegralImage>, labels:Vec<bool>) {
		let mut examples = examples;
		let mut labels = labels;
		let mut classifiers_to_keep:usize = 500; // Viola + Jones used 38, but they were smaller.
		//let mut candidate_classifiers: Vec<WeakClassifier> = (0..2000).map(|_|{WeakClassifier::new_random(DETECTOR_SIZE)}).collect();
		let mut candidate_classifiers = make_2x2_haar(DETECTOR_SIZE as usize, DETECTOR_SIZE as usize);
		candidate_classifiers.extend(make_3x3_haar(DETECTOR_SIZE as usize, DETECTOR_SIZE as usize));
		let mut classifiers = Vec::<WeakClassifier>::with_capacity(classifiers_to_keep);
		let mut performance = Vec::<(u32, u32, u32, u32, f32)>::with_capacity(classifiers_to_keep);
		
		// Pick the weak learner that minimizes the number of misclassifications for the data.
		while classifiers.len() < classifiers_to_keep && !candidate_classifiers.is_empty() {
			let mut lowest_score = 10.0f32;
			let mut best_classifier_index = 0usize;
			let mut best_error_rates = (10000, 100000, 1000000, 1000000, 1f32);
			let mut have_candidate = false;
			
			// Parallel evaluate classifiers.
			let classifier_performance:Vec::<(usize, (u32, u32, u32, u32, f32))> = candidate_classifiers.par_iter().enumerate().map(|(idx, c)|{
				(idx, evaluate_classifier(&examples, &labels, &c))
			}).collect();

			lowest_score = 100_000.0;
			println!("Start lowest gini: {}", lowest_score);
			for (idx, error_rates) in classifier_performance {
				let (true_positives, true_negatives, false_positives, false_negatives, gini) = error_rates;
				if true_positives == 0 || true_negatives == 0 || false_positives == 0 || false_negatives == 0 {
					continue // Ignore those that miss one category completely.
				}
				let score = gini; // We want a low gini score, but also want really few false negatives.
				if score < lowest_score { // Normally we would just compare gini, but we want to emphasize very low false negative rates.
					lowest_score = score;
					best_classifier_index = idx;
					best_error_rates = error_rates;
					println!("TP/TN/FP/FN/Gini:{:?}", &error_rates);
					have_candidate = true;
				}
			}
			println!("Len Classifiers: {}\t\tCandidates: {}", &classifiers.len(), &candidate_classifiers.len());
			
			if lowest_score.is_nan() || lowest_score.is_infinite() {
				break
			}
			
			assert!(have_candidate); // Training can fail if we never find a non-trivial classifier.
			
			// Calculate the amount of say for the classifier.
			let total_error = 1e-3f64 + (best_error_rates.2 + best_error_rates.3) as f64 / examples.len() as f64;
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
			let new_indices:Vec<usize> = (0..examples.len()).into_par_iter().map(|_|{
				let mut rng = thread_rng();
				let mut n = rng.gen::<f32>();
				assert!(n >= 0f32 && n <= 1f32);
				let mut i = 0;
				for (idx, w) in new_weights.iter().enumerate() {
					if n < *w {
						i = idx;
						break;
					} else {
						n -= *w;
					}
				}
				i
			}).collect();
			
			let mut new_examples = vec![];
			let mut new_labels = vec![];
			for i in new_indices {
				new_examples.push(examples[i]);
				new_labels.push(labels[i]);
			}
			examples = new_examples;
			labels = new_labels;
			
			classifiers.push(classifier);
			performance.push(best_error_rates);
		}
		
		// Now that we have all our classifiers, we should sort them by their False negative rate with GINI as the tie breaker.
		let mut classifiers_with_performance: Vec::<(f32, &WeakClassifier)> = classifiers.iter().zip(performance).map(|(c, perf)| {
			let (true_positives_a, true_negatives_a, false_positives_a, false_negatives_a, gini_a) = perf;
			let metric = (1+false_negatives_a) as f32 * gini_a;
			(metric, c)
		}).collect();
		classifiers_with_performance.sort_by(|a, b|{
			a.0.partial_cmp(&b.0).unwrap()
		});
		self.classifier_ensemble.clear();
		for (_, c) in classifiers_with_performance {
			self.classifier_ensemble.push(c.clone());
		}
	}
}

//
// Weak Classifier
//

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakClassifier {
	rectangles: Vec<(u32, u32, u32, u32)>, // x0, y0, x1, y1
	weight: f32,
}

impl WeakClassifier {
	fn predict(&self, integral_image:&IntegralImage) -> bool {
		let mut positive_regions = 0;
		let mut negative_regions = 0;
		for (i, (ax, ay, bx, by)) in self.rectangles.iter().enumerate() {
			let positive_region = (i%2 == 1);
			
			//let region_size = ((dx*dy) as f32).abs();
			let region_sum = integral_image.get_region_sum(
				*ax as usize, *ay as usize,
				*bx as usize, *by as usize
			);
			
			if positive_region {
				positive_regions += region_sum;
			} else {
				negative_regions += region_sum;
			}
		}
		
		return positive_regions > negative_regions;
	}
}

fn make_2x2_haar(max_width:usize, max_height:usize) -> Vec<WeakClassifier> {
	let max_width:u32 = max_width as u32;
	let max_height:u32 = max_height as u32;
	let mut res = Vec::<WeakClassifier>::new();
	for y in 0..max_height-2 {
		for x in 0..max_width-2 {
			// Vertical bars.
			res.push(
				WeakClassifier {
					rectangles: vec![
						(x, y, x+1, y+2),
						(x+1, y, x+2, y+2)
					],
					weight: 0.0
				}
			);
			// Horizontal
			res.push(
				WeakClassifier {
					rectangles: vec![
						(x, y, x+2, y+1),
						(x, y+1, x+2, y+2)
					],
					weight: 0.0
				}
			);
		}
	}
	res
}

fn make_3x3_haar(max_width:usize, max_height:usize) -> Vec<WeakClassifier> {
	let max_width:u32 = max_width as u32;
	let max_height:u32 = max_height as u32;
	let mut res = Vec::<WeakClassifier>::new();
	for y in 0..max_height-3 {
		for x in 0..max_width-3 {
			// Vertical bars.
			res.push(
				WeakClassifier {
					rectangles: vec![
						(x, y, x+1, y+3),
						(x+1, y, x+2, y+3),
						(x+3, y, x+3, y+3),
					],
					weight: 0.0
				}
			);
			// Horizontal
			res.push(
				WeakClassifier {
					rectangles: vec![
						(x, y, x+3, y+1),
						(x, y+1, x+3, y+2),
						(x, y+2, x+3, y+3),
					],
					weight: 0.0
				}
			);
		}
	}
	res
}

fn gini_index(true_positive:u32, true_negative:u32, false_positive:u32, false_negative:u32) -> f32 {
	let total_count = true_positive + true_negative + false_positive + false_negative + 1;
	let num_positive = true_positive + false_negative + 1;
	let num_negative = true_negative + false_positive + 1;
	
	let p_positive = num_positive as f32 / total_count as f32;
	let p_negative = num_negative as f32 / total_count as f32;
	
	let p_true_positive = true_positive as f32 / num_positive as f32;
	let p_false_negative = false_negative as f32 / num_positive as f32;
	let p_true_negative = true_negative as f32 / num_negative as f32;
	let p_false_positive = false_positive as f32 / num_negative as f32;
	
	let gini_positive = 1.0f32 - (p_true_positive*p_true_positive + p_false_negative*p_false_negative);
	let gini_negative = 1.0f32 - (p_true_negative*p_true_negative + p_false_positive*p_false_positive);
	let result = gini_positive*p_positive + gini_negative*p_negative;
	//assert!(!result.is_nan());
	result
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
			rectangles: vec![
				(0, 2, 1, 7),
				(2, 2, 3, 7),
				(4, 2, 5, 7)
			],
			weight: 0.1
		};
		
		assert!(column_detector.predict(&img));
		//assert!(angle_detector.predict(&img));
		//assert!(!bar_detector.predict(&img));
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
		//assert_approx_eq!(gini_index(4, 4, 0, 2), 0.2666);
		
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
			let resized = IntegralImage::new_from_image_data_subsampled(128, 128, &img_buffer, 128/DETECTOR_SIZE as usize, 128/DETECTOR_SIZE as usize);
			examples.push(resized);
			labels.push(true);
			if examples.len() > 5000 {
				break;
			}
		}
		while let Ok(bytes_read) = negative.read(&mut img_buffer) {
			let resized = IntegralImage::new_from_image_data_subsampled(128, 128, &img_buffer, 128/DETECTOR_SIZE as usize, 128/DETECTOR_SIZE as usize);
			examples.push(resized);
			labels.push(false);
			if examples.len() > 100000 {
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