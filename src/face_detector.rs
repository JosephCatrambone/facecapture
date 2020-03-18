
use rand::{thread_rng, Rng};
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

fn detect_face(image_width:usize, image_height:usize, image_data:Vec<u8>) -> Vec::<DetectedFace> {
	let mut results = vec![];
	
	
	
	results
}

//
// Integral image
//

pub struct IntegralImage {
	pub width: usize,
	pub height: usize,
	pub pixel_data: Vec::<u8>,
	pub integral_data: Vec::<u32>
}

impl IntegralImage {
	pub fn new_from_image_data(width: usize, height: usize, pixel_data: Vec<u8>) -> Self {
		let mut integral_data = vec![0u32; width * height];
		for y in 0..height {
			for x in 0..width {
				let i = x + y*width;
				
				if x > 0 {
					integral_data[i] += integral_data[i-1];
				}
				if y > 0 {
					integral_data[i] += integral_data[i-width];
				}
				if x > 0 && y > 0 {
					integral_data[i] -= integral_data[(i-1)-width];
				}
				integral_data[i] += pixel_data[i] as u32;
			}
		}
		IntegralImage {
			width,
			height,
			pixel_data,
			integral_data
		}
	}
	
	/// Get the value of the block from (x0,y0) to (x1,y1) inclusive.
	/// This operation is constant time.
	pub fn get_region_sum(&self, x0:usize, y0:usize, x1:usize, y1:usize) -> i32 {
		// A B C D
		// E F G H
		// I J K L
		// M N O P
		// If we want to find the sum of the KLOP area, we can take  P, the sum of the rect,
		// less H, the sum of that top half, less N, the sum of the left half, plus F, since we took
		// out that top-left section twice.
		assert!(x1 > x0 && y1 > y0);
		let bottom_right = self.integral_data[x1 + (y1*self.width)];
		let left = if x0 == 0 { 0 } else { self.integral_data[(x0-1) + (y0*self.width)] };
		let top = if y0 == 0 { 0 } else { self.integral_data[x0 + ((y0-1)*self.width)]};
		let top_left = if x0 == 0 || y0 == 0 { 0 } else { self.integral_data[(x0-1) + (y0-1)*self.width] };
		((bottom_right + top_left) as i32 - left as i32) - top as i32
	}
}

//
// Weak Classifier
//

#[derive(Debug, Copy, Clone)]
pub struct WeakClassifier {
	rectangles: [u32; 4*3], // x0, y0, x1, y1 * 3
	invert_rects: bool,
	threshold: f32,
	weight: f32,
}

impl WeakClassifier {
	fn new_random(max_detector_size:u32) -> Self {
		// Patches can take one of three configurations: stacked horizontal, stacked vertical, blocks.
		let mut rng = thread_rng();
		let class_type: u32 = rng.gen::<u32>() % 2;
		let rectangles = match class_type {
			0 | 1 => {
				// Pick two random points for the x divs.
				let x1 = rng.gen_range(0, max_detector_size-1);
				let x2 = rng.gen_range(x1+1, max_detector_size);
				let y1 = rng.gen_range(0, (max_detector_size/2)-1);
				let y2 = max_detector_size - y1;
				
				if class_type == 0 {
					[
						0, y1, x1, y2,
						x1, y1, x2, y2,
						x2, y1, max_detector_size, y2
					]
				} else {
					[ // Transpose
						y1, 0, y2, x1,
						y1, x1, y2, x2,
						y1, x2, y2, max_detector_size
					]
				}
			},
			_ => panic!()
		};
		
		let invert_rects = rng.gen_bool(0.5);
		
		WeakClassifier {
			rectangles,
			invert_rects,
			threshold: 0.0,
			weight: 1.0
		}
	}
	
	fn predict(&self, integral_image:&IntegralImage) -> bool {
		let mut region_size = 0;
		let mut region_sum = 0;
		for i in 0..3 {
			let ax = self.rectangles[i*4+0];
			let ay = self.rectangles[i*4+1];
			let bx = self.rectangles[i*4+2];
			let by = self.rectangles[i*4+3];
			let dy = by-ay;
			let dx = bx-ax;
			
			if dx == 0 || dy == 0 {
				continue;
			}
			
			region_size += dx*dy;
			let temp_region_sum = integral_image.get_region_sum(
				ax as usize, ay as usize,
				bx as usize, by as usize
			) * (-1 * (i as i32%2)); // Invert the middle region.
			if self.invert_rects {
				region_sum -= temp_region_sum;
			} else {
				region_sum += temp_region_sum;
			}
		}
		
		return (region_sum as f32/region_size as f32) > self.threshold;
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

pub fn train_face_classifier(examples:Vec<&IntegralImage>, labels:Vec<bool>, candidate_classifiers:&mut Vec<WeakClassifier>) -> Vec<WeakClassifier> {
	if candidate_classifiers.is_empty() {
		return vec![];
	}
	
	// Pick the weak learner that minimizes the number of misclassifications for the data.
	let mut lowest_gini = 1.0f32; // Can't be higher.
	let mut best_classifier_index = 0usize;
	let mut best_classifier_error_count = 0;
	
	for (idx, classifier) in candidate_classifiers.iter().enumerate() {
		let mut true_positives = 0;
		let mut true_negatives = 0;
		let mut false_positives = 0;
		let mut false_negatives = 0;
		for (x, y) in examples.iter().zip(&labels) {
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
		if gini < lowest_gini {
			lowest_gini = gini;
			best_classifier_index = idx;
			best_classifier_error_count = false_negatives + false_positives;
		}
	}
	
	// Calculate the amount of say for the classifier.
	let total_error = 1e-6f64 + best_classifier_error_count as f64;
	let amount_of_say = 0.5 * ((1.0 - total_error) / total_error).log2();
	let mut classifier = candidate_classifiers.remove(best_classifier_index);
	classifier.weight = amount_of_say as f32;
	
	// Reweight the examples.
	let base_sample_weight:f32 = 1.0f32 / examples.len() as f32;
	let mut new_weights_temp:Vec<f32> = (&examples).iter().zip(&labels).map(|(x,y)|{
		// For correctly classified examples, new weight is sample_weight * e^-amount_of_say.
		// For incorrectly classified examples, new weight is sample_weight * e^amount_of_say.
		let correct = classifier.predict(x) == *y;
		if correct {
			base_sample_weight * (-classifier.weight).exp()
		} else {
			base_sample_weight * (classifier.weight).exp()
		}
	}).collect();
	let sum_weights:f32 = new_weights_temp.iter().sum();
	let new_weights:Vec<f32> = new_weights_temp.iter().map(|w|{ (*w)/sum_weights }).collect();
	
	// Sample from the examples.
	// _Instead of using weighted gini index_, make a new collection via randomly sampling _based on the weight_.
	let mut new_examples:Vec<&IntegralImage> = vec![];
	let mut new_labels:Vec<bool> = vec![];
	
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
	
	let mut new_classifiers = vec![classifier];
	new_classifiers.extend(train_face_classifier(new_examples, new_labels, candidate_classifiers));
	return new_classifiers;
}

#[cfg(test)]
mod tests {
	// Note this useful idiom: importing names from outer (for mod tests) scope.
	use super::*;
	
	macro_rules! assert_approx_eq {
		($a:expr, $b:expr) => {
			assert!(($a as f32 - $b as f32).abs() < 1e-3f32, "{} !~= {}", $a, $b);
		};
	}
	
	#[test]
	fn integral_sanity() {
		let img = IntegralImage::new_from_image_data(5, 4, vec![
			1, 0, 1, 0, 0,
			2, 0, 0, 0, 0,
			3, 0, 1, 1, 0,
			0, 0, 0, 0, 0
		]);
		assert_eq!(img.get_region_sum(0, 0, 4, 3), 1+1+2+3+1+1);
		assert_eq!(img.get_region_sum(0, 0, 1, 1), 1+2);
		assert_eq!(img.integral_data, vec![
			1, 1, 2, 2, 2,
			3, 3, 4, 4, 4,
			6, 6, 8, 9, 9,
			6, 6, 8, 9, 9
		]);
	}
	
	
	#[test]
	fn weak_classifier_sanity() {
		let img_data = vec![
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0,
			0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0,
			0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
			0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
			0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
			0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
		];
		let img = IntegralImage::new_from_image_data(10, 12, img_data);
		let column_detector = WeakClassifier {
			rectangles: [0, 2, 1, 5,  2, 2, 3, 6,  4, 2, 5, 6],
			invert_rects: true, // Dark, light, dark.
			threshold: 0.0,
			weight: 0.0
		};
		
		assert!(column_detector.predict(&img));
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
	fn test_train_face_classifier() {
		let mut rng = thread_rng();
		
		// Load a bunch of data.
		let mut examples:Vec<IntegralImage> = vec![];
		let mut labels:Vec<bool> = vec![];
		let mut train_x:Vec<&IntegralImage> = vec![];
		let mut train_y:Vec<bool> = vec![];
		let mut test_x:Vec<&IntegralImage> = vec![];
		let mut test_y:Vec<bool> = vec![];
		
		// Split training and test.
		for i in 0..examples.len() {
			if rng.gen_bool(0.9) {
				train_x.push(&examples[i]);
				train_y.push(labels[i]);
			} else {
				test_x.push(&examples[i]);
				test_y.push(labels[i]);
			}
		}
		
		let classifiers:Vec<WeakClassifier> = (0..2000).map(|_|{WeakClassifier::new_random(32u32)}).collect();
		
	}
}