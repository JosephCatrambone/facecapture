use crate::integral_image::IntegralImage;

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaarFeature {
	rectangles: Vec<(u32, u32, u32, u32)>, // x0, y0, x1, y1
	weight: f32,
}

impl HaarFeature {
	pub fn probability(&self, integral_image:&IntegralImage) -> f32 {
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
		return positive_regions as f32 / (positive_regions + negative_regions) as f32;
	}
	
	pub fn predict(&self, integral_image:&IntegralImage) -> bool {
		self.probability(integral_image) > 0.5f32
	}
}

pub fn make_2x2_haar(max_width:usize, max_height:usize) -> Vec<HaarFeature> {
	let max_width:u32 = max_width as u32;
	let max_height:u32 = max_height as u32;
	let mut res = Vec::<HaarFeature>::new();
	for y in 0..max_height-2 {
		for x in 0..max_width-2 {
			// Vertical bars.
			res.push(
				HaarFeature {
					rectangles: vec![
						(x, y, x+1, y+2),
						(x+1, y, x+2, y+2)
					],
					weight: 0.0
				}
			);
			// Horizontal
			res.push(
				HaarFeature {
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

pub fn make_3x3_haar(max_width:usize, max_height:usize) -> Vec<HaarFeature> {
	let max_width:u32 = max_width as u32;
	let max_height:u32 = max_height as u32;
	let mut res = Vec::<HaarFeature>::new();
	for y in 0..max_height-3 {
		for x in 0..max_width-3 {
			// Vertical bars.
			res.push(
				HaarFeature {
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
				HaarFeature {
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

fn evaluate_classifier(examples:&Vec<&IntegralImage>, labels:&Vec<bool>, classifier:&HaarFeature) -> (u32, u32, u32, u32, f32) {
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
		let column_detector = HaarFeature {
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
}