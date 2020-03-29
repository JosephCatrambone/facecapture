
use nalgebra::linalg::SVD;
use nalgebra::{DMatrix, Vector, DVector};
use serde::{Deserialize, Serialize};

const LATENT_SIZE:usize = 512;
const IMAGE_WIDTH:usize = 64;
const IMAGE_HEIGHT:usize = 64;

#[derive(Serialize, Deserialize)]
struct Expression {
	name: String,
	latent_point: Vec<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
struct ExpressionGroup {
	name: String,
	expressions: Vec<Expression>,
	mutually_exclusive: bool,  // If true, only one of these can be active at a time.
}

#[derive(Serialize, Deserialize)]
pub struct ExpressionDetector {
	expression_groups: Vec<ExpressionGroup>,
	mean_face: Vec<f32>,
	selected_group: Option<ExpressionGroup>,
	selected_expression: Option<Expression>,
	//face_profile: DMatrix<f32>,
	// low-dim = sigma_inv * U_trans * q
}

impl ExpressionDetector {
	pub fn new() -> Self {
		ExpressionDetector {
			expression_groups: vec![],
			mean_face: vec![],
			selected_group: None,
			selected_expression: None,
		}
	}
	
	pub fn get_expression_list(&self) -> Vec<(String, Vec<String>)> {
		let mut expression_list = vec![];
		
		for xg in &self.expression_groups {
			let mut subexpr = vec![];
			for xpr in &xg.expressions {
				subexpr.push(xpr.name.clone());
			}
			expression_list.push((xg.name.clone(), subexpr));
		}
		
		expression_list
	}
	
	pub fn get_expression_weights(&self, image_width:usize, image_height:usize, image_data:&Vec<u8>, roi:(u32, u32, u32, u32)) -> Vec<(String, Vec<(String, f32)>)> {
		// Returns a list of { group: [expression 1: amount, expression 2: amount, expression 3: amount, ...], group 2: [expression 1: amount, ...]
		let mut expression_list = vec![];
		
		// Extract the ROI from the given face.
		todo!();
		
		for xg in &self.expression_groups {
			let mut subexpr = vec![];
			for xpr in xg.expressions {
				subexpr.push((xpr.name, 0f32));
			}
			expression_list.push((xg.name.clone(), subexpr));
		}
		
		expression_list
	}
}
