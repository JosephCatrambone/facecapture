
use std::io::Cursor;
use serde::{Deserialize, Serialize};
use tch;
use std::collections::HashMap;
use std::convert::TryInto;
use std::borrow::BorrowMut;
use rand::AsByteSliceMut;

// Keep these compatible with the expression detector.
const LATENT_SIZE:usize = 1024;
const IMAGE_WIDTH:usize = 48;
const IMAGE_HEIGHT:usize = 48;

#[derive(Serialize, Deserialize)]
struct Expression {
	name: String,
	latent_point: Vec<f32>,
}

pub struct ExpressionDetector {
	model: tch::CModule,
	expressions: Vec<Expression>,
	//face_profile: DMatrix<f32>,
	// low-dim = sigma_inv * U_trans * q
}

impl ExpressionDetector {
	pub fn new() -> Self {
		let mut static_model_data = include_bytes!("../ml/expression_detector_cexport.pt");
		
		let module = tch::CModule::load_data(&mut Cursor::new(&static_model_data[..]));
		match module {
			Ok(m) => {
				ExpressionDetector {
					model: m,
					expressions: vec![]
				}
			},
			Err(e) => {
				eprintln!("{}", e);
				panic!(e);
			}
		}
	}
	
	pub fn get_expressions(&self) -> Vec<String> {
		let mut expression_list = vec![];
		
		for xpr in &self.expressions {
			expression_list.push(xpr.name.clone());
		}
		
		expression_list
	}
	
	pub fn get_expression_weights(&self, image_width:u32, image_height:u32, image_data:&Vec<u8>, roi:(u32, u32, u32, u32)) -> HashMap<String, f32> {
		// Returns a list of { group: [expression 1: amount, expression 2: amount, expression 3: amount, ...], group 2: [expression 1: amount, ...]
		let mut expression_list = HashMap::with_capacity(self.expressions.len());
		
		// Extract the ROI from the given face.
		let mut face:Vec<f64> = Vec::with_capacity((roi.2*roi.3) as usize);
		for y in roi.1 .. (roi.1+roi.3) {
			for x in roi.0 .. (roi.0+roi.2) {
				face.push(image_data[(x + y*image_width) as usize] as f64 / 255.0f64);
			}
		}
		
		// Embed the extracted face:
		let embedding:Vec<f64> = match self.model.forward_is(&[tch::IValue::DoubleList(face)]).unwrap() {
			tch::IValue::DoubleList(x) => x,
			_ => panic!("Fuck")
		};
		let mut embedding_magnitude = 0f32;
		for i in 0..LATENT_SIZE {
			embedding_magnitude += (embedding[i]*embedding[i]) as f32;
		}
		
		// Calc cosine product with all expressions.
		for xpr in &self.expressions {
			let mut similarity = 0f32;
			let mut magnitude = 0.0f32;
			for (idx, a) in xpr.latent_point.iter().enumerate() {
				let a = *a;
				let b = embedding[idx] as f32;
				magnitude += a*a;
				similarity += a*b;
			}
			expression_list.insert(xpr.name.clone(), similarity/(magnitude*embedding_magnitude));
		}
		
		expression_list
	}
}
