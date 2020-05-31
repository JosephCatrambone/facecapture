
use std::io::Cursor;
use serde::{Deserialize, Serialize};
use tch;
use std::collections::HashMap;

// Required imports:
use tch::nn::Module;

// Keep these compatible with the expression detector.
const LATENT_SIZE:usize = 1024;
const IMAGE_WIDTH:usize = 48;
const IMAGE_HEIGHT:usize = 48;

#[derive(Serialize, Deserialize)]
struct Expression {
	name: String,
	latent_point: Vec<f64>,
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
	
	pub fn add_expression(&mut self, image_width:u32, image_height:u32, image_data:&Vec<u8>, roi:(u32, u32, u32, u32), name:String) {
		// Cut the region out, resize it to our face detector.
		let mut face:Vec<f64> = image_and_roi_to_vec(image_width, image_height, image_data, roi).iter().map(|f|{ *f as f64 / 255.0f64 }).collect();
		let mut tensor = tch::Tensor::of_slice(face.as_slice()).view([1, 1, IMAGE_HEIGHT as i64, IMAGE_WIDTH as i64]);
		
		// Calculate embedding.
		//let t = tch::Tensor::of_slice(face.as_slice());
		//let t = IValue::DoubleList(face);
		let mut embedding:Vec<f64> = Vec::<f64>::from(self.model.forward(&tensor));
		/*
		tch::no_grad(||{
			embedding = Vec::<f64>::from(self.model.forward(&tensor));
		});
		 */
		
		// Insert a new expression.
		self.expressions.push(Expression{
			name,
			latent_point: embedding
		});
	}
	
	pub fn get_expression_count(&self) -> usize { self.expressions.len() }
	
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
		let mut face:Vec<f64> = image_and_roi_to_vec(image_width, image_height, image_data, roi).iter().map(|f|{ *f as f64 / 255.0f64 }).collect();
		let mut tensor = tch::Tensor::of_slice(face.as_slice()).view([1, 1, IMAGE_HEIGHT as i64, IMAGE_WIDTH as i64]);
		
		// Embed the extracted face:
		let embedding:Vec<f64> = Vec::<f64>::from(self.model.forward(&tensor));
		let mut embedding_magnitude = 0f64;
		for i in 0..LATENT_SIZE {
			embedding_magnitude += (embedding[i]*embedding[i]) as f64;
		}
		
		// Calc cosine product with all expressions.
		for xpr in &self.expressions {
			let mut similarity = 0f64;
			let mut magnitude = 0.0f64;
			for (idx, a) in xpr.latent_point.iter().enumerate() {
				let a = *a;
				let b = embedding[idx];
				magnitude += a*a;
				similarity += a*b;
			}
			expression_list.insert(xpr.name.clone(), (similarity/(magnitude*embedding_magnitude)) as f32);
		}
		
		expression_list
	}
}

fn image_and_roi_to_vec(image_width:u32, image_height:u32, image_data:&Vec<u8>, roi:(u32,u32,u32,u32)) -> Vec<u8> {
	let mut result:Vec<u8> = vec![];
	
	// Calculate the mapping from x/y in the final image to x/y in the ROI of the source image.
	// x goes from 0 -> IMAGE_WIDTH.  We want it to go from 0 -> ROI_w.
	// x * ROI_w/IMAGE_WIDTH + roi_x
	let x_to_src = roi.2 as f32 / IMAGE_WIDTH as f32;
	let y_to_src = roi.3 as f32 / IMAGE_HEIGHT as f32;
	
	for y in 0..IMAGE_HEIGHT {
		for x in 0..IMAGE_WIDTH {
			let src_x = (x as f32 * x_to_src) as usize + roi.0 as usize;
			let src_y = (y as f32 * y_to_src) as usize + roi.1 as usize;
			result.push(image_data[src_x + src_y*image_width as usize]);
		}
	}
	
	result
}
