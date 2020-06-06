
use std::io::Cursor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tract_core;
use tract_ndarray;
use tract_onnx;
use tract_onnx::prelude::*;

// Keep these compatible with the expression detector.
const LATENT_SIZE:usize = 1024;
const DETECTOR_WIDTH:usize = 48;
const DETECTOR_HEIGHT:usize = 48;

#[derive(Serialize, Deserialize)]
struct Expression {
	name: String,
	latent_point: Vec<f32>,
}

pub struct ExpressionDetector {
	model: SimplePlan<tract_core::model::fact::TypedFact, Box<dyn tract_core::ops::TypedOp>, tract_core::model::model::ModelImpl<tract_core::model::fact::TypedFact, std::boxed::Box<dyn tract_core::ops::TypedOp>>>,
	expressions: Vec<Expression>,
	//face_profile: DMatrix<f32>,
	// low-dim = sigma_inv * U_trans * q
}

impl ExpressionDetector {
	pub fn new() -> Self {
		let mut static_model_data = include_bytes!("../ml/expression_detector_cpu.onnx");
		let mut cursor = Cursor::new(static_model_data);
		let m = onnx()
			.model_for_read(&mut cursor).unwrap()
			.with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1, DETECTOR_HEIGHT, DETECTOR_WIDTH))).unwrap()
			.into_optimized().unwrap()
			.into_runnable().unwrap();
		
		ExpressionDetector {
			model: m,
			expressions: vec![]
		}
	}
	
	pub fn add_expression(&mut self, image_width:u32, image_height:u32, image_data:&Vec<u8>, roi:(u32, u32, u32, u32), name:String) {
		// Cut the region out, resize it to our face detector.
		let mut face = image_and_roi_to_tensor(image_width, image_height, image_data, roi);
		
		// Calculate embedding.
		let embedding = (self.model.run(tvec!(face)) as TractResult<()>).unwrap()[0].to_array_view::<f32>().unwrap();
		
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
		let mut face = image_and_roi_to_tensor(image_width, image_height, image_data, roi);
		
		// Embed the extracted face:
		let embedding = (self.model.run(tvec!(face)) as TractResult<()>).unwrap()[0].to_array_view::<f32>().unwrap();
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
	let x_to_src = roi.2 as f32 / DETECTOR_WIDTH as f32;
	let y_to_src = roi.3 as f32 / DETECTOR_HEIGHT as f32;
	
	for y in 0..DETECTOR_HEIGHT {
		for x in 0..DETECTOR_WIDTH {
			let src_x = (x as f32 * x_to_src) as usize + roi.0 as usize;
			let src_y = (y as f32 * y_to_src) as usize + roi.1 as usize;
			result.push(image_data[src_x + src_y*image_width as usize]);
		}
	}
	
	result
}

// Given a source image of the form w,h,data and an roi with (x, y, w, h), extract a tensor.
fn image_and_roi_to_tensor(image_width:u32, _image_height:u32, image_data:&Vec<u8>, roi:(u32, u32, u32, u32)) -> Tensor {
	let x_to_src = roi.2 as f32 / DETECTOR_WIDTH as f32;
	let y_to_src = roi.3 as f32 / DETECTOR_HEIGHT as f32;
	tract_ndarray::Array4::from_shape_fn((1, 1, DETECTOR_HEIGHT, DETECTOR_WIDTH), |(_b, c, y, x)| {
		let src_x = (x as f32 * x_to_src) as usize + roi.0 as usize;
		let src_y = (y as f32 * y_to_src) as usize + roi.1 as usize;
		image_data[src_x + src_y*image_width as usize] as f32 / 255.0f32
	}).into()
}
