
use std::io;
use std::io::prelude::*;
use std::io::Read;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tch;
use tch::nn::ModuleT;
use std::borrow::BorrowMut;

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
	model: tch::CModule,
	model_store: tch::nn::VarStore,
	expressions: Vec<Expression>,
	//face_profile: DMatrix<f32>,
	// low-dim = sigma_inv * U_trans * q
}

impl ExpressionDetector {
	pub fn new() -> Self {
		let mut static_model_data:Vec<u8> = include_bytes!("../ml/expression_detector_cexport_cpu.pt").to_vec();
		let m = match tch::CModule::load_data::<&[u8]>(&mut static_model_data.as_slice()) {
			Ok(model) => model,
			Err(e) => {
				dbg!(e);
				panic!("Goddamnit.");
			}
		};
		
		//let m = tch::CModule::load("./ml/expression_detector_cpu.onnx").unwrap();
		let vs = tch::nn::VarStore::new(tch::Device::Cpu);
		
		ExpressionDetector {
			model: m,
			model_store: vs,
			expressions: vec![]
		}
	}
	
	pub fn add_expression(&mut self, image_width:u32, image_height:u32, image_data:&Vec<u8>, roi:(u32, u32, u32, u32), name:String) {
		// Cut the region out, resize it to our face detector.
		let mut face = image_and_roi_to_tensor(image_width, image_height, image_data, roi);
		
		// Calculate embedding.
		let tensor: tch::Tensor = self.model.forward_t(&face, false);
		let embedding = (0..LATENT_SIZE).into_iter().map(|i| { tensor.double_value(&[0, i as i64]) as f32}).collect();
		
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
		let tensor: tch::Tensor = self.model.forward_t(&face, false);
		let embedding:Vec<f32> = (0..LATENT_SIZE).into_iter().map(|i| { tensor.double_value(&[0, i as i64]) as f32}).collect();
		let mut embedding_magnitude = 0f32;
		for i in 0..LATENT_SIZE {
			embedding_magnitude += (embedding[i]*embedding[i]);
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
fn image_and_roi_to_tensor(image_width:u32, _image_height:u32, image_data:&Vec<u8>, roi:(u32, u32, u32, u32)) -> tch::Tensor {
	//let mut result = tch::Tensor::zeros(&[1, 1, DETECTOR_HEIGHT, DETECTOR_WIDTH], (tch::Kind::Float, tch::Device::Cpu));
	let data:Vec<f32> = image_and_roi_to_vec(image_width, _image_height, image_data, roi).iter().map(|f|{ *f as f32 / 255.0f32 }).collect();
	tch::Tensor::of_slice(data.as_slice()).view([1i64, 1i64, DETECTOR_HEIGHT as i64, DETECTOR_WIDTH as i64])
}
