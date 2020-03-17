
use super::integral_image::*;

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
		assert!(x0 >= 0 && y0 >= 0 && x1 > x0 && y1 > y0);
		let bottom_right = self.integral_data[x1 + (y1*self.width)];
		let left = if x0 == 0 { 0 } else { self.integral_data[(x0-1) + (y0*self.width)] };
		let top = if y0 == 0 { 0 } else { self.integral_data[x0 + ((y0-1)*self.width)]};
		let top_left = if x0 == 0 || y0 == 0 { 0 } else { self.integral_data[(x0-1) + (y0-1)*self.width] };
		((bottom_right + top_left) as i32 - left as i32) - top as i32
	}
}

#[cfg(test)]
mod tests {
	// Note this useful idiom: importing names from outer (for mod tests) scope.
	use super::*;
	
	#[test]
	fn sanity() {
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
}