
pub struct IntegralImage {
	pub width: usize,
	pub height: usize,
	pub integral_data: Vec::<u32>,
}

impl IntegralImage {
	pub fn new_from_image_data(width: usize, height: usize, pixel_data: &Vec<u8>) -> Self {
		IntegralImage::new_from_image_data_subsampled(width, height, pixel_data, width, height)
	}
	
	pub fn new_from_image_data_subsampled(width: usize, height: usize, pixel_data: &Vec<u8>, target_width:usize, target_height:usize) -> Self {
		assert_eq!(width*height, pixel_data.len());
		let x_step = width/target_width;
		let y_step = height/target_height;
		let mut integral_data = vec![0u32; target_width*target_height];
		for y in 0..target_height {
			for x in 0..target_width {
				let original_index = (x*x_step) + ((y*y_step)*width);
				let new_index = x + y*target_width;
				
				if x > 0 {
					integral_data[new_index] += integral_data[new_index-1];
				}
				if y > 0 {
					integral_data[new_index] += integral_data[new_index-target_width];
				}
				if x > 0 && y > 0 {
					integral_data[new_index] -= integral_data[(new_index-1)-target_width];
				}
				integral_data[new_index] += pixel_data[original_index] as u32;
			}
		}
		IntegralImage {
			width:target_width,
			height:target_height,
			integral_data,
		}
	}
	
	// THIS PERFORMS A COPY!  TODO: Make a 'view' method or something.
	pub fn new_from_region(&self, x0:usize, y0:usize, x1:usize, y1:usize) -> Self {
		let mut integral_data = Vec::<u32>::with_capacity((x1-x0)*(y1-y0));
		for y in y0..y1 {
			for x in x0..x1 {
				integral_data.push(self.integral_data[x + y*self.width]);
			}
		}
		IntegralImage {
			width: x1-x0,
			height: y1-y0,
			integral_data,
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
		assert!(x1 >= x0 && y1 >= y0 && x1 < self.width && y1 < self.height);
		let bottom_right = self.integral_data[x1 + (y1*self.width)];
		let left = if x0 == 0 { 0 } else { self.integral_data[(x0-1) + (y1*self.width)] };
		let top = if y0 == 0 { 0 } else { self.integral_data[x1 + ((y0-1)*self.width)]};
		let top_left = if x0 == 0 || y0 == 0 { 0 } else { self.integral_data[(x0-1) + (y0-1)*self.width] };
		((bottom_right + top_left) as i32 - left as i32) - top as i32
	}
}

#[cfg(test)]
mod tests {
	// Note this useful idiom: importing names from outer (for mod tests) scope.
	use super::*;
	
	#[test]
	fn integral_sanity() {
		let img = IntegralImage::new_from_image_data(5, 4, &vec![
			1, 0, 1, 0, 0,
			2, 0, 0, 0, 0,
			3, 0, 1, 0, 0,
			0, 0, 0, 0, 1
		]);
		assert_eq!(img.get_region_sum(0, 0, 4, 3), 1 + 1 + 2 + 3 + 1 + 1);
		assert_eq!(img.get_region_sum(0, 0, 1, 1), 1 + 2);
		assert_eq!(img.get_region_sum(2, 0, 2, 2), 1 + 1);
		assert_eq!(img.get_region_sum(0, 2, 2, 2), 3 + 1);
		assert_eq!(img.integral_data, vec![
			1, 1, 2, 2, 2,
			3, 3, 4, 4, 4,
			6, 6, 8, 8, 8,
			6, 6, 8, 8, 9
		]);
		
		let img = IntegralImage::new_from_image_data(12, 10, &vec![
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
		]);
		assert_eq!(img.get_region_sum(0, 2, 1, 5), 0);
		assert_eq!(img.get_region_sum(2, 2, 3, 6), 10);
		assert_eq!(img.get_region_sum(4, 2, 5, 6), 2);
	}
}