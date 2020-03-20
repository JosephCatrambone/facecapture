
use nalgebra::linalg::SVD;
use nalgebra::{DMatrix, Vector, DVector};

const LATENT_SIZE:usize = 512;

/*
	let dm1 = DMatrix::from_diagonal_element(4, 3, 1.0);
    let dm2 = DMatrix::identity(4, 3);
    let dm3 = DMatrix::from_fn(4, 3, |r, c| if r == c { 1.0 } else { 0.0 });
    let dm4 = DMatrix::from_iterator(
        4,
        3,
        [
            // Components listed column-by-column.
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]
        .iter()
        .cloned(),
    );
*/

struct Expression {
	name: String,
	latent_point: [f32; LATENT_SIZE],
}

struct ExpressionGroup<'a> {
	expressions: Vec<&'a Expression>,
	mutually_exclusive: bool,  // If true, only one of these can be active at a time.
}

pub struct ExpressionDetector<'a> {
	expressions: Vec<Expression>,
	expression_groups: Vec<ExpressionGroup<'a>>,
	mean_face: Vec<f32>,
	//face_profile: DMatrix<f32>,
	// low-dim = sigma_inv * U_trans * q
}

impl<'a> ExpressionDetector<'_> {
	pub fn new_from_face_images(image_width: u32, image_height: u32, image_data:&Vec<Vec<u8>>) -> Self {
		// Given a bunch of captures of a face, compute the eigenfaces and build a dimensionality reduction matrix so we can get a latent space.
		let mut mean_face = vec![0f32; (image_width*image_height) as usize];
		for img_data in image_data {
			for (i, pixel_data) in img_data.iter().enumerate() {
				mean_face[i] += (*pixel_data) as f32 / 255f32;
			}
		}
		
		// Now that we have a bunch of face captures, build a matrix from them.  Each row should be a face.
		let mut face_matrix = DMatrix::from_fn(image_data.len(), (image_width*image_height) as usize, |r, c|{
			image_data[r][c] as f32/255f32 - mean_face[c]
		});
		let svd = SVD::new(face_matrix, true, true); // TODO: use try_new.
		//let u = svd.u.unwrap();
		//let s = svd.singular_values;
		//let v = svd.v_t.unwrap().transpose();
		
		ExpressionDetector {
			expressions: vec![],
			expression_groups: vec![],
			mean_face,
			//DMatrix::identity
		}
	}
}
