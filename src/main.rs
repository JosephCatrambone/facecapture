mod expression_detector;
mod face_detector;
mod integral_image;

use macroquad::*;
use macroquad::Image as MQImage;
//use camera_capture::image::{ImageBuffer, Rgb, ConvertBuffer};
use std::str::FromStr;
use std::sync::mpsc::{channel, Sender, Receiver};

const COLOR_CHANNELS:usize = 4;

#[macroquad::main("Title!")]
async fn main() {
	debug!("Starting...");
	
	let cli_args = std::env::args();
	
	let camera_resolution = parse_pair::<usize>("640x480", 'x').unwrap();
	
	// Set up the camera capture thread.
	let (sender, receiver) = channel();

	// Spawn off an expensive computation
	let camera_thread = std::thread::spawn(move|| {
		//sender.send(expensive_computation()).unwrap();
		let cam_cap = camera_capture::create(0);
		if let Err(e) = cam_cap {
			eprintln!("Failed to create capture builder: {}", e);
			std::process::abort();
		}
		let frame_iter = cam_cap.unwrap()
			.resolution(camera_resolution.0 as u32, camera_resolution.1 as u32).unwrap()
			.fps(30.0).unwrap()
			.start();
		if let Err(e) = frame_iter {
			eprintln!("Unable to fetch image iterator from camera: {}", e);
			std::process::abort();
		}
		let frame_iter = frame_iter.unwrap();
		for f in frame_iter {
			if sender.send(f).is_err() {
				break;
			}
		}
	});
	
	// We'll reuse this simple image for drawing the camera output.
	let mut image = MQImage {
		width: camera_resolution.0 as u16,
		height: camera_resolution.1 as u16,
		bytes: vec![0u8; camera_resolution.0*camera_resolution.1*COLOR_CHANNELS]
	};
	let mut texture = load_texture_from_image(&image); // GPU alloc.
	
	// Main application loop.
	loop {
		clear_background(RED);
		
		// Maybe update a frame.
		if let Ok(mut img) = receiver.try_recv() {
			for y in 0..camera_resolution.1 {
				for x in 0..camera_resolution.0 {
					let i = (x + y * camera_resolution.0) * COLOR_CHANNELS;
					let pxl = img.get_pixel(x as u32, y as u32).data;
					image.bytes[i + 0] = pxl[0];
					image.bytes[i + 1] = pxl[1];
					image.bytes[i + 2] = pxl[2];
					image.bytes[i + 3] = 255u8; // A
				}
			}
			update_texture(texture, &image);
		}
		
		draw_texture(
			texture,
			screen_width() / 2. - texture.width() / 2.,
			screen_height() / 2. - texture.height() / 2.,
			WHITE,
		);
		
		next_frame().await
	}
	
	drop(receiver); // Do we need to do this?
	camera_thread.join().unwrap();
}

fn parse_pair<T>(s:&str, separator:char) -> Option<(T, T)> where T: FromStr {
	match(s.find(separator)) {
		None => None,
		Some(index) => {
			match(T::from_str(&s[..index]), T::from_str(&s[index+1..])) {
				(Ok(a), Ok(b)) => Some((a, b)),
				_ => None
			}
		}
	}
}
