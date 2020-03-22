mod expression_detector;
mod face_detector;
mod integral_image;
mod decision_tree;

use macroquad::*;
use macroquad::Image as MQImage;
//use std::cmp::max;
//use camera_capture::image::{ImageBuffer, Rgb, ConvertBuffer};
use std::str::FromStr;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::time::{SystemTime, UNIX_EPOCH};

use face_detector::FaceDetector;

const COLOR_CHANNELS:usize = 4;

#[macroquad::main("Title!")]
async fn main() {
	debug!("Starting...");
	
	let fd: FaceDetector = serde_json::from_str(include_str!("../face_detector.json")).unwrap();
	
	let cli_args = std::env::args();
	
	let camera_resolution = parse_pair::<usize>("640x480", 'x').unwrap();
	
	// Track the number of dropped frames.
	let max_frame_time = 64;
	let max_dropped_frames = 8;
	let mut dropped_frames = 0;
	let mut rolling_average_dropped_frames = 0.0;
	
	// Set up the camera capture thread.
	// Using sync channel instead of channel so we block when the buffer is full.
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
	let mut bw_image = MQImage {
		width: camera_resolution.0 as u16,
		height: camera_resolution.1 as u16,
		bytes: vec![0u8; camera_resolution.0*camera_resolution.1]
	};
	let texture = load_texture_from_image(&image); // GPU alloc.
	
	// Main application loop.
	loop {
		clear_background(RED);
		
		let frame_start_time = SystemTime::now();
		
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
					bw_image.bytes[i/COLOR_CHANNELS] = ((pxl[0] as u32 + pxl[1] as u32 + pxl[2] as u32) / 3) as u8; // TODO: Proper color interp.
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
		
		// Draw detected faces.
		let faces = fd.detect_face(camera_resolution.0, camera_resolution.1, &bw_image.bytes);
		for f in faces {
			let x_offset = screen_width() / 2.0 - texture.width() / 2.0;
			let y_offset = screen_height() / 2.0 - texture.height() / 2.0;
			let intensity = 8u8 + (240f32 * f.confidence * f.confidence) as u8;
			draw_rectangle_lines(f.x as f32 + x_offset, f.y as f32 + y_offset, f.width as f32, f.height as f32, Color([0, 255, intensity, intensity]));
		}
		
		// Count the number of frames we've done and drop them if we've taken too long.
		let frame_time = frame_start_time.elapsed();
		if let Ok(dt) = frame_time {
			let mut ft = dt.as_millis();
			while ft >= max_frame_time {
				let _ = receiver.try_recv();
				ft = ft - max_frame_time;
				dropped_frames += 1;
			}
		}
		
		rolling_average_dropped_frames = 0.9*rolling_average_dropped_frames + 0.1*dropped_frames as f32;
		draw_text(format!("Avg dropped frames: {}", rolling_average_dropped_frames).as_str(), 10f32, 10f32, 16f32, WHITE);
		dropped_frames = 0;
		
		next_frame().await
	}
	
	//drop(receiver); // Do we need to do this?
	//camera_thread.join().unwrap();
}

fn parse_pair<T>(s:&str, separator:char) -> Option<(T, T)> where T: FromStr {
	match s.find(separator) {
		None => None,
		Some(index) => {
			match(T::from_str(&s[..index]), T::from_str(&s[index+1..])) {
				(Ok(a), Ok(b)) => Some((a, b)),
				_ => None
			}
		}
	}
}
