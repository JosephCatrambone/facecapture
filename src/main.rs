mod expression_detector;
mod face_detector;
mod haar_like_feature;
mod integral_image;
mod decision_tree;

use face_detector::FaceDetector;

use camera_capture::ImageIterator;
//use camera_capture::image::{ImageBuffer, Rgb, ConvertBuffer};
use image::ImageBuffer;
use macroquad::*;
use macroquad::Image as MQImage;
use std::borrow::BorrowMut;
use std::cell::Cell;
//use std::cmp::max;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::time::{SystemTime, UNIX_EPOCH};

const COLOR_CHANNELS:usize = 4;

enum ApplicationModes {
	SetupCamera,
	SetupExpression,
	Preview,
	Recording,
}

struct ApplicationState {
	camera_resolution: (u32, u32),
	focus_area: (u32, u32, u32, u32), // The relative offset from the (0,0) of the camera.
	bw_image_source: MQImage,
	cpu_image_buffer: MQImage,
	gpu_image: Texture2D,
	mode: ApplicationModes,
}

#[macroquad::main("Title!")]
async fn main() {
	debug!("Starting...");
	
	let cli_args = std::env::args();
	
	// Set up camera capture.
	let camera_resolution = parse_pair::<usize>("640x480", 'x').unwrap();
	let mut frame_iter = open_webcam(0, (camera_resolution.0 as u32, camera_resolution.1 as u32));
	
	// Start application state.
	let mut application_state = Cell::new(init_application(camera_resolution));
	
	let fd: FaceDetector = serde_json::from_str(include_str!("../face_detector.json")).unwrap();
	//let fd = FaceDetector::new();
	
	// Track the number of dropped frames.
	let max_frame_time = 64;
	let max_dropped_frames:u32 = 8;
	let mut dropped_frames:u32 = 0;
	let mut rolling_average_dropped_frames = 0.0;
	
	// Main application loop.
	loop {
		clear_background(RED);
		
		let frame_start_time = SystemTime::now();
		
		// Maybe update a frame.
		update_capture_data(&mut application_state, &mut frame_iter);
		
		// Draw frame to screen.
		draw_capture_frame(&mut application_state);
		
		// Draw detected faces.
		/*
		let faces = fd.detect_face(camera_resolution.0, camera_resolution.1, &bw_image.bytes);
		for f in faces {
			let x_offset = screen_width() / 2.0 - texture.width() / 2.0;
			let y_offset = screen_height() / 2.0 - texture.height() / 2.0;
			let intensity = 1u8 + (240f32 * f.confidence) as u8;
			draw_rectangle_lines(f.x as f32 + x_offset + 2f32, f.y as f32 + y_offset + 2f32, f.width as f32 - 2f32, f.height as f32 - 2f32, Color([0, 255, intensity, intensity]));
			//draw_rectangle(f.x as f32 + x_offset + 2f32, f.y as f32 + y_offset + 2f32, f.width as f32 - 2f32, f.height as f32 - 2f32, Color([0, 255, intensity, intensity]));
		}
		*/
		
		// Count the number of frames we've done and drop them if we've taken too long.
		let frame_time = frame_start_time.elapsed();
		if let Ok(dt) = frame_time {
			let mut ft = dt.as_millis();
			while ft >= max_frame_time {
				//let _ = receiver.try_recv();
				ft = ft - max_frame_time;
				dropped_frames += 1;
			}
		}
		
		rolling_average_dropped_frames = 0.9*rolling_average_dropped_frames + 0.1*dropped_frames as f32;
		draw_text(format!("Avg dropped frames: {}", rolling_average_dropped_frames).as_str(), 10f32, 10f32, 16f32, WHITE);
		dropped_frames = 0;
		
		next_frame().await
	}

	//camera_thread.join().unwrap();
}

fn open_webcam(i:u32, camera_resolution: (u32, u32)) -> ImageIterator {
	// We had initially put this on a separate thread, but the ESCAPI driver does not behave nicely when multithreaded.
	
	let cam_cap = camera_capture::create(i);
	
	if let Err(e) = cam_cap {
		eprintln!("Failed to create capture builder: {}", e);
		std::process::abort();
	}
	
	let frame_iter = cam_cap.unwrap()
		.resolution(camera_resolution.0, camera_resolution.1).unwrap()
		.fps(30.0).unwrap()
		.start();
	
	if let Err(e) = frame_iter {
		eprintln!("Unable to fetch image iterator from camera: {}", e);
		std::process::abort();
	}
	let mut frame_iter = frame_iter.unwrap();
	
	frame_iter
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

fn update_capture_data(application_state: &mut Cell<ApplicationState>, frame_iter:&mut ImageIterator) {
	if let Some(mut camera_frame) = frame_iter.next() { // receiver.try_recv() {
		let mut astate = application_state.get_mut();
		let mut image = &mut astate.cpu_image_buffer;
		let mut bw_image = &mut astate.bw_image_source;
		for y in 0..image.height as u32 {
			for x in 0..image.width as u32 {
				let i = (x + y * image.width as u32) as usize * COLOR_CHANNELS;
				let pxl = camera_frame.get_pixel(x as u32, y as u32).data;
				image.bytes[i + 0] = pxl[0];
				image.bytes[i + 1] = pxl[1];
				image.bytes[i + 2] = pxl[2];
				image.bytes[i + 3] = 255u8; // A
				bw_image.bytes[i/COLOR_CHANNELS] = ((pxl[0] as u32 + pxl[1] as u32 + pxl[2] as u32) / 3) as u8; // TODO: Proper color interp.
			}
		}
		update_texture(astate.gpu_image, &image);
	}
}

fn draw_capture_frame(application_state: &mut Cell<ApplicationState>) {
	let texture = application_state.get_mut().gpu_image;
	draw_texture(
		texture,
		screen_width() / 2. - texture.width() / 2.,
		screen_height() / 2. - texture.height() / 2.,
		WHITE,
	);
}

fn init_application(camera_resolution:(usize, usize)) -> ApplicationState {
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
	
	ApplicationState {
		camera_resolution: (camera_resolution.0 as u32, camera_resolution.1 as u32),
		focus_area: (0, 0, 0, 0),
		bw_image_source: bw_image,
		cpu_image_buffer: image,
		gpu_image: texture,
		mode: ApplicationModes::Preview
	}
}

fn preview() {

}

fn setup_camera() {

}

fn recording() {

}

fn expression_setup() {

}
