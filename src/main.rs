mod expression_detector;
mod face_detector;
mod haar_like_feature;
mod integral_image;
mod decision_tree;

use expression_detector::ExpressionDetector;
use face_detector::FaceDetector;

use camera_capture::ImageIterator;
//use camera_capture::image::{ImageBuffer, Rgb, ConvertBuffer};
use image::ImageBuffer;
use macroquad as mq;
use quad_gl::Texture2D;
//use std::cell::Cell;
//use std::cmp::max;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::time::{SystemTime, UNIX_EPOCH};

const COLOR_CHANNELS:usize = 4;
const FONT_SIZE:f32 = 12f32;
const EXPRESSION_WINDOW_ID: u64 = 1;
const EXPRESSION_WINDOW_NAME_TEXT_ID: u64 = 2;
const ROI_WINDOW_ID: u64 = 3;
const ROI_WINDOW_LEFT_ID: u64 = 4;
const ROI_WINDOW_TOP_ID: u64 = 5;
const ROI_WINDOW_RIGHT_ID: u64 = 6;
const ROI_WINDOW_BOTTOM_ID: u64 = 7;
const ROI_WINDOW_DARKEN_ID: u64 = 8;

struct ApplicationState {
	camera_resolution: (u32, u32),
	focus_area: (u32, u32, u32, u32), // The relative offset from the (0,0) of the camera. X,Y,W,H
	bw_image_source: mq::Image,
	cpu_image_buffer: mq::Image,
	gpu_image: Texture2D,
	
	expression_detector: ExpressionDetector,
	
	new_expression_name: String,
	recording: bool,
	darken_roi: f32,
}

#[macroquad::main("Title!")]
async fn main() {
	dbg!("Starting...");
	
	let cli_args = std::env::args();
	
	// Set up camera capture.
	let camera_resolution = parse_pair::<usize>("640x480", 'x').unwrap();
	let mut frame_iter = open_webcam(0, (camera_resolution.0 as u32, camera_resolution.1 as u32));
	
	// Start application state.
	let mut application_state = init_application(camera_resolution);
	
	let fd: FaceDetector = serde_json::from_str(include_str!("../face_detector.json")).unwrap();
	//let fd = FaceDetector::new();
	
	// Track the number of dropped frames.
	let max_frame_time = 64;
	let max_dropped_frames:u32 = 8;
	let mut dropped_frames:u32 = 0;
	let mut rolling_average_dropped_frames = 0.0;
	
	// Compatibility between the UI toolkit and our application state.
	// UI Needs F32, we would like u32.
	let mut roi_left: f32 = 0f32;
	let mut roi_top: f32 = 0f32;
	let mut roi_right: f32 = 0f32;
	let mut roi_bottom: f32 = 0f32;
	
	// Main application loop.
	loop {
		mq::clear_background(mq::WHITE);
		
		let frame_start_time = SystemTime::now();
		
		// Maybe update a frame.
		update_capture_data(&mut application_state, &mut frame_iter);
		
		// Draw frame to screen.
		draw_capture_frame(&mut application_state);
		
		// Draw the ROI setup.
		mq::draw_window(
			ROI_WINDOW_ID,
			mq::vec2(10f32, 10f32),
			mq::vec2(400f32, 400f32),
			mq::WindowParams {
				..Default::default()
			},
			|ui| {
				// ui.tree_node
				application_state.new_expression_name.clear();
				ui.slider(ROI_WINDOW_LEFT_ID, "ROI LEFT", 0f32..application_state.camera_resolution.0 as f32, &mut roi_left);
				ui.slider(ROI_WINDOW_TOP_ID, "ROI TOP", 0f32..application_state.camera_resolution.1 as f32, &mut roi_top);
				ui.slider(ROI_WINDOW_RIGHT_ID, "ROI RIGHT", roi_left..application_state.camera_resolution.0 as f32, &mut roi_right);
				ui.slider(ROI_WINDOW_BOTTOM_ID, "ROI BOTTOM", roi_top..application_state.camera_resolution.1 as f32, &mut roi_bottom);
				application_state.focus_area = (roi_left as u32, roi_top as u32, (roi_right-roi_left) as u32, (roi_bottom-roi_top) as u32);
				ui.separator();
				ui.slider(ROI_WINDOW_DARKEN_ID, "Shade ROI", 0f32..1f32, &mut application_state.darken_roi);
			}
		);
		
		// Draw expression magnitudes and offer new captures.
		mq::draw_window(
			EXPRESSION_WINDOW_ID,
			mq::vec2(0f32, 0f32),
			mq::vec2(300f32, 600f32),
			mq::WindowParams{
				label: "Expresions".to_string(),
				..Default::default()
			},
			|ui| {
				if !application_state.recording {
					ui.input_field(EXPRESSION_WINDOW_NAME_TEXT_ID, "", &mut application_state.new_expression_name);
					if ui.button(None, "+ Add Expression") {
						capture_expression(&mut application_state);
						application_state.new_expression_name.clear();
					}
					ui.separator();
				}
			}
		);
		
		// Draw detected faces.
		/*
		let faces = fd.detect_face(camera_resolution.0, camera_resolution.1, &application_state.bw_image_source.bytes);
		for f in faces {
			let x_offset = screen_width() / 2.0 - application_state.gpu_image.width() / 2.0;
			let y_offset = screen_height() / 2.0 - application_state.gpu_image.height() / 2.0;
			let intensity = 1u8 + (240f32 * f.confidence) as u8;
			draw_rectangle_lines(f.x as f32 + x_offset + 2f32, f.y as f32 + y_offset + 2f32, f.width as f32 - 2f32, f.height as f32 - 2f32, 1.0f32, Color([0, 255, intensity, intensity]));
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
		mq::draw_text(format!("Avg dropped frames: {}", rolling_average_dropped_frames).as_str(), 10f32, 10f32, FONT_SIZE, mq::WHITE);
		dropped_frames = 0;
		
		mq::next_frame().await
	}
	
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

fn update_capture_data(application_state: &mut ApplicationState, frame_iter:&mut ImageIterator) {
	if let Some(mut camera_frame) = frame_iter.next() { // receiver.try_recv() {
		let mut image = &mut application_state.cpu_image_buffer;
		let mut bw_image = &mut application_state.bw_image_source;
		for y in 0..image.height as u32 {
			for x in 0..image.width as u32 {
				let i = (x + y * image.width as u32) as usize * COLOR_CHANNELS;
				let pxl = camera_frame.get_pixel(x as u32, y as u32).0;
				image.bytes[i + 0] = pxl[0];
				image.bytes[i + 1] = pxl[1];
				image.bytes[i + 2] = pxl[2];
				image.bytes[i + 3] = 255u8; // A
				bw_image.bytes[i/COLOR_CHANNELS] = ((pxl[0] as u32 + pxl[1] as u32 + pxl[2] as u32) / 3) as u8; // TODO: Proper color interp.
			}
		}
		mq::update_texture(application_state.gpu_image, &image);
	}
}

fn draw_capture_frame(application_state: &mut ApplicationState) {
	// Draw the frame
	let texture = application_state.gpu_image;
	let x_offset = mq::screen_width() / 2.0 - texture.width() / 2.0;
	let y_offset = mq::screen_height() / 2.0 - texture.height() / 2.0;
	mq::draw_texture(texture, x_offset, y_offset, mq::WHITE);
	
	if application_state.darken_roi > 0.01f32 {
		// Draw some dark boundaries around the image's ROI.
		// Rather than draw a rectangle directly we draw dark areas _around_ the region of interest.
		let shade_color = mq::Color([0, 0, 0, (255f32*application_state.darken_roi) as u8]);
		let roi_x = application_state.focus_area.0 as f32 + x_offset;
		let roi_y = application_state.focus_area.1 as f32 + y_offset;
		let roi_w = application_state.focus_area.2 as f32;
		let roi_h = application_state.focus_area.3 as f32;
		mq::draw_rectangle(0f32, 0f32, mq::screen_width(), roi_y, shade_color); // Top
		mq::draw_rectangle(0f32, 0f32, roi_x, mq::screen_height(), shade_color); // Left boundary
		mq::draw_rectangle(roi_x + roi_w, 0f32, (mq::screen_width() - roi_w), mq::screen_height(), shade_color); // Right shade
		mq::draw_rectangle(0f32, roi_y + roi_h, mq::screen_width(), (mq::screen_height() - roi_h), shade_color); // Bottom shade
	}
}

fn init_application(camera_resolution:(usize, usize)) -> ApplicationState {
	// We'll reuse this simple image for drawing the camera output.
	let mut image = mq::Image {
		width: camera_resolution.0 as u16,
		height: camera_resolution.1 as u16,
		bytes: vec![0u8; camera_resolution.0*camera_resolution.1*COLOR_CHANNELS]
	};
	let mut bw_image = mq::Image {
		width: camera_resolution.0 as u16,
		height: camera_resolution.1 as u16,
		bytes: vec![0u8; camera_resolution.0*camera_resolution.1]
	};
	let texture = mq::load_texture_from_image(&image); // GPU alloc.
	
	ApplicationState {
		camera_resolution: (camera_resolution.0 as u32, camera_resolution.1 as u32),
		focus_area: (0, 0, 0, 0),
		bw_image_source: bw_image,
		cpu_image_buffer: image,
		gpu_image: texture,
		expression_detector: ExpressionDetector::new(),
		
		new_expression_name: "".to_string(),
		recording: false,
		darken_roi: 0.5f32,
	}
}

fn preview(application_state:&mut ApplicationState) {
	// Draw the frame
	let texture = application_state.gpu_image;
	let x_offset = mq::screen_width() / 2.0 - texture.width() / 2.0;
	let y_offset = mq::screen_height() / 2.0 - texture.height() / 2.0;
	mq::draw_texture(texture, x_offset, y_offset, mq::WHITE);
	
	mq::draw_text("PREVIEW MODE -- [W]: Setup Camera  [E]: Setup Expresion  [R]: Record", 10f32, mq::screen_height()-16f32, FONT_SIZE, mq::Color([255, 255, 255, 255]));
}

fn setup_camera(application_state:&mut ApplicationState) {
	// Draw the frame
	let texture = application_state.gpu_image;
	let x_offset = mq::screen_width() / 2.0 - texture.width() / 2.0;
	let y_offset = mq::screen_height() / 2.0 - texture.height() / 2.0;
	mq::draw_texture(texture, x_offset, y_offset, mq::WHITE);
	
	// Draw some dark boundaries around the image's ROI.
	// Rather than draw a rectangle directly we draw dark areas _around_ the region of interest.
	let shade_color = mq::Color([0, 0, 0, 100]);
	let roi_x = application_state.focus_area.0 as f32 + x_offset;
	let roi_y = application_state.focus_area.1 as f32 + y_offset;
	let roi_w = application_state.focus_area.2 as f32;
	let roi_h = application_state.focus_area.3 as f32;
	mq::draw_rectangle(0f32, 0f32, mq::screen_width(), roi_y, shade_color); // Top
	mq::draw_rectangle(0f32, 0f32, roi_x, mq::screen_height(), shade_color); // Left boundary
	mq::draw_rectangle(roi_x + roi_w, 0f32, (mq::screen_width() - roi_w), mq::screen_height(), shade_color); // Right shade
	mq::draw_rectangle(0f32, roi_y + roi_h, mq::screen_width(), (mq::screen_height() - roi_h), shade_color); // Bottom shade
	
	// Show help text.
	mq::draw_text("Setting ROI.  [LEFT MOUSE]: Top Left.  [RIGHT MOUSE]: Bottom Right.  [ESCAPE]: Back.", 16f32, mq::screen_height()-24f32, FONT_SIZE, mq::Color([255, 255, 255, 255]));
	
	// Maybe update our shaded area.
	if mq:: is_mouse_button_down(mq::MouseButton::Left) || mq::is_mouse_button_down(mq::MouseButton::Right) {
		let (x,y) = mq::mouse_position(); // X and Y are in terms of the screen size.  Remove the offset of the drawing before making them our ROI.
		let x = (x - x_offset) as u32;
		let y = (y - y_offset) as u32;
		
		if mq::is_mouse_button_down(mq::MouseButton::Left) {
			application_state.focus_area = (x, y, application_state.focus_area.2, application_state.focus_area.3);
		} else if mq::is_mouse_button_down(mq::MouseButton::Right) {
			application_state.focus_area = (application_state.focus_area.0, application_state.focus_area.1, x - application_state.focus_area.0, y - application_state.focus_area.1);
		}
	}
}

fn setup_expression(application_state:&mut ApplicationState) {
	// Draw the frame
	let texture = application_state.gpu_image;
	let x_offset = 10f32;
	let y_offset = 10f32;
	mq::draw_texture_rec(texture, x_offset, y_offset, application_state.focus_area.2 as f32, application_state.focus_area.3 as f32, application_state.focus_area.0 as f32, application_state.focus_area.1 as f32, application_state.focus_area.2 as f32, application_state.focus_area.3 as f32, mq::WHITE);
	
	// Show help text.
	mq::draw_text("[A] Add Expression, [S] Add Sample, [D] Delete Expression", 16f32, mq::screen_height()-24f32, FONT_SIZE, mq::Color([255, 255, 255, 255]));
	
	// Draw all the expressions.
	
	
	// Check for clicks on the mouse.
	
	// Check keyboard input.
	if mq::is_key_down(mq::KeyCode::Escape) {
		application_state.recording = false;
	}
}

fn capture_expression(application_state: &mut ApplicationState) {
	//let new_expr_name = application_state.new_expression_name.clone();
	
	//application_state.new_expression_name.clear();
}