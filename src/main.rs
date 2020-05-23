mod decision_tree;
mod expression_detector;
mod face_detector;
mod haar_like_feature;
mod integral_image;
mod webcam_tools;

use expression_detector::ExpressionDetector;
use face_detector::FaceDetector;

use fltk::prelude::*;
use fltk;
use image::{ImageBuffer, RgbImage};
//use std::cell::Cell;
//use std::cmp::max;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Sender, Receiver};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::path::Path;

const WINDOW_SIZE:(usize, usize) = (1920, 1080);
const COMPONENT_PADDING:i32 = 10;

fn main() {
	dbg!("Test");
	let cli_args = std::env::args();
	let camera_resolution = parse_pair::<u32>("640x480", 'x').unwrap();
	let app = fltk::app::App::default().set_scheme(fltk::app::AppScheme::Gtk);
	
	/*
	let mut wind = Window::new(100, 100, WINDOW_SIZE.0, WINDOW_SIZE.1, "FaceCapture");
	let mut frame = Frame::new(5, 5, 790, 590, "");
	frame.set_color(Color::White);
	frame.set_frame(FrameType::DownBox);
	let mut x = 0;
	let mut y = 0;
	wind.end();
	wind.show();
	frame.handle(Box::new(move |ev| {
		// println!("{:?}", ev);
		set_draw_color(Color::Red);
		set_line_style(LineStyle::Solid, 3);
		match ev {
			app::Event::Push => {
				let coords = app::event_coords();
				x = coords.0;
				y = coords.1;
				draw_point(x, y);
				true
			}
			app::Event::Drag => {
				let coords = app::event_coords();
				// println!("{:?}", coords);
				if coords.0 < 5 || coords.0 > 795 || coords.1 < 5 || coords.1 > 595 {
					return false;
				}
				draw_line(x, y, coords.0, coords.1);
				x = coords.0;
				y = coords.1;
				true
			}
			_ => false,
		}
	}));
	app.run().unwrap();
	*/
	
	let mut wind = fltk::window::Window::default()
		.with_size(WINDOW_SIZE.0 as i32, WINDOW_SIZE.1 as i32)
		.center_screen()
		.with_label("Face Capture 2020");
	
	let mut image_frame = fltk::frame::Frame::default()
		.with_size(camera_resolution.0 as i32, camera_resolution.1 as i32)
		.center_of(&wind);
		//.with_label("0");
	
	// Expressions section.
	let mut expression_frame = fltk::frame::Frame::default()
		.with_size(300, 800)
		.right_of(&image_frame, COMPONENT_PADDING)
		.set_frame(FrameType::DownBox);
	
	// Animation control buttons.
	let mut animation_control = fltk::group::Group::default()
		.with_size(camera_resolution.0 as i32, 50)
		.below_of(&image_frame, 10);
	let mut current_frame = fltk::valuator::Counter::default()
		.with_size(250, 50)
		.below_of(&image_frame, 10);
	let mut record_button = fltk::button::Button::default()
		.with_label("@circle")
		.with_size(50, 50)
		.right_of(&current_frame, 250 + 10);
	let mut new_take = fltk::button::Button::default()
		//.size_of(&frame)
		.with_size(100, 50)
		.with_label("New Take")
		.right_of(&record_button, 100 + 10);
	animation_control.end();
	
	wind.make_resizable(false);
	wind.end();
	wind.show();
	
	/* Set up writing to frame from another thread. */
	let mut app_frame = Arc::from(Mutex::from(image_frame));
	
	/* Event handling */
	//but_inc.set_callback
	/*
	image_frame.handle(Box::new(move |ev| {
		// println!("{:?}", ev);
		set_draw_color(Color::Red);
		set_line_style(LineStyle::Solid, 3);
		match ev {
			app::Event::Push => {
				let coords = app::event_coords();
				x = coords.0;
				y = coords.1;
				draw_point(x, y);
				true
			}
			app::Event::Drag => {
				let coords = app::event_coords();
				// println!("{:?}", coords);
				if coords.0 < 5 || coords.0 > 795 || coords.1 < 5 || coords.1 > 595 {
					return false;
				}
				draw_line(x, y, coords.0, coords.1);
				x = coords.0;
				y = coords.1;
				true
			}
			_ => false,
		}
	}));
	*/
	
	{
		//let mut frame = Arc::clone(frame);
		let mut frame:Arc<Mutex<fltk::frame::Frame>> = app_frame.clone();
		std::thread::spawn(move || {
			dbg!("Opening iterator.");
			loop {
				let mut iterator = webcam_tools::open_webcam(0, camera_resolution);
				for img in iterator {
					let mut frame = frame.lock().unwrap();
					let img = RgbImage::from(img);
					let fimg:fltk::image::RgbImage = fltk::image::RgbImage::new(&img.to_vec(), img.width() as i32, img.height() as i32, 3);
					frame.set_image(&fimg);
					//frame.set_label("Foo!");
					std::thread::sleep(Duration::from_millis(10));
					//std::thread::yield_now();
				};
				eprintln!("Iterator wrapped up.  Trying to reopen camera.");
			}
		});
	}
	
	app.run().unwrap();
	dbg!("Done with main.");
}

//let fd: FaceDetector = serde_json::from_str(include_str!("../face_detector.json")).unwrap();
//let fd = FaceDetector::new();


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