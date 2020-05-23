use camera_capture::ImageIterator;
use std::sync::mpsc::{self, TryRecvError, Receiver};
use image::RgbImage;
use std::time::Duration;

pub fn open_webcam(i:u32, camera_resolution: (u32, u32)) -> ImageIterator {
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

pub fn open_webcam_channel(device_id:u32, camera_resolution: (u32, u32), buffer_capacity: usize) -> (Receiver<RgbImage>, std::thread::JoinHandle<()>) {
	let (mut tx, mut rx) = mpsc::sync_channel(buffer_capacity);
	let thread = std::thread::spawn(move ||{
		let mut iterator = open_webcam(device_id, camera_resolution);
		for img in iterator {
			dbg!("Got image.");
			let rgb_img = RgbImage::from(img);
			dbg!("Made into RGB.");
			if tx.send(rgb_img).is_err() {
				panic!("Failed to push RGB Image into channel.");
			}
			dbg!("Sent.");
			std::thread::sleep(Duration::from_millis(10));
		}
	});
	std::thread::sleep(Duration::from_secs(1)); // One second hack to open camera before returning channel.
	(rx, thread)
}