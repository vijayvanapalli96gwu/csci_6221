use opencv::{
    prelude::*,
    highgui,
    imgproc,
    core,
    videoio,
};
use anyhow::{bail, Result};
use tch::nn::{self, OptimizerConfig};
use tch::vision::{imagenet, resnet};
use std::convert::TryFrom;

pub fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    println!("{:?}", args);
    let (weights, dataset_dir) = match args.as_slice() {
        [_, w, d] => (std::path::Path::new(w), d.to_owned()),
        _ => bail!("usage: main resnet18.ot dataset-path"),
    };
    // Load the dataset and resize it to the usual imagenet dimension of 224x224.
    let dataset = imagenet::load_from_dir(dataset_dir)?;
    println!("{dataset:?}");

    // Create the model and load the weights from the file.
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net = resnet::resnet18_no_final_layer(&vs.root());
    vs.load(weights)?;

    // Pre-compute the final activations.
    let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
    let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false));

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
    let mut sgd = nn::Sgd::default().build(&vs, 1e-3)?;

    for epoch_idx in 1..1000 {
        let predicted = train_images.apply(&linear);
        let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
        sgd.backward_step(&loss);

        let test_accuracy = test_images.apply(&linear).accuracy_for_logits(&dataset.test_labels);
        println!("{} {:.2}%", epoch_idx, 100. * f64::try_from(test_accuracy)?);
        //println!("Success");
        //println!("{} {}%", epoch_idx, 100. * f64::try_from(predicted.get1(e))?);
    }

    // vs.save("asl.pt").unwrap();
	// Load the image file and resize it to the usual imagenet dimension of 224x224.
	// let image = imagenet::load_image_and_resize224(r"C:\Users\jithe\OneDrive\Desktop\ASL\asl\mini\signdata\val\hi\positive_287.png")?
	// 	.to_device(vs.device());

    // Testing on a single image after training.
    let test_img = imagenet::load_image_and_resize224(r"C:\Users\jithe\OneDrive\Desktop\asp\dataset\val\victory\victory_18.png")?;
    let test_output = tch::no_grad(|| test_img.unsqueeze(0).apply_t(&net, false));
    let predicted_1 = test_output.apply(&linear);
    let predicted_index = predicted_1.argmax(-1, true);
    
    // Map the predicted index to the corresponding class label.
    let class_labels = vec!["C","hi","victory"];
    let num_classes = class_labels.len();
    let predicted_class = class_labels[predicted_index.int64_value(&[]) as usize % num_classes];


    // Print the predicted class.
    println!("Predicted class for the test image: {:?}", predicted_class);

    let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;  // 0 is the default camera
    let win_name = "frame";
    highgui::named_window(win_name, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(win_name, 640, 480)?;
    let mut frame = Mat::default();
    let is_video_on = capture.is_opened()?;
    const IMG_WIDTH: i32 = 32; 
    const IMG_HEIGHT: i32 = 32; 
    const DELAY: i32 = 100;

    loop{
        let key = highgui::wait_key(DELAY)?; 

        capture.read(&mut frame)?;
        let mut resized = Mat::default();   
        imgproc::resize(&frame, &mut resized, core::Size{width: IMG_WIDTH, height: IMG_HEIGHT}, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;
        let mut rgb_resized = Mat::default();  
        imgproc::cvt_color(&resized, &mut rgb_resized, imgproc::COLOR_BGR2RGB, 0)?;    
        // get data from Mat 
        let h = resized.size()?.height;
        let w = resized.size()?.width;   
        let resized_data = resized.data_bytes_mut()?; 
        // convert bytes to tensor
        let tensor = tch::Tensor::from_data_size(resized_data, &[h as i64, w as i64, 3], tch::Kind::Uint8);  
        // normalize image tensor
        let tensor = tensor.to_kind(tch::Kind::Float) / 255;
        // carry tensor to cuda
        let tensor = tensor.to_device(tch::Device::Cpu); 
        // convert (H, W, C) to (C, H, W)
        let tensor = tensor.permute(&[2, 0, 1]); 
        // add batch dim (convert (C, H, W) to (N, C, H, W)) 
        // let normalized_tensor = tensor.unsqueeze(0); 
        let normalized_tensor = tch::no_grad(||tensor.unsqueeze(0).apply_t(&net, false));   
      
        let class_labels = vec!["C","hi","victory"];
        let predicted_1 = normalized_tensor.apply(&linear);
        let predicted_index = predicted_1.argmax(-1, true);
        let num_classes = class_labels.len();
        let predicted_class = class_labels[predicted_index.int64_value(&[]) as usize % num_classes];


        // Print the predicted class.
        println!("Predicted class for the test image: {:?}", predicted_class);
        let text = predicted_class;
        // let font = opencv::imgproc::get_font(opencv::imgproc::FONT_HERSHEY_SIMPLEX, 1.0, 1, 0, 1, 0).unwrap();
        let color = core::Scalar::new(255.0, 0.0, 0.0, 0.0); // BGR color format

        // Define the position to place the text
        let org = core::Point::new(50, 50); // Adjust the coordinates as needed
        // Write text on the image
        imgproc::put_text(&mut frame, text, org, 2,2.0, color, 2, opencv::imgproc::LINE_8, false).unwrap();


        highgui::imshow(win_name, &frame)?;
        if key == 113 { 
                        highgui::destroy_all_windows()?;
                        println!("Pressed q. Aborting program.");
                        break;
                    }

    }



//     // model trained on 32x32 images. (CIFAR10)
//     const CIFAR_WIDTH: i32 = 640; 
//     const CIFAR_HEIGHT: i32 = 480; 

//     // time that a frame will stay on screen in ms
//     const DELAY: i32 = 3000;

//     // create video stream 
//     let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;  // 0 is the default camera
//      let opened = videoio::VideoCapture::is_opened(&capture)?;
//      if !opened {
//          panic!("Unable to open default camera!");
//      }

//     // create empty window named 'frame'
//     let win_name = "frame";
//     highgui::named_window(win_name, highgui::WINDOW_NORMAL)?;
//     highgui::resize_window(win_name, 640, 480)?;
    
//     // create empty Mat to store image data
//     let mut frame = Mat::default();
//     loop {
//         // read frame to empty mat 
//         capture.read(&mut frame)?;
//         // resize image
//         let mut resized = Mat::default();   
//         imgproc::resize(&frame, &mut resized, core::Size{width: CIFAR_WIDTH, height: CIFAR_HEIGHT}, 0.0, 0.0, opencv::imgproc::INTER_LINEAR)?;
//         // convert bgr image to rgb
//         let mut rgb_resized = Mat::default();  
//         imgproc::cvt_color(&resized, &mut rgb_resized, imgproc::COLOR_BGR2RGB, 0)?;    
//         // get data from Mat 
//         let h = resized.size()?.height;
//         let w = resized.size()?.width;   
//         let resized_data = resized.data_bytes_mut()?; 
//         // convert bytes to tensor
//         let tensor = tch::Tensor::from_data_size(resized_data, &[h as i64, w as i64, 3], tch::Kind::Uint8);  
//         // normalize image tensor
//         let tensor = tensor.to_kind(tch::Kind::Float) / 255;
//         // carry tensor to cuda
//         let tensor = tensor.to_device(tch::Device::Cpu); 
//         // convert (H, W, C) to (C, H, W)
//         let tensor = tensor.permute(&[2, 0, 1]); 
//         // add batch dim (convert (C, H, W) to (N, C, H, W)) 
//         let normalized_tensor = tch::no_grad(||tensor.unsqueeze(0).apply_t(&net, false));   

//         // make prediction and time it. 
//         // let start = time::Instant::now();
//         // let probabilites = model.forward_ts(&[normalized_tensor])?.softmax(-1, tch::Kind::Float);  
//         // let predicted_class = i32::try_from(probabilites.argmax(None, false));
//         // let probability_of_class = f32::try_from(probabilites.max()); 
//         // let duration = start.elapsed();
//         // println!("Predicted class: {:?}, probability of it: {:?}, prediction time: {:?}", class_map[&predicted_class], probability_of_class, duration); 
//         let class_labels = vec!["good", "hi", "victory"];
//         let predicted_1 = normalized_tensor.apply(&linear);
//         let predicted_index = predicted_1.argmax(-1, true);
//         let num_classes = class_labels.len();
//         let predicted_class = class_labels[predicted_index.int64_value(&[]) as usize % num_classes];


//         // Print the predicted class.
//         println!("Predicted class for the test image: {:?}", predicted_class);

//         // show image 
//         highgui::imshow(win_name, &frame)?;
        
//         let key = highgui::wait_key(DELAY)?; 

//         // if button q pressed, abort.
//         if key == 113 { 
//             highgui::destroy_all_windows()?;
//             println!("Pressed q. Aborting program.");
//             break;
//         }
// }

    Ok(())
}