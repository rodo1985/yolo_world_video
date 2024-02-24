# YOLO-World Video Processing on Hugging Face Spaces

Welcome to our YOLO-World video processing project on Hugging Face Spaces! This project leverages the power of YOLO (You Only Look Once) models for efficient and accurate real-time object detection in videos. Our interface allows users to upload videos, select object detection parameters, and visualize the processed video with detected objects highlighted.

## Features

- **Video Upload:** Users can upload their videos for object detection.
- **Model Selection:** Choose between different YOLO models (`yolov8s-world.pt`, `yolov8m-world.pt`, `yolov8l-world.pt`) for varying levels of accuracy and processing speed.
- **Custom Object Detection:** Enter specific categories for detection to tailor the model to your needs.
- **Adjustable Confidence and IoU Thresholds:** Fine-tune the detection sensitivity and intersection-over-union thresholds for optimal accuracy.
- **Real-Time Progress:** Track the processing progress with a real-time progress bar.

## How It Works

1. **Upload a Video:** Begin by uploading a video file that you want to process.
2. **Set Parameters:** Enter the categories you're interested in detecting (comma-separated), select a YOLO model, and adjust the confidence and IoU thresholds as needed.
3. **Process Video:** Click the "Process video" button to start the object detection process. The system will analyze each frame of the video, detect objects based on your parameters, and highlight them.
4. **View Results:** Once processing is complete, the output video will be displayed, showing the detected objects with bounding boxes.

## Technologies Used

- **Gradio:** For creating the interactive web interface.
- **YOLO (You Only Look Once):** For real-time object detection.
- **OpenCV:** For video processing and rendering.
- **Hugging Face Spaces:** Hosting the interactive application.

## Try It Out

Ready to see YOLO-World in action? [Visit our Hugging Face Space](#) to start detecting objects in your videos!

## Local Setup (Optional)

If you prefer to run this project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-repo/yolo-world-space.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Open your web browser and navigate to the URL displayed in your terminal.

## Contribute

We welcome contributions! Whether it's improving the detection algorithm, enhancing the interface, or fixing bugs, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- The YOLO authors for their groundbreaking work in real-time object detection.
- The Gradio and Hugging Face teams for their amazing tools that make deploying AI apps easier.

---

Happy Detecting!

