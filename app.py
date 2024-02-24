import utils
import gradio as gr


# constants
DEFAULT_MODEL = "yolov8l-world.pt"
VIDEO_EXAMPLES = [
    ['https://media.roboflow.com/supervision/video-examples/croissant-1280x720.mp4',
        'croissant', 0.01, 0.2],
    ['https://media.roboflow.com/supervision/video-examples/suitcases-1280x720.mp4',
        'suitcase', 0.1, 0.2],
    ['https://media.roboflow.com/supervision/video-examples/tokyo-walk-1280x720.mp4',
        'woman walking', 0.1, 0.2],
    ['https://media.roboflow.com/supervision/video-examples/wooly-mammoth-1280x720.mp4',
        'mammoth', 0.01, 0.2],
]
CACHE_EXAMPLES = False

# load the YOLO model
model = utils.yolo_model(DEFAULT_MODEL)

# CSS to hide the footer and reduce the height of the Gradio interface
css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"

with gr.Blocks(css=css) as demo:

    # title and description
    gr.Markdown("# YOLO-World video processing")
    gr.Markdown(
        "Upload a video and click the button to process it with YOLO-World")

    # define the structure of the interface
    with gr.Row():
        input_video = gr.Video(label="Input video")
        output_video = gr.Video(label="Output video")

    with gr.Row():
        # prompt and button
        prompt = gr.Textbox(
            label='Categories',
            placeholder='comma separated list of categories',
            scale=5
        )
        button = gr.Button("Process video")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                confidence = gr.Slider(
                    minimum=0, maximum=1, value=0.25, label="Confidence threshold", step=0.01)
            with gr.Row():
                iou = gr.Slider(minimum=0, maximum=1, value=0.7,
                                label="IoU threshold", step=0.01)

        with gr.Column():
            model_to_load = gr.Dropdown(["yolov8s-world.pt", "yolov8m-world.pt", "yolov8l-world.pt"],
                                        label="Model", info="Select the model to use", value=DEFAULT_MODEL)
            gr.Button("Upload model").click(model.load, inputs=[model_to_load])

    # examples
    with gr.Row():
        gr.Examples(
            examples=VIDEO_EXAMPLES,
            inputs=[input_video, prompt, confidence, iou],
            outputs=output_video,
            fn=model.process,
            cache_examples=CACHE_EXAMPLES,
        )

    # click event
    button.click(model.process, inputs=[
                 input_video, prompt, confidence, iou], outputs=output_video)
    gr.close_all()

demo.launch()
