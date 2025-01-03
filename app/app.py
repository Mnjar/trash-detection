import gradio as gr
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('app/trash_detection.pt')  # Ganti dengan path model YOLO Anda

def predict(image):
    """
    Function to make predictions using YOLO model.
    Args:
        image (PIL.Image): Input image.
    Returns:
        List[List]: Predictions with labels, confidence, and bounding boxes.
    """
    # Convert PIL image to numpy array
    img = np.array(image)

    # Get predictions from the model
    results = model(img)

    # Extract predictions
    predictions = results.pandas().xyxy[0]  # Pandas DataFrame of predictions
    # Select necessary columns and convert to list
    output = predictions[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

    return output

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),  # Input image as PIL
    outputs=gr.Dataframe(
        headers=["Label", "Confidence", "Xmin", "Ymin", "Xmax", "Ymax"],
        label="Predictions"
    ),
    title="YOLO Object Detection",
    description="Upload an image to detect objects using YOLO."
)

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)
