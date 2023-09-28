# - Importing the dependencies
from ultralytics import YOLO
from PIL import Image, ImageFilter, ImageDraw
import io
import streamlit as st


# - CSS Styling
st.markdown(
    """
    <style>
    .centered-heading {
        text-align: center;
        padding-bottom: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# - Defining a function to load the YOLO model
@st.cache_data()
def load_model():
    return YOLO('human_face_detection_model.pt')


# - Loading the pre-trained YOLOv8l model
model = load_model()

        
# - Defining a function to perform object detection and blur faces
def perform_object_detection_and_blur(uploaded_image, model, image_index):
    img_bytes = uploaded_image.read()
    image = Image.open(io.BytesIO(img_bytes))
    results = model(source=image)

    # - Creating a copy of the original image for blurring
    blurred_image = image.copy()
    boxes = []

    for result in results:
        boxes += result.boxes

    # - Displaying the image with bounding boxes
    display_detected_objects(image, boxes, image_index)

    # - Bluring and saving each detected face
    for j, box in enumerate(boxes):
        x_center, y_center, box_width, box_height = box.xywh[0]
        x1, y1, x2, y2 = calculate_coordinates(box)

        mask = create_mask(image, x1, y1, x2, y2)
        apply_blur_to_face(blurred_image, x1, y1, x2, y2)

    save_and_display_blurred_image(blurred_image, boxes, image_index)
    

# - Defining a function to display the original image with bounding boxes
def display_detected_objects(image, boxes, image_index):
    im_array = image.copy()
    for box in boxes:
        draw = ImageDraw.Draw(im_array)
        draw.rectangle(calculate_coordinates(box), outline="red", width=2)
    st.image(im_array, caption=f"Human Face Detection Result - Image {image_index + 1}", use_column_width=True)


# - Defining a function to calculate the coordinates of a bounding box
def calculate_coordinates(box):
    x_center, y_center, box_width, box_height = box.xywh[0]
    x1 = int(x_center - (box_width / 2))
    y1 = int(y_center - (box_height / 2))
    x2 = int(x_center + (box_width / 2))
    y2 = int(y_center + (box_height / 2))
    return x1, y1, x2, y2


# - Defining a function to create a mask for the face region
def create_mask(image, x1, y1, x2, y2):
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


# - Defining a function to apply Gaussian blur to the face region
def apply_blur_to_face(image, x1, y1, x2, y2):
    face_region = image.crop((x1, y1, x2, y2))
    face_region = face_region.filter(ImageFilter.GaussianBlur(radius=20))
    image.paste(face_region, (x1, y1))
    

# - Defining a function to save and display the blurred image
def save_and_display_blurred_image(blurred_image, boxes, image_index):
    blurred_image.save(f"blurred_image_{image_index + 1}.png", format="PNG")
    st.image(blurred_image, caption=f"{len(boxes)} human faces are detected and blurred.", use_column_width=True)
    st.download_button(
        label=f"Download Blurred Image {image_index + 1}",
        data=open(f"blurred_image_{image_index + 1}.png", "rb").read(),
        file_name=f"blurred_image_{image_index + 1}.png",
        key=f"download_blurred_image_{image_index + 1}",
    )
    

# - Streamlit application topic
st.markdown("<h1 class='centered-heading'>Human Face Blurring App (YoloV8)</h1>", unsafe_allow_html=True)


# - Uploading multiple images
with st.form("my-form", clear_on_submit=True):
        uploaded_images = st.file_uploader("Upload an image or multiple images with human faces", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("submit")


# - Main
if uploaded_images:
    for i, uploaded_image in enumerate(uploaded_images):
        perform_object_detection_and_blur(uploaded_image, model, i)