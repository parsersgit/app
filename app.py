from io import BytesIO
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
POTHOLE_MODEL_PATH = APP_DIR / "weights" / "best.pt"
PEOPLE_MODEL_NAME = "yolo26n-seg.pt"
INFERENCE_DEVICE = "cpu"

POTHOLE_COLOR = (255, 92, 40)  # RGB
PERSON_COLOR = (35, 145, 255)  # RGB


st.set_page_config(
    page_title="Сегментация людей и дорожных ям",
    page_icon="🛣️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[YOLO, YOLO]:
    pothole_model = YOLO(str(POTHOLE_MODEL_PATH))
    people_model = YOLO(PEOPLE_MODEL_NAME)
    return pothole_model, people_model


def detection_count(result) -> int:
    return 0 if result.boxes is None else len(result.boxes)


def draw_result(
    image_rgb: np.ndarray,
    result,
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    canvas = image_rgb.copy()
    height, width = canvas.shape[:2]

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        for mask in masks:
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue

            blended = (
                canvas[mask_bool].astype(np.float32) * 0.58
                + np.asarray(color, dtype=np.float32) * 0.42
            )
            canvas[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)

            mask_uint8 = (mask_bool.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, color, 2)

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = box.tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {confidence:.0%}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            text_top = max(0, y1 - text_height - baseline - 6)
            cv2.rectangle(
                canvas,
                (x1, text_top),
                (min(width - 1, x1 + text_width + 8), y1),
                color,
                thickness=-1,
            )
            cv2.putText(
                canvas,
                text,
                (x1 + 4, max(text_height + 2, y1 - baseline - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return canvas


def run_inference(image: Image.Image, confidence: float):
    pothole_model, people_model = load_models()

    started_at = perf_counter()
    pothole_result = pothole_model.predict(
        source=image,
        conf=confidence,
        imgsz=640,
        device=INFERENCE_DEVICE,
        verbose=False,
    )[0]
    people_result = people_model.predict(
        source=image,
        classes=[0],  # класс person
        conf=confidence,
        imgsz=640,
        device=INFERENCE_DEVICE,
        verbose=False,
    )[0]
    elapsed = perf_counter() - started_at

    rendered = np.asarray(image).copy()
    # Встроенный шрифт OpenCV не поддерживает кириллицу, поэтому подписи
    # на изображении оставлены короткими и понятными на английском.
    rendered = draw_result(rendered, pothole_result, POTHOLE_COLOR, "pothole")
    rendered = draw_result(rendered, people_result, PERSON_COLOR, "person")

    return (
        rendered,
        detection_count(pothole_result),
        detection_count(people_result),
        elapsed,
    )


st.title("Сегментация людей и дорожных ям")
st.write(
    "Две модели обрабатывают одно изображение: готовая модель находит людей, "
    "а дообученная — дорожные ямы."
)

with st.sidebar:
    st.header("Настройки")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Чем выше значение, тем меньше сомнительных объектов покажет модель.",
    )
    st.markdown("🟧 **Ямы**  \n🟦 **Люди**")
    st.caption("Обработка выполняется локально на процессоре.")

if not POTHOLE_MODEL_PATH.exists():
    st.error(
        "Не найден файл модели ям. Скопируйте обученный best.pt в папку "
        "weights и перезапустите приложение."
    )
    st.code("weights/best.pt")
    st.stop()

source_mode = st.radio(
    "Источник изображения",
    ("Файл", "Камера"),
    horizontal=True,
)

if source_mode == "Файл":
    source_file = st.file_uploader(
        "Выберите фотографию",
        type=("jpg", "jpeg", "png", "webp"),
    )
else:
    source_file = st.camera_input("Сделайте фотографию")

if source_file is not None:
    try:
        input_image = ImageOps.exif_transpose(Image.open(source_file)).convert("RGB")
    except Exception as error:
        st.error(f"Не удалось открыть изображение: {error}")
        st.stop()

    if st.button("Запустить сегментацию", type="primary"):
        try:
            with st.spinner("Модели обрабатывают изображение..."):
                output_image, potholes, people, elapsed = run_inference(
                    input_image, confidence
                )
        except Exception as error:
            st.error(
                "Не удалось запустить модели. Проверьте подключение к интернету "
                "при первом запуске и файл weights/best.pt."
            )
            st.exception(error)
            st.stop()

        pothole_metric, people_metric, time_metric = st.columns(3)
        pothole_metric.metric("Найдено ям", potholes)
        people_metric.metric("Найдено людей", people)
        time_metric.metric("Время обработки", f"{elapsed:.2f} с")

        original_column, result_column = st.columns(2)
        with original_column:
            st.subheader("Исходное изображение")
            st.image(input_image, use_container_width=True)
        with result_column:
            st.subheader("Результат")
            st.image(output_image, use_container_width=True)

        output_buffer = BytesIO()
        Image.fromarray(output_image).save(output_buffer, format="JPEG", quality=95)
        st.download_button(
            "Скачать результат",
            data=output_buffer.getvalue(),
            file_name="segmentation_result.jpg",
            mime="image/jpeg",
        )
