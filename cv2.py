import cv2
# pip install opencv-python


def cv2imread(img_path) -> np.ndarray:
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return rgb_img


def cv2imwrite(rgb_img, img_path) -> None:
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, bgr_img)