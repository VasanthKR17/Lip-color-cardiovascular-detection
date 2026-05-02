import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

class LipExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # Initialize FaceMesh object with static_image_mode as True for image processing
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Mediapipe lip indices
        # Outer lips contour
        self.LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        # Inner lips (optional but good context)
        self.LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

    def extract_lips_and_color(self, image):
        """
        Extracts the lip region from an image and computes the average HSV color.
        :param image: PIL Image or OpenCV NumPy array (RGB)
        :return: Extracted raw lip image, Masked image cutout, Average HSV, Average RGB
        """
        if isinstance(image, Image.Image):
            rgb_img = np.array(image.convert('RGB'))
        else:
            rgb_img = image
            
        h, w, _ = rgb_img.shape
        results = self.face_mesh.process(rgb_img)

        # Return Nones if no face is detected
        if not results.multi_face_landmarks:
            return None, None, None, None

        landmarks = results.multi_face_landmarks[0]
        
        # Get coordinates for the outer lips outline
        lip_coords = []
        for index in self.LIPS_OUTER:
            pt = landmarks.landmark[index]
            x, y = int(pt.x * w), int(pt.y * h)
            lip_coords.append((x, y))
            
        lip_coords = np.array(lip_coords, dtype=np.int32)
        
        # Create a contour mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [lip_coords], 255)
        
        # Apply mask to RGB image
        masked_img = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
        
        # Determine Bounding box for cropping to focus entirely on the lips
        x_min, y_min = np.min(lip_coords, axis=0)
        x_max, y_max = np.max(lip_coords, axis=0)
        
        # Add slight padding for visual aesthetics
        pad = 5
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)
        
        # Crop the lip section based on bounding box
        cropped_masked_lip = masked_img[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        raw_cropped_lip = rgb_img[y_min:y_max, x_min:x_max]
        
        # Calculate Average colors strictly inside the lip boundaries
        # Convert cropped image to HSV for hue analysis
        hsv_cropped = cv2.cvtColor(cropped_masked_lip, cv2.COLOR_RGB2HSV)
        
        # Use simple mean pixel values of non-zero mask pixels
        mean_hsv = cv2.mean(hsv_cropped, mask=cropped_mask)[:3]
        mean_rgb = cv2.mean(cropped_masked_lip, mask=cropped_mask)[:3]
        
        return raw_cropped_lip, cropped_masked_lip, mean_hsv, mean_rgb
