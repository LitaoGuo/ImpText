import cv2
import numpy as np
from PIL import Image
import os

class ImageRobustnessToolkit:
    def _load_image(self, image_source):
        """
        Helper to load image from path or return existing numpy array.
        Returns BGR numpy array (OpenCV format).
        """
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image not found: {image_source}")
            # cv2.imread handles most formats, but let's be robust with unicode paths if needed
            # For now standard imread
            img = cv2.imread(image_source)
            if img is None:
                raise ValueError(f"Failed to load image: {image_source}")
            return img
        elif isinstance(image_source, np.ndarray):
            return image_source.copy()
        elif isinstance(image_source, Image.Image):
            return cv2.cvtColor(np.array(image_source), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("Unsupported image source type")

    def _save_or_return(self, img, output_path):
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, img)
            return output_path
        return img

    def adaptive_thresholding(self, image_source, output_path=None):
        """
        Adaptive thresholding.
        Estimate local thresholds to suppress background texture and isolate text strokes.
        """
        img = self._load_image(image_source)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blockSize=11, C=2
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        return self._save_or_return(binary, output_path)

    def canny_edge_extraction(self, image_source, output_path=None):
        """
        Canny edge extraction.
        Keep high-frequency contours while discarding color and texture distractions.
        """
        img = self._load_image(image_source)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        # Invert to white background, black lines
        edges_inverted = 255 - edges
        return self._save_or_return(edges_inverted, output_path)

    def channel_extraction(self, image_source, channel='r', output_path=None):
        """
        Channel extraction.
        Extract R, G, B, or the saturation channel from HSV.
        channel: 'r', 'g', 'b', or 's'
        """
        img = self._load_image(image_source)
        if channel.lower() in ['r', 'g', 'b']:
            b, g, r = cv2.split(img)
            if channel.lower() == 'r':
                res = r
            elif channel.lower() == 'g':
                res = g
            else:
                res = b
        elif channel.lower() == 's':
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            res = s
        else:
            raise ValueError("Channel must be 'r', 'g', 'b', or 's'")
            
        return self._save_or_return(res, output_path)

    def jpeg_purify(self, image_source, quality=50, output_path=None):
        """
        JPEG purify.
        Re-encode through lossy compression to suppress high-frequency perturbations.
        """
        img = self._load_image(image_source)
        # Encode to buffer
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if not result:
            raise ValueError("Failed to encode image")
        # Decode back
        decimg = cv2.imdecode(encimg, 1)
        return self._save_or_return(decimg, output_path)

    def posterization(self, image_source, levels=4, output_path=None):
        """
        Posterization.
        Reduce color levels so distracting gradients merge into larger flat regions.
        """
        img = self._load_image(image_source)
        # Simple quantization: floor(value / (256/levels)) * (256/levels)
        # But let's use a slightly better mapping to center values
        indices = np.arange(0, 256)
        divider = np.linspace(0, 255, levels+1)[1]
        quantiz = np.int16(indices / divider) * divider
        quantiz = np.clip(quantiz, 0, 255)
        # Adjust to center of bin? Or just start. Let's keep simple.
        
        # Look Up Table
        lut = quantiz.astype('uint8')
        res = cv2.LUT(img, lut)
        return self._save_or_return(res, output_path)

    def sharpening(self, image_source, output_path=None):
        """
        Sharpening.
        Increase edge contrast.
        """
        img = self._load_image(image_source)
        # Kernel for sharpening
        kernel = np.array([[0, -1, 0], 
                           [-1, 5,-1], 
                           [0, -1, 0]])
        sharpened = cv2.filter2D(img, -1, kernel)
        return self._save_or_return(sharpened, output_path)

    def anisotropic_stretch(self, image_source, scale_x=1.5, scale_y=1.0, output_path=None):
        """
        Anisotropic stretch.
        Independently adjust horizontal or vertical scale.
        """
        img = self._load_image(image_source)
        height, width = img.shape[:2]
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return self._save_or_return(resized, output_path)

    def grid_slice(self, image_source, rows=2, cols=2, output_dir=None):
        """
        Grid slice.
        Split a large image into local patches.
        Returns a list of images (or paths if output_dir provided).
        """
        img = self._load_image(image_source)
        height, width = img.shape[:2]
        h_step = height // rows
        w_step = width // cols
        
        slices = []
        paths = []
        
        for r in range(rows):
            for c in range(cols):
                y1 = r * h_step
                y2 = (r + 1) * h_step if r < rows - 1 else height
                x1 = c * w_step
                x2 = (c + 1) * w_step if c < cols - 1 else width
                
                slice_img = img[y1:y2, x1:x2]
                
                if output_dir:
                    filename = f"slice_{r}_{c}.png"
                    path = os.path.join(output_dir, filename)
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(path, slice_img)
                    paths.append(path)
                else:
                    slices.append(slice_img)
                    
        return paths if output_dir else slices

    def clahe(self, image_source, clip_limit=2.0, tile_grid_size=(8,8), output_path=None):
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Enhance local contrast.
        """
        img = self._load_image(image_source)
        # CLAHE is typically applied to Lightness channel in LAB or just Gray
        # Let's support both. If input is BGR, convert to LAB, apply to L, convert back.
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return self._save_or_return(final, output_path)

    def downscale_2x(self, image_source, output_path=None):
        """
        Downscale 2x.
        Reduce image width and height to one half of the original size.
        """
        img = self._load_image(image_source)
        height, width = img.shape[:2]
        new_width = width // 2
        new_height = height // 2
        
        # Ensure at least 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return self._save_or_return(resized, output_path)

    def downscale_4x(self, image_source, output_path=None):
        """
        Downscale 4x.
        Reduce image width and height to one quarter of the original size.
        """
        img = self._load_image(image_source)
        height, width = img.shape[:2]
        new_width = width // 4
        new_height = height // 4
        
        # Ensure at least 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return self._save_or_return(resized, output_path)

    def morphological_closing(self, image_source, kernel_size=(3, 3), output_path=None):
        """
        Morphological closing.
        Connect broken strokes and fill small gaps.
        """
        img = self._load_image(image_source)
        
        # 1. Ensure grayscale input.
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 2. Invert colors because OpenCV morphology treats foreground as white.
        img_inv = cv2.bitwise_not(gray)
        
        # 3. Define the structuring element.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        # 4. Apply closing: dilation followed by erosion.
        closed_inv = cv2.morphologyEx(img_inv, cv2.MORPH_CLOSE, kernel)
        
        # 5. Invert back to dark text on a light background.
        result = cv2.bitwise_not(closed_inv)
        
        return self._save_or_return(result, output_path)

    def blackhat_extraction(self, image_source, kernel_size=(15, 15), output_path=None):
        """
        Black-hat transform.
        Suppress uneven background texture and extract dark text.
        """
        img = self._load_image(image_source)
        
        # 1. Convert to grayscale.
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 2. Define the structuring element.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        # 3. Apply black-hat transform: closing image minus original image.
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # 4. Normalize to 0-255.
        normalized = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        
        # 5. Invert to dark text on a light background.
        result = cv2.bitwise_not(normalized)
        
        return self._save_or_return(result, output_path)
