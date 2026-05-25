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
        1. 自适应二值化 (Adaptive Thresholding)
        根据局部光照自动计算阈值，去除背景纹理，提取文字骨架。
        """
        img = self._load_image(image_source)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blockSize=11, C=2
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        return self._save_or_return(binary, output_path)

    def canny_edge_extraction(self, image_source, output_path=None):
        """
        2. 边缘检测 (Canny Edge Extraction)
        丢弃颜色纹理，仅提取高频轮廓，并反转为“白底黑线”以便识别。
        """
        img = self._load_image(image_source)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        # Invert to white background, black lines
        edges_inverted = 255 - edges
        return self._save_or_return(edges_inverted, output_path)

    def channel_extraction(self, image_source, channel='r', output_path=None):
        """
        3. 通道提取 (Channel Extraction)
        分离 R、G、B 或 HSV 中的 S (饱和度) 通道。
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
        4. JPEG 重编码 (JPEG Purify)
        通过有损压缩和解压，破坏对抗样本中肉眼不可见的高频微扰动。
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
        5. 色调分离/量化 (Posterization)
        大幅减少颜色数量（如降至 4 色），将渐变干扰合并为单一色块。
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
        6. 图像锐化 (Sharpening)
        增强图像边缘对比度。
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
        7. 异向拉伸 (Anisotropic Stretch)
        独立调整横向或纵向比例。
        """
        img = self._load_image(image_source)
        height, width = img.shape[:2]
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return self._save_or_return(resized, output_path)

    def grid_slice(self, image_source, rows=2, cols=2, output_dir=None):
        """
        8. 网格切片 (Grid Slice)
        将大图切割为多个局部小图。
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
        9. 局部直方图均衡化 (CLAHE)
        增强局部区域的对比度。
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
        10. 缩小分辨率2倍 (Downscale 2x)
        将图像长宽各缩小为原来的 1/2
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
        10. 缩小分辨率4倍 (Downscale 4x)
        将图像长宽各缩小为原来的 1/4 (即分辨率缩小 16 倍，长宽缩小 4 倍)
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
        11. 形态学闭运算 (Morphological Closing)
        修复针式打印点阵、连接断裂笔画。
        """
        img = self._load_image(image_source)
        
        # 1. 确保是灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 2. 颜色反转: OpenCV 形态学假设前景是白色的，而文档文字通常是黑色的。
        img_inv = cv2.bitwise_not(gray)
        
        # 3. 定义结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        # 4. 执行闭运算 (先膨胀后腐蚀)
        closed_inv = cv2.morphologyEx(img_inv, cv2.MORPH_CLOSE, kernel)
        
        # 5. 反转回“白底黑字”
        result = cv2.bitwise_not(closed_inv)
        
        return self._save_or_return(result, output_path)

    def blackhat_extraction(self, image_source, kernel_size=(15, 15), output_path=None):
        """
        12. 黑帽变换 (Black-Hat Transform)
        去除复杂背景底纹，提取深色文字。
        """
        img = self._load_image(image_source)
        
        # 1. 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 2. 定义结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        # 3. 执行黑帽变换
        # 结果 = 闭运算图 - 原图
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # 4. 增强结果: 归一化到 0-255
        normalized = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        
        # 5. 反转为符合人类阅读习惯的“白底黑字”
        result = cv2.bitwise_not(normalized)
        
        return self._save_or_return(result, output_path)
